import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from blockmanager import BlockManager
from cache import PagedKVCache
from engine import Engine
from model import ModelConfig, Qwen25_15B
from weights import load_weights


BASE_TEXT = (
    "Pico-vLLM H200 profiling workload. "
    "This request stresses long-context prefill, paged KV cache writes, "
    "decode attention reads, and scheduler interaction under concurrency. "
)


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * q)))
    return ordered[idx]


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def make_prompt(tokenizer, target_tokens: int) -> str:
    base_ids = tokenizer.encode(BASE_TEXT)
    if not base_ids:
        raise RuntimeError("Tokenizer produced an empty base prompt.")
    repeats = (target_tokens + len(base_ids) - 1) // len(base_ids)
    prompt_ids = (base_ids * repeats)[:target_tokens]
    return tokenizer.decode(prompt_ids)


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def collect_requests(engine: Engine):
    requests = []
    requests.extend(engine.scheduler.waiting)
    requests.extend(engine.scheduler.prefilling)
    requests.extend(engine.scheduler.decoding)
    requests.extend(engine.scheduler.finished)
    return {request.request_id: request for request in requests}


def collect_env(rank: int, local_rank: int, world_size: int, args) -> dict:
    env = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "use_cuda_graph": not args.disable_cuda_graph,
        "enable_prefix_cache": not args.disable_prefix_cache,
        "dtype": "bfloat16",
    }
    if torch.cuda.is_available():
        env.update(
            {
                "gpu_name": torch.cuda.get_device_name(local_rank),
                "device_count": torch.cuda.device_count(),
                "cuda_runtime": torch.version.cuda,
            }
        )
    return env


def create_engine(args, device: torch.device, rank: int, world_size: int, max_prompt_len: int, max_concurrency: int):
    cfg = ModelConfig(tp_size=world_size, tp_rank=rank)
    if cfg.local_num_key_value_heads <= 0:
        raise ValueError(
            "Current Qwen2.5-1.5B config has too few KV heads for this TP size. "
            f"num_key_value_heads={cfg.num_key_value_heads}, tp_size={world_size}. "
            "Use TP<=2 for this model, or implement KV head replication for larger TP."
        )

    model = Qwen25_15B(cfg)
    model = load_weights(
        model,
        args.weights,
        tp_size=world_size,
        tp_rank=rank,
        dtype=torch.bfloat16,
    )
    model = model.to(torch.bfloat16).to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.weights)
    max_seq_len = max_prompt_len + args.max_new_tokens + 1
    needed_blocks = ((max_seq_len + cfg.BLOCK_SIZE - 1) // cfg.BLOCK_SIZE) * max_concurrency
    num_gpu_blocks = args.num_gpu_blocks or max(needed_blocks + args.block_slack, 1024)

    block_manager = BlockManager(
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=0,
        block_size=cfg.BLOCK_SIZE,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.local_num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    cache_kwargs = {
        "block_manager": block_manager,
        "num_layers": cfg.num_hidden_layers,
        "max_seq_len": max_seq_len,
        "num_kv_heads": cfg.local_num_key_value_heads,
        "head_dim": cfg.head_dim,
        "device": device,
        "dtype": torch.bfloat16,
    }
    engine = Engine(
        model=model,
        tokenizer=tokenizer,
        block_manager=block_manager,
        cache_cls=PagedKVCache,
        cache_kwargs=cache_kwargs,
        device=device,
        use_cuda_graph=not args.disable_cuda_graph,
        max_batch_size=max(args.max_batch_size, max_concurrency),
        rank=rank,
        local_tp_size=world_size,
        role="pd",
        enable_prefix_cache=not args.disable_prefix_cache,
        detokenize_outputs=getattr(args, "detokenize_outputs", True),
        ignore_eos=getattr(args, "ignore_eos", False),
    )
    engine.scheduler.max_num_seqs = max(args.max_num_seqs, max_concurrency)
    return engine, tokenizer, block_manager


def run_workload(
    args,
    engine: Engine,
    tokenizer,
    block_manager: BlockManager,
    prompt_len: int,
    concurrency: int,
    device: torch.device,
    prompts: list[str] | None = None,
):
    if prompts is None:
        prompts = [make_prompt(tokenizer, prompt_len) for _ in range(concurrency)]
    if len(prompts) != concurrency:
        raise ValueError(f"Expected {concurrency} prompts, got {len(prompts)}")
    prompt_token_counts = [len(tokenizer.encode(prompt)) for prompt in prompts]
    request_ids = [
        engine.submit(prompt, max_new_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0)
        for prompt in prompts
    ]
    engine.mark_finished()

    generated_lens = {request_id: 0 for request_id in request_ids}
    first_token_ms: dict[int, float] = {}
    last_token_ms: dict[int, float] = {}
    itl_ms: dict[int, list[float]] = {request_id: [] for request_id in request_ids}
    step_ms: list[float] = []
    active_decode_counts: list[int] = []
    completed = 0

    cuda_sync(device)
    if args.cuda_profiler_api and device.type == "cuda":
        torch.cuda.cudart().cudaProfilerStart()
    start = time.perf_counter()

    while not engine.is_done():
        if args.nvtx and device.type == "cuda":
            torch.cuda.nvtx.range_push(f"h200_engine_step_{len(step_ms)}")
        step_start = time.perf_counter()
        completed_outputs = engine.step()
        cuda_sync(device)
        step_end = time.perf_counter()
        if args.nvtx and device.type == "cuda":
            torch.cuda.nvtx.range_pop()

        now_ms = (step_end - start) * 1000.0
        step_ms.append((step_end - step_start) * 1000.0)
        active_decode_counts.append(engine.scheduler.num_decoding)
        completed += len(completed_outputs)

        for request_id, request in collect_requests(engine).items():
            if request_id not in generated_lens:
                continue
            new_len = len(request.generated_ids)
            old_len = generated_lens[request_id]
            if new_len > old_len:
                if old_len == 0:
                    first_token_ms[request_id] = now_ms
                else:
                    itl_ms[request_id].append(now_ms - last_token_ms[request_id])
                last_token_ms[request_id] = now_ms
                generated_lens[request_id] = new_len

    if args.cuda_profiler_api and device.type == "cuda":
        torch.cuda.cudart().cudaProfilerStop()
    cuda_sync(device)
    total_ms = (time.perf_counter() - start) * 1000.0

    all_itl = [value for values in itl_ms.values() for value in values]
    total_output_tokens = sum(generated_lens.values())
    result = {
        "prompt_len": prompt_len,
        "concurrency": concurrency,
        "max_new_tokens": args.max_new_tokens,
        "prompt_token_counts": prompt_token_counts,
        "prompt_token_count": prompt_token_counts[0] if len(set(prompt_token_counts)) == 1 else 0,
        "completed": completed,
        "total_output_tokens_observed": total_output_tokens,
        "total_ms": total_ms,
        "output_tokens_per_s": total_output_tokens / total_ms * 1000.0 if total_ms > 0 else 0.0,
        "ttft_ms": summarize(list(first_token_ms.values())),
        "itl_ms": summarize(all_itl),
        "step_ms": summarize(step_ms),
        "steps": len(step_ms),
        "max_active_decoding": max(active_decode_counts) if active_decode_counts else 0,
        "gpu_blocks_total": block_manager.num_physical_gpu_blocks,
        "gpu_blocks_free": len(block_manager.gpu_free_blocks),
    }
    if device.type == "cuda":
        result["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["cuda_max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    return result


def main():
    parser = argparse.ArgumentParser(description="H200 long-context baseline benchmark for Pico-vLLM.")
    parser.add_argument("--weights", default="./weights")
    parser.add_argument("--prompt-lens", default="1024,4096,8192")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-gpu-blocks", type=int, default=0)
    parser.add_argument("--block-slack", type=int, default=256)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--cuda-profiler-api", action="store_true")
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    prompt_lens = parse_csv_ints(args.prompt_lens)
    concurrencies = parse_csv_ints(args.concurrency)
    if not prompt_lens or not concurrencies:
        raise ValueError("--prompt-lens and --concurrency must not be empty.")

    env = collect_env(rank, local_rank, world_size, args)

    output_path = args.output
    if not output_path:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = f"logs/benchmarks/h200_baseline/{stamp}_rank{rank}.jsonl"
    if rank == 0:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "env", **env}, ensure_ascii=False) + "\n")

    for prompt_len in prompt_lens:
        for concurrency in concurrencies:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            engine, tokenizer, block_manager = create_engine(
                args,
                device=device,
                rank=rank,
                world_size=world_size,
                max_prompt_len=prompt_len,
                max_concurrency=concurrency,
            )
            result = run_workload(args, engine, tokenizer, block_manager, prompt_len, concurrency, device)
            result = {"type": "result", **env, **result}
            if rank == 0:
                print(json.dumps(result, ensure_ascii=False), flush=True)
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            del engine
            del block_manager
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
