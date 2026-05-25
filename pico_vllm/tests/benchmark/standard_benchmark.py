#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PICO_DIR = REPO_ROOT / "pico_vllm"
BENCHMARKS_DIR = PICO_DIR / "benchmarks"
sys.path.insert(0, str(PICO_DIR))
sys.path.insert(0, str(BENCHMARKS_DIR))

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

BASE_TEXT = (
    "Pico-vLLM engine benchmark workload. "
    "The same token-id trace is used for Pico-vLLM and vLLM. "
    "This prompt is repeated and token-trimmed to create deterministic input sizes. "
)


@dataclass(frozen=True)
class WorkloadCase:
    input_tokens: int
    output_tokens: int
    concurrency: int


@dataclass(frozen=True)
class RequestTrace:
    prompt_token_ids: list[list[int]]
    prompt_token_count: int
    prompt_token_ids_sha256: str
    prompt_text_sha256: str


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cuda_sync(device: torch.device | None = None) -> None:
    if torch.cuda.is_available():
        if device is None:
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize(device)


def make_prompt_token_ids(tokenizer, target_tokens: int) -> list[int]:
    base_ids = tokenizer.encode(BASE_TEXT)
    if not base_ids:
        raise RuntimeError("Tokenizer produced an empty base prompt.")
    repeats = (target_tokens + len(base_ids) - 1) // len(base_ids)
    prompt_ids = (base_ids * repeats)[:target_tokens]
    if len(prompt_ids) != target_tokens:
        raise RuntimeError(f"Unable to build exact token trace: target={target_tokens}, actual={len(prompt_ids)}")
    return list(prompt_ids)


def token_ids_hash(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_trace(tokenizer, case: WorkloadCase) -> RequestTrace:
    prompt_ids = make_prompt_token_ids(tokenizer, case.input_tokens)
    prompt_text = tokenizer.decode(prompt_ids)
    return RequestTrace(
        prompt_token_ids=[list(prompt_ids) for _ in range(case.concurrency)],
        prompt_token_count=len(prompt_ids),
        prompt_token_ids_sha256=token_ids_hash(prompt_ids),
        prompt_text_sha256=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
    )


def env_record(args, rank: int, local_rank: int, world_size: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "type": "env",
        "timestamp_utc": utc_now(),
        "benchmark_suite": "engine",
        "backend": args.backend,
        "engine": args.engine,
        "model": args.model,
        "weights": args.weights,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "use_cuda_graph": not args.disable_cuda_graph,
        "enable_prefix_cache": args.enable_prefix_cache,
        "detokenize_outputs": False,
        "ignore_eos": True,
    }
    if args.backend == "vllm":
        try:
            import vllm

            record["vllm"] = vllm.__version__
        except Exception as exc:
            record["vllm_import_error"] = str(exc)
    if torch.cuda.is_available():
        record.update(
            {
                "gpu_name": torch.cuda.get_device_name(local_rank),
                "device_count": torch.cuda.device_count(),
            }
        )
    return record


def collect_pico_requests(engine) -> dict[int, Any]:
    requests = []
    requests.extend(engine.scheduler.waiting)
    requests.extend(engine.scheduler.prefilling)
    requests.extend(engine.scheduler.decoding)
    requests.extend(engine.scheduler.finished)
    return {request.request_id: request for request in requests}

def run_pico_engine_workload(args, case: WorkloadCase, trace: RequestTrace, rank: int, world_size: int, local_rank: int) -> dict[str, Any] | None:
    import benchmark_h200_baseline as pico_baseline

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    pico_args = argparse.Namespace(
        weights=args.weights,
        max_new_tokens=case.output_tokens,
        num_gpu_blocks=args.num_gpu_blocks,
        block_slack=args.block_slack,
        max_batch_size=max(args.max_batch_size, case.concurrency),
        max_num_seqs=max(args.max_num_seqs, case.concurrency),
        disable_cuda_graph=args.disable_cuda_graph,
        disable_prefix_cache=not args.enable_prefix_cache,
        cuda_profiler_api=False,
        nvtx=False,
        detokenize_outputs=False,
        ignore_eos=True,
    )
    engine, _tokenizer, block_manager = pico_baseline.create_engine(
        pico_args,
        device=device,
        rank=rank,
        world_size=world_size,
        max_prompt_len=case.input_tokens,
        max_concurrency=case.concurrency,
    )

    request_ids = [
        engine.submit_token_ids(prompt_ids, max_new_tokens=case.output_tokens, temperature=0.0, top_p=1.0)
        for prompt_ids in trace.prompt_token_ids
    ]
    engine.mark_finished()

    def current_output_lens() -> list[int]:
        requests = collect_pico_requests(engine)
        return [len(requests[request_id].generated_ids) for request_id in request_ids if request_id in requests]

    def unfinished_request_count() -> int:
        return sum(length < case.output_tokens for length in current_output_lens())

    cuda_sync(device)
    start = time.perf_counter()
    steps = 0
    active_counts: list[int] = []
    while True:
        active_counts.append(unfinished_request_count())
        engine.step()
        steps += 1
        output_lens = current_output_lens()
        if len(output_lens) == len(request_ids) and all(length >= case.output_tokens for length in output_lens):
            break
        if engine.is_done():
            break
    cuda_sync(device)
    total_ms = (time.perf_counter() - start) * 1000.0

    # Keep request cleanup out of the measured generation window.
    while not engine.is_done():
        engine.step()

    requests = collect_pico_requests(engine)
    output_lens = [len(requests[request_id].generated_ids) for request_id in request_ids if request_id in requests]
    raw_output_tokens = sum(output_lens)
    completed = sum(length >= case.output_tokens for length in output_lens)

    result: dict[str, Any] = {
        "completed": completed,
        "raw_completed": completed,
        "total_output_tokens_observed": raw_output_tokens,
        "engine_total_ms": total_ms,
        "engine_steps": steps,
        "max_active_requests": max(active_counts) if active_counts else 0,
        "per_request_output_tokens": output_lens,
        "gpu_blocks_total": block_manager.num_physical_gpu_blocks,
        "gpu_blocks_free": len(block_manager.gpu_free_blocks),
    }
    if device.type == "cuda":
        result["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["cuda_max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device)

    del engine
    del block_manager
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if rank != 0:
        return None
    return normalize_result(args, case, trace, result)

def run_vllm_engine_workload(args, case: WorkloadCase, trace: RequestTrace) -> dict[str, Any]:
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    max_model_len = args.vllm_max_model_len or (case.input_tokens + case.output_tokens + 1)
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.weights,
        "dtype": args.vllm_dtype,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_seqs": max(args.max_num_seqs, case.concurrency),
        "enable_prefix_caching": args.enable_prefix_cache,
        "enforce_eager": args.disable_cuda_graph,
        "disable_log_stats": True,
    }
    llm = LLM(**llm_kwargs)
    # echo: sampling params will effect inference speed, need a solid config
    def make_sampling_params() -> SamplingParams:
        return SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=case.output_tokens,
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

    for index, prompt_ids in enumerate(trace.prompt_token_ids):
        llm.llm_engine.add_request(
            str(index),
            TokensPrompt(prompt_token_ids=list(prompt_ids)),
            make_sampling_params(),
        )

    cuda_sync()
    start = time.perf_counter()
    steps = 0
    max_active = 0
    completed = 0
    output_lens: dict[str, int] = {}
    while llm.llm_engine.has_unfinished_requests():
        if hasattr(llm.llm_engine, "get_num_unfinished_requests"):
            max_active = max(max_active, int(llm.llm_engine.get_num_unfinished_requests()))
        step_outputs = llm.llm_engine.step()
        steps += 1
        for output in step_outputs:
            if not getattr(output, "finished", False):
                continue
            completed += 1
            output_lens[output.request_id] = sum(len(completion.token_ids) for completion in output.outputs)
    cuda_sync()
    total_ms = (time.perf_counter() - start) * 1000.0

    per_request_output_tokens = [output_lens.get(str(index), 0) for index in range(case.concurrency)]
    raw_output_tokens = sum(per_request_output_tokens)
    result: dict[str, Any] = {
        "completed": completed,
        "raw_completed": completed,
        "total_output_tokens_observed": raw_output_tokens,
        "engine_total_ms": total_ms,
        "engine_steps": steps,
        "max_active_requests": max_active,
        "per_request_output_tokens": per_request_output_tokens,
        "vllm_engine_kwargs": llm_kwargs,
    }
    if torch.cuda.is_available():
        result["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["cuda_max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return normalize_result(args, case, trace, result)


def normalize_result(args, case: WorkloadCase, trace: RequestTrace, result: dict[str, Any]) -> dict[str, Any]:
    expected_tokens = case.concurrency * case.output_tokens
    raw_output_tokens = int(result.get("total_output_tokens_observed", 0))
    raw_completed = int(result.get("raw_completed", result.get("completed", 0)))
    total_ms = float(result.get("engine_total_ms", 0.0))
    return {
        "type": "result",
        "timestamp_utc": utc_now(),
        "benchmark_suite": "engine",
        "backend": args.backend,
        "engine": args.engine,
        "model": args.model,
        "input_tokens": case.input_tokens,
        "output_tokens": case.output_tokens,
        "concurrency": case.concurrency,
        "prompt_token_count": trace.prompt_token_count,
        "prompt_token_ids_sha256": trace.prompt_token_ids_sha256,
        "prompt_text_sha256": trace.prompt_text_sha256,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "detokenize_outputs": False,
        "enable_prefix_cache": args.enable_prefix_cache,
        "completed": min(raw_completed, case.concurrency),
        "raw_completed": raw_completed,
        "expected_output_tokens": expected_tokens,
        "total_output_tokens_observed": raw_output_tokens,
        "raw_total_output_tokens_observed": raw_output_tokens,
        "engine_total_ms": total_ms,
        "engine_output_tokens_per_s": raw_output_tokens / total_ms * 1000.0 if total_ms > 0 else 0.0,
        "engine_request_per_s": raw_completed / total_ms * 1000.0 if total_ms > 0 else 0.0,
        "engine_steps": result.get("engine_steps", 0),
        "max_active_requests": result.get("max_active_requests", 0),
        "cuda_max_memory_allocated_bytes": result.get("cuda_max_memory_allocated_bytes", 0),
        "cuda_max_memory_reserved_bytes": result.get("cuda_max_memory_reserved_bytes", 0),
        "raw": result,
    }


def write_result(output: Path, record: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_result_rows(jsonl_path: Path) -> list[dict[str, Any]]:
    if not jsonl_path.exists():
        return []
    return [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]


def write_csv(jsonl_path: Path, csv_path: Path) -> None:
    rows = [row for row in load_result_rows(jsonl_path) if row.get("type") == "result"]
    if not rows:
        return
    fields = [
        "engine", "backend", "model", "input_tokens", "prompt_token_count", "output_tokens", "concurrency",
        "completed", "raw_completed", "expected_output_tokens", "total_output_tokens_observed",
        "engine_total_ms", "engine_output_tokens_per_s", "engine_request_per_s", "engine_steps",
        "max_active_requests", "cuda_max_memory_allocated_bytes", "cuda_max_memory_reserved_bytes",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(jsonl_path: Path, md_path: Path) -> None:
    rows = [row for row in load_result_rows(jsonl_path) if row.get("type") == "result"]
    if not rows:
        return
    headers = ["engine", "in", "out", "conc", "tok/s", "req/s", "ms", "steps", "completed"]
    lines = ["# Engine Benchmark Summary", "", "| " + " | ".join(headers) + " |", "|" + "|".join([":---"] * len(headers)) + "|"]
    for row in rows:
        lines.append(
            "| {engine} | {input_tokens} | {output_tokens} | {concurrency} | {tps:.2f} | {rps:.2f} | {ms:.2f} | {steps} | {completed}/{concurrency} |".format(
                engine=row.get("engine"),
                input_tokens=row.get("input_tokens"),
                output_tokens=row.get("output_tokens"),
                concurrency=row.get("concurrency"),
                tps=row.get("engine_output_tokens_per_s", 0.0),
                rps=row.get("engine_request_per_s", 0.0),
                ms=row.get("engine_total_ms", 0.0),
                steps=row.get("engine_steps", 0),
                completed=row.get("completed", 0),
            )
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(jsonl_path: Path, png_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = [row for row in load_result_rows(jsonl_path) if row.get("type") == "result"]
    if not rows:
        return
    labels = [f"{r['engine']}\nin={r['input_tokens']} out={r['output_tokens']} c={r['concurrency']}" for r in rows]
    tps = [r.get("engine_output_tokens_per_s", 0.0) for r in rows]
    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(rows) * 1.2), 4))
    ax.bar(labels, tps)
    ax.set_title("Engine output tokens/s")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure engine benchmark for Pico-vLLM and vLLM.")
    parser.add_argument("--backend", choices=["pico", "vllm"], default="pico")
    parser.add_argument("--engine", default="pico", help="Result label, e.g. pico or vllm.")
    parser.add_argument("--model", default="./weights", help="Model path/label. Used by vLLM as model path.")
    parser.add_argument("--weights", default="./weights", help="Local tokenizer/weights path used for trace construction and Pico.")
    parser.add_argument("--input-lens", default="128,512,2048")
    parser.add_argument("--output-lens", default="32")
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--num-gpu-blocks", type=int, default=0)
    parser.add_argument("--block-slack", type=int, default=256)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--enable-prefix-cache", action="store_true", help="Enable prefix cache. Disabled by default for raw engine comparison.")
    parser.add_argument("--vllm-dtype", default="bfloat16")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-max-model-len", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Regenerate CSV/Markdown/PNG reports from --output JSONL.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.report_only:
        if not args.output:
            raise ValueError("--report-only requires --output")
        output = Path(args.output)
        write_csv(output, output.with_suffix(".csv"))
        write_markdown(output, output.with_suffix(".md"))
        write_plot(output, output.with_suffix(".png"))
        return 0
    # echo: generate workload shape
    input_lens = parse_csv_ints(args.input_lens)
    output_lens = parse_csv_ints(args.output_lens)
    concurrencies = parse_csv_ints(args.concurrency)
    cases = [WorkloadCase(i, o, c) for i in input_lens for o in output_lens for c in concurrencies]
    if not cases:
        raise ValueError("No benchmark cases selected.")
    # echo: rank config
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.backend == "pico" and world_size > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    # echo: tokenizer config
    tokenizer = AutoTokenizer.from_pretrained(args.weights)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = Path(args.output or f"logs/benchmarks/standard/{stamp}_{args.engine}_rank{rank}.jsonl")
    if rank == 0:
        write_result(output, env_record(args, rank, local_rank, world_size))

    for case in cases:
        trace = make_trace(tokenizer, case)
        if args.backend == "pico":
            record = run_pico_engine_workload(args, case, trace, rank, world_size, local_rank)
        else:
            record = run_vllm_engine_workload(args, case, trace) if rank == 0 else None
        if record is not None:
            print(json.dumps(record, ensure_ascii=False), flush=True)
            write_result(output, record)

    if args.backend == "pico" and world_size > 1:
        dist.destroy_process_group()

    if rank == 0 and not args.no_report:
        write_csv(output, output.with_suffix(".csv"))
        write_markdown(output, output.with_suffix(".md"))
        write_plot(output, output.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
