#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
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

import benchmark_h200_baseline as pico_baseline

BASE_TEXT = (
    "Pico-vLLM standard benchmark workload. "
    "The same request shape is used for Pico-vLLM, vLLM, and SGLang. "
    "This prompt is repeated and token-trimmed to create deterministic input sizes. "
)


@dataclass(frozen=True)
class WorkloadCase:
    input_tokens: int
    output_tokens: int
    concurrency: int


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


def make_prompt(tokenizer, target_tokens: int) -> str:
    base_ids = tokenizer.encode(BASE_TEXT)
    if not base_ids:
        raise RuntimeError("Tokenizer produced an empty base prompt.")
    repeats = (target_tokens + len(base_ids) - 1) // len(base_ids)
    prompt_ids = (base_ids * repeats)[:target_tokens]
    return tokenizer.decode(prompt_ids)


def env_record(args, rank: int, local_rank: int, world_size: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "type": "env",
        "timestamp_utc": utc_now(),
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
    }
    if torch.cuda.is_available():
        record.update(
            {
                "gpu_name": torch.cuda.get_device_name(local_rank),
                "device_count": torch.cuda.device_count(),
            }
        )
    return record


def run_pico_case(args, case: WorkloadCase, rank: int, world_size: int, local_rank: int) -> dict[str, Any] | None:
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
        disable_prefix_cache=args.disable_prefix_cache,
        cuda_profiler_api=False,
        nvtx=False,
    )
    engine, tokenizer, block_manager = pico_baseline.create_engine(
        pico_args,
        device=device,
        rank=rank,
        world_size=world_size,
        max_prompt_len=case.input_tokens,
        max_concurrency=case.concurrency,
    )
    result = pico_baseline.run_workload(
        pico_args,
        engine,
        tokenizer,
        block_manager,
        case.input_tokens,
        case.concurrency,
        device,
    )
    del engine
    del block_manager
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if rank != 0:
        return None
    return normalize_result(args, case, result)


def decode_sse_line(line: bytes) -> dict[str, Any] | None:
    text = line.decode("utf-8", errors="replace").strip()
    if not text.startswith("data:"):
        return None
    payload = text[len("data:"):].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


def openai_stream_one(args, prompt: str, output_tokens: int) -> dict[str, Any]:
    url = args.api_base.rstrip("/") + "/completions"
    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")

    first_ms = None
    last_ms = None
    inter_token_ms: list[float] = []
    token_count = 0
    generated_text_parts: list[str] = []
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=args.request_timeout_s) as response:
        for raw_line in response:
            event = decode_sse_line(raw_line)
            if event is None:
                continue
            choices = event.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("text", "")
            if text == "":
                continue
            generated_text_parts.append(text)
            now_ms = (time.perf_counter() - started) * 1000.0
            if first_ms is None:
                first_ms = now_ms
            elif last_ms is not None:
                inter_token_ms.append(now_ms - last_ms)
            last_ms = now_ms
            token_count += 1
    total_ms = (time.perf_counter() - started) * 1000.0
    return {
        "ttft_ms": first_ms or total_ms,
        "itl_ms": inter_token_ms,
        "total_ms": total_ms,
        "output_tokens": token_count,
        "generated_text": "".join(generated_text_parts),
    }


def run_openai_case(args, case: WorkloadCase) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.weights)
    prompts = [make_prompt(tokenizer, case.input_tokens) for _ in range(case.concurrency)]
    started = time.perf_counter()
    request_results = []
    with ThreadPoolExecutor(max_workers=case.concurrency) as pool:
        futures = [pool.submit(openai_stream_one, args, prompt, case.output_tokens) for prompt in prompts]
        for future in as_completed(futures):
            request_results.append(future.result())
    wall_ms = (time.perf_counter() - started) * 1000.0

    all_itl = [v for result in request_results for v in result["itl_ms"]]
    ttfts = [result["ttft_ms"] for result in request_results]
    total_output = 0
    for result in request_results:
        generated_text = result.get("generated_text", "")
        if generated_text:
            total_output += len(tokenizer.encode(generated_text, add_special_tokens=False))
        else:
            total_output += result["output_tokens"]
    result = {
        "completed": len(request_results),
        "total_output_tokens_observed": total_output,
        "total_ms": wall_ms,
        "output_tokens_per_s": total_output / wall_ms * 1000.0 if wall_ms > 0 else 0.0,
        "ttft_ms": summarize(ttfts),
        "itl_ms": summarize(all_itl),
        "step_ms": summarize([]),
        "steps": 0,
        "max_active_decoding": case.concurrency,
    }
    return normalize_result(args, case, result)


def normalize_result(args, case: WorkloadCase, result: dict[str, Any]) -> dict[str, Any]:
    expected_tokens = case.concurrency * case.output_tokens
    observed_tokens = min(int(result.get("total_output_tokens_observed", 0)), expected_tokens)
    total_ms = float(result.get("total_ms", 0.0))
    return {
        "type": "result",
        "timestamp_utc": utc_now(),
        "backend": args.backend,
        "engine": args.engine,
        "model": args.model,
        "input_tokens": case.input_tokens,
        "output_tokens": case.output_tokens,
        "concurrency": case.concurrency,
        "completed": min(int(result.get("completed", 0)), case.concurrency),
        "total_output_tokens_observed": observed_tokens,
        "total_ms": total_ms,
        "output_tokens_per_s": observed_tokens / total_ms * 1000.0 if total_ms > 0 else 0.0,
        "ttft_ms_avg": result.get("ttft_ms", {}).get("avg", 0.0),
        "ttft_ms_p50": result.get("ttft_ms", {}).get("p50", 0.0),
        "ttft_ms_p95": result.get("ttft_ms", {}).get("p95", 0.0),
        "itl_ms_avg": result.get("itl_ms", {}).get("avg", 0.0),
        "itl_ms_p50": result.get("itl_ms", {}).get("p50", 0.0),
        "itl_ms_p95": result.get("itl_ms", {}).get("p95", 0.0),
        "raw": result,
    }


def write_result(output: Path, record: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(jsonl_path: Path, csv_path: Path) -> None:
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("type") == "result"]
    if not rows:
        return
    fields = [
        "engine", "backend", "model", "input_tokens", "output_tokens", "concurrency",
        "completed", "total_output_tokens_observed", "total_ms", "output_tokens_per_s",
        "ttft_ms_avg", "ttft_ms_p50", "ttft_ms_p95", "itl_ms_avg", "itl_ms_p50", "itl_ms_p95",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(jsonl_path: Path, md_path: Path) -> None:
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("type") == "result"]
    if not rows:
        return
    headers = ["engine", "in", "out", "conc", "tok/s", "TTFT p95", "ITL p95"]
    lines = ["# Benchmark Summary", "", "| " + " | ".join(headers) + " |", "|" + "|".join([":---"] * len(headers)) + "|"]
    for row in rows:
        lines.append(
            "| {engine} | {input_tokens} | {output_tokens} | {concurrency} | {tps:.2f} | {ttft:.2f} | {itl:.2f} |".format(
                engine=row.get("engine"),
                input_tokens=row.get("input_tokens"),
                output_tokens=row.get("output_tokens"),
                concurrency=row.get("concurrency"),
                tps=row.get("output_tokens_per_s", 0.0),
                ttft=row.get("ttft_ms_p95", 0.0),
                itl=row.get("itl_ms_p95", 0.0),
            )
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(jsonl_path: Path, png_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("type") == "result"]
    if not rows:
        return
    labels = [f"{r['engine']}\nin={r['input_tokens']} c={r['concurrency']}" for r in rows]
    tps = [r.get("output_tokens_per_s", 0.0) for r in rows]
    ttft = [r.get("ttft_ms_p95", 0.0) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(rows) * 1.2), 4))
    axes[0].bar(labels, tps)
    axes[0].set_title("Output tokens/s")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(labels, ttft)
    axes[1].set_title("TTFT p95 ms")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard single-node benchmark for Pico-vLLM/vLLM/SGLang.")
    parser.add_argument("--backend", choices=["pico", "openai"], default="pico")
    parser.add_argument("--engine", default="pico", help="Result label, e.g. pico, vllm, sglang.")
    parser.add_argument("--model", default="./weights")
    parser.add_argument("--weights", default="./weights", help="Local tokenizer/weights path used for prompt construction and Pico.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible API base for vLLM/SGLang.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--input-lens", default="128,512,2048")
    parser.add_argument("--output-lens", default="32")
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--num-gpu-blocks", type=int, default=0)
    parser.add_argument("--block-slack", type=int, default=256)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--disable-prefix-cache", action="store_true")
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

    input_lens = parse_csv_ints(args.input_lens)
    output_lens = parse_csv_ints(args.output_lens)
    concurrencies = parse_csv_ints(args.concurrency)
    cases = [WorkloadCase(i, o, c) for i in input_lens for o in output_lens for c in concurrencies]
    if not cases:
        raise ValueError("No benchmark cases selected.")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.backend == "pico" and world_size > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = Path(args.output or f"logs/benchmarks/standard/{stamp}_{args.engine}_rank{rank}.jsonl")
    if rank == 0:
        write_result(output, env_record(args, rank, local_rank, world_size))

    for case in cases:
        if args.backend == "pico":
            record = run_pico_case(args, case, rank, world_size, local_rank)
        else:
            record = run_openai_case(args, case) if rank == 0 else None
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
