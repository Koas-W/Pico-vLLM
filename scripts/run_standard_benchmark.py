#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / "pico_vllm" / "tests" / "benchmark" / "standard_benchmark.py"
VALID_ENGINES = ("pico", "vllm")


def default_vllm_python() -> str:
    local = REPO_ROOT / ".venv-vllm019" / "bin" / "python"
    return str(local) if local.exists() else sys.executable


def parse_engines(value: str) -> list[str]:
    engines = [part.strip() for part in value.split(",") if part.strip()]
    if not engines:
        raise argparse.ArgumentTypeError("at least one engine is required")
    invalid = [engine for engine in engines if engine not in VALID_ENGINES]
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown engine(s): {', '.join(invalid)}")
    seen = set()
    unique = []
    for engine in engines:
        if engine not in seen:
            seen.add(engine)
            unique.append(engine)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure engine benchmark comparisons for Pico-vLLM and vLLM.")
    parser.add_argument("--engines", type=parse_engines, default=parse_engines("pico"), help="Comma-separated engines: pico,vllm.")
    parser.add_argument("--model", default="./weights", help="Model label/path. Used by vLLM as model path.")
    parser.add_argument("--weights", default="./weights", help="Local tokenizer/weights path used by benchmark.")
    parser.add_argument("--input-lens", default="128,512,2048")
    parser.add_argument("--output-lens", default="32")
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--output", default="logs/benchmarks/standard/engine_compare.jsonl")
    parser.add_argument("--pico-python", default=sys.executable, help="Python executable used for Pico engine benchmark.")
    parser.add_argument("--vllm-python", default=default_vllm_python(), help="Python executable used for vLLM engine benchmark.")
    parser.add_argument("--num-gpu-blocks", type=int, default=0)
    parser.add_argument("--block-slack", type=int, default=256)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--enable-prefix-cache", action="store_true", help="Enable prefix cache. Disabled by default for raw engine comparison.")
    parser.add_argument("--vllm-dtype", default="bfloat16")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-max-model-len", type=int, default=0)
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def common_benchmark_args(args: argparse.Namespace, python: str, backend: str, engine: str) -> list[str]:
    command = [
        python,
        str(BENCHMARK_SCRIPT),
        "--backend",
        backend,
        "--engine",
        engine,
        "--model",
        args.model,
        "--weights",
        args.weights,
        "--input-lens",
        args.input_lens,
        "--output-lens",
        args.output_lens,
        "--concurrency",
        args.concurrency,
        "--num-gpu-blocks",
        str(args.num_gpu_blocks),
        "--block-slack",
        str(args.block_slack),
        "--max-batch-size",
        str(args.max_batch_size),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--output",
        args.output,
        "--no-report",
    ]
    if args.disable_cuda_graph:
        command.append("--disable-cuda-graph")
    if args.enable_prefix_cache:
        command.append("--enable-prefix-cache")
    return command


def run_pico(args: argparse.Namespace) -> None:
    run_command(common_benchmark_args(args, args.pico_python, "pico", "pico"))


def run_vllm(args: argparse.Namespace) -> None:
    command = common_benchmark_args(args, args.vllm_python, "vllm", "vllm")
    command.extend(
        [
            "--vllm-dtype",
            args.vllm_dtype,
            "--vllm-gpu-memory-utilization",
            str(args.vllm_gpu_memory_utilization),
            "--vllm-max-model-len",
            str(args.vllm_max_model_len),
        ]
    )
    run_command(command)


def regenerate_report(args: argparse.Namespace) -> None:
    run_command([sys.executable, str(BENCHMARK_SCRIPT), "--report-only", "--output", args.output])


def main() -> int:
    args = parse_args()
    for engine in args.engines:
        if engine == "pico":
            run_pico(args)
        elif engine == "vllm":
            run_vllm(args)
        else:
            raise AssertionError(engine)
    regenerate_report(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("ERROR interrupted", file=sys.stderr, flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
