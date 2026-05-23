#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pico_vllm" / "tests"))

from ci_utils import TestEnvironment, detect_environment


Requirement = Callable[[TestEnvironment], tuple[bool, str]]


@dataclass(frozen=True)
class Step:
    layer: str
    name: str
    command: list[str]
    requirement: Requirement
    default: bool = True


def always(_: TestEnvironment) -> tuple[bool, str]:
    return True, ""


def requires_cuda(env: TestEnvironment) -> tuple[bool, str]:
    if env.cuda_available:
        return True, ""
    return False, "CUDA is not available"


def requires_two_gpus(env: TestEnvironment) -> tuple[bool, str]:
    if env.gpu_count >= 2:
        return True, ""
    return False, f"requires at least 2 CUDA devices, found {env.gpu_count}"


def requires_weights(env: TestEnvironment) -> tuple[bool, str]:
    if not env.weights_available:
        return False, "./weights is not available"
    if not env.transformers_available:
        return False, "transformers is not available"
    if not env.safetensors_available:
        return False, "safetensors is not available"
    return True, ""


def pytest_cmd(*paths: str) -> list[str]:
    return [sys.executable, "-m", "pytest", "-q", "-s", *paths]


def torchrun_cmd(nproc: int, script: str, *args: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        script,
        *args,
    ]


def build_steps() -> list[Step]:
    return [
        Step(
            layer="00_env",
            name="syntax compile",
            command=[sys.executable, "-m", "compileall", "-q", "pico_vllm", "scripts"],
            requirement=always,
        ),
        Step(
            layer="00_env",
            name="environment detection",
            command=pytest_cmd("pico_vllm/tests/ci/00_env/test_environment.py"),
            requirement=always,
        ),
        Step(
            layer="01_ops",
            name="CPU torch operators",
            command=pytest_cmd("pico_vllm/tests/ci/01_ops/test_torch_cpu_ops.py"),
            requirement=always,
        ),
        Step(
            layer="01_ops",
            name="CUDA paged prefill attention",
            command=pytest_cmd("pico_vllm/tests/ci/01_ops/test_triton_prefill_attention.py"),
            requirement=requires_cuda,
        ),
        Step(
            layer="02_single_card",
            name="CPU tiny random model inference",
            command=pytest_cmd("pico_vllm/tests/ci/02_single_card/test_torch_cpu_tiny_model.py"),
            requirement=always,
        ),
        Step(
            layer="02_single_card",
            name="single CUDA card tiny model",
            command=pytest_cmd("pico_vllm/tests/ci/02_single_card/test_tiny_model_smoke.py"),
            requirement=requires_cuda,
        ),
        Step(
            layer="02_single_card",
            name="CPU real Qwen weights inference",
            command=pytest_cmd("pico_vllm/tests/ci/02_single_card/test_torch_cpu_real_model_inference.py"),
            requirement=requires_weights,
            default=False,
        ),
    ]


def print_environment(env: TestEnvironment) -> None:
    for line in format_environment(env):
        print(line, flush=True)


def format_environment(env: TestEnvironment) -> list[str]:
    return [
        "Detected test environment:",
        f"  python: {env.python}",
        f"  torch: {env.torch_version}",
        f"  cuda_available: {env.cuda_available}",
        f"  gpu_count: {env.gpu_count}",
        f"  triton_available: {env.triton_available}",
        f"  pytest_available: {env.pytest_available}",
        f"  transformers_available: {env.transformers_available}",
        f"  safetensors_available: {env.safetensors_available}",
        f"  weights_available: {env.weights_available}",
        f"  torchrun_available: {env.torchrun_available}",
        "",
    ]


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip().lower())
    return value.strip("_") or "step"


def emit(line: str, log_file=None) -> None:
    print(line, flush=True)
    if log_file is not None:
        log_file.write(line + "\n")
        log_file.flush()


def tee_process(command: list[str], log_file) -> int:
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
        log_file.flush()
    return process.wait()


def make_log_dir(log_dir_arg: str | None) -> Path:
    if log_dir_arg:
        log_dir = Path(log_dir_arg)
        if not log_dir.is_absolute():
            log_dir = REPO_ROOT / log_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = REPO_ROOT / "logs" / "local_ci" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the layered Pico-vLLM local CI gate.")
    parser.add_argument(
        "--layer",
        action="append",
        choices=[
            "00_env",
            "01_ops",
            "02_single_card",
            "03_single_node_multi_card",
            "04_multi_card",
        ],
        help="Run only the selected layer. May be passed multiple times.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the selected steps and exit.",
    )
    parser.add_argument(
        "--log-dir",
        help="Directory for per-step logs. Defaults to logs/local_ci/<timestamp>.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = detect_environment(REPO_ROOT)
    print_environment(env)

    selected_layers = set(args.layer or [])
    steps = build_steps()
    if selected_layers:
        steps = [step for step in steps if step.layer in selected_layers]
    else:
        steps = [step for step in steps if step.default]

    if args.list:
        for step in steps:
            ok, reason = step.requirement(env)
            status = "run" if ok else f"skip: {reason}"
            print(f"[{step.layer}] {step.name}: {status}", flush=True)
        return 0

    log_dir = make_log_dir(args.log_dir)
    summary_path = log_dir / "summary.log"
    with summary_path.open("w", encoding="utf-8") as summary:
        for line in format_environment(env):
            summary.write(line + "\n")
        summary.write(f"log_dir: {log_dir}\n\n")

    print(f"Logs: {log_dir}", flush=True)

    failed = False
    for index, step in enumerate(steps, start=1):
        ok, reason = step.requirement(env)
        label = f"[{step.layer}] {step.name}"
        log_path = log_dir / f"{index:02d}_{step.layer}_{slugify(step.name)}.log"

        with log_path.open("w", encoding="utf-8") as log_file:
            for line in format_environment(env):
                log_file.write(line + "\n")
            log_file.write(f"label: {label}\n")
            log_file.write(f"command: {' '.join(step.command)}\n")
            log_file.write(f"log_path: {log_path}\n\n")

            with summary_path.open("a", encoding="utf-8") as summary:
                summary.write(f"{label}: {log_path}\n")

            emit(f"LOG  {label}: {log_path}", log_file)

            if not ok:
                emit(f"SKIP {label}: {reason}", log_file)
                with summary_path.open("a", encoding="utf-8") as summary:
                    summary.write(f"  status: SKIP ({reason})\n")
                continue

            emit(f"RUN  {label}", log_file)
            emit("     " + " ".join(step.command), log_file)
            started = time.time()
            returncode = tee_process(step.command, log_file)
            elapsed = time.time() - started
            if returncode != 0:
                emit(f"FAIL {label}: exit code {returncode}, elapsed={elapsed:.2f}s", log_file)
                with summary_path.open("a", encoding="utf-8") as summary:
                    summary.write(f"  status: FAIL exit_code={returncode} elapsed={elapsed:.2f}s\n")
                failed = True
                break

            emit(f"PASS {label}: elapsed={elapsed:.2f}s\n", log_file)
            with summary_path.open("a", encoding="utf-8") as summary:
                summary.write(f"  status: PASS elapsed={elapsed:.2f}s\n")

    if failed:
        print(f"Local CI failed. Logs: {log_dir}", flush=True)
        return 1

    print(f"Local CI completed. Logs: {log_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
