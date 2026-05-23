#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
import threading


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / "pico_vllm" / "tests" / "benchmark" / "standard_benchmark.py"
VALID_ENGINES = ("pico", "vllm", "sglang")


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


def parse_extra_args(value: str) -> list[str]:
    return [part for part in value.split() if part]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pico-vLLM/vLLM/SGLang standard benchmark comparisons.")
    parser.add_argument("--engines", type=parse_engines, default=parse_engines("pico"), help="Comma-separated engines: pico,vllm,sglang.")
    parser.add_argument("--server-mode", choices=["external", "docker", "command"], default="docker", help="How non-Pico servers are provided.")
    parser.add_argument("--model", default="./weights", help="Model label/path passed to benchmark requests.")
    parser.add_argument("--weights", default="./weights", help="Local tokenizer/weights path used by benchmark.")
    parser.add_argument("--input-lens", default="128,512,2048")
    parser.add_argument("--output-lens", default="32")
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--output", default="logs/benchmarks/standard/compare.jsonl")
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--server-timeout-s", type=float, default=900.0)
    parser.add_argument("--vllm-api-base", default="")
    parser.add_argument("--sglang-api-base", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--num-gpu-blocks", type=int, default=0)
    parser.add_argument("--block-slack", type=int, default=256)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--keep-containers", action="store_true", help="Leave benchmark server containers running after the run.")
    parser.add_argument("--vllm-image", default="vllm/vllm-openai:latest")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-extra-args", default="", help="Extra args appended to vLLM inside the container.")
    parser.add_argument("--vllm-command", default="", help="Command used for --server-mode command.")
    parser.add_argument("--sglang-image", default="lmsysorg/sglang:latest-cu129")
    parser.add_argument("--sglang-port", type=int, default=30000)
    parser.add_argument("--sglang-extra-args", default="", help="Extra args appended to SGLang launch_server inside the container.")
    parser.add_argument("--sglang-command", default="", help="Command used for --server-mode command.")
    return parser.parse_args()


class ManagedServer:
    def __init__(self, name: str, command: list[str], keep: bool) -> None:
        self.name = name
        self.command = command
        self.keep = keep
        self.process: subprocess.Popen[str] | None = None
        self.output_lines: list[str] = []
        self.reader_thread: threading.Thread | None = None

    def __enter__(self) -> "ManagedServer":
        print("START " + self.name, flush=True)
        print("RUN " + " ".join(self.command), flush=True)
        self.process = subprocess.Popen(
            self.command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self.reader_thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.process is None:
            return
        if self.keep:
            print(f"KEEP {self.name}: pid={self.process.pid}", flush=True)
            return
        if self.process.poll() is None:
            print("STOP " + self.name, flush=True)
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=30)

    def poll(self) -> int | None:
        if self.process is None:
            return None
        return self.process.poll()

    def _read_output(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            self.output_lines.append(line.rstrip("\n"))
            if len(self.output_lines) > 200:
                self.output_lines = self.output_lines[-200:]

    def log_tail(self, lines: int = 80) -> str:
        return "\n".join(self.output_lines[-lines:])


def run_command(command: list[str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def request_json(url: str, api_key: str, timeout_s: float) -> dict:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = response.read().decode("utf-8", errors="replace")
    return json.loads(payload) if payload else {}


def wait_for_openai_server(api_base: str, api_key: str, timeout_s: float, server: ManagedServer | None = None) -> None:
    deadline = time.time() + timeout_s
    url = api_base.rstrip("/") + "/models"
    last_error = ""
    while time.time() < deadline:
        if server is not None:
            returncode = server.poll()
            if returncode is not None:
                raise RuntimeError(f"{server.name} exited before ready with code {returncode}\n{server.log_tail()}")
        try:
            request_json(url, api_key, timeout_s=5.0)
            print(f"READY {api_base}", flush=True)
            return
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(2.0)
    tail = "" if server is None else "\n" + server.log_tail()
    raise RuntimeError(f"Timed out waiting for {api_base}; last error: {last_error}{tail}")


def common_benchmark_args(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
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
        "--request-timeout-s",
        str(args.request_timeout_s),
        "--output",
        args.output,
        "--no-report",
    ]
    if args.api_key:
        command.extend(["--api-key", args.api_key])
    if args.disable_cuda_graph:
        command.append("--disable-cuda-graph")
    if args.disable_prefix_cache:
        command.append("--disable-prefix-cache")
    return command


def run_pico(args: argparse.Namespace) -> None:
    command = common_benchmark_args(args)
    command.extend(
        [
            "--backend",
            "pico",
            "--engine",
            "pico",
            "--num-gpu-blocks",
            str(args.num_gpu_blocks),
            "--block-slack",
            str(args.block_slack),
            "--max-batch-size",
            str(args.max_batch_size),
            "--max-num-seqs",
            str(args.max_num_seqs),
        ]
    )
    run_command(command)


def run_openai_engine(args: argparse.Namespace, engine: str, api_base: str) -> None:
    wait_for_openai_server(api_base, args.api_key, args.server_timeout_s)
    command = common_benchmark_args(args)
    command.extend(["--backend", "openai", "--engine", engine, "--api-base", api_base])
    run_command(command)


def docker_mount_model_path(args: argparse.Namespace) -> tuple[str, str]:
    weights_path = (REPO_ROOT / args.weights).resolve() if not Path(args.weights).is_absolute() else Path(args.weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"weights path does not exist: {weights_path}")
    return str(weights_path), "/model"


def vllm_docker_command(args: argparse.Namespace) -> tuple[list[str], str]:
    host_model_path, container_model_path = docker_mount_model_path(args)
    api_base = args.vllm_api_base or f"http://127.0.0.1:{args.vllm_port}/v1"
    command = [
        args.docker_bin,
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc=host",
        "-p",
        f"{args.vllm_port}:8000",
        "-v",
        f"{host_model_path}:{container_model_path}:ro",
    ]
    if "HF_TOKEN" in os.environ:
        command.extend(["--env", "HF_TOKEN"])
    command.extend(
        [
            args.vllm_image,
            "--model",
            container_model_path,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--generation-config",
            "vllm",
        ]
    )
    command.extend(parse_extra_args(args.vllm_extra_args))
    return command, api_base


def sglang_docker_command(args: argparse.Namespace) -> tuple[list[str], str]:
    host_model_path, container_model_path = docker_mount_model_path(args)
    api_base = args.sglang_api_base or f"http://127.0.0.1:{args.sglang_port}/v1"
    command = [
        args.docker_bin,
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc=host",
        "--shm-size",
        "32g",
        "-p",
        f"{args.sglang_port}:30000",
        "-v",
        f"{host_model_path}:{container_model_path}:ro",
    ]
    if "HF_TOKEN" in os.environ:
        command.extend(["--env", "HF_TOKEN"])
    command.extend(
        [
            args.sglang_image,
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            container_model_path,
            "--host",
            "0.0.0.0",
            "--port",
            "30000",
        ]
    )
    command.extend(parse_extra_args(args.sglang_extra_args))
    return command, api_base


def run_vllm_docker(args: argparse.Namespace) -> None:
    command, api_base = vllm_docker_command(args)
    with ManagedServer("vllm", command, keep=args.keep_containers) as server:
        wait_for_openai_server(api_base, args.api_key, args.server_timeout_s, server=server)
        benchmark_command = common_benchmark_args(args)
        benchmark_command.extend(["--backend", "openai", "--engine", "vllm", "--api-base", api_base])
        run_command(benchmark_command)


def run_sglang_docker(args: argparse.Namespace) -> None:
    command, api_base = sglang_docker_command(args)
    with ManagedServer("sglang", command, keep=args.keep_containers) as server:
        wait_for_openai_server(api_base, args.api_key, args.server_timeout_s, server=server)
        benchmark_command = common_benchmark_args(args)
        benchmark_command.extend(["--backend", "openai", "--engine", "sglang", "--api-base", api_base])
        run_command(benchmark_command)


def run_command_server(args: argparse.Namespace, engine: str, command_text: str, api_base: str) -> None:
    if not command_text:
        raise ValueError(f"--{engine}-command is required when --server-mode command")
    command = shlex.split(command_text)
    with ManagedServer(engine, command, keep=args.keep_containers) as server:
        wait_for_openai_server(api_base, args.api_key, args.server_timeout_s, server=server)
        benchmark_command = common_benchmark_args(args)
        benchmark_command.extend(["--backend", "openai", "--engine", engine, "--api-base", api_base])
        run_command(benchmark_command)


def regenerate_report(args: argparse.Namespace) -> None:
    run_command([sys.executable, str(BENCHMARK_SCRIPT), "--report-only", "--output", args.output])


def main() -> int:
    args = parse_args()
    for engine in args.engines:
        if engine == "pico":
            run_pico(args)
        elif engine == "vllm":
            api_base = args.vllm_api_base or f"http://127.0.0.1:{args.vllm_port}/v1"
            if args.server_mode == "docker":
                run_vllm_docker(args)
            elif args.server_mode == "command":
                run_command_server(args, "vllm", args.vllm_command, api_base)
            else:
                run_openai_engine(args, "vllm", api_base)
        elif engine == "sglang":
            api_base = args.sglang_api_base or f"http://127.0.0.1:{args.sglang_port}/v1"
            if args.server_mode == "docker":
                run_sglang_docker(args)
            elif args.server_mode == "command":
                run_command_server(args, "sglang", args.sglang_command, api_base)
            else:
                run_openai_engine(args, "sglang", api_base)
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
