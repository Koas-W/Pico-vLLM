#!/usr/bin/env python3
"""Acceptance matrix benchmark: pico (flash) vs vLLM, fair config.

For each (input_len x output_len) cell, runs `--reps` times and keeps the BEST
(min total latency) to suppress shared-GPU contention noise. The model is loaded
ONCE per engine (reps/cells reuse it) to keep total time reasonable.

Fairness:
  - vLLM keeps its strong defaults: CUDA graphs (enforce_eager=False), V1 engine,
    async scheduling ON. Engine forced in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0)
    only so client-side timing is valid (no perf impact at concurrency 1).
  - prefix caching OFF for BOTH (independent reps, raw per-request comparison).
  - greedy (temperature 0), ignore_eos, detokenize off, bf16, concurrency 1.

Run (one engine per process):
  PYTHONPATH=pico_vllm:pico_vllm/benchmarks .venv-vllm019/bin/python \
      pico_vllm/tests/benchmark/accept_benchmark.py --engine pico  --out /tmp/acc.jsonl
  PYTHONPATH=pico_vllm:pico_vllm/benchmarks .venv-vllm019/bin/python \
      pico_vllm/tests/benchmark/accept_benchmark.py --engine vllm  --out /tmp/acc.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve()
PICO = HERE.parents[2]
for p in (str(PICO), str(PICO / "benchmarks")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

INPUTS = [64, 512, 2048, 8192]
OUTPUTS = [16, 128, 1024]


def synth_ids(n):
    # deterministic, avoid low special-token ids
    return [100 + (i % 20000) for i in range(n)]


def record(out_path, row):
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def run_pico(args):
    import benchmark_h200_baseline as base

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    eargs = Namespace(
        weights=args.weights, max_new_tokens=max(OUTPUTS), num_gpu_blocks=0,
        block_slack=256, max_batch_size=1, max_num_seqs=1,
        disable_cuda_graph=False, disable_prefix_cache=True,
        detokenize_outputs=False, ignore_eos=True,
    )
    engine, _, _ = base.create_engine(
        eargs, device=device, rank=0, world_size=1,
        max_prompt_len=max(INPUTS), max_concurrency=1,
    )

    for inp in INPUTS:
        ids = synth_ids(inp)
        for out in OUTPUTS:
            best_total, best_ttft = float("inf"), float("inf")
            for _ in range(args.reps):
                rid = engine.submit_token_ids(ids, out, 0.0, 1.0)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                ttft = None
                done = False
                while not done:
                    completed = engine.step()
                    if ttft is None:
                        torch.cuda.synchronize()
                        ttft = time.perf_counter() - t0
                    for cid, _ in completed:
                        if cid == rid:
                            done = True
                torch.cuda.synchronize()
                total = time.perf_counter() - t0
                best_total = min(best_total, total)
                best_ttft = min(best_ttft, ttft)
                # The engine retires/frees a finished request lazily on the NEXT
                # schedule(); drain so its KV blocks are returned before next rep
                # (otherwise blocks leak and long inputs OOM the block manager).
                while engine.scheduler.num_decoding > 0:
                    engine.step()
            record(args.out, {
                "engine": "pico", "input": inp, "output": out,
                "total_ms": best_total * 1000, "ttft_ms": best_ttft * 1000,
                "decode_tok_s": out / best_total,
            })
            print(f"[pico] in={inp} out={out}: ttft={best_ttft*1000:.1f}ms "
                  f"total={best_total*1000:.1f}ms tok/s={out/best_total:.1f}", flush=True)


def run_vllm(args):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    llm = LLM(
        model=args.weights, dtype="bfloat16", gpu_memory_utilization=0.5,
        max_model_len=max(INPUTS) + max(OUTPUTS) + 8, max_num_seqs=1,
        enable_prefix_caching=False, enforce_eager=False, disable_log_stats=True,
        async_scheduling=not args.vllm_no_async,
    )
    eng = llm.llm_engine
    uid = 0

    for inp in INPUTS:
        ids = synth_ids(inp)
        for out in OUTPUTS:
            best_total, best_ttft = float("inf"), float("inf")
            for _ in range(args.reps):
                sp = SamplingParams(
                    temperature=0.0, top_p=1.0, max_tokens=out, ignore_eos=True,
                    detokenize=False, output_kind=RequestOutputKind.FINAL_ONLY,
                )
                eng.add_request(str(uid), TokensPrompt(prompt_token_ids=ids), sp)
                uid += 1
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                ttft = None
                while eng.has_unfinished_requests():
                    eng.step()
                    if ttft is None:
                        torch.cuda.synchronize()
                        ttft = time.perf_counter() - t0
                torch.cuda.synchronize()
                total = time.perf_counter() - t0
                best_total = min(best_total, total)
                best_ttft = min(best_ttft, ttft)
            record(args.out, {
                "engine": "vllm", "input": inp, "output": out,
                "total_ms": best_total * 1000, "ttft_ms": best_ttft * 1000,
                "decode_tok_s": out / best_total,
            })
            print(f"[vllm] in={inp} out={out}: ttft={best_ttft*1000:.1f}ms "
                  f"total={best_total*1000:.1f}ms tok/s={out/best_total:.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["pico", "vllm"], required=True)
    ap.add_argument("--weights", default="./weights")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="/tmp/acc.jsonl")
    ap.add_argument("--inputs", default="")
    ap.add_argument("--outputs", default="")
    ap.add_argument("--vllm-no-async", action="store_true")
    args = ap.parse_args()

    global INPUTS, OUTPUTS
    if args.inputs:
        INPUTS = [int(x) for x in args.inputs.split(",")]
    if args.outputs:
        OUTPUTS = [int(x) for x in args.outputs.split(",")]

    if args.engine == "pico":
        run_pico(args)
    else:
        run_vllm(args)


if __name__ == "__main__":
    main()
