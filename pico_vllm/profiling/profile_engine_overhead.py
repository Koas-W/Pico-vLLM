#!/usr/bin/env python3
"""Measure Engine.step() Python-side overhead vs pure GPU model time.

Does NOT modify any existing code. Builds the engine through the existing
benchmark_h200_baseline.create_engine, runs a normal decode workload, and
compares:

  engine_step_ms : wall time of one engine.step()  (host prep + GPU + sampling)
  pure_gpu_ms    : wall time of one captured CUDA-graph replay (GPU only)
  overhead_ms    : engine_step_ms - pure_gpu_ms   <-- the Python-side overhead

A lightweight CPU breakdown is gathered by wrapping (at runtime, in this
script) scheduler.schedule / engine._decode_step_graph / sampler.sample_batch.
The decode path forces a GPU sync inside sampler (.item()), so the "sample"
bucket also absorbs the GPU wait; CPU-only sample ~= sample - pure_gpu.

Run from repo root, e.g.:
  PYTHONPATH=pico_vllm:pico_vllm/benchmarks .venv-vllm019/bin/python \
      pico_vllm/profiling/profile_engine_overhead.py --prompt-len 512 --concurrency 1
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
PICO_DIR = HERE.parents[1]            # .../pico_vllm
BENCH_DIR = PICO_DIR / "benchmarks"
for _p in (str(PICO_DIR), str(BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import benchmark_h200_baseline as base
import sampler as sampler_mod


def pctile(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int((len(s) - 1) * q)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="./weights")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--warmup-steps", type=int, default=8)
    ap.add_argument("--gpu-replays", type=int, default=50)
    ap.add_argument("--num-gpu-blocks", type=int, default=0)
    ap.add_argument("--block-slack", type=int, default=256)
    ap.add_argument("--disable-prefix-cache", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # batch == concurrency so the captured graph has no ghost padding and
    # pure_gpu_ms equals the real per-step GPU cost.
    eargs = argparse.Namespace(
        weights=args.weights,
        max_new_tokens=args.max_new_tokens,
        num_gpu_blocks=args.num_gpu_blocks,
        block_slack=args.block_slack,
        max_batch_size=args.concurrency,
        max_num_seqs=args.concurrency,
        disable_cuda_graph=False,
        disable_prefix_cache=args.disable_prefix_cache,
        detokenize_outputs=False,
        ignore_eos=True,
    )
    engine, tokenizer, block_manager = base.create_engine(
        eargs, device=device, rank=0, world_size=1,
        max_prompt_len=args.prompt_len, max_concurrency=args.concurrency,
    )

    # ---- runtime wrapping for a CPU breakdown (no source edits) ----
    buckets = {"schedule": 0.0, "decode_prep": 0.0, "sample": 0.0}
    measuring = {"on": False}

    def wrap(get, set_, key):
        orig = get()

        def wrapped(*a, **k):
            t = time.perf_counter()
            r = orig(*a, **k)
            if measuring["on"]:
                buckets[key] += time.perf_counter() - t
            return r

        set_(wrapped)
        return orig

    wrap(lambda: engine.scheduler.schedule,
         lambda f: setattr(engine.scheduler, "schedule", f), "schedule")
    wrap(lambda: engine._decode_step_graph,
         lambda f: setattr(engine, "_decode_step_graph", f), "decode_prep")
    # engine calls `sampler.sample_batch(...)`, so patch the module attribute.
    orig_sample = sampler_mod.sample_batch

    def wrapped_sample(*a, **k):
        t = time.perf_counter()
        r = orig_sample(*a, **k)
        if measuring["on"]:
            buckets["sample"] += time.perf_counter() - t
        return r

    sampler_mod.sample_batch = wrapped_sample

    # ---- submit workload ----
    prompt = base.make_prompt(tokenizer, args.prompt_len)
    for _ in range(args.concurrency):
        engine.submit(prompt, max_new_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0)
    engine.mark_finished()

    step_ms = []
    pure_gpu_ms = None
    measured_steps = 0
    torch.cuda.synchronize()

    step_idx = 0
    while not engine.is_done():
        t0 = time.perf_counter()
        engine.step()  # sampler .item() already syncs the GPU each step
        dt = (time.perf_counter() - t0) * 1000.0

        if step_idx > args.warmup_steps:  # skip prefill(step 0) + warmup decodes
            measuring["on"] = True
            step_ms.append(dt)
            measured_steps += 1

        # measure pure GPU once, mid-decode, with real (populated) buffers
        if (pure_gpu_ms is None and step_idx == args.warmup_steps + 5
                and getattr(engine, "cuda_graph", None) is not None):
            torch.cuda.synchronize()
            tg = time.perf_counter()
            for _ in range(args.gpu_replays):
                engine.cuda_graph.replay()
            torch.cuda.synchronize()
            pure_gpu_ms = (time.perf_counter() - tg) * 1000.0 / args.gpu_replays

        step_idx += 1

    measuring["on"] = False

    # ---- report ----
    C = args.concurrency
    step_avg = statistics.fmean(step_ms) if step_ms else 0.0
    step_p50 = pctile(step_ms, 0.50)
    pure_gpu_ms = pure_gpu_ms or 0.0
    overhead = step_avg - pure_gpu_ms
    sched = buckets["schedule"] / measured_steps * 1000.0 if measured_steps else 0.0
    prep = buckets["decode_prep"] / measured_steps * 1000.0 if measured_steps else 0.0
    samp = buckets["sample"] / measured_steps * 1000.0 if measured_steps else 0.0
    cpu_sample = max(0.0, samp - pure_gpu_ms)  # sample bucket includes GPU wait

    def line(label, ms):
        share = (ms / step_avg * 100.0) if step_avg else 0.0
        print(f"  {label:<28} {ms:8.3f} ms   {share:5.1f}%")

    print("=" * 64)
    print(f"GPU: {torch.cuda.get_device_name(0)}  sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
    print(f"prompt_len={args.prompt_len}  concurrency={C}  "
          f"measured_decode_steps={measured_steps}")
    print("=" * 64)
    print(f"  engine_step  (avg)           {step_avg:8.3f} ms   100.0%")
    print(f"  engine_step  (p50)           {step_p50:8.3f} ms")
    print("-" * 64)
    line("pure_gpu (graph replay)", pure_gpu_ms)
    line("python overhead (step-gpu)", overhead)
    print("-" * 64)
    print("  CPU breakdown per step:")
    line("  scheduler.schedule", sched)
    line("  _decode_step_graph (host prep)", prep)
    line("  sampler.sample_batch (CPU)", cpu_sample)
    print("-" * 64)
    tok_s_real = C / (step_avg / 1000.0) if step_avg else 0.0
    tok_s_gpu = C / (pure_gpu_ms / 1000.0) if pure_gpu_ms else 0.0
    print(f"  tokens/s  (real engine)      {tok_s_real:8.1f}")
    print(f"  tokens/s  (GPU-bound ceiling){tok_s_gpu:8.1f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
