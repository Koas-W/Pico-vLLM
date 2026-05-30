"""Run prefill TTFT at a fixed input length REPS times, print every value.

Used to A/B compare engine-side optimizations (e.g., before vs after
vectorizing `get_prefill_slot_mapping`) with full distribution visibility,
not just best-of-N.

  PYTHONPATH=pico_vllm:pico_vllm/benchmarks .venv-vllm019/bin/python \
      pico_vllm/profiling/bench_ttft_repeat.py --input 8192 --reps 10
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve()
PICO = HERE.parents[1]
for p in (str(PICO), str(PICO / "benchmarks")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch


def synth_ids(n):
    return [100 + (i % 20000) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=int, default=8192)
    ap.add_argument("--output", type=int, default=16)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2, help="warmup reps not counted")
    args = ap.parse_args()

    import benchmark_h200_baseline as base

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    eargs = Namespace(
        weights="./weights", max_new_tokens=args.output, num_gpu_blocks=0,
        block_slack=256, max_batch_size=1, max_num_seqs=1,
        disable_cuda_graph=False, disable_prefix_cache=True,
        detokenize_outputs=False, ignore_eos=True,
    )
    engine, _, _ = base.create_engine(
        eargs, device=device, rank=0, world_size=1,
        max_prompt_len=args.input, max_concurrency=1,
    )

    ids = synth_ids(args.input)
    ttfts = []
    for rep in range(args.warmup + args.reps):
        rid = engine.submit_token_ids(ids, args.output, 0.0, 1.0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ttft = None
        done = False
        while not done:
            completed = engine.step()
            if ttft is None:
                torch.cuda.synchronize()
                ttft = (time.perf_counter() - t0) * 1000.0
            for cid, _ in completed:
                if cid == rid:
                    done = True
        # drain so blocks are returned before next rep
        while engine.scheduler.num_decoding > 0:
            engine.step()
        if rep >= args.warmup:
            ttfts.append(ttft)
            print(f"  rep {rep - args.warmup + 1}: TTFT = {ttft:.2f} ms")

    print()
    print(f"input={args.input}, output={args.output}, reps={args.reps}")
    print(f"  min   : {min(ttfts):.2f} ms")
    print(f"  median: {statistics.median(ttfts):.2f} ms")
    print(f"  mean  : {statistics.fmean(ttfts):.2f} ms")
    print(f"  max   : {max(ttfts):.2f} ms")
    print(f"  stdev : {statistics.stdev(ttfts) if len(ttfts) > 1 else 0:.2f} ms")


if __name__ == "__main__":
    main()
