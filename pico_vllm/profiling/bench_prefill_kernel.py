"""Micro-benchmark the prefill attention kernel: legacy vs autotuned flash.

Times the kernel in isolation (full prefill, B=1) across sequence lengths and
prints the autotuned config chosen for the flash kernel.

  PYTHONPATH=pico_vllm:pico_vllm/tests .venv-vllm019/bin/python pico_vllm/profiling/bench_prefill_kernel.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))            # pico_vllm
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))  # build_cache

import torch
from model import ModelConfig
from ops.triton.attention import paged_prefill_attention
from ops.triton.flash_prefill import paged_prefill_attention_flash, _flash_prefill_grouped_kernel
from test_prefill_attention import build_cache

DEVICE = "cuda"
DTYPE = torch.bfloat16


def time_kernel(fn, iters=30):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    fn()  # warm
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main():
    cfg = ModelConfig()
    torch.manual_seed(0)
    print(f"{'seqlen':>7} {'legacy ms':>10} {'flash ms':>9} {'speedup':>8}")
    for L in (512, 1024, 2048, 4096, 8192):
        k_cache, v_cache, block_table = build_cache(cfg, [L], cfg.MAX_BLOCKS_PER_SEQ)
        ctx = torch.tensor([L], dtype=torch.int32, device=DEVICE)
        ntl = torch.tensor([L], dtype=torch.int32, device=DEVICE)
        qsl = torch.tensor([0], dtype=torch.int32, device=DEVICE)
        q = torch.randn(L, cfg.num_attention_heads, cfg.head_dim, device=DEVICE, dtype=DTYPE)
        mb = cfg.MAX_BLOCKS_PER_SEQ

        leg = time_kernel(lambda: paged_prefill_attention(q, k_cache, v_cache, block_table, ctx, ntl, qsl, MAX_BLOCKS_PER_SEQ=mb))
        fla = time_kernel(lambda: paged_prefill_attention_flash(q, k_cache, v_cache, block_table, ctx, ntl, qsl, MAX_BLOCKS_PER_SEQ=mb))
        bc = _flash_prefill_grouped_kernel.best_config
        cfg_str = f"BM={bc.kwargs.get('BLOCK_M')},nw={bc.num_warps},ns={bc.num_stages}"
        print(f"{L:>7} {leg:>10.3f} {fla:>9.3f} {leg/fla:>7.2f}x   [{cfg_str}]")


if __name__ == "__main__":
    main()
