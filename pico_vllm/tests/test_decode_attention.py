"""Correctness check for the decode attention kernels.

Compares the legacy kernel and the new GQA-grouped split-KV flash-decoding
kernel against a PyTorch SDPA reference, over several batch / context-length
configurations (including unequal context lengths per request, to exercise the
per-row split boundaries and masking).

Run from repo root:
  PYTHONPATH=pico_vllm .venv-vllm019/bin/python pico_vllm/tests/test_decode_attention.py
"""
import sys

import torch
import torch.nn.functional as F

from blockmanager import BlockManager
from model import ModelConfig
from ops.triton.attention import paged_decode_attention
from ops.triton.flash_decode import paged_decode_attention_flash

DEVICE = "cuda"
DTYPE = torch.bfloat16
BLOCK_SIZE = 16


def build_cache(cfg, context_lens, max_blocks):
    """Allocate a block manager and fill random K/V for the given seqs.

    Returns k_cache_layer, v_cache_layer (layer 0), block_table (B, max_blocks).
    """
    B = len(context_lens)
    total_blocks_needed = sum((L + BLOCK_SIZE - 1) // BLOCK_SIZE for L in context_lens)
    bm = BlockManager(
        num_gpu_blocks=total_blocks_needed + 8, num_cpu_blocks=0,
        block_size=BLOCK_SIZE, num_layers=1,
        num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        dtype=DTYPE, device=DEVICE,
    )
    k_cache = bm.gpu_kv_cache[0, 0]  # (num_blocks, n_kv, block_size, head_dim)
    v_cache = bm.gpu_kv_cache[1, 0]

    block_table = torch.full((B, max_blocks), -1, dtype=torch.int32, device=DEVICE)
    next_block = 0
    for i, L in enumerate(context_lens):
        nblk = (L + BLOCK_SIZE - 1) // BLOCK_SIZE
        for j in range(nblk):
            phys = next_block
            next_block += 1
            block_table[i, j] = phys
            # how many valid tokens in this block
            valid = min(BLOCK_SIZE, L - j * BLOCK_SIZE)
            k_cache[phys, :, :valid, :] = torch.randn(
                cfg.num_key_value_heads, valid, cfg.head_dim, device=DEVICE, dtype=DTYPE
            )
            v_cache[phys, :, :valid, :] = torch.randn(
                cfg.num_key_value_heads, valid, cfg.head_dim, device=DEVICE, dtype=DTYPE
            )
    return k_cache, v_cache, block_table


def sdpa_reference(q, k_cache, v_cache, block_table, context_lens, cfg):
    B = q.shape[0]
    outs = []
    for i in range(B):
        L = int(context_lens[i].item())
        nblk = (L + BLOCK_SIZE - 1) // BLOCK_SIZE
        phys = block_table[i, :nblk]
        k = (k_cache[phys].permute(1, 0, 2, 3)
             .reshape(cfg.num_key_value_heads, -1, cfg.head_dim)[:, :L]
             .repeat_interleave(cfg.num_kv_groups, dim=0).unsqueeze(0))
        v = (v_cache[phys].permute(1, 0, 2, 3)
             .reshape(cfg.num_key_value_heads, -1, cfg.head_dim)[:, :L]
             .repeat_interleave(cfg.num_kv_groups, dim=0).unsqueeze(0))
        out_i = F.scaled_dot_product_attention(q[i].unsqueeze(0), k, v, is_causal=False)
        outs.append(out_i)
    return torch.cat(outs, dim=0)


def run_case(cfg, context_lens, label):
    B = len(context_lens)
    max_blocks = cfg.MAX_BLOCKS_PER_SEQ
    k_cache, v_cache, block_table = build_cache(cfg, context_lens, max_blocks)
    ctx = torch.tensor(context_lens, dtype=torch.int32, device=DEVICE)

    q = torch.randn(B, cfg.num_attention_heads, 1, cfg.head_dim, device=DEVICE, dtype=DTYPE)

    legacy = paged_decode_attention(q, k_cache, v_cache, block_table, ctx, MAX_BLOCKS_PER_SEQ=max_blocks)
    flash = paged_decode_attention_flash(q, k_cache, v_cache, block_table, ctx, MAX_BLOCKS_PER_SEQ=max_blocks)
    ref = sdpa_reference(q, k_cache, v_cache, block_table, ctx, cfg)

    def err(a, b):
        diff = (a.float() - b.float()).abs()
        return diff.max().item(), diff.mean().item()

    lmax, lmean = err(legacy, ref)
    fmax, fmean = err(flash, ref)
    flmax, flmean = err(flash, legacy)
    ok = fmax < 0.1
    print(f"[{label}] B={B} ctx={context_lens}")
    print(f"    legacy vs SDPA : max={lmax:.4f} mean={lmean:.5f}")
    print(f"    flash  vs SDPA : max={fmax:.4f} mean={fmean:.5f}")
    print(f"    flash  vs legacy: max={flmax:.4f} mean={flmean:.5f}  {'OK' if ok else 'FAIL'}")
    return ok


def main():
    cfg = ModelConfig()
    torch.manual_seed(0)
    cases = [
        ([64], "single short"),
        ([512], "single mid"),
        ([2048], "single long"),
        ([2048, 2048, 2048, 2048], "batch equal long"),
        ([37, 512, 1031, 2048], "batch unequal"),
        ([16, 1, 33], "batch tiny/edge"),
    ]
    all_ok = True
    for ctx, label in cases:
        all_ok &= run_case(cfg, ctx, label)
    print("=" * 50)
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
