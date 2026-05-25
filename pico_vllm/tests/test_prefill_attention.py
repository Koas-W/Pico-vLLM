"""Correctness check for the prefill attention kernels.

Compares the legacy prefill kernel and the new GQA-grouped causal flash prefill
kernel against a PyTorch SDPA causal reference, including a prefix>0 (prefix
cache hit) case.

Run from repo root:
  PYTHONPATH=pico_vllm .venv-vllm019/bin/python pico_vllm/tests/test_prefill_attention.py
"""
import sys

import torch
import torch.nn.functional as F

from blockmanager import BlockManager
from model import ModelConfig
from ops.triton.attention import paged_prefill_attention
from ops.triton.flash_prefill import paged_prefill_attention_flash

DEVICE = "cuda"
DTYPE = torch.bfloat16
BLOCK_SIZE = 16


def build_cache(cfg, total_lens, max_blocks):
    B = len(total_lens)
    needed = sum((L + BLOCK_SIZE - 1) // BLOCK_SIZE for L in total_lens)
    bm = BlockManager(
        num_gpu_blocks=needed + 8, num_cpu_blocks=0, block_size=BLOCK_SIZE, num_layers=1,
        num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim, dtype=DTYPE, device=DEVICE,
    )
    k_cache = bm.gpu_kv_cache[0, 0]
    v_cache = bm.gpu_kv_cache[1, 0]
    block_table = torch.full((B, max_blocks), -1, dtype=torch.int32, device=DEVICE)
    nb = 0
    for i, L in enumerate(total_lens):
        for j in range((L + BLOCK_SIZE - 1) // BLOCK_SIZE):
            phys = nb; nb += 1
            block_table[i, j] = phys
            valid = min(BLOCK_SIZE, L - j * BLOCK_SIZE)
            k_cache[phys, :, :valid, :] = torch.randn(cfg.num_key_value_heads, valid, cfg.head_dim, device=DEVICE, dtype=DTYPE)
            v_cache[phys, :, :valid, :] = torch.randn(cfg.num_key_value_heads, valid, cfg.head_dim, device=DEVICE, dtype=DTYPE)
    return k_cache, v_cache, block_table


def sdpa_ref(q_i, k_cache, v_cache, phys, total_len, new_len, cfg):
    # q_i: (new_len, N_HEAD, HD)
    k = (k_cache[phys].permute(1, 0, 2, 3).reshape(cfg.num_key_value_heads, -1, cfg.head_dim)[:, :total_len]
         .repeat_interleave(cfg.num_kv_groups, dim=0))         # (N_HEAD, total, HD)
    v = (v_cache[phys].permute(1, 0, 2, 3).reshape(cfg.num_key_value_heads, -1, cfg.head_dim)[:, :total_len]
         .repeat_interleave(cfg.num_kv_groups, dim=0))
    prefix = total_len - new_len
    q = q_i.permute(1, 0, 2)                                    # (N_HEAD, new, HD)
    # causal mask: query at global pos (prefix+i) attends key j <= prefix+i
    qpos = torch.arange(prefix, total_len, device=DEVICE).unsqueeze(1)   # (new,1)
    kpos = torch.arange(total_len, device=DEVICE).unsqueeze(0)           # (1,total)
    mask = (kpos <= qpos)                                       # (new, total)
    out = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                                         attn_mask=mask.unsqueeze(0).unsqueeze(0))
    return out.squeeze(0).permute(1, 0, 2)                      # (new, N_HEAD, HD)


def run_case(cfg, specs, label):
    # specs: list of (total_len, new_len). prefix = total - new.
    total_lens = [t for t, _ in specs]
    new_lens = [n for _, n in specs]
    max_blocks = cfg.MAX_BLOCKS_PER_SEQ
    k_cache, v_cache, block_table = build_cache(cfg, total_lens, max_blocks)

    total_new = sum(new_lens)
    q = torch.randn(total_new, cfg.num_attention_heads, cfg.head_dim, device=DEVICE, dtype=DTYPE)
    ctx = torch.tensor(total_lens, dtype=torch.int32, device=DEVICE)
    ntl = torch.tensor(new_lens, dtype=torch.int32, device=DEVICE)
    qsl = torch.tensor([sum(new_lens[:i]) for i in range(len(new_lens))], dtype=torch.int32, device=DEVICE)

    legacy = paged_prefill_attention(q, k_cache, v_cache, block_table, ctx, ntl, qsl, MAX_BLOCKS_PER_SEQ=max_blocks)
    flash = paged_prefill_attention_flash(q, k_cache, v_cache, block_table, ctx, ntl, qsl, MAX_BLOCKS_PER_SEQ=max_blocks)

    refs = []
    for i, (t, n) in enumerate(specs):
        nblk = (t + BLOCK_SIZE - 1) // BLOCK_SIZE
        phys = block_table[i, :nblk]
        q_i = q[qsl[i]: qsl[i] + n]
        refs.append(sdpa_ref(q_i, k_cache, v_cache, phys, t, n, cfg))
    ref = torch.cat(refs, dim=0)

    def err(a, b):
        diff = (a.float() - b.float()).abs()
        return diff.max().item(), diff.mean().item()

    lmax, _ = err(legacy, ref)
    fmax, fmean = err(flash, ref)
    ok = fmax < 0.1
    print(f"[{label}] specs(total,new)={specs}")
    print(f"    legacy vs SDPA: max={lmax:.4f}")
    print(f"    flash  vs SDPA: max={fmax:.4f} mean={fmean:.5f}  {'OK' if ok else 'FAIL'}")
    return ok


def main():
    cfg = ModelConfig()
    torch.manual_seed(0)
    cases = [
        ([(64, 64)], "single short full"),
        ([(512, 512)], "single mid full"),
        ([(2048, 2048)], "single long full"),
        ([(2048, 2048), (2048, 2048)], "batch equal full"),
        ([(1031, 1031), (512, 512), (37, 37)], "batch unequal full"),
        ([(2048, 512)], "prefix hit (prefix=1536)"),
        ([(1024, 1)], "near-decode (new=1)"),
    ]
    ok = True
    for specs, label in cases:
        ok &= run_case(cfg, specs, label)
    print("=" * 50)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
