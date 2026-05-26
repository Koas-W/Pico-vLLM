"""GQA-grouped, split-KV flash-decoding attention kernel.

Drop-in replacement for `attention.paged_decode_attention` with the SAME
signature and the SAME (B, N_HEAD, 1, HEAD_DIM) output layout, so the model /
engine need no changes. Used by default; set env `PICO_ATTN=legacy`
(see ops/triton/backend.py) to fall back to the original kernel.

Optimizations over the legacy decode kernel:
  1. Split-KV (flash-decoding): grid gains a `num_splits` axis so the KV
     history is processed by many CTAs in parallel instead of one CTA serially
     walking all blocks -> saturates the H200's 132 SMs. Wall time goes from
     O(context) to ~O(context / num_splits).
  2. GQA grouping: one program serves all `group_size` query heads that share a
     KV head, so K/V is loaded from HBM ONCE per group instead of `group_size`
     times (removes the legacy 6x KV read amplification). Q@K and P@V use tl.dot
     (tensor cores), with the group padded to 16 rows for MMA.
  3. Light autotune over num_warps/num_stages (resolved during the engine's
     pre-capture warmup, so it is CUDA-graph safe).

CUDA-graph safety: `num_splits` is a launch-time constant (env `PICO_DECODE_SPLITS`,
default 32), never derived from a runtime `.item()`. Per-split work adapts to
`context_lens` INSIDE the kernel. Intermediate partial buffers are allocated with
torch.empty inside the wrapper, so during capture they are taken over by the
graph memory pool (shapes are constant across replays).
"""
import os

import torch
import triton
import triton.language as tl


_NUM_SPLITS = int(os.environ.get("PICO_DECODE_SPLITS", "32"))


# NOTE: autotune is intentionally NOT used. The engine warms up / captures the
# CUDA graph with context_lens=0 (empty-loop), so an autotuner would pick a
# config on meaningless work and bake it into the graph. A fixed num_warps is
# more honest; real autotune needs a realistic-ctx warmup (follow-up).
@triton.jit
def _flash_decode_partial_kernel(
    q,                  # (B, N_HEAD, HEAD_DIM)  (decode: the seq=1 dim collapsed)
    k_cache,            # (num_blocks, N_KV_HEAD, BLOCK_SIZE, HEAD_DIM)
    v_cache,
    block_table,        # (B, MAX_BLOCKS_PER_SEQ) int32
    context_lens,       # (B,) int32
    scale,
    partial_acc,        # (B, N_HEAD, NUM_SPLITS, HEAD_DIM) fp32
    partial_m,          # (B, N_HEAD, NUM_SPLITS) fp32
    partial_l,          # (B, N_HEAD, NUM_SPLITS) fp32
    MAX_BLOCKS_PER_SEQ: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    N_KV_HEAD: tl.constexpr,
    N_HEAD: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    PADDED_GROUP: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_kv = tl.program_id(1)
    pid_s = tl.program_id(2)

    ctx = tl.load(context_lens + pid_b)
    q_head0 = pid_kv * GROUP_SIZE

    row = tl.arange(0, PADDED_GROUP)          # (PG,)
    row_mask = row < GROUP_SIZE
    d = tl.arange(0, HEAD_DIM)                 # (HD,)
    qheads = q_head0 + row                     # (PG,) absolute q-head ids

    # partial output pointers for this (b, group, split)
    pm_ptr = partial_m + pid_b * (N_HEAD * NUM_SPLITS) + qheads * NUM_SPLITS + pid_s
    pl_ptr = partial_l + pid_b * (N_HEAD * NUM_SPLITS) + qheads * NUM_SPLITS + pid_s
    pacc_ptr = (partial_acc
                + pid_b * (N_HEAD * NUM_SPLITS * HEAD_DIM)
                + qheads[:, None] * (NUM_SPLITS * HEAD_DIM)
                + pid_s * HEAD_DIM
                + d[None, :])

    # online-softmax state (per padded q-head row)
    m_i = tl.full((PADDED_GROUP,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((PADDED_GROUP,), dtype=tl.float32)
    acc = tl.zeros((PADDED_GROUP, HEAD_DIM), dtype=tl.float32)

    # load Q group: (PG, HD)
    q_ptrs = q + pid_b * (N_HEAD * HEAD_DIM) + qheads[:, None] * HEAD_DIM + d[None, :]
    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0)

    # this split's block range; empty splits skip the loop and write the
    # sentinel (m=-inf, l=0, acc=0) so the combine step ignores them.
    total_blocks = tl.cdiv(ctx, BLOCK_SIZE)
    blocks_per_split = tl.cdiv(total_blocks, NUM_SPLITS)
    start_block = pid_s * blocks_per_split
    end_block = tl.minimum(start_block + blocks_per_split, total_blocks)

    offs = tl.arange(0, BLOCK_SIZE)
    for blk in range(start_block, end_block):
        phys = tl.load(block_table + pid_b * MAX_BLOCKS_PER_SEQ + blk)
        phys = tl.maximum(phys, 0).to(tl.int64)
        base = phys * (N_KV_HEAD * BLOCK_SIZE * HEAD_DIM) + pid_kv * (BLOCK_SIZE * HEAD_DIM)

        tok_pos = blk * BLOCK_SIZE + offs                 # (BS,)
        kv_valid = tok_pos < ctx
        kv_ptrs = base + offs[:, None] * HEAD_DIM + d[None, :]   # (BS, HD)
        k_blk = tl.load(k_cache + kv_ptrs, mask=kv_valid[:, None], other=0.0)
        v_blk = tl.load(v_cache + kv_ptrs, mask=kv_valid[:, None], other=0.0)

        # S = Q @ K^T : (PG, BS)
        s = tl.dot(q_tile, tl.trans(k_blk)).to(tl.float32) * scale
        s = tl.where(kv_valid[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])                    # (PG, BS)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_blk.dtype), v_blk).to(tl.float32)
        m_i = m_new

    tl.store(pm_ptr, m_i, mask=row_mask)
    tl.store(pl_ptr, l_i, mask=row_mask)
    tl.store(pacc_ptr, acc, mask=row_mask[:, None])


@triton.jit
def _flash_decode_combine_kernel(
    partial_acc,        # (B, N_HEAD, NUM_SPLITS, HEAD_DIM) fp32
    partial_m,          # (B, N_HEAD, NUM_SPLITS) fp32
    partial_l,          # (B, N_HEAD, NUM_SPLITS) fp32
    context_lens,       # (B,) int32
    out,                # (B, N_HEAD, HEAD_DIM)
    HEAD_DIM: tl.constexpr,
    N_HEAD: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    d = tl.arange(0, HEAD_DIM)
    out_ptr = out + pid_b * (N_HEAD * HEAD_DIM) + pid_h * HEAD_DIM + d

    ctx = tl.load(context_lens + pid_b)
    if ctx == 0:
        tl.store(out_ptr, tl.zeros((HEAD_DIM,), dtype=out.dtype.element_ty))
        return

    s_idx = tl.arange(0, NUM_SPLITS)
    m_s = tl.load(partial_m + pid_b * (N_HEAD * NUM_SPLITS) + pid_h * NUM_SPLITS + s_idx)
    l_s = tl.load(partial_l + pid_b * (N_HEAD * NUM_SPLITS) + pid_h * NUM_SPLITS + s_idx)
    acc_ptrs = (partial_acc
                + pid_b * (N_HEAD * NUM_SPLITS * HEAD_DIM)
                + pid_h * (NUM_SPLITS * HEAD_DIM)
                + s_idx[:, None] * HEAD_DIM
                + d[None, :])
    acc_s = tl.load(acc_ptrs)                            # (NS, HD)

    m_g = tl.max(m_s)                                    # scalar
    factor = tl.exp(m_s - m_g)                           # (NS,)
    l_g = tl.sum(l_s * factor)
    acc_g = tl.sum(acc_s * factor[:, None], axis=0)      # (HD,)
    out_v = acc_g / l_g
    tl.store(out_ptr, out_v.to(out.dtype.element_ty))


@torch.compiler.disable
def paged_decode_attention_flash(
    q, k_cache, v_cache, block_table, context_lens,
    MAX_BLOCKS_PER_SEQ, BLOCK_SIZE=16,
):
    """Same signature/return as attention.paged_decode_attention.

    q: (B, N_HEAD, 1, HEAD_DIM)  ->  out: (B, N_HEAD, 1, HEAD_DIM)
    """
    B, N_HEAD, _, HEAD_DIM = q.shape
    N_KV_HEAD = k_cache.shape[1]
    GROUP_SIZE = N_HEAD // N_KV_HEAD
    PADDED_GROUP = max(16, triton.next_power_of_2(GROUP_SIZE))
    NUM_SPLITS = _NUM_SPLITS
    scale = 1.0 / (HEAD_DIM ** 0.5)

    out = torch.empty(B, N_HEAD, 1, HEAD_DIM, device=q.device, dtype=q.dtype)
    partial_acc = torch.empty(B, N_HEAD, NUM_SPLITS, HEAD_DIM, device=q.device, dtype=torch.float32)
    partial_m = torch.empty(B, N_HEAD, NUM_SPLITS, device=q.device, dtype=torch.float32)
    partial_l = torch.empty(B, N_HEAD, NUM_SPLITS, device=q.device, dtype=torch.float32)

    _flash_decode_partial_kernel[(B, N_KV_HEAD, NUM_SPLITS)](
        q, k_cache, v_cache, block_table, context_lens, scale,
        partial_acc, partial_m, partial_l,
        MAX_BLOCKS_PER_SEQ=MAX_BLOCKS_PER_SEQ,
        BLOCK_SIZE=BLOCK_SIZE,
        HEAD_DIM=HEAD_DIM,
        N_KV_HEAD=N_KV_HEAD,
        N_HEAD=N_HEAD,
        GROUP_SIZE=GROUP_SIZE,
        PADDED_GROUP=PADDED_GROUP,
        NUM_SPLITS=NUM_SPLITS,
        num_warps=4,
    )
    _flash_decode_combine_kernel[(B, N_HEAD)](
        partial_acc, partial_m, partial_l, context_lens, out,
        HEAD_DIM=HEAD_DIM,
        N_HEAD=N_HEAD,
        NUM_SPLITS=NUM_SPLITS,
    )
    return out
