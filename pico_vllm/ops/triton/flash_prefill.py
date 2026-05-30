"""GQA-grouped, causal prefill attention kernel.

Companion to flash_decode.py, gated by the SAME env var `PICO_ATTN`
(default flash; PICO_ATTN=legacy falls back). Drop-in for
`attention.paged_prefill_attention` (same signature and
(total_new_tokens, N_HEAD, HEAD_DIM) output layout) -> no model/engine edits.

Split-KV does NOT apply to prefill (it already parallelizes over the query
dimension). The matching/extra optimizations here are:
  1. GQA grouping: one program serves all `group_size` q-heads sharing a KV head
     -> K/V loaded from HBM once per group (removes the legacy 6x KV re-read).
     The group's queries are stacked into the GEMM M dimension (group*BLOCK_M),
     so Q@K and P@V stay tensor-core GEMMs.
  2. Causal early-exit: a Q tile only loops KV blocks up to its own causal
     horizon instead of the whole sequence (~2x less work on the triangle).

Prefill runs eagerly (not under CUDA graph), so a host-side .item() for the grid
is fine, exactly like the legacy wrapper.
"""
import torch
import triton
import triton.language as tl


# Block config: tuned offline with triton.autotune over BLOCK_M in {8,16,32} x
# num_warps {4,8} x num_stages {2,3}. BLOCK_M=8 / num_warps=4 / num_stages=3 won
# at EVERY sequence length (512..8192) -- larger BLOCK_M spills registers because
# the GQA-grouped acc is (GROUP_SIZE*BLOCK_M padded to pow2) x HEAD_DIM fp32. The
# winner is hardcoded rather than left as runtime autotune, because autotune's
# per-key cold-start benchmarking would otherwise land on the first prefill of
# each new length and cause large one-off latency spikes (it blew a single-shot
# benchmark from ~90ms to ~1.6s). See profiling/bench_prefill_kernel.py.
_PREFILL_BLOCK_M = 8
_PREFILL_NUM_WARPS = 4
_PREFILL_NUM_STAGES = 3


@triton.jit
def _flash_prefill_grouped_kernel(
    q,                  # (total_new_tokens, N_HEAD, HEAD_DIM)
    k_cache, v_cache,   # (num_blocks, N_KV_HEAD, BLOCK_SIZE, HEAD_DIM)
    block_table,        # (B, MAX_BLOCKS_PER_SEQ)
    context_lens,       # (B,) total_len = prefix + new
    new_token_lens,     # (B,) M
    q_start_loc,        # (B,)
    scale, out,         # (total_new_tokens, N_HEAD, HEAD_DIM)
    MAX_BLOCKS_PER_SEQ: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    N_KV_HEAD: tl.constexpr,
    N_HEAD: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    PADDED_M: tl.constexpr,    # next_pow2(GROUP_SIZE * BLOCK_M)
):
    pid_b = tl.program_id(0)
    pid_qblk = tl.program_id(1)
    pid_kv = tl.program_id(2)

    new_len = tl.load(new_token_lens + pid_b)
    total_len = tl.load(context_lens + pid_b)
    prefix_len = total_len - new_len
    q_offset = tl.load(q_start_loc + pid_b)

    q_tile_start = pid_qblk * BLOCK_M
    if q_tile_start >= new_len:
        return

    # Stacked rows: r -> head_in_group = r // BLOCK_M, qpos_local = r % BLOCK_M
    r = tl.arange(0, PADDED_M)
    head_in_group = r // BLOCK_M
    qpos_local = r % BLOCK_M
    abs_head = pid_kv * GROUP_SIZE + head_in_group           # (PM,)
    q_row = q_offset + q_tile_start + qpos_local             # (PM,) row in q tensor
    row_valid = (head_in_group < GROUP_SIZE) & (qpos_local < (new_len - q_tile_start))

    d = tl.arange(0, HEAD_DIM)
    q_ptrs = q + q_row[:, None] * (N_HEAD * HEAD_DIM) + abs_head[:, None] * HEAD_DIM + d[None, :]
    q_tile = tl.load(q_ptrs, mask=row_valid[:, None], other=0.0)  # (PM, HD)

    q_pos_global = prefix_len + q_tile_start + qpos_local         # (PM,) for causal

    m_i = tl.full((PADDED_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((PADDED_M,), dtype=tl.float32)
    acc = tl.zeros((PADDED_M, HEAD_DIM), dtype=tl.float32)

    # causal early-exit: only KV up to the last query's horizon in this tile
    causal_max = prefix_len + q_tile_start + BLOCK_M             # exclusive-ish upper bound
    kv_upper = tl.minimum(total_len, causal_max)
    num_kv_blocks = tl.cdiv(kv_upper, BLOCK_SIZE)

    offs = tl.arange(0, BLOCK_SIZE)
    for blk in range(0, num_kv_blocks):
        phys = tl.load(block_table + pid_b * MAX_BLOCKS_PER_SEQ + blk)
        phys = tl.maximum(phys, 0).to(tl.int64)
        base = phys * (N_KV_HEAD * BLOCK_SIZE * HEAD_DIM) + pid_kv * (BLOCK_SIZE * HEAD_DIM)

        k_pos = blk * BLOCK_SIZE + offs                          # (BS,)
        kv_valid = k_pos < total_len
        kv_ptrs = base + offs[:, None] * HEAD_DIM + d[None, :]
        k_blk = tl.load(k_cache + kv_ptrs, mask=kv_valid[:, None], other=0.0)
        v_blk = tl.load(v_cache + kv_ptrs, mask=kv_valid[:, None], other=0.0)

        s = tl.dot(q_tile, tl.trans(k_blk)).to(tl.float32) * scale   # (PM, BS)
        causal = q_pos_global[:, None] >= k_pos[None, :]
        s = tl.where(causal & kv_valid[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_blk.dtype), v_blk).to(tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]
    out_ptrs = out + q_row[:, None] * (N_HEAD * HEAD_DIM) + abs_head[:, None] * HEAD_DIM + d[None, :]
    tl.store(out_ptrs, acc.to(out.dtype.element_ty), mask=row_valid[:, None])


@torch.compiler.disable
def paged_prefill_attention_flash(
    q, k_cache, v_cache, block_table,
    context_lens, new_token_lens, q_start_loc,
    MAX_BLOCKS_PER_SEQ, BLOCK_SIZE=16, BLOCK_M=16,
    max_new_len: int | None = None,
):
    """Same signature/return as attention.paged_prefill_attention."""
    total_new_tokens, N_HEAD, HEAD_DIM = q.shape
    N_KV_HEAD = k_cache.shape[1]
    GROUP_SIZE = N_HEAD // N_KV_HEAD
    B = context_lens.shape[0]
    scale = 1.0 / (HEAD_DIM ** 0.5)

    out = torch.empty_like(q)
    # 调用方(model.forward)在 prefill 入口处一次性算好 max_new_len,
    # 避免每层都做一次 .max().item() 的 host↔device 同步(28 层 × 5-10µs)。
    if max_new_len is None:
        max_new_len = int(new_token_lens.max().item())

    BLOCK_M = _PREFILL_BLOCK_M
    PADDED_M = triton.next_power_of_2(GROUP_SIZE * BLOCK_M)
    num_q_blocks = (max_new_len + BLOCK_M - 1) // BLOCK_M

    grid = (B, num_q_blocks, N_KV_HEAD)
    _flash_prefill_grouped_kernel[grid](
        q, k_cache, v_cache, block_table,
        context_lens, new_token_lens, q_start_loc,
        scale, out,
        MAX_BLOCKS_PER_SEQ=MAX_BLOCKS_PER_SEQ,
        BLOCK_SIZE=BLOCK_SIZE,
        HEAD_DIM=HEAD_DIM,
        N_KV_HEAD=N_KV_HEAD,
        N_HEAD=N_HEAD,
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_M=BLOCK_M,
        PADDED_M=PADDED_M,
        num_warps=_PREFILL_NUM_WARPS,
        num_stages=_PREFILL_NUM_STAGES,
    )
    return out
