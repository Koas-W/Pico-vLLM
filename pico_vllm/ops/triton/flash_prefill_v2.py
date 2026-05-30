"""v2 GQA-grouped, causal paged prefill attention kernel (Hopper-tuned).

Drop-in for `attention.paged_prefill_attention` / `flash_prefill.
paged_prefill_attention_flash`: same call signature (incl. the `max_new_len`
passthrough) and the same (total_new_tokens, N_HEAD, HEAD_DIM) output layout,
so it can replace v1 with no model/engine edits.

What changed vs v1 (flash_prefill.py), and why -- all three come straight from
the vLLM unified-attention design ("The Anatomy of a Triton Attention Kernel",
arXiv:2511.11581, and vllm/v1/attention/ops/triton_unified_attention.py):

  1. KV TILE DECOUPLED FROM PAGE SIZE.
     v1 walked the paged KV cache one 16-token page at a time, so the K^T GEMM
     had contraction/N = BLOCK_SIZE = 16 and the P@V GEMM had K = 16 -- far too
     small to saturate Hopper tensor cores, and 512 inner iterations at 8K.
     v2 uses BLOCK_N (default 64) independent of the 16-token page: for each
     tile it gathers the per-token physical block from block_table
     (phys = block_table[pos // BLOCK_SIZE], off = pos % BLOCK_SIZE) and loads a
     (BLOCK_N, HEAD_DIM) K/V tile. The 16 tokens inside a page stay contiguous,
     so a 64-wide tile is 4 contiguous 16-token runs -> good L2 locality, and
     the GEMMs now run at N/K = 64.

  2. FLAT POWER-OF-2 GQA HEAD PACKING.
     v1 packed group*BLOCK_M_tokens then padded to next_pow2 -> a constant 25%
     wasted M rows for group=6 (48 -> 64). v2 fixes the M tile to a power of two
     (BLOCK_M, default 64) and derives BLOCK_Q = BLOCK_M // group query tokens
     per tile. M row r decodes as token_local = r // group, head = r % group.
     For group=6, BLOCK_M=64 -> BLOCK_Q=10, 60/64 rows used (6% waste).

  3. LARGER, HOPPER-FRIENDLY TILES + OFFLINE AUTOTUNE.
     BLOCK_M/BLOCK_N/num_warps/num_stages are module constants (overridable by
     the sweep script) rather than a runtime @triton.autotune, because per-key
     cold-start benchmarking lands on the first prefill of each new length and
     causes large one-off latency spikes (see flash_prefill.py history). The
     winning config is hardcoded after an offline sweep.

Split-KV does not apply to prefill (it already parallelizes over the query
dimension). Causal early-exit is kept: a Q tile only scans KV up to its own
causal horizon.
"""
import torch
import triton
import triton.language as tl


# Offline-tuned config for H200 / group=6 / head_dim=128. Picked from a sweep of
# BLOCK_M in {16,32,64,128} x BLOCK_N in {16,32,64,128} x num_warps {4,8} x
# num_stages {2,3,4} across seqlens 64..8192 (see profiling/bench_prefill_kernel.py).
# Winner BM=64/BN=32/w=4/st=3 is the most length-robust: best at 256-512 and 2048,
# and within 0.4% of the per-length optimum at 8192 (1.30x vs v1). Larger
# BM=128/BN=128/w=8 ties at long seqlens but loses on short prompts, and the
# original w=8 default was actually a trap (slower than v1 at long seqlens).
# Hardcoded rather than runtime @triton.autotune: per-key cold-start benchmarking
# lands on the first prefill of each new length and causes one-off latency spikes.
_V2_BLOCK_M = 64        # M-dim tile (query rows = tokens * group), power of two
_V2_BLOCK_N = 32        # KV-dim tile, decoupled from the 16-token page size
_V2_NUM_WARPS = 4
_V2_NUM_STAGES = 3


@triton.jit
def _flash_prefill_v2_kernel(
    q,                  # (total_new_tokens, N_HEAD, HEAD_DIM)
    k_cache, v_cache,   # (num_blocks, N_KV_HEAD, BLOCK_SIZE, HEAD_DIM)
    block_table,        # (B, MAX_BLOCKS_PER_SEQ)
    context_lens,       # (B,) total_len = prefix + new
    new_token_lens,     # (B,) M (new tokens this prefill)
    q_start_loc,        # (B,) start row of each request inside q
    scale, out,         # (total_new_tokens, N_HEAD, HEAD_DIM)
    MAX_BLOCKS_PER_SEQ: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,    # paged cache page size (16)
    HEAD_DIM: tl.constexpr,
    N_KV_HEAD: tl.constexpr,
    N_HEAD: tl.constexpr,
    GROUP_SIZE: tl.constexpr,    # q heads per kv head (6)
    BLOCK_M: tl.constexpr,       # M tile (power of two)
    BLOCK_Q: tl.constexpr,       # query tokens per tile = BLOCK_M // GROUP_SIZE
    BLOCK_N: tl.constexpr,       # KV tile (decoupled from BLOCK_SIZE)
):
    pid_b = tl.program_id(0)
    pid_qblk = tl.program_id(1)
    pid_kv = tl.program_id(2)

    new_len = tl.load(new_token_lens + pid_b)
    total_len = tl.load(context_lens + pid_b)
    prefix_len = total_len - new_len
    q_offset = tl.load(q_start_loc + pid_b)

    q_tok_start = pid_qblk * BLOCK_Q          # first query token of this tile
    if q_tok_start >= new_len:
        return

    # --- M-dim layout: row r -> (token_local, head_in_group) ----------------
    # token outer, head inner (matches vLLM: token = r // group, head = r % group)
    r = tl.arange(0, BLOCK_M)
    token_local = r // GROUP_SIZE             # (BLOCK_M,)
    head_in_group = r % GROUP_SIZE
    abs_head = pid_kv * GROUP_SIZE + head_in_group
    q_row = q_offset + q_tok_start + token_local
    row_valid = (token_local < BLOCK_Q) & ((q_tok_start + token_local) < new_len)

    d = tl.arange(0, HEAD_DIM)
    q_ptrs = q + q_row[:, None] * (N_HEAD * HEAD_DIM) + abs_head[:, None] * HEAD_DIM + d[None, :]
    q_tile = tl.load(q_ptrs, mask=row_valid[:, None], other=0.0)   # (BLOCK_M, HEAD_DIM)

    # global key position of each query row (for causal mask); head-independent
    q_pos_global = prefix_len + q_tok_start + token_local         # (BLOCK_M,)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # causal early-exit: only scan keys up to the last query's horizon
    causal_max = prefix_len + q_tok_start + BLOCK_Q              # exclusive-ish
    kv_upper = tl.minimum(total_len, causal_max)
    num_kv_tiles = tl.cdiv(kv_upper, BLOCK_N)

    offs_n = tl.arange(0, BLOCK_N)
    for i in range(0, num_kv_tiles):
        kv_pos = i * BLOCK_N + offs_n                            # (BLOCK_N,)
        kv_valid = kv_pos < total_len

        # gather per-token physical block (tile spans BLOCK_N/BLOCK_SIZE pages)
        table_idx = kv_pos // BLOCK_SIZE
        phys = tl.load(block_table + pid_b * MAX_BLOCKS_PER_SEQ + table_idx,
                       mask=kv_valid, other=0)
        phys = tl.maximum(phys, 0).to(tl.int64)
        in_blk = kv_pos % BLOCK_SIZE
        kv_base = (phys * (N_KV_HEAD * BLOCK_SIZE * HEAD_DIM)
                   + pid_kv * (BLOCK_SIZE * HEAD_DIM)
                   + in_blk * HEAD_DIM)                          # (BLOCK_N,)

        kv_ptrs = kv_base[:, None] + d[None, :]                  # (BLOCK_N, HEAD_DIM)
        k_tile = tl.load(k_cache + kv_ptrs, mask=kv_valid[:, None], other=0.0)
        v_tile = tl.load(v_cache + kv_ptrs, mask=kv_valid[:, None], other=0.0)

        s = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * scale  # (BLOCK_M, BLOCK_N)
        causal = q_pos_global[:, None] >= kv_pos[None, :]
        s = tl.where(causal & kv_valid[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile).to(tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]
    out_ptrs = out + q_row[:, None] * (N_HEAD * HEAD_DIM) + abs_head[:, None] * HEAD_DIM + d[None, :]
    tl.store(out_ptrs, acc.to(out.dtype.element_ty), mask=row_valid[:, None])


@torch.compiler.disable
def paged_prefill_attention_v2(
    q, k_cache, v_cache, block_table,
    context_lens, new_token_lens, q_start_loc,
    MAX_BLOCKS_PER_SEQ, BLOCK_SIZE=16, BLOCK_M=16,
    max_new_len: int | None = None,
    # v2 tunables (sweep script overrides these; default = offline winner)
    block_m: int = _V2_BLOCK_M,
    block_n: int = _V2_BLOCK_N,
    num_warps: int = _V2_NUM_WARPS,
    num_stages: int = _V2_NUM_STAGES,
):
    """Same signature/return as flash_prefill.paged_prefill_attention_flash.

    `BLOCK_M` in the positional args is the legacy v1 knob and is ignored here;
    v2 uses `block_m`/`block_n` (M tile / KV tile). `max_new_len` lets the
    caller skip the per-call .item() sync (engine passes new_len directly).
    """
    total_new_tokens, N_HEAD, HEAD_DIM = q.shape
    N_KV_HEAD = k_cache.shape[1]
    GROUP_SIZE = N_HEAD // N_KV_HEAD
    B = context_lens.shape[0]
    scale = 1.0 / (HEAD_DIM ** 0.5)

    out = torch.empty_like(q)
    if max_new_len is None:
        max_new_len = int(new_token_lens.max().item())

    # M tile must hold at least one full group; BLOCK_Q query tokens per tile.
    assert block_m % GROUP_SIZE == 0 or block_m >= GROUP_SIZE, "block_m too small for group"
    BLOCK_Q = max(1, block_m // GROUP_SIZE)
    num_q_blocks = (max_new_len + BLOCK_Q - 1) // BLOCK_Q

    grid = (B, num_q_blocks, N_KV_HEAD)
    _flash_prefill_v2_kernel[grid](
        q, k_cache, v_cache, block_table,
        context_lens, new_token_lens, q_start_loc,
        scale, out,
        MAX_BLOCKS_PER_SEQ=MAX_BLOCKS_PER_SEQ,
        BLOCK_SIZE=BLOCK_SIZE,
        HEAD_DIM=HEAD_DIM,
        N_KV_HEAD=N_KV_HEAD,
        N_HEAD=N_HEAD,
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_M=block_m,
        BLOCK_Q=BLOCK_Q,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
