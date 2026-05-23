import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ops.torch import TorchOps


pytestmark = [pytest.mark.cpu, pytest.mark.ops]


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x, cos, sin):
    return x * cos.unsqueeze(2) + _rotate_half(x) * sin.unsqueeze(2)


def _scatter_full_kv(k_cache, v_cache, block_table, batch_idx, k_full, v_full, block_size):
    total_len = k_full.shape[0]
    num_blocks = (total_len + block_size - 1) // block_size
    for block_idx in range(num_blocks):
        phys_id = int(block_table[batch_idx, block_idx].item())
        start = block_idx * block_size
        end = min(start + block_size, total_len)
        k_cache[phys_id, :, : end - start, :] = k_full[start:end].transpose(0, 1)
        v_cache[phys_id, :, : end - start, :] = v_full[start:end].transpose(0, 1)


def _ref_decode_attention(q, k_full, v_full):
    num_heads = q.shape[0]
    head_dim = q.shape[-1]
    num_kv_heads = k_full.shape[1]
    kv_groups = num_heads // num_kv_heads

    k = k_full.repeat_interleave(kv_groups, dim=1).transpose(0, 1)
    v = v_full.repeat_interleave(kv_groups, dim=1).transpose(0, 1)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / (head_dim ** 0.5)
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


def _ref_prefill_attention(q, k_full, v_full, prefix_len):
    new_len, num_heads, head_dim = q.shape
    total_len = k_full.shape[0]
    num_kv_heads = k_full.shape[1]
    kv_groups = num_heads // num_kv_heads

    q_h = q.transpose(0, 1).float()
    k_h = k_full.repeat_interleave(kv_groups, dim=1).transpose(0, 1).float()
    v_h = v_full.repeat_interleave(kv_groups, dim=1).transpose(0, 1).float()

    scores = torch.matmul(q_h, k_h.transpose(-1, -2)) / (head_dim ** 0.5)
    q_pos = torch.arange(prefix_len, prefix_len + new_len)
    k_pos = torch.arange(total_len)
    scores = scores.masked_fill(~(q_pos[:, None] >= k_pos[None, :]).unsqueeze(0), float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v_h).transpose(0, 1).to(q.dtype)


def test_rms_norm_and_swiglu_match_torch_reference():
    torch.manual_seed(0)
    ops = TorchOps()

    x = torch.randn(2, 3, 8)
    norm = ops.create_rms_norm(8, eps=1e-6)
    norm.weight.data = torch.randn(8)
    ref = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * norm.weight
    assert torch.allclose(norm(x), ref, atol=1e-6, rtol=1e-6)

    gate_up = torch.randn(2, 3, 16)
    gate, up = gate_up.chunk(2, dim=-1)
    assert torch.allclose(ops.swiglu(gate_up), F.silu(gate) * up, atol=1e-6, rtol=1e-6)


def test_store_kvcache_and_decode_rope_cache():
    torch.manual_seed(1)
    ops = TorchOps()
    block_size = 4
    num_blocks = 4
    num_kv_heads = 2
    head_dim = 8

    k_cache = torch.zeros(num_blocks, num_kv_heads, block_size, head_dim)
    v_cache = torch.zeros_like(k_cache)
    k = torch.randn(3, num_kv_heads, head_dim)
    v = torch.randn(3, num_kv_heads, head_dim)
    slot_mapping = torch.tensor([0, 5, 10], dtype=torch.int32)

    ops.store_kvcache(k, v, k_cache, v_cache, slot_mapping, block_size=block_size)

    for token_idx, slot in enumerate(slot_mapping.tolist()):
        block_id = slot // block_size
        offset = slot % block_size
        assert torch.allclose(k_cache[block_id, :, offset, :], k[token_idx])
        assert torch.allclose(v_cache[block_id, :, offset, :], v[token_idx])

    batch_size = 2
    num_q_heads = 4
    q = torch.randn(batch_size, 1, num_q_heads, head_dim)
    k_dec = torch.randn(batch_size, 1, num_kv_heads, head_dim)
    v_dec = torch.randn(batch_size, 1, num_kv_heads, head_dim)
    cos = torch.randn(batch_size, 1, head_dim)
    sin = torch.randn(batch_size, 1, head_dim)
    context_lens = torch.tensor([7, 0], dtype=torch.int32)
    slots = torch.tensor([7, -1], dtype=torch.int32)

    k_cache.zero_()
    v_cache.zero_()
    q_rot = ops.decode_rope_and_cache(
        q, k_dec, v_dec, cos, sin, k_cache, v_cache, slots, context_lens
    )

    ref_q = _apply_rope(q, cos, sin)
    ref_k = _apply_rope(k_dec, cos, sin)
    assert torch.allclose(q_rot[0], ref_q[0])
    assert torch.count_nonzero(q_rot[1]) == 0
    assert torch.allclose(k_cache[1, :, 3, :], ref_k[0, 0])
    assert torch.allclose(v_cache[1, :, 3, :], v_dec[0, 0])
    assert torch.count_nonzero(k_cache[3]) == 0
    assert torch.count_nonzero(v_cache[3]) == 0


def test_paged_decode_attention_matches_reference():
    torch.manual_seed(2)
    ops = TorchOps()
    block_size = 4
    batch_size = 2
    num_blocks = 8
    num_heads = 4
    num_kv_heads = 2
    head_dim = 8

    k_cache = torch.zeros(num_blocks, num_kv_heads, block_size, head_dim)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.full((batch_size, 4), -1, dtype=torch.int32)
    block_table[0, :2] = torch.tensor([3, 1], dtype=torch.int32)
    block_table[1, :2] = torch.tensor([4, 6], dtype=torch.int32)
    context_lens = torch.tensor([5, 7], dtype=torch.int32)

    k_full = []
    v_full = []
    for batch_idx, seq_len in enumerate(context_lens.tolist()):
        k_b = torch.randn(seq_len, num_kv_heads, head_dim)
        v_b = torch.randn(seq_len, num_kv_heads, head_dim)
        _scatter_full_kv(k_cache, v_cache, block_table, batch_idx, k_b, v_b, block_size)
        k_full.append(k_b)
        v_full.append(v_b)

    q = torch.randn(batch_size, num_heads, 1, head_dim)
    out = ops.paged_decode_attention(
        q, k_cache, v_cache, block_table, context_lens, MAX_BLOCKS_PER_SEQ=4, BLOCK_SIZE=block_size
    )
    ref = torch.stack(
        [_ref_decode_attention(q[i], k_full[i], v_full[i]) for i in range(batch_size)]
    )

    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_paged_prefill_attention_matches_reference():
    torch.manual_seed(3)
    ops = TorchOps()
    block_size = 4
    batch_size = 2
    num_blocks = 8
    num_heads = 4
    num_kv_heads = 2
    head_dim = 8
    new_lens = [3, 2]
    prefix_lens = [2, 4]

    k_cache = torch.zeros(num_blocks, num_kv_heads, block_size, head_dim)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.full((batch_size, 4), -1, dtype=torch.int32)
    block_table[0, :2] = torch.tensor([0, 5], dtype=torch.int32)
    block_table[1, :2] = torch.tensor([2, 7], dtype=torch.int32)

    q_parts = []
    q_start_loc = []
    context_lens = []
    refs = []
    offset = 0

    for batch_idx, (new_len, prefix_len) in enumerate(zip(new_lens, prefix_lens)):
        total_len = prefix_len + new_len
        q_b = torch.randn(new_len, num_heads, head_dim)
        k_b = torch.randn(total_len, num_kv_heads, head_dim)
        v_b = torch.randn(total_len, num_kv_heads, head_dim)

        _scatter_full_kv(k_cache, v_cache, block_table, batch_idx, k_b, v_b, block_size)
        refs.append(_ref_prefill_attention(q_b, k_b, v_b, prefix_len))
        q_parts.append(q_b)
        q_start_loc.append(offset)
        context_lens.append(total_len)
        offset += new_len

    q = torch.cat(q_parts, dim=0)
    out = ops.paged_prefill_attention(
        q,
        k_cache,
        v_cache,
        block_table,
        torch.tensor(context_lens, dtype=torch.int32),
        torch.tensor(new_lens, dtype=torch.int32),
        torch.tensor(q_start_loc, dtype=torch.int32),
        MAX_BLOCKS_PER_SEQ=4,
        BLOCK_SIZE=block_size,
        BLOCK_M=block_size,
    )

    ref = torch.cat(refs, dim=0)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
