import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import OpsBackend


def _float_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos.unsqueeze(2) + _rotate_half(x) * sin.unsqueeze(2)


def _store_kvcache_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    valid_mask: torch.Tensor | None = None,
) -> None:
    total_tokens = k.shape[0]
    slots = slot_mapping.reshape(-1)
    if slots.numel() != total_tokens:
        raise ValueError(
            f"slot_mapping has {slots.numel()} entries, but k/v have "
            f"{total_tokens} tokens."
        )

    slots = slots.to(device=k_cache.device, dtype=torch.long)
    active = slots >= 0
    if valid_mask is not None:
        active = active & valid_mask.reshape(-1).to(device=k_cache.device, dtype=torch.bool)
    if not bool(active.any()):
        return

    active_slots = slots[active]
    block_ids = active_slots // block_size
    block_offsets = active_slots % block_size

    k_cache[block_ids, :, block_offsets, :] = k[active].to(k_cache.dtype)
    v_cache[block_ids, :, block_offsets, :] = v[active].to(v_cache.dtype)


def _token_valid_mask(
    context_lens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    context_lens = context_lens.reshape(-1).to(device=device)
    total_tokens = batch_size * seq_len

    if context_lens.numel() == total_tokens:
        return context_lens > 0
    if context_lens.numel() == batch_size:
        return (context_lens > 0).repeat_interleave(seq_len)
    raise ValueError(
        "context_lens must have either B entries or B * seq_len entries; "
        f"got {context_lens.numel()} for B={batch_size}, seq_len={seq_len}."
    )


def _gather_paged_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if seq_len <= 0:
        empty_shape = (k_cache.shape[1], 0, k_cache.shape[-1])
        return (
            torch.empty(empty_shape, dtype=k_cache.dtype, device=k_cache.device),
            torch.empty(empty_shape, dtype=v_cache.dtype, device=v_cache.device),
        )

    num_blocks = (seq_len + block_size - 1) // block_size
    phys_ids = block_table_row[:num_blocks].to(device=k_cache.device, dtype=torch.long)
    if bool((phys_ids < 0).any()):
        raise ValueError("block_table contains -1 for a block required by context_lens.")

    k = k_cache.index_select(0, phys_ids).permute(1, 0, 2, 3)
    k = k.reshape(k_cache.shape[1], num_blocks * block_size, k_cache.shape[-1])
    k = k[:, :seq_len, :]

    v = v_cache.index_select(0, phys_ids).permute(1, 0, 2, 3)
    v = v.reshape(v_cache.shape[1], num_blocks * block_size, v_cache.shape[-1])
    v = v[:, :seq_len, :]
    return k, v


class TorchRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_2d = x.reshape(-1, self.hidden_size)
        compute_dtype = _float_compute_dtype(x_2d.dtype)

        x_compute = x_2d.to(compute_dtype)
        variance = x_compute.pow(2).mean(dim=-1, keepdim=True)
        out = x_compute * torch.rsqrt(variance + self.eps)
        out = out * self.weight.to(dtype=compute_dtype, device=x.device)
        return out.to(dtype=x.dtype).reshape(x_shape)


class TorchOps(OpsBackend):
    """Pure PyTorch CPU backend for inference operators."""

    name = "torch"
    device_type = "cpu"
    supports_cuda_graph = False

    def create_rms_norm(self, hidden_size: int, eps: float = 1e-6) -> nn.Module:
        return TorchRMSNorm(hidden_size, eps)

    def swiglu(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        compute_dtype = _float_compute_dtype(gate_up.dtype)
        out = F.silu(gate.to(compute_dtype)) * up.to(compute_dtype)
        return out.to(dtype=gate_up.dtype)

    def store_kvcache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int = 16,
    ) -> None:
        _store_kvcache_impl(k, v, k_cache, v_cache, slot_mapping, block_size)

    def decode_rope_and_cache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache_k: torch.Tensor,
        kv_cache_v: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, num_q_heads, head_dim = q.shape
        q_rot = _apply_rope(q, cos, sin)
        k_rot = _apply_rope(k, cos, sin)

        valid = _token_valid_mask(context_lens, batch_size, seq_len, q.device)
        q_rot_2d = q_rot.reshape(batch_size * seq_len, num_q_heads, head_dim)
        k_rot_2d = k_rot.reshape(batch_size * seq_len, k.shape[2], head_dim)
        v_2d = v.reshape(batch_size * seq_len, v.shape[2], head_dim)

        _store_kvcache_impl(
            k_rot_2d,
            v_2d,
            kv_cache_k,
            kv_cache_v,
            slot_mapping,
            block_size=kv_cache_k.shape[2],
            valid_mask=valid,
        )

        if not bool(valid.all()):
            q_rot_2d = q_rot_2d.clone()
            q_rot_2d[~valid] = 0
            q_rot = q_rot_2d.reshape_as(q_rot)
        return q_rot

    def fused_rope_and_cache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache_k: torch.Tensor,
        kv_cache_v: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        check_ctx_len: bool = True,
    ) -> torch.Tensor:
        """CPU fused RoPE(Q, K) + KV cache write (parity with TritonOps).

        The unified eager forward path now calls ``fused_rope_and_cache`` for
        both prefill and eager decode, so the CPU backend must provide it too.
        ``check_ctx_len`` exists only for signature parity with the Triton
        kernel's ghost-row guard; on CPU the valid-token masking is derived from
        ``context_lens`` inside ``decode_rope_and_cache`` (``_token_valid_mask``
        accepts both ``(B,)`` prefill and ``(B,)``/``(B*seq,)`` decode shapes),
        and prefill carries no ghost rows, so no extra handling is needed here.
        """
        del check_ctx_len
        return self.decode_rope_and_cache(
            q, k, v, cos, sin,
            kv_cache_k, kv_cache_v,
            slot_mapping,
            context_lens,
        )

    def paged_decode_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        MAX_BLOCKS_PER_SEQ: int,
        BLOCK_SIZE: int = 16,
    ) -> torch.Tensor:
        del MAX_BLOCKS_PER_SEQ

        batch_size, num_heads, _, head_dim = q.shape
        num_kv_heads = k_cache.shape[1]
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")

        compute_dtype = _float_compute_dtype(q.dtype)
        scale = head_dim ** -0.5
        out = torch.zeros_like(q)

        for batch_idx in range(batch_size):
            seq_len = int(context_lens[batch_idx].item())
            if seq_len <= 0:
                continue

            k, v = _gather_paged_kv(
                k_cache, v_cache, block_table[batch_idx], seq_len, BLOCK_SIZE
            )
            kv_groups = num_heads // num_kv_heads
            k = k.repeat_interleave(kv_groups, dim=0).to(compute_dtype)
            v = v.repeat_interleave(kv_groups, dim=0).to(compute_dtype)

            q_i = q[batch_idx, :, :, :].to(compute_dtype)
            scores = torch.matmul(q_i, k.transpose(-1, -2)) * scale
            probs = torch.softmax(scores, dim=-1)
            out[batch_idx] = torch.matmul(probs, v).to(dtype=q.dtype)

        return out

    def paged_prefill_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        new_token_lens: torch.Tensor,
        q_start_loc: torch.Tensor,
        MAX_BLOCKS_PER_SEQ: int,
        BLOCK_SIZE: int = 16,
        BLOCK_M: int = 16,
        max_new_len: int | None = None,
    ) -> torch.Tensor:
        del MAX_BLOCKS_PER_SEQ, BLOCK_M, max_new_len

        _, num_heads, head_dim = q.shape
        num_kv_heads = k_cache.shape[1]
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")

        compute_dtype = _float_compute_dtype(q.dtype)
        scale = head_dim ** -0.5
        out = torch.zeros_like(q)
        batch_size = context_lens.shape[0]

        for batch_idx in range(batch_size):
            total_len = int(context_lens[batch_idx].item())
            new_len = int(new_token_lens[batch_idx].item())
            q_offset = int(q_start_loc[batch_idx].item())
            if total_len <= 0 or new_len <= 0:
                continue

            prefix_len = total_len - new_len
            if prefix_len < 0:
                raise ValueError("context_lens must be greater than or equal to new_token_lens.")

            k, v = _gather_paged_kv(
                k_cache, v_cache, block_table[batch_idx], total_len, BLOCK_SIZE
            )
            kv_groups = num_heads // num_kv_heads
            k = k.repeat_interleave(kv_groups, dim=0).to(compute_dtype)
            v = v.repeat_interleave(kv_groups, dim=0).to(compute_dtype)

            q_b = q[q_offset : q_offset + new_len].transpose(0, 1).to(compute_dtype)
            scores = torch.matmul(q_b, k.transpose(-1, -2)) * scale

            q_pos = torch.arange(
                prefix_len,
                prefix_len + new_len,
                device=q.device,
                dtype=torch.long,
            )
            k_pos = torch.arange(total_len, device=q.device, dtype=torch.long)
            causal_mask = q_pos[:, None] >= k_pos[None, :]
            scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

            probs = torch.softmax(scores, dim=-1)
            out_b = torch.matmul(probs, v).transpose(0, 1)
            out[q_offset : q_offset + new_len] = out_b.to(dtype=q.dtype)

        return out
