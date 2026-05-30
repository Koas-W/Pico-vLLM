import os

import torch
import torch.nn as nn

from ..base import OpsBackend


class TritonOps(OpsBackend):
    """Triton backend adapter over the existing kernel wrappers."""

    name = "triton"
    device_type = "cuda"
    supports_cuda_graph = True

    # Attention kernel selection (read once; constant for the run, so the choice
    # is baked into the CUDA graph at capture time). Controls BOTH decode and
    # prefill. Default "flash": GQA-grouped split-KV decode + (for prefill) the
    # page-decoupled, offline-tuned v2 kernel (flash_prefill_v2.py).
    # PICO_ATTN options for the PREFILL path:
    #   (default)      -> v2 page-decoupled kernel (BLOCK_N decoupled from page)
    #   PICO_ATTN=v1   -> previous GQA-grouped flash prefill (flash_prefill.py)
    #   PICO_ATTN=legacy -> original per-page prefill kernel (attention.py)
    # Decode follows the same flash/legacy split as before.
    _attn = os.environ.get("PICO_ATTN", "flash").lower()

    def create_rms_norm(self, hidden_size: int, eps: float = 1e-6) -> nn.Module:
        from .rms_norm import FastRMSNorm

        return FastRMSNorm(hidden_size, eps=eps)

    def swiglu(self, gate_up: torch.Tensor) -> torch.Tensor:
        from .swiglu import fused_swiglu

        return fused_swiglu(gate_up)

    def store_kvcache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int = 16,
    ) -> None:
        from .store_kvcache import store_kvcache

        store_kvcache(k, v, k_cache, v_cache, slot_mapping, block_size=block_size)

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
        from .fused_rope_kvcache_store import fused_decode_rope_and_cache

        return fused_decode_rope_and_cache(
            q, k, v, cos, sin,
            kv_cache_k, kv_cache_v,
            slot_mapping,
            context_lens,
        )

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
        """Generic fused RoPE (Q, K) + KV cache write. Same kernel as the decode
        path; `check_ctx_len=False` for prefill (no ghost padding rows, and
        per-token tok_idx would be out of bounds on the per-request
        context_lens tensor)."""
        from .fused_rope_kvcache_store import fused_decode_rope_and_cache

        return fused_decode_rope_and_cache(
            q, k, v, cos, sin,
            kv_cache_k, kv_cache_v,
            slot_mapping,
            context_lens,
            check_ctx_len=check_ctx_len,
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
        if self._attn == "legacy":
            from .attention import paged_decode_attention

            return paged_decode_attention(
                q,
                k_cache,
                v_cache,
                block_table,
                context_lens,
                MAX_BLOCKS_PER_SEQ=MAX_BLOCKS_PER_SEQ,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        from .flash_decode import paged_decode_attention_flash

        return paged_decode_attention_flash(
            q,
            k_cache,
            v_cache,
            block_table,
            context_lens,
            MAX_BLOCKS_PER_SEQ=MAX_BLOCKS_PER_SEQ,
            BLOCK_SIZE=BLOCK_SIZE,
        )

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
        # Default: v2 page-decoupled prefill kernel.
        if self._attn not in ("legacy", "v1"):
            from .flash_prefill_v2 import paged_prefill_attention_v2

            return paged_prefill_attention_v2(
                q,
                k_cache,
                v_cache,
                block_table,
                context_lens,
                new_token_lens,
                q_start_loc,
                MAX_BLOCKS_PER_SEQ=MAX_BLOCKS_PER_SEQ,
                BLOCK_SIZE=BLOCK_SIZE,
                BLOCK_M=BLOCK_M,
                max_new_len=max_new_len,
            )

        # Optional fallback: previous v1 GQA-grouped flash prefill.
        if self._attn == "v1":
            from .flash_prefill import paged_prefill_attention_flash

            return paged_prefill_attention_flash(
                q,
                k_cache,
                v_cache,
                block_table,
                context_lens,
                new_token_lens,
                q_start_loc,
                MAX_BLOCKS_PER_SEQ=MAX_BLOCKS_PER_SEQ,
                BLOCK_SIZE=BLOCK_SIZE,
                BLOCK_M=BLOCK_M,
                max_new_len=max_new_len,
            )

        from .attention import paged_prefill_attention

        return paged_prefill_attention(
            q,
            k_cache,
            v_cache,
            block_table,
            context_lens,
            new_token_lens,
            q_start_loc,
            MAX_BLOCKS_PER_SEQ=MAX_BLOCKS_PER_SEQ,
            BLOCK_SIZE=BLOCK_SIZE,
            BLOCK_M=BLOCK_M,
            max_new_len=max_new_len,
        )
