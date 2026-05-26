import pytest
import torch

from blockmanager import BlockManager
from model import ModelConfig, Qwen25_15B


pytestmark = [pytest.mark.cpu, pytest.mark.single_card]


def test_tiny_random_model_prefill_and_forward_decode_on_cpu():
    torch.manual_seed(0)
    dtype = torch.float32
    device = torch.device("cpu")

    cfg = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        use_cuda=False,
    )
    model = Qwen25_15B(cfg).to(device=device, dtype=dtype).eval()
    bm = BlockManager(
        num_gpu_blocks=8,
        num_cpu_blocks=0,
        block_size=cfg.BLOCK_SIZE,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=dtype,
        device=device,
    )

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
    seq_len = input_ids.shape[1]
    block_table = torch.full((1, 4), -1, dtype=torch.int32, device=device)
    block_table[0, 0] = 0

    with torch.no_grad():
        logits = model(
            input_ids,
            kv_cache_k=bm.gpu_kv_cache[0],
            kv_cache_v=bm.gpu_kv_cache[1],
            position_ids=torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0),
            slot_mapping=torch.arange(seq_len, dtype=torch.int32, device=device),
            is_prefill=True,
            block_table=block_table,
            context_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
            new_token_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
            q_start_loc=torch.tensor([0], dtype=torch.int32, device=device),
        )

        decode_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decode_logits = model.forward_decode(
            decode_ids,
            kv_cache_k=bm.gpu_kv_cache[0],
            kv_cache_v=bm.gpu_kv_cache[1],
            position_ids=torch.tensor([[seq_len]], dtype=torch.long, device=device),
            slot_mapping=torch.tensor([seq_len], dtype=torch.int32, device=device),
            block_table=block_table,
            context_lens=torch.tensor([seq_len + 1], dtype=torch.int32, device=device),
        )

    assert model.ops.device_type == "cpu"
    assert logits.shape == (1, seq_len, cfg.vocab_size)
    assert decode_logits.shape == (1, 1, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(decode_logits).all()
