import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# test_tp_generate.py
import torch
import torch.distributed as dist
import os
from transformers import AutoTokenizer
from model import Qwen25_15B, ModelConfig
from weights import load_weights
from cache import PagedKVCache, BlockManager

def main():
    tp_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if tp_size > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    cfg = ModelConfig(tp_size=tp_size)
    model = Qwen25_15B(cfg)
    model = load_weights(model, "./weights", tp_size=tp_size, rank=rank)
    model = model.to(torch.bfloat16).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("./weights")
    BLOCK_SIZE = 16

    bm = BlockManager(
        num_gpu_blocks=200, num_cpu_blocks=0,
        block_size=BLOCK_SIZE, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.local_num_key_value_heads,
        head_dim=cfg.head_dim, dtype=torch.bfloat16,
    )
    cache = PagedKVCache(
        block_manager=bm, num_layers=cfg.num_hidden_layers,
        max_seq_len=512, num_kv_heads=cfg.local_num_key_value_heads,
        head_dim=cfg.head_dim, device=device, dtype=torch.bfloat16,
    )

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    seq_len = input_ids.shape[1]

    # === Prefill ===
    cache._allocate_for_prefill(seq_len)
    slot_mapping = cache.get_prefill_slot_mapping(seq_len).to(device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(
            input_ids,
            kv_cache_k=bm.gpu_kv_cache[0], kv_cache_v=bm.gpu_kv_cache[1],
            position_ids=position_ids, slot_mapping=slot_mapping,
            is_prefill=True,
        )

    next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    generated = [next_token.item()]
    cache._seq_len += seq_len

    # === Decode 20 steps ===
    for step in range(20):
        cache.prepare_decode_step()
        slot = cache.get_decode_slot()
        pos = cache._seq_len

        slot_mapping = torch.tensor([slot], dtype=torch.int32, device=device)
        position_ids = torch.tensor([[pos]], dtype=torch.long, device=device)
        block_table = cache.get_block_table().unsqueeze(0).to(device)
        context_lens = torch.tensor([cache._seq_len + 1], dtype=torch.int32, device=device)

        with torch.no_grad():
            logits = model(
                next_token,
                kv_cache_k=bm.gpu_kv_cache[0], kv_cache_v=bm.gpu_kv_cache[1],
                position_ids=position_ids, slot_mapping=slot_mapping,
                is_prefill=False, block_table=block_table, context_lens=context_lens,
            )

        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        generated.append(next_token.item())
        cache._seq_len += 1

    if rank == 1:
        text = tokenizer.decode(generated)
        print(f"\ntp_size={tp_size}")
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{text}'")

    if tp_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()