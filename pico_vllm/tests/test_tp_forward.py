import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# test_tp_forward.py
import torch
import os
from transformers import AutoTokenizer
from model import Qwen25_15B, ModelConfig
from weights import load_weights
from cache import PagedKVCache, BlockManager
from comm import create_comm_backend, set_default_comm_backend

def main():
    tp_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    comm_backend = create_comm_backend()

    if tp_size > 1:
        comm_backend.init_process_group()
        set_default_comm_backend(comm_backend)
        rank = comm_backend.get_rank()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    cfg = ModelConfig(tp_size=tp_size, tp_rank=rank, comm_backend=comm_backend if tp_size > 1 else None)
    model = Qwen25_15B(cfg)
    model = load_weights(model, "./weights", tp_size=tp_size, tp_rank=rank)
    model = model.to(torch.bfloat16).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("./weights")

    # BlockManager 用 local kv heads
    BLOCK_SIZE = 16
    bm = BlockManager(
        num_gpu_blocks=200,
        num_cpu_blocks=0,
        block_size=BLOCK_SIZE,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.local_num_key_value_heads,   # ← local
        head_dim=cfg.head_dim,
        dtype=torch.bfloat16,
    )

    cache = PagedKVCache(
        block_manager=bm,
        num_layers=cfg.num_hidden_layers,
        max_seq_len=512,
        num_kv_heads=cfg.local_num_key_value_heads,   # ← local
        head_dim=cfg.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    seq_len = input_ids.shape[1]

    # 分配 block 并构造 prefill 所需参数
    cache._allocate_for_prefill(seq_len)
    slot_mapping = cache.get_prefill_slot_mapping(seq_len).to(device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(
            input_ids,
            kv_cache_k=bm.gpu_kv_cache[0],
            kv_cache_v=bm.gpu_kv_cache[1],
            position_ids=position_ids,
            slot_mapping=slot_mapping,
            is_prefill=True,
        )

    # 只在 rank 0 打印结果（所有 rank 的 logits 应该相同）
    if rank == 0:
        last_logits = logits[0, -1]
        top5 = last_logits.topk(5)
        print(f"\ntp_size={tp_size}, prompt='{prompt}'")
        print("Top 5 predictions:")
        for i in range(5):
            token = tokenizer.decode([top5.indices[i].item()])
            print(f"  {token!r:15s}  logit={top5.values[i].item():.2f}")
        print(f"Next token: {tokenizer.decode([top5.indices[0].item()])}")

    if tp_size > 1:
        comm_backend.destroy_process_group()

if __name__ == "__main__":
    main()
