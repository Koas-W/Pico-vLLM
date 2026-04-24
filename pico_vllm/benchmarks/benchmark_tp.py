import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# benchmark_tp.py
import torch
import torch.distributed as dist
import time
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
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    cfg = ModelConfig(tp_size=tp_size)
    model = Qwen25_15B(cfg)
    model = load_weights(model, "./weights", tp_size=tp_size, rank=rank)
    model = model.to(dtype).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("./weights")

    BLOCK_SIZE = 16
    MAX_SEQ_LEN = 1024
    MAX_BLOCKS = MAX_SEQ_LEN // BLOCK_SIZE
    B = 1

    bm = BlockManager(
        num_gpu_blocks=200, num_cpu_blocks=0,
        block_size=BLOCK_SIZE, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.local_num_key_value_heads,
        head_dim=cfg.head_dim, dtype=dtype,
    )
    cache = PagedKVCache(
        block_manager=bm, num_layers=cfg.num_hidden_layers,
        max_seq_len=MAX_SEQ_LEN, num_kv_heads=cfg.local_num_key_value_heads,
        head_dim=cfg.head_dim, device=device, dtype=dtype,
    )

    # ============================================================
    # 静态 buffer
    # ============================================================
    static_input_ids    = torch.zeros(B, 1, dtype=torch.long, device=device)
    static_slot_mapping = torch.zeros(B, dtype=torch.int32, device=device)
    static_position_ids = torch.zeros(B, 1, dtype=torch.long, device=device)
    static_block_table  = torch.full((B, MAX_BLOCKS), -1, dtype=torch.int32, device=device)
    static_context_lens = torch.zeros(B, dtype=torch.int32, device=device)

    # ============================================================
    # Prefill
    # ============================================================
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    seq_len = input_ids.shape[1]

    cache._allocate_for_prefill(seq_len)
    slot_mapping = cache.get_prefill_slot_mapping(seq_len).to(device)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(
            input_ids,
            kv_cache_k=bm.gpu_kv_cache[0], kv_cache_v=bm.gpu_kv_cache[1],
            position_ids=position_ids, slot_mapping=slot_mapping,
            is_prefill=True,
        )

    cache._seq_len += seq_len
    next_token = logits[0, -1].argmax()

    if rank == 0:
        print(f"Prefill done: '{tokenizer.decode([next_token.item()])}'")

    # ============================================================
    # 准备 Decode 静态数据 + 预热
    # ============================================================
    cache.prepare_decode_step()
    static_input_ids[0, 0] = next_token.item()
    static_slot_mapping[0] = cache.get_decode_slot()
    static_position_ids[0, 0] = cache.seq_len
    bt = cache.get_block_table()
    static_block_table[0, :bt.shape[0]].copy_(bt)
    static_context_lens[0] = cache.seq_len + 1

    for _ in range(3):
        with torch.no_grad():
            _ = model.forward_decode(
                static_input_ids,
                kv_cache_k=bm.gpu_kv_cache[0], kv_cache_v=bm.gpu_kv_cache[1],
                position_ids=static_position_ids, slot_mapping=static_slot_mapping,
                block_table=static_block_table, context_lens=static_context_lens,
            )
    torch.cuda.synchronize()

    # ============================================================
    # CUDA Graph capture
    # ============================================================
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_output = model.forward_decode(
            static_input_ids,
            kv_cache_k=bm.gpu_kv_cache[0], kv_cache_v=bm.gpu_kv_cache[1],
            position_ids=static_position_ids, slot_mapping=static_slot_mapping,
            block_table=static_block_table, context_lens=static_context_lens,
        )

    next_token = static_output[0, -1].argmax()

    # ============================================================
    # Decode Profiling（零同步）
    # ============================================================
    PROFILING_TOKENS = 100

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for step in range(PROFILING_TOKENS):
        cache._seq_len += 1
        cache.prepare_decode_step()

        static_input_ids.copy_(next_token, non_blocking=True)
        static_slot_mapping.fill_(cache.get_decode_slot())
        static_position_ids.fill_(cache._seq_len)
        static_context_lens.fill_(cache._seq_len + 1)

        bt = cache.get_block_table()
        static_block_table[0, :bt.shape[0]].copy_(bt, non_blocking=True)

        g.replay()
        next_token = static_output[0, -1].argmax()

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    tok_per_sec = PROFILING_TOKENS / total_time
    ms_per_tok = (total_time / PROFILING_TOKENS) * 1000

    if rank == 0:
        print("-" * 50)
        print(f"  tp_size:    {tp_size}")
        print(f"  生成数量:   {PROFILING_TOKENS} tokens")
        print(f"  总耗时:     {total_time:.4f} 秒")
        print(f"  单步延迟:   {ms_per_tok:.2f} ms/tok")
        print(f"  平均吞吐:   {tok_per_sec:.2f} tokens/s")
        print("-" * 50)

    # ============================================================
    # 清理
    # ============================================================
    if tp_size > 1:
        del g, static_output
        torch.cuda.synchronize()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()