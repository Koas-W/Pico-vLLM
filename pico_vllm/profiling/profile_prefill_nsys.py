# profile_prefill_nsys.py
# Isolate the PREFILL forward and dump a kernel breakdown, to see how big a
# share the prefill attention kernel is (analogous to the decode profile).
# Run:
#   PROMPT_LEN=4096 PYTHONPATH=pico_vllm nsys profile -c cudaProfilerApi \
#       .venv-vllm019/bin/python pico_vllm/profiling/profile_prefill_nsys.py
import os
import torch
from model import Qwen25_15B, ModelConfig
from weights import load_weights
from cache import PagedKVCache, BlockManager

device = "cuda"
cfg = ModelConfig()
BLOCK_SIZE = 16
PROMPT_LEN = int(os.environ.get("PROMPT_LEN", "4096"))
MAX_SEQ_LEN = PROMPT_LEN + 16
NUM_GPU_BLOCKS = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE + 8

model = Qwen25_15B(cfg)
model = load_weights(model, "./weights")
model = model.to(torch.bfloat16).to(device)
model.eval()

bm = BlockManager(
    num_gpu_blocks=NUM_GPU_BLOCKS, num_cpu_blocks=0,
    block_size=BLOCK_SIZE, num_layers=cfg.num_hidden_layers,
    num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim, dtype=torch.bfloat16,
)
cache = PagedKVCache(
    block_manager=bm, num_layers=cfg.num_hidden_layers, max_seq_len=MAX_SEQ_LEN,
    num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim, device=device, dtype=torch.bfloat16,
)

prompt_ids = (torch.arange(PROMPT_LEN, device=device) % 1000).unsqueeze(0)
seq_len = prompt_ids.shape[1]


def prefill_once():
    cache.reset()  # free blocks back to the manager between iterations
    cache._allocate_for_prefill(seq_len)
    slot_mapping = cache.get_prefill_slot_mapping(seq_len)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    bt = cache.get_block_table()
    block_table = torch.full((1, cfg.MAX_BLOCKS_PER_SEQ), -1, dtype=torch.int32, device=device)
    block_table[0, :bt.shape[0]] = bt
    context_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    new_token_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    q_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
    with torch.no_grad():
        model(
            prompt_ids,
            kv_cache_k=bm.gpu_kv_cache[0], kv_cache_v=bm.gpu_kv_cache[1],
            position_ids=position_ids, slot_mapping=slot_mapping, is_prefill=True,
            block_table=block_table, context_lens=context_lens,
            new_token_lens=new_token_lens, q_start_loc=q_start_loc,
        )


for _ in range(3):
    prefill_once()
torch.cuda.synchronize()

torch.cuda.cudart().cudaProfilerStart()
for _ in range(10):
    prefill_once()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print(f"Done prefill profiling, PROMPT_LEN={PROMPT_LEN}")
