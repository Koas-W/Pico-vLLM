# profile_nsys.py
import torch
import os
from transformers import AutoTokenizer
from model import Qwen25_15B, ModelConfig
from weights import load_weights
from cache import PagedKVCache, BlockManager

device = 'cuda'
cfg = ModelConfig()
BLOCK_SIZE = 16

# prompt 长度可通过环境变量 PROMPT_LEN 控制（默认 64）
PROMPT_LEN = int(os.environ.get("PROMPT_LEN", "64"))
DECODE_STEPS = 20
MAX_SEQ_LEN = PROMPT_LEN + DECODE_STEPS + 16
NUM_GPU_BLOCKS = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE + 8

model = Qwen25_15B(cfg)
model = load_weights(model, "./weights")
model = model.to(torch.bfloat16).to(device)
model.eval()
# model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
tokenizer = AutoTokenizer.from_pretrained("./weights")

bm = BlockManager(
    num_gpu_blocks=NUM_GPU_BLOCKS, num_cpu_blocks=0,
    block_size=BLOCK_SIZE, num_layers=cfg.num_hidden_layers,
    num_kv_heads=cfg.num_key_value_heads,
    head_dim=cfg.head_dim, dtype=torch.bfloat16,
)
cache = PagedKVCache(
    block_manager=bm, num_layers=cfg.num_hidden_layers,
    max_seq_len=MAX_SEQ_LEN, num_kv_heads=cfg.num_key_value_heads,
    head_dim=cfg.head_dim, device=device, dtype=torch.bfloat16,
)

# 合成一个 PROMPT_LEN 长度的 prompt（profiling 不关心 token 语义）
prompt_ids = (torch.arange(PROMPT_LEN, device=device) % 1000).unsqueeze(0)

seq_len = prompt_ids.shape[1]

# prefill
cache._allocate_for_prefill(seq_len)
slot_mapping = cache.get_prefill_slot_mapping(seq_len)
position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

# 当前 paged prefill 接口需要的额外参数（无 prefix，整段都是新 token）
bt = cache.get_block_table()
block_table = torch.full((1, cfg.MAX_BLOCKS_PER_SEQ), -1, dtype=torch.int32, device=device)
block_table[0, :bt.shape[0]] = bt
context_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
new_token_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
q_start_loc = torch.tensor([0], dtype=torch.int32, device=device)

with torch.no_grad():
    logits = model(
        prompt_ids,
        kv_cache_k=bm.gpu_kv_cache[0],
        kv_cache_v=bm.gpu_kv_cache[1],
        position_ids=position_ids,
        slot_mapping=slot_mapping,
        is_prefill=True,
        block_table=block_table,
        context_lens=context_lens,
        new_token_lens=new_token_lens,
        q_start_loc=q_start_loc,
    )
cache._seq_len += seq_len
last_token = logits[0, -1:].argmax(-1, keepdim=True)

# ============================================================
# 走引擎真实 decode 路径：forward_decode（融合 RoPE）+ CUDA Graph
# 镜像 engine._build_cuda_graph / _decode_step_graph（B=1）
# ============================================================
static_input_ids    = torch.zeros(1, 1, dtype=torch.long, device=device)
static_slot_mapping = torch.zeros(1, dtype=torch.int32, device=device)
static_position_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
static_block_table  = torch.full((1, cfg.MAX_BLOCKS_PER_SEQ), -1, dtype=torch.int32, device=device)
static_context_lens = torch.zeros(1, dtype=torch.int32, device=device)

def fill_static_for_decode(token_id: int):
    """按当前 cache._seq_len 就地更新静态 buffer（不前进 seq_len）"""
    cache.prepare_decode_step()
    static_input_ids[0, 0] = token_id
    static_slot_mapping[0] = cache.get_decode_slot()
    static_position_ids[0, 0] = cache._seq_len
    static_context_lens[0] = cache._seq_len + 1
    bt = cache.get_block_table()
    static_block_table.fill_(-1)
    static_block_table[0, :bt.shape[0]] = bt

# 预热（触发 triton autotune/编译）
last_id = int(last_token.item())
for _ in range(3):
    fill_static_for_decode(last_id)
    with torch.no_grad():
        _ = model.forward_decode(
            static_input_ids,
            kv_cache_k=bm.gpu_kv_cache[0], kv_cache_v=bm.gpu_kv_cache[1],
            position_ids=static_position_ids, slot_mapping=static_slot_mapping,
            block_table=static_block_table, context_lens=static_context_lens,
        )
    cache._seq_len += 1
torch.cuda.synchronize()

# CUDA Graph capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model.forward_decode(
        static_input_ids,
        kv_cache_k=bm.gpu_kv_cache[0], kv_cache_v=bm.gpu_kv_cache[1],
        position_ids=static_position_ids, slot_mapping=static_slot_mapping,
        block_table=static_block_table, context_lens=static_context_lens,
    )

# nsys 捕获：更新 buffer + replay，模拟真实 decode 循环
torch.cuda.cudart().cudaProfilerStart()

for _ in range(20):
    fill_static_for_decode(last_id)
    g.replay()
    last_id = int(static_output[0, -1].argmax().item())
    cache._seq_len += 1

torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()

print("Done. Run with: nsys profile -c cudaProfilerApi python profile_nsys.py")

cache.reset()