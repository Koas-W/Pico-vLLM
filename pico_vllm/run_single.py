"""Single-GPU smoke test: feed a few prompts through the Engine and print the
generated text, to sanity-check that generation works end-to-end.

Uses the default attention backend (flash). Set PICO_ATTN=legacy to compare:
  PYTHONPATH=pico_vllm .venv-vllm019/bin/python pico_vllm/run_single.py            # flash (default)
  PICO_ATTN=legacy PYTHONPATH=pico_vllm .venv-vllm019/bin/python pico_vllm/run_single.py
"""
import os

import torch
from transformers import AutoTokenizer

from model import Qwen25_15B, ModelConfig
from weights import load_weights
from engine import Engine
from blockmanager import BlockManager
from cache import PagedKVCache

PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
    "The opposite of hot is",
    "Question: What is 2 + 2? Answer:",
    "def add(a, b):\n    return",
]

cfg = ModelConfig()
device = cfg.device
use_cuda = device.type == "cuda"

model = Qwen25_15B(cfg)
model = load_weights(model, "./weights")
model = model.to(torch.bfloat16).to(device)

tokenizer = AutoTokenizer.from_pretrained("./weights")

bm = BlockManager(
    num_gpu_blocks=500, num_cpu_blocks=0,
    block_size=16, num_layers=cfg.num_hidden_layers,
    num_kv_heads=cfg.num_key_value_heads,
    head_dim=cfg.head_dim, dtype=torch.bfloat16,
    device=device,
)

engine = Engine(
    model=model, tokenizer=tokenizer, block_manager=bm,
    cache_cls=PagedKVCache, device=device,
    use_cuda_graph=use_cuda,
    max_batch_size=max(8, len(PROMPTS)),
    enable_prefix_cache=True,
)
engine.scheduler.max_num_seqs = max(8, len(PROMPTS))

print(f"attention backend: PICO_ATTN={os.environ.get('PICO_ATTN', 'flash (default)')}")

id_to_prompt = {}
for p in PROMPTS:
    rid = engine.submit(p, max_new_tokens=24, temperature=0.0, top_p=1.0)
    id_to_prompt[rid] = p
engine.mark_finished()

outputs = {}
while not engine.is_done():
    for req_id, text in engine.step():
        outputs[req_id] = text

print("=" * 70)
for rid in sorted(outputs):
    prompt = id_to_prompt[rid]
    full = outputs[rid]
    completion = full[len(prompt):] if full.startswith(prompt) else full
    print(f"[{rid}] PROMPT:     {prompt!r}")
    print(f"     COMPLETION: {completion!r}")
    print("-" * 70)
