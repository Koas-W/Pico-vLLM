# test_weights.py
import torch
import torch.distributed as dist
import os
from model import Qwen25_15B, ModelConfig
from weights import load_weights

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
    # 模型初始化时也需要知道 tp_size，先用完整尺寸创建，看权重能不能对上
    model = Qwen25_15B(cfg)
    model = load_weights(model, "./weights", tp_size=tp_size, rank=rank)
    model = model.to(torch.bfloat16).to(device)

    # 检查几个关键层的 shape
    layer0 = model.layers[0]
    print(f"[Rank {rank}] qkv_proj.weight: {layer0.attn.qkv_proj.weight.shape}")
    print(f"[Rank {rank}] o_proj.weight:   {layer0.attn.o_proj.weight.shape}")
    print(f"[Rank {rank}] gate_up.weight:  {layer0.ffn.gate_up_proj.weight.shape}")
    print(f"[Rank {rank}] down_proj.weight:{layer0.ffn.down_proj.weight.shape}")
    print(f"[Rank {rank}] norm1.weight:    {layer0.norm1.weight.shape}")

    if tp_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()