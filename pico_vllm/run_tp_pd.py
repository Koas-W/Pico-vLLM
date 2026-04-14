# run_tp_pd.py
import torch
import torch.distributed as dist
import os
from transformers import AutoTokenizer
from model import Qwen25_15B, ModelConfig
from weights import load_weights
from cache import PagedKVCache, BlockManager
from engine import Engine

def main():
    dist.init_process_group(backend="nccl",
        device_id=torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # === 拓扑配置 ===
    tp_size = 2
    p_ranks = [0, 1]      # P 组
    d_ranks = [2, 3]      # D 组

    # 判断当前 rank 的角色
    if rank in p_ranks:
        role = "p"
        tp_rank = p_ranks.index(rank)
        peer_ranks = [d_ranks[tp_rank]]    # P rank 0 → D rank 2
    else:
        role = "d"
        tp_rank = d_ranks.index(rank)
        peer_ranks = [p_ranks[tp_rank]]    # D rank 2 → P rank 0

    # 创建 TP 子组（所有 rank 都要参与 new_group 调用，即使自己不在组内）
    p_tp_group = dist.new_group(p_ranks)
    d_tp_group = dist.new_group(d_ranks)
    tp_group = p_tp_group if role == "p" else d_tp_group

    # === 模型、权重、Engine 初始化 ===
    cfg = ModelConfig(tp_size=tp_size, tp_rank=tp_rank, tp_group=tp_group)
    model = Qwen25_15B(cfg)   # 传 tp_group
    model = load_weights(model, "./weights", tp_size=tp_size, tp_rank=tp_rank)
    model = model.to(torch.bfloat16).to(device)

    tokenizer = AutoTokenizer.from_pretrained("./weights")
    BLOCK_SIZE = 16

    bm = BlockManager(
        num_gpu_blocks=200, num_cpu_blocks=0,
        block_size=BLOCK_SIZE, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.local_num_key_value_heads,
        head_dim=cfg.head_dim, dtype=torch.bfloat16,
    )

    engine = Engine(
        model=model, tokenizer=tokenizer, block_manager=bm,
        cache_cls=PagedKVCache, device=device,
        use_cuda_graph=True,
        local_tp_size=tp_size, rank=rank, peer_ranks=peer_ranks,
    )

    # 提交请求
    engine.submit("The capital of France is", max_new_tokens=20, temperature=1, top_p=0.9)
    engine.submit("He kiss her with love", max_new_tokens=20, temperature=1, top_p=0.9)
    engine.submit("Do you know that aging means", max_new_tokens=20, temperature=1, top_p=0.9)

    # 运行直到完成
    flag=0
    num_requests=3
    while True:
        completed = engine.step()
        for req_id, text in completed:
            if rank == 0:
                print(f"[Request {req_id}] {text}")
        if completed:
            flag+=completed.__len__()
        if flag>=num_requests:
            break

    print(f"[Rank {rank}] generation done", flush=True)
    if tp_size > 1:
        # 释放 CUDA Graph 持有的 NCCL 资源
        if engine.use_cuda_graph:
            del engine.cuda_graph
            del engine.static_output
            torch.cuda.synchronize()
        dist.destroy_process_group()
        print(f"[Rank {rank}] destroyed process group", flush=True)

if __name__ == "__main__":
    main()