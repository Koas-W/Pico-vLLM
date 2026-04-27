import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# test_hetero_tp_pd.py
import torch
import os, sys
from transformers import AutoTokenizer
from model import Qwen25_15B, ModelConfig
from weights import load_weights
from cache import PagedKVCache, BlockManager
from engine import Engine
from comm import create_comm_backend, set_default_comm_backend

def main():
    comm_backend = create_comm_backend()
    comm_backend.init_process_group(
        device_id=torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    )
    set_default_comm_backend(comm_backend)
    rank = comm_backend.get_rank()
    world_size = comm_backend.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # === 拓扑：P(TP=2) + D(TP=1) ===
    tp_p, tp_d = 2, 1
    p_ranks = [0, 1]
    d_ranks = [2]
    # === 拓扑：P(TP=1) + D(TP=2) ===
    # tp_p, tp_d = 1, 2
    # p_ranks = [0]
    # d_ranks = [1, 2]

    # 创建 TP 子组（所有 rank 必须参与 new_group 调用）
    p_tp_group = comm_backend.new_group(p_ranks)
    d_tp_group = comm_backend.new_group(d_ranks)

    if rank in p_ranks:
        role = "p"
        tp_rank = p_ranks.index(rank)
        local_tp_size = tp_p
        tp_group = p_tp_group
        remote_tp_size = tp_d

        if tp_p >= tp_d:
            # 多对一或一对一：每个 P rank 发给一个 D rank
            dst = tp_rank * tp_d // tp_p
            peer_ranks = [d_ranks[dst]]
            # 同一个 D rank 对应的多个 P rank 里，第一个是 primary
            is_primary = (tp_rank == dst * tp_p // tp_d)
        else:
            # 一对多：一个 P rank 发给多个 D rank
            start = tp_rank * tp_d // tp_p
            end = (tp_rank + 1) * tp_d // tp_p
            peer_ranks = [d_ranks[j] for j in range(start, end)]
            is_primary = True  # 唯一的发送者，必须发 meta

    else:
        role = "d"
        tp_rank = d_ranks.index(rank)
        local_tp_size = tp_d
        tp_group = d_tp_group
        remote_tp_size = tp_p
        is_primary = True  # D 侧不区分 primary

        if tp_p > tp_d:
            # 多对一：从多个 P rank 收
            start = tp_rank * tp_p // tp_d
            end = (tp_rank + 1) * tp_p // tp_d
            peer_ranks = [p_ranks[i] for i in range(start, end)]
        else:
            # 一对一或一对多（从 D 角度看是一对一）：从一个 P rank 收
            src = tp_rank * tp_p // tp_d
            peer_ranks = [p_ranks[src]]

    # P2P warmup
    warmup = torch.zeros(1, device=device)
    if rank == 0:
        comm_backend.send(warmup, dst=2)
        comm_backend.recv(warmup, src=2)
    elif rank == 2:
        comm_backend.recv(warmup, src=0)
        comm_backend.send(warmup, dst=0)
    comm_backend.barrier()

    cfg = ModelConfig(tp_size=local_tp_size, tp_rank=tp_rank, tp_group=tp_group, comm_backend=comm_backend)
    model = Qwen25_15B(cfg)
    model = load_weights(model, "./weights", tp_size=local_tp_size, tp_rank=tp_rank)
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
        use_cuda_graph=(role == "d"),
        local_tp_size=local_tp_size, 
        remote_tp_size=remote_tp_size, 
        is_primary=is_primary,
        rank=rank, role=role,
        peer_ranks=peer_ranks,
        comm_backend=comm_backend,
    )

    # P 组内所有 rank 提交相同请求
    prompts = [
        ("The capital of France is", 20),
        ("1 + 1 =", 20),
    ]

    if role == "p":
        for prompt, max_tokens in prompts:
            engine.submit(prompt, max_new_tokens=max_tokens, temperature=0, top_p=1.0)
        engine.mark_finished()

    results = {}
    while not engine.is_done():
        completed = engine.step()
        for req_id, text in completed:
            results[req_id] = text

    # 只在 D rank 打印结果
    if role == "d" and tp_rank == 0:
        print(f"\n=== Heterogeneous TP+PD: P(TP={tp_p}) + D(TP={tp_d}) ===")
        for req_id in sorted(results):
            print(f"  [Request {req_id}] {results[req_id]}")

    comm_backend.barrier()
    if hasattr(engine, 'cuda_graph'):
        del engine.cuda_graph
        if hasattr(engine, 'static_output'):
            del engine.static_output
        torch.cuda.synchronize()
    comm_backend.destroy_process_group()


if __name__ == "__main__":
    main()
