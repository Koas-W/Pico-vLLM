import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Prefix Cache Benchmark
======================
非侵入式地测量 prefix cache 对 TTFT 的影响。

Workload 设计：
  - 模拟多轮对话：固定 system prompt + 不同 user query
  - 逐个请求提交并完成，隔离每个请求的 TTFT
  - 分 warm-up（首次，miss）和 steady（后续，hit）两阶段

测量方式：
  - TTFT = 第一次 step() 的耗时（该 step 只做 prefill + 采样首 token）
  - 用 torch.cuda.Event 精确计时，避免 CPU-GPU 同步误差

用法：
  python benchmark_prefix_cache.py           # 单卡
  torchrun --nproc_per_node=2 benchmark_prefix_cache.py  # 多卡 TP
"""

import torch
import time
import os
import json
from dataclasses import dataclass, field
from transformers import AutoTokenizer

# ────────────────────────────────────────────
# Part 1: Workload Generator
# ────────────────────────────────────────────

# 三种不同长度和风格的 system prompt，模拟真实场景
SYSTEM_PROMPTS = {
    "assistant": (
        "You are a helpful AI assistant. You provide clear, accurate, and concise answers. "
        "You should always be polite and professional. When you don't know the answer, "
        "you should say so honestly rather than making something up. You are trained by "
        "a leading AI research lab and your knowledge covers a wide range of topics including "
        "science, technology, history, literature, mathematics, programming, and more. "
        "You should format your responses clearly, using bullet points or numbered lists "
        "when appropriate. You should provide examples when they help illustrate a concept. "
        "Always consider the user's level of expertise and adjust your explanation accordingly. "
        "If a question is ambiguous, ask for clarification before answering."
    ),
    "coder": (
        "You are an expert software engineer and programming assistant. You write clean, "
        "efficient, and well-documented code. You follow best practices including proper "
        "error handling, type hints, and comprehensive testing. When reviewing code, you "
        "look for bugs, performance issues, security vulnerabilities, and style problems. "
        "You are proficient in Python, C++, Rust, JavaScript, and many other languages. "
        "You understand system design, distributed systems, databases, and cloud architecture. "
        "When explaining technical concepts, you use precise terminology but also provide "
        "intuitive analogies. You always consider edge cases and potential failure modes. "
        "For complex problems, you break them down into smaller, manageable steps. "
        "You prefer simple solutions over clever ones and readability over brevity."
    ),
    "analyst": (
        "You are a senior data analyst and business intelligence expert. You help users "
        "understand data, identify trends, and make data-driven decisions. You are skilled "
        "in statistical analysis, data visualization, and machine learning. You can work "
        "with SQL, Python, R, and various BI tools. When presenting findings, you always "
        "consider the audience and tailor your communication style accordingly. You provide "
        "context for numbers, explain statistical significance, and highlight actionable "
        "insights. You are careful about correlation vs causation and always mention "
        "limitations of the analysis. You suggest follow-up analyses when appropriate "
        "and help stakeholders understand the confidence level of your conclusions. "
        "You format your analysis results in clear tables and summaries."
    ),
}

# 每种 system prompt 对应的 user query（模拟多轮对话中的不同用户问题）
USER_QUERIES = {
    "assistant": [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "What are the main differences between Python and JavaScript?",
        "How does photosynthesis work?",
        "What is the theory of relativity?",
        "Describe the water cycle in detail.",
        "What are the benefits of regular exercise?",
        "Explain how neural networks learn.",
        "What caused World War I?",
        "How do vaccines work?",
    ],
    "coder": [
        "Write a binary search function in Python.",
        "How do I implement a thread pool in C++?",
        "Explain the difference between async and threading.",
        "Review this code: def f(x): return x*2+1",
        "What is the time complexity of merge sort?",
        "How do I set up a CI/CD pipeline?",
        "Explain dependency injection with an example.",
        "What are the SOLID principles?",
        "How do I optimize a slow SQL query?",
        "Write a unit test for a REST API endpoint.",
    ],
    "analyst": [
        "How do I calculate customer lifetime value?",
        "Explain A/B testing methodology.",
        "What metrics should I track for a SaaS product?",
        "How do I detect outliers in a dataset?",
        "Explain the difference between precision and recall.",
        "How do I build a cohort analysis?",
        "What is the best way to visualize time series data?",
        "How do I measure statistical significance?",
        "Explain feature importance in random forests.",
        "How do I handle missing data in a dataset?",
    ],
}


@dataclass
class BenchmarkRequest:
    """一个 benchmark 请求的完整描述"""
    request_id: int = -1
    system_prompt_key: str = ""
    system_prompt: str = ""
    user_query: str = ""
    full_prompt: str = ""
    token_count: int = 0          # tokenize 后的 token 数
    ttft_ms: float = 0.0          # 测量结果
    total_time_ms: float = 0.0
    total_tokens_generated: int = 0
    prefix_hit: bool = False      # 是否命中 prefix cache


def build_prompt(system_prompt: str, user_query: str) -> str:
    """组装成 chat 格式的 prompt（简化版，不依赖 chat template）"""
    return f"System: {system_prompt}\n\nUser: {user_query}\n\nAssistant:"


def generate_workload(tokenizer, num_rounds: int = 3) -> list[BenchmarkRequest]:
    """
    生成仿真 workload。

    模式：每种 system prompt 发 num_rounds 轮请求。
    第 1 轮是 cold start（miss），后续轮次是 warm（应该 hit）。
    """
    requests = []

    for sp_key, sp_text in SYSTEM_PROMPTS.items():
        queries = USER_QUERIES[sp_key]
        for round_idx in range(num_rounds):
            query = queries[round_idx % len(queries)]
            full_prompt = build_prompt(sp_text, query)
            token_count = len(tokenizer.encode(full_prompt))

            req = BenchmarkRequest(
                system_prompt_key=sp_key,
                system_prompt=sp_text,
                user_query=query,
                full_prompt=full_prompt,
                token_count=token_count,
            )
            requests.append(req)

    return requests


# ────────────────────────────────────────────
# Part 2: TTFT Measurement
# ────────────────────────────────────────────

def measure_ttft_single_request(engine, prompt: str, max_new_tokens: int = 10) -> tuple[float, float, int, str]:
    """
    提交单个请求，精确测量 TTFT 和总时间。

    返回：(ttft_ms, total_ms, num_generated_tokens, output_text)

    设计：
      1. submit → scheduler.waiting 里只有这一个请求
      2. 第一次 step() → prefill + 采样首 token → 这就是 TTFT
      3. 后续 step() → decode 直到完成
    """
    # ── TTFT 测量 ──
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    # 提交请求
    req_id = engine.submit(prompt, max_new_tokens=max_new_tokens, temperature=0, top_p=1.0)

    # 第一次 step：prefill
    total_start.record()
    start_event.record()
    completed = engine.step()
    end_event.record()
    torch.cuda.synchronize()
    ttft_ms = start_event.elapsed_time(end_event)

    # 检查是否第一步就完成了（max_new_tokens=1 或 EOS）
    output_text = ""
    if completed:
        for rid, text in completed:
            if rid == req_id:
                output_text = text

    # 后续 decode steps
    num_steps = 1
    while not output_text:
        completed = engine.step()
        num_steps += 1
        for rid, text in completed:
            if rid == req_id:
                output_text = text

    total_end.record()
    torch.cuda.synchronize()
    total_ms = total_start.elapsed_time(total_end)

    return ttft_ms, total_ms, num_steps, output_text


def run_benchmark(engine, tokenizer, workload: list[BenchmarkRequest],
                  label: str, rank: int = 0) -> list[BenchmarkRequest]:
    """
    顺序执行 workload 中的所有请求，逐个测量 TTFT。
    """
    results = []

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(f"  {'#':>3}  {'prompt_key':<12} {'tokens':>6}  {'TTFT(ms)':>10}  {'Total(ms)':>10}  {'steps':>5}")
        print(f"  {'-'*3}  {'-'*12} {'-'*6}  {'-'*10}  {'-'*10}  {'-'*5}")

    # warm up：跑一个无关请求，让 CUDA context 完全就绪
    _ = measure_ttft_single_request(engine, "Hello", max_new_tokens=3)

    for i, req in enumerate(workload):
        ttft_ms, total_ms, steps, output = measure_ttft_single_request(
            engine, req.full_prompt, max_new_tokens=10
        )

        req.request_id = i
        req.ttft_ms = ttft_ms
        req.total_time_ms = total_ms
        req.total_tokens_generated = steps

        # 检查是否 prefix cache 命中（通过 TTFT 推断）
        results.append(req)

        if rank == 0:
            print(f"  {i:>3}  {req.system_prompt_key:<12} {req.token_count:>6}  "
                  f"{ttft_ms:>10.2f}  {total_ms:>10.2f}  {steps:>5}")

    return results


def print_summary(results_on: list[BenchmarkRequest],
                  results_off: list[BenchmarkRequest],
                  prefix_cache_stats: dict | None = None):
    """打印对比摘要"""

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")

    # 按 system_prompt_key 分组
    keys = list(SYSTEM_PROMPTS.keys())

    print(f"\n  {'prompt':<12} {'round':>5}  {'TTFT_ON(ms)':>12} {'TTFT_OFF(ms)':>12} {'speedup':>8}")
    print(f"  {'-'*12} {'-'*5}  {'-'*12} {'-'*12} {'-'*8}")

    total_on = 0.0
    total_off = 0.0
    cold_on = 0.0
    cold_off = 0.0
    warm_on = 0.0
    warm_off = 0.0
    num_warm = 0

    for key in keys:
        reqs_on = [r for r in results_on if r.system_prompt_key == key]
        reqs_off = [r for r in results_off if r.system_prompt_key == key]

        for j, (ron, roff) in enumerate(zip(reqs_on, reqs_off)):
            speedup = roff.ttft_ms / ron.ttft_ms if ron.ttft_ms > 0 else float('inf')
            marker = " (cold)" if j == 0 else " (warm)"
            print(f"  {key:<12} {j:>5}  {ron.ttft_ms:>12.2f} {roff.ttft_ms:>12.2f} {speedup:>7.2f}x{marker}")

            total_on += ron.ttft_ms
            total_off += roff.ttft_ms
            if j == 0:
                cold_on += ron.ttft_ms
                cold_off += roff.ttft_ms
            else:
                warm_on += ron.ttft_ms
                warm_off += roff.ttft_ms
                num_warm += 1

    print(f"\n  Average TTFT:")
    n = len(results_on)
    print(f"    Overall:  ON={total_on/n:.2f}ms  OFF={total_off/n:.2f}ms  speedup={total_off/total_on:.2f}x")

    num_cold = len(keys)
    print(f"    Cold ({num_cold}):  ON={cold_on/num_cold:.2f}ms  OFF={cold_off/num_cold:.2f}ms  "
          f"speedup={cold_off/cold_on:.2f}x")

    if num_warm > 0:
        print(f"    Warm ({num_warm}):  ON={warm_on/num_warm:.2f}ms  OFF={warm_off/num_warm:.2f}ms  "
              f"speedup={warm_off/warm_on:.2f}x")

    if prefix_cache_stats:
        hr = prefix_cache_stats.get('hit_tokens', 0)
        mr = prefix_cache_stats.get('miss_tokens', 0)
        total = hr + mr
        print(f"\n  Prefix Cache Stats:")
        print(f"    Hit tokens:  {hr}")
        print(f"    Miss tokens: {mr}")
        print(f"    Hit rate:    {hr/total*100:.1f}%" if total > 0 else "    Hit rate:    N/A")


# ────────────────────────────────────────────
# Part 3: Main
# ────────────────────────────────────────────

def create_engine(enable_prefix_cache: bool, tp_size: int, rank: int, device):
    """创建 Engine 实例，和交互界面的初始化方式一致"""
    from model import Qwen25_15B, ModelConfig
    from weights import load_weights
    from cache import PagedKVCache
    from blockmanager import BlockManager
    from engine import Engine
    from comm import get_default_comm_backend

    comm_backend = get_default_comm_backend() if tp_size > 1 else None
    cfg = ModelConfig(tp_size=tp_size, tp_rank=rank, comm_backend=comm_backend)
    model = Qwen25_15B(cfg)
    model = load_weights(model, "./weights", tp_size=tp_size, tp_rank=rank)
    model = model.to(torch.bfloat16).to(device)

    tokenizer = AutoTokenizer.from_pretrained("./weights")

    BLOCK_SIZE = 16
    bm = BlockManager(
        num_gpu_blocks=500,
        num_cpu_blocks=0,
        block_size=BLOCK_SIZE,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.local_num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.bfloat16,
    )

    engine = Engine(
        model=model,
        tokenizer=tokenizer,
        block_manager=bm,
        cache_cls=PagedKVCache,
        device=device,
        use_cuda_graph=True,
        local_tp_size=tp_size,
        rank=rank,
        comm_backend=comm_backend,
        enable_prefix_cache=enable_prefix_cache,
        # max_batch_size=1,
    )

    return engine, tokenizer, bm


def main():
    from comm import create_comm_backend, set_default_comm_backend

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

    NUM_ROUNDS = 3  # 每种 system prompt 发几轮

    # ── Phase 1: Prefix Cache ON ──
    engine_on, tokenizer, bm_on = create_engine(
        enable_prefix_cache=True, tp_size=tp_size, rank=rank, device=device
    )
    workload_on = generate_workload(tokenizer, num_rounds=NUM_ROUNDS)
    results_on = run_benchmark(engine_on, tokenizer, workload_on,
                               "Prefix Cache: ON", rank=rank)

    # 收集 prefix cache 统计
    stats = None
    if engine_on.prefix_cache is not None:
        stats = dict(engine_on.prefix_cache.stats)
        if rank == 0:
            print(f"\n  [Stats] hit_rate={engine_on.prefix_cache.hit_rate()*100:.1f}%")

    # 释放显存
    del engine_on, bm_on
    torch.cuda.empty_cache()

    # ── Phase 2: Prefix Cache OFF ──
    engine_off, tokenizer, bm_off = create_engine(
        enable_prefix_cache=False, tp_size=tp_size, rank=rank, device=device
    )
    workload_off = generate_workload(tokenizer, num_rounds=NUM_ROUNDS)
    results_off = run_benchmark(engine_off, tokenizer, workload_off,
                                "Prefix Cache: OFF", rank=rank)

    del engine_off, bm_off
    torch.cuda.empty_cache()

    # ── Phase 3: Summary ──
    if rank == 0:
        print_summary(results_on, results_off, stats)

    if tp_size > 1:
        comm_backend.destroy_process_group()


if __name__ == "__main__":
    main()
