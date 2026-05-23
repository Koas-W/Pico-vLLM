# Pico-vLLM H200 Roadmap

目标：把 Pico-vLLM 在 8x H200 上推进成可复现实验平台，重点覆盖长上下文、多卡通信、KV cache 传输和 profiling/benchmark 基线。当前阶段不跟踪远程分支，所有开发从本地分支 `echo/profiling_and_more_benchmark` 开始。

## 判断

8x H200 的价值不在继续压 Qwen2.5-1.5B 单卡 tok/s，而在放大三类瓶颈：

- 长上下文下的 KV cache 容量、读带宽和 prefill/decode 调度冲突。
- 多卡 TP/PD 下的 NCCL 通信、KV 传输和计算通信重叠。
- Hopper FP8 对 KV cache 带宽和并发上下文容量的收益。

当前代码已经具备 PagedAttention、Prefix Cache、CUDA Graph、Continuous Batching、TP、PD 分离和 Triton kernel 基础。H200 工作应先补可复现实验基线，再做调度和通信优化。

## P0: H200 Benchmark/Profiling Baseline

交付物：

- 统一的 H200 engine benchmark，覆盖 TTFT、ITL、step latency、吞吐、显存峰值、block 使用量。
- JSONL 结果输出，包含 GPU 型号、world size、dtype、prompt length、concurrency、prefix cache、CUDA Graph 等元数据。
- 支持 `nsys -c cudaProfilerApi` 的 profiler window，避免采集模型加载和 warmup 噪声。
- 单卡、TP=2、PD 组合的首批基线报告。

首批 workload：

- prompt length: 1K / 4K / 8K / 16K / 32K。
- concurrency: 1 / 2 / 4 / 8。
- max_new_tokens: 32 / 128。
- prefix cache: off / on。
- CUDA Graph: off / on。

注意：当前 Qwen2.5-1.5B 配置只有 2 个 KV heads，现有 TP 分片路径天然适合 TP=1/2。要把 8 张 H200 全部用于 TP，需要先处理 KV head replication 或换更大模型配置。短期 8 卡实验优先用 PD 拆分、并发实例和 TP=2 组合。

## P1: Chunked Prefill Scheduler

目标：长 prompt 不能用一次完整 prefill 阻塞 decode。调度器需要 token budget：

- decode 优先占用每步 budget。
- waiting 请求按 chunk 进入 prefilling。
- 长 prompt 被拆成多个 chunk，并在 chunk 之间允许 decode 插队。
- chunk 继续写同一个 paged KV cache，prefill 完成后再进入 decoding。

实验对照：

- baseline: 当前 FCFS whole-prefill。
- fixed chunk: 固定 chunk size。
- adaptive chunk: 根据活跃 decode 数和 prompt length 调整 chunk。
- decode-priority budget: 每步先保 decode，再分配 prefill token budget。

指标：

- TTFT avg/p50/p95/p99。
- ITL avg/p50/p95/p99。
- decode stall rate。
- total throughput。
- active requests 和 block 使用水位。

## P1: TP Async All-Reduce And Overlap

当前通信点：

- attention `o_proj` 后 all-reduce。
- FFN `down_proj` 后 all-reduce。

阶段：

- baseline: 同步 `dist.all_reduce`。
- async v1: `async_op=True`，测量通信可隐藏比例。
- overlap v2: 分层重排，让上一段 all-reduce 与下一段可提前计算部分重叠。
- graph-aware v3: 固定 batch/topology 下兼容 CUDA Graph replay。

风险：

- 当前模型对 Qwen2.5-1.5B 的 TP 大于 2 时 KV heads 会被整除到 0，需要先设计 GQA KV head replication 或换 7B/14B/32B 配置。
- CUDA Graph 与异步 collective 的交互需要单独正确性和死锁测试。

## P2: FP8 KV Cache

目标：优先做 KV cache FP8，不先做权重量化。

设计：

- KV cache 存储支持 bf16 / fp8_e4m3 / fp8_e5m2。
- scale 粒度先做 per-tensor，再做 per-head/per-block。
- paged attention 读取 KV 时 dequant。

实验：

- bf16 KV baseline。
- fp8 KV per-tensor scale。
- fp8 KV per-head scale。
- fp8 KV per-block scale。

指标：

- decode bandwidth proxy。
- tokens/s。
- max concurrent long-context requests。
- logit drift / perplexity drift。

## P2: PD KV Transfer Backend

当前 `kv_transfer.py` 已有 Sync/Async 抽象，异步版基于 `dist.isend/irecv`。下一步先做传输模式实验，再替换后端。

实验：

- P:D = 1:1 / 2:6 / 4:4 / 6:2。
- TP_P != TP_D 的异构组合。
- whole-prompt KV transfer vs chunk/layer/block streaming。
- 当前 gather-contiguous-send-scatter 路径的拷贝开销。

后端路线：

- torch distributed P2P baseline。
- NCCL P2P/collective 变体。
- NIXL backend。

## P3: Prefix Cache COW And Offload

目标：补齐共享 prefix block 的内存语义。

关键问题：

- 共享 prefix block 被多个 request 引用时，decode append 何时需要 COW。
- 哪些场景只追加新 block，不会污染共享 block。
- radix node 生命周期和 block ref count 如何一致。
- GPU-only、GPU+CPU offload、recompute-aware eviction 的取舍。

压力测试：

- shared prefix: 8K / 32K / 64K / 128K。
- hit ratio: 0% / 25% / 50% / 75% / 95%。
- eviction: LRU / cost-aware / recompute-aware。

## Current Branch Tasks

- [x] 创建本地分支 `echo/profiling_and_more_benchmark`。
- [x] 添加 H200 roadmap。
- [x] 添加 H200 baseline benchmark/profiling 入口。
- [ ] 在 8x H200 上跑单卡 baseline。
- [ ] 在 8x H200 上跑 TP=2 baseline。
- [ ] 基于结果选择 chunked prefill 的默认 token budget 和 chunk size。
