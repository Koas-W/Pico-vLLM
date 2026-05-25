# Benchmark Working State

## Direction

当前 benchmark 方向已经收敛为 pure engine benchmark。Serving / OpenAI-compatible / HTTP / SSE / Docker server / external server 相关 benchmark 入口都不再作为当前工作目标。

核心目标：Pico-vLLM Engine 和 vLLM Engine 在同一份 token-id trace 上的纯 engine 吞吐对比。

## 已改动

- `pico_vllm/tests/benchmark/standard_benchmark.py`
  - 改为 engine-only benchmark。
  - backend 只保留 `pico` 和 `vllm`。
  - 删除 OpenAI client、streaming、TTFT、ITL、server mode metadata。
  - workload 现在生成 token-id trace，两边直接消费同一份 token IDs。
  - 默认 `detokenize_outputs=False`、`ignore_eos=True`。
  - 默认关闭 prefix cache；需要时显式传 `--enable-prefix-cache`。
  - Pico 计时窗口已改为生成到目标 token 数即停止，request cleanup/drain 放在计时外，避免比 vLLM 多计一个无生成 step。

- `scripts/run_standard_benchmark.py`
  - 删除 server/docker/external/SGLang 相关逻辑。
  - 只负责分别启动 Pico engine benchmark 和 vLLM engine benchmark。
  - vLLM 可通过 `--vllm-python .venv-vllm019/bin/python` 指定带 vLLM 的 Python。

- `pico_vllm/engine.py`
  - 新增 `submit_token_ids()`，避免 pure engine benchmark 把 tokenizer 计入路径。
  - 新增 `detokenize_outputs`，benchmark 中关闭完成输出的 decode。
  - 新增 `ignore_eos`，benchmark 中强制按 `max_new_tokens` 生成。

- `pico_vllm/scheduler.py`
  - 修复 `max_new_tokens=1` 时 prefill 已生成首 token 后仍进入下一轮 decode 的问题。

- `pico_vllm/tests/benchmark/README.md`
  - 改为 engine-only 使用说明。

## 当前 benchmark 口径

计时窗口：

```text
submit all token-id requests
cuda synchronize
start timer
while expected output tokens are not fully observed:
    engine.step()
cuda synchronize
stop timer
cleanup/drain finished requests outside timer
```

不包含：

- HTTP / JSON / SSE
- OpenAI-compatible server
- tokenizer encode
- detokenizer decode
- server queue / network / client streaming

主要指标：

- `engine_total_ms`
- `engine_output_tokens_per_s`
- `engine_request_per_s`
- `engine_steps`
- `max_active_requests`
- `total_output_tokens_observed`
- `expected_output_tokens`

## 复现命令

Pico + vLLM engine 对比：

```bash
.venv/bin/python scripts/run_standard_benchmark.py   --engines pico,vllm   --weights ./weights   --model ./weights   --input-lens 8,32,128   --output-lens 32   --concurrency 1,4   --vllm-python .venv-vllm019/bin/python   --vllm-gpu-memory-utilization 0.8   --output /tmp/pico_bench_check/engine_compare.jsonl
```

只跑 Pico：

```bash
.venv/bin/python scripts/run_standard_benchmark.py   --engines pico   --weights ./weights   --model ./weights   --input-lens 8,32,128   --output-lens 32   --concurrency 1,4   --output /tmp/pico_bench_check/pico_engine.jsonl
```

只跑 vLLM：

```bash
.venv-vllm019/bin/python pico_vllm/tests/benchmark/standard_benchmark.py   --backend vllm   --engine vllm   --weights ./weights   --model ./weights   --input-lens 8,32,128   --output-lens 32   --concurrency 1,4   --vllm-gpu-memory-utilization 0.8   --output /tmp/pico_bench_check/vllm_engine.jsonl
```

## 已验证

- Python 语法检查通过。
- GPU 0 最小 smoke 已跑：`input_tokens=8`、`concurrency=1`、`output_tokens=1,4`。
- 修正前 Pico 会多计一个完成迁移/cleanup step：`output_tokens=1` 时 `engine_steps=2`，`output_tokens=4` 时 `engine_steps=5`。
- 修正后 Pico 与目标输出 token 数对齐：`output_tokens=1` 时 `engine_steps=1`，`output_tokens=4` 时 `engine_steps=4`。
- 正式 engine 矩阵已跑：`input_tokens=8,32,128`、`output_tokens=32`、`concurrency=1,4`。
  - 产物：`/tmp/pico_bench_check/engine_matrix_20260525.jsonl`、`.csv`、`.md`、`.png`。
  - 每组 Pico/vLLM 的 prompt hash 一致，`expected_output_tokens == total_output_tokens_observed`。
  - vLLM `concurrency=4` 组记录 `engine_steps=33`，Pico 为 32；这是 vLLM engine 调度行为，token/s 仍按实际 engine wall time 和实际输出 token 计算。

## 待验证

- 如需报告用数字，建议再跑 3-5 次取中位数；当前矩阵是单次 run。

## 注意

旧的 serving / OpenAI server 结果只能作为历史参考，不再进入当前报告口径。
