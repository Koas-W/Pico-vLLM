# Benchmark Working State

最后更新: 2026-05-24 UTC

这个文件是 Pico-vLLM / vLLM / SGLang benchmark 工作的恢复锚点。明天重新打开时，先看这里就能知道已经做了什么、还差什么、下一步怎么继续。

## 已完成

- 给 `pico_vllm/tests/benchmark/standard_benchmark.py` 加了 `--report-only`，可以从 JSONL 重新生成 CSV / Markdown / PNG。
- 新增了 `scripts/run_standard_benchmark.py` 作为统一编排入口。
- 编排器支持 `--server-mode external`、`docker`、`command`。
- 已用本地 mock OpenAI server 验证过 `command` 模式。
- 更新了 [README.md](/home/chen/workspace/Pico-vLLM/README.md) 和 [pico_vllm/tests/benchmark/README.md](/home/chen/workspace/Pico-vLLM/pico_vllm/tests/benchmark/README.md)。
- 确认这个租用集群本身就在 Kubernetes/containerd 容器里，嵌套 Docker 不是安全路径。
- 在工作区里建了隔离的 vLLM 环境：
  - `.venv-vllm`：vLLM 0.21 试验环境。
  - `.venv-vllm019`：vLLM 0.19.0，可用环境。
- 验证 `.venv-vllm019` 状态正常：
  - `torch 2.10.0+cu128`
  - `vllm 0.19.0`
  - `pip check` 通过
  - `torch.cuda.is_available() == True`
- 在 GPU 0 上跑通了 vLLM smoke。
- 在 GPU 0 上跑通了 Pico vs vLLM smoke。

## 当前状态

最近一次成功的对比是：

- `pico` vs `vllm`
- `input_tokens=8`
- `output_tokens=1`
- `concurrency=1`
- 仅用 `GPU 0`

结果：

- `pico`：
  - `ttft_ms ≈ 98.77`
  - `output_tokens_per_s ≈ 9.13`
- `vllm`：
  - `ttft_ms ≈ 329.67`
  - `output_tokens_per_s ≈ 3.03`

产物：

- `/tmp/pico_bench_check/pico_vllm_compare.jsonl`
- `/tmp/pico_bench_check/pico_vllm_compare.csv`
- `/tmp/pico_bench_check/pico_vllm_compare.md`

## 进行中

- 先给 vLLM 加一版“调优后再测”的 smoke。
- 再决定是否把 `.venv-vllm019` 作为默认 vLLM smoke 路径。
- 暂时不做 SGLang。

## 待做

- 把 Pico vs vLLM 从单点 smoke 扩成小矩阵：
  - `input_tokens=8,32,128`
  - `concurrency=1,4`
- 如果 vLLM 调优版 smoke 表现明显更好，就把这组参数保留下来。
- 新跑之前先确认 GPU 0 空闲。

## 明天怎么接

1. 先确认 `.venv-vllm019` 还能正常导入。

```bash
.venv-vllm019/bin/python -c "import torch, vllm; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count()); print('vllm', vllm.__version__)"
.venv-vllm019/bin/python -m pip check
```

2. 再跑一版 vLLM 调优 smoke。

```bash
.venv/bin/python scripts/run_standard_benchmark.py \
  --engines vllm \
  --server-mode command \
  --vllm-command 'env CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=.venv-vllm019/lib/python3.12/site-packages/torch/lib:.venv-vllm019/lib/python3.12/site-packages/nvidia/cu13/lib NVCC_APPEND_FLAGS=-I/home/chen/workspace/Pico-vLLM/.venv-vllm019/lib/python3.12/site-packages/nvidia/curand/include .venv-vllm019/bin/vllm serve ./weights --host 127.0.0.1 --port 18080 --generation-config vllm --max-model-len 512 --gpu-memory-utilization 0.4' \
  --vllm-api-base http://127.0.0.1:18080/v1 \
  --weights ./weights \
  --model ./weights \
  --input-lens 8 \
  --output-lens 1 \
  --concurrency 1 \
  --server-timeout-s 300 \
  --request-timeout-s 60 \
  --output /tmp/pico_bench_check/vllm019_smoke.jsonl
```

3. 比较调优前后效果，再决定是否做更大矩阵。

4. 继续之前先看 GPU 状态。

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

## 环境事实

- 集群已经在容器里：
  - `/proc/1/cgroup` 显示 `kubepods` / `cri-containerd`。
- 没有可用的嵌套容器运行时：
  - `docker`: not found
  - `podman`: not found
  - `apptainer`: not found
  - `singularity`: not found
  - `enroot`: not found
- 安全策略：
  - 不装 Docker
  - 不用 sudo
  - 不改系统包
  - 只用工作区内的隔离 venv

## 保留文件

不要删除这些路径：

```text
BENCHMARK_WORKING_STATE.md
.venv-vllm/
.venv-vllm019/
scripts/run_standard_benchmark.py
```
