# Engine Benchmark

This benchmark is for pure engine performance only. It intentionally excludes OpenAI-compatible serving, HTTP, SSE streaming, Docker servers, and external server mode.

CI answers: does the implementation function correctly?
Engine benchmark answers: how fast do Pico-vLLM and vLLM engines process the same token-id trace?

## Workload

Each case is defined by:

- fixed input token length
- fixed output token count
- fixed concurrency
- greedy decoding (`temperature=0`, `top_p=1`)
- `ignore_eos=True`, so each request is expected to generate exactly `output_tokens`
- `detokenize_outputs=False`, so text decoding is not part of the measured path
- prefix cache disabled by default; use `--enable-prefix-cache` only when explicitly benchmarking cache behavior

The benchmark constructs one token-id prompt per case and repeats it for the selected concurrency. Pico-vLLM and vLLM consume the same token IDs; prompt hashes are written to JSONL for verification.

## One-command Comparison

```bash
.venv/bin/python scripts/run_standard_benchmark.py   --engines pico,vllm   --weights ./weights   --model ./weights   --input-lens 128,512,2048   --output-lens 32   --concurrency 1,4,8   --vllm-python .venv-vllm019/bin/python   --output logs/benchmarks/standard/engine_compare.jsonl
```

Run a subset with `--engines pico` or `--engines vllm`.

## Direct Pico-vLLM

```bash
.venv/bin/python pico_vllm/tests/benchmark/standard_benchmark.py   --backend pico   --engine pico   --weights ./weights   --model ./weights   --input-lens 128,512,2048   --output-lens 32   --concurrency 1,4,8   --output logs/benchmarks/standard/pico_engine.jsonl
```

For TP=2 with the current Qwen2.5-1.5B implementation:

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2   pico_vllm/tests/benchmark/standard_benchmark.py   --backend pico   --engine pico_tp2   --weights ./weights   --model ./weights   --input-lens 128,512   --output-lens 32   --concurrency 1,4   --output logs/benchmarks/standard/pico_engine_tp2.jsonl
```

## Direct vLLM Engine

Use a Python environment with vLLM installed:

```bash
.venv-vllm019/bin/python pico_vllm/tests/benchmark/standard_benchmark.py   --backend vllm   --engine vllm   --weights ./weights   --model ./weights   --input-lens 128,512,2048   --output-lens 32   --concurrency 1,4,8   --vllm-gpu-memory-utilization 0.8   --output logs/benchmarks/standard/vllm_engine.jsonl
```

## Outputs

The JSONL result rows include:

- `benchmark_suite=engine`
- `prompt_token_count`
- `prompt_token_ids_sha256`
- `expected_output_tokens`
- `total_output_tokens_observed`
- `engine_total_ms`
- `engine_output_tokens_per_s`
- `engine_request_per_s`
- `engine_steps`
- `max_active_requests`

CSV, Markdown, and PNG reports are regenerated automatically unless `--no-report` is set.
