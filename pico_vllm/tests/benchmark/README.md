# Standard Benchmark

This branch is for end-to-end performance tests, separate from `pico_vllm/tests/ci`.

CI answers: does the implementation function correctly?
Benchmark answers: how fast is Pico-vLLM compared with vLLM and SGLang on the same workload?

## Common Workload

The benchmark uses the intersection that Pico-vLLM, vLLM, and SGLang can all support on one node:

- single-node 1-8 GPU runs
- fixed input token lengths
- fixed output token counts
- deterministic greedy decoding (`temperature=0`)
- configurable concurrency
- TTFT, ITL, total latency, and output tokens/s

## Pico-vLLM

```bash
.venv/bin/python pico_vllm/tests/benchmark/standard_benchmark.py   --backend pico   --engine pico   --weights ./weights   --model ./weights   --input-lens 128,512,2048   --output-lens 32   --concurrency 1,4,8   --output logs/benchmarks/standard/qwen15b_compare.jsonl
```

For TP=2 with the current Qwen2.5-1.5B implementation:

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2   pico_vllm/tests/benchmark/standard_benchmark.py   --backend pico   --engine pico_tp2   --weights ./weights   --model ./weights   --input-lens 128,512   --output-lens 32   --concurrency 1,4   --output logs/benchmarks/standard/qwen15b_compare.jsonl
```

## vLLM / SGLang

Start the framework's OpenAI-compatible server separately, then run the same benchmark against it. Append to the same JSONL path to get one combined CSV/Markdown/PNG report.

```bash
.venv/bin/python pico_vllm/tests/benchmark/standard_benchmark.py   --backend openai   --engine vllm   --api-base http://127.0.0.1:8000/v1   --weights ./weights   --model ./weights   --input-lens 128,512,2048   --output-lens 32   --concurrency 1,4,8   --output logs/benchmarks/standard/qwen15b_compare.jsonl
```

```bash
.venv/bin/python pico_vllm/tests/benchmark/standard_benchmark.py   --backend openai   --engine sglang   --api-base http://127.0.0.1:30000/v1   --weights ./weights   --model ./weights   --input-lens 128,512,2048   --output-lens 32   --concurrency 1,4,8   --output logs/benchmarks/standard/qwen15b_compare.jsonl
```

## Outputs

For `--output logs/benchmarks/standard/qwen15b_compare.jsonl`, rank 0 also writes:

- `logs/benchmarks/standard/qwen15b_compare.csv`
- `logs/benchmarks/standard/qwen15b_compare.md`
- `logs/benchmarks/standard/qwen15b_compare.png`

The JSONL file is append-friendly. Use the same output path for Pico-vLLM, vLLM, and SGLang runs to build one comparison table and plot.
