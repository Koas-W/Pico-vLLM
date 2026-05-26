import os
import resource
import statistics
import textwrap
import time

import pytest
import torch

from blockmanager import BlockManager
from model import ModelConfig, Qwen25_15B
from weights import load_weights


pytestmark = [pytest.mark.cpu, pytest.mark.single_card, pytest.mark.weights, pytest.mark.slow]


def _rate(numerator: int | float, seconds: float) -> float | None:
    if seconds <= 0:
        return None
    return numerator / seconds


def _fmt_float(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}{suffix}"


def _fmt_seconds(value: float) -> str:
    return f"{value:.3f}s"


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 1000:.2f}ms"


def _max_rss_mib() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _print_wrapped(label: str, text: str, width: int = 100) -> None:
    print(f"  {label}:", flush=True)
    wrapped = textwrap.fill(text, width=width) if text else ""
    for line in wrapped.splitlines() or [""]:
        print(f"    {line}", flush=True)


def _print_final_report(
    *,
    prompt: str,
    generated_text: str,
    generated_token_ids: list[int],
    dtype: torch.dtype,
    stop_reason: str,
    prompt_tokens: int,
    generated_tokens: int,
    max_new_tokens: int,
    load_time: float,
    prefill_time: float,
    decode_model_time: float,
    decode_loop_time: float,
    total_time: float,
    decode_latencies: list[float],
) -> None:
    decode_forwards = len(decode_latencies)
    generation_model_time = prefill_time + decode_model_time
    generation_wall_time = prefill_time + decode_loop_time
    total_context_tokens = prompt_tokens + generated_tokens

    latency_min = min(decode_latencies) if decode_latencies else None
    latency_max = max(decode_latencies) if decode_latencies else None
    latency_avg = statistics.fmean(decode_latencies) if decode_latencies else None
    latency_p50 = statistics.median(decode_latencies) if decode_latencies else None

    print("", flush=True)
    print("=" * 72, flush=True)
    print("Pico-vLLM CPU Real Weights Inference Report", flush=True)
    print("=" * 72, flush=True)

    print("Output", flush=True)
    _print_wrapped("prompt", prompt)
    _print_wrapped("generated_text", generated_text)
    print(f"  generated_token_ids: {generated_token_ids}", flush=True)

    print("Run", flush=True)
    print(f"  dtype: {dtype}", flush=True)
    print(f"  stop_reason: {stop_reason}", flush=True)
    print(f"  max_new_tokens: {max_new_tokens}", flush=True)
    print(f"  prompt_tokens: {prompt_tokens}", flush=True)
    print(f"  generated_tokens: {generated_tokens}", flush=True)
    print(f"  total_context_tokens: {total_context_tokens}", flush=True)
    print(f"  decode_forwards: {decode_forwards}", flush=True)

    print("Timing", flush=True)
    print(f"  load_time: {_fmt_seconds(load_time)}", flush=True)
    print(f"  prefill_time_ttft_excl_load: {_fmt_seconds(prefill_time)}", flush=True)
    print(f"  time_to_first_token_incl_load: {_fmt_seconds(load_time + prefill_time)}", flush=True)
    print(f"  decode_model_time: {_fmt_seconds(decode_model_time)}", flush=True)
    print(f"  decode_loop_wall_time: {_fmt_seconds(decode_loop_time)}", flush=True)
    print(f"  generation_model_time_excl_load: {_fmt_seconds(generation_model_time)}", flush=True)
    print(f"  generation_wall_time_excl_load: {_fmt_seconds(generation_wall_time)}", flush=True)
    print(f"  total_e2e_time: {_fmt_seconds(total_time)}", flush=True)

    print("Throughput", flush=True)
    print(
        "  prefill_prompt_tps: "
        f"{_fmt_float(_rate(prompt_tokens, prefill_time), ' tok/s')}",
        flush=True,
    )
    print(
        "  decode_forward_tps: "
        f"{_fmt_float(_rate(decode_forwards, decode_model_time), ' tok/s')}",
        flush=True,
    )
    print(
        "  output_tps_excl_load: "
        f"{_fmt_float(_rate(generated_tokens, generation_model_time), ' tok/s')}",
        flush=True,
    )
    print(
        "  total_token_tps_excl_load: "
        f"{_fmt_float(_rate(total_context_tokens, generation_model_time), ' tok/s')}",
        flush=True,
    )
    print(
        "  output_tps_e2e_incl_load: "
        f"{_fmt_float(_rate(generated_tokens, total_time), ' tok/s')}",
        flush=True,
    )

    print("Decode Latency", flush=True)
    print(f"  avg: {_fmt_ms(latency_avg)}", flush=True)
    print(f"  p50: {_fmt_ms(latency_p50)}", flush=True)
    print(f"  min: {_fmt_ms(latency_min)}", flush=True)
    print(f"  max: {_fmt_ms(latency_max)}", flush=True)

    print("Memory", flush=True)
    print(f"  max_rss: {_max_rss_mib():.2f} MiB", flush=True)
    print("=" * 72, flush=True)
    print("", flush=True)


def test_real_weights_cpu_prefill_and_decode_smoke():
    transformers = pytest.importorskip("transformers")

    torch.set_grad_enabled(False)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    prompt = os.environ.get("PICO_VLLM_CPU_REAL_PROMPT", "Hello")
    max_new_tokens = int(os.environ.get("PICO_VLLM_CPU_REAL_MAX_NEW_TOKENS", "32"))
    assert max_new_tokens > 0, "PICO_VLLM_CPU_REAL_MAX_NEW_TOKENS must be positive"

    device = torch.device("cpu")
    cfg = ModelConfig(use_cuda=False)

    started = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained("./weights")
    eos_token_id = tokenizer.eos_token_id
    assert eos_token_id is not None, "tokenizer must define eos_token_id for EOF generation"

    load_started = time.time()
    model = Qwen25_15B(cfg).eval()
    model = load_weights(model, "./weights", verbose=True).eval()
    load_time = time.time() - load_started
    dtype = next(model.parameters()).dtype

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    seq_len = input_ids.shape[1]
    total_len = seq_len + max_new_tokens
    num_blocks = (total_len + cfg.BLOCK_SIZE - 1) // cfg.BLOCK_SIZE

    bm = BlockManager(
        num_gpu_blocks=num_blocks,
        num_cpu_blocks=0,
        block_size=cfg.BLOCK_SIZE,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=dtype,
        device=device,
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)

    prefill_started = time.time()
    logits = model(
        input_ids,
        kv_cache_k=bm.gpu_kv_cache[0],
        kv_cache_v=bm.gpu_kv_cache[1],
        position_ids=torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0),
        slot_mapping=torch.arange(seq_len, dtype=torch.int32, device=device),
        is_prefill=True,
        block_table=block_table,
        context_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
        new_token_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
        q_start_loc=torch.tensor([0], dtype=torch.int32, device=device),
    )
    prefill_time = time.time() - prefill_started

    assert model.ops.device_type == "cpu"
    assert logits.shape == (1, seq_len, cfg.vocab_size)
    assert torch.isfinite(logits).all()

    generated = []
    decode_latencies = []
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    stop_reason = "max_new_tokens"
    previous_decode_latency = None
    decode_started = time.time()
    print("Decode token trace:", flush=True)
    for step in range(max_new_tokens):
        token_id = int(next_id.item())
        generated.append(token_id)
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        source = "prefill" if step == 0 else "decode"
        latency = "n/a" if previous_decode_latency is None else _fmt_ms(previous_decode_latency)
        print(
            f"  #{step + 1:03d} source={source:<7} token_id={token_id:<8} "
            f"latency={latency:<10} text={token_text!r}",
            flush=True,
        )

        if token_id == eos_token_id:
            stop_reason = "eos"
            break
        if step + 1 >= max_new_tokens:
            break

        position = seq_len + step
        decode_step_started = time.time()
        decode_logits = model.forward_decode(
            next_id,
            kv_cache_k=bm.gpu_kv_cache[0],
            kv_cache_v=bm.gpu_kv_cache[1],
            position_ids=torch.tensor([[position]], dtype=torch.long, device=device),
            slot_mapping=torch.tensor([position], dtype=torch.int32, device=device),
            block_table=block_table,
            context_lens=torch.tensor([position + 1], dtype=torch.int32, device=device),
        )

        assert decode_logits.shape == (1, 1, cfg.vocab_size)
        assert torch.isfinite(decode_logits).all()
        previous_decode_latency = time.time() - decode_step_started
        decode_latencies.append(previous_decode_latency)
        next_id = decode_logits[:, -1, :].argmax(dim=-1, keepdim=True)

    decode_loop_time = time.time() - decode_started
    decode_model_time = sum(decode_latencies)
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    generated_tokens = len(generated)
    total_time = time.time() - started

    _print_final_report(
        prompt=prompt,
        generated_text=decoded,
        generated_token_ids=generated,
        dtype=dtype,
        stop_reason=stop_reason,
        prompt_tokens=seq_len,
        generated_tokens=generated_tokens,
        max_new_tokens=max_new_tokens,
        load_time=load_time,
        prefill_time=prefill_time,
        decode_model_time=decode_model_time,
        decode_loop_time=decode_loop_time,
        total_time=total_time,
        decode_latencies=decode_latencies,
    )
