#!/usr/bin/env python3
"""AWQ/GPTQ Performance Benchmark for Ascend NPU — Python API version.

Compares dense (FP16/BF16) vs quantized (AWQ / GPTQ-Int4 / GPTQ-Int8) on the
same hardware and config — a paired comparison as required by the scoring
rules. Measures throughput / TTFT / TPOT / peak HBM for each.

Each model runs in a fresh subprocess to avoid NPU state issues between loads.

Environment notes:
  * MUST run from /data/ascend/vllm-ascend/ (CWD namespace conflict, see
    CLAUDE.md §7 problem 9).
  * LD_LIBRARY_PATH is injected into each subprocess so vllm-ascend custom ops
    (libvllm_ascend_kernels.so) load — otherwise all custom ops are disabled
    and performance collapses (see CLAUDE.md §7 problem 12).

Save full stdout to logs:

    cd /data/ascend/vllm-ascend
    source /data/ascend/.venv/bin/activate
    python benchmarks/benchmark_awq_gptq.py --label baseline \\
      2>&1 | tee /data/ascend/logs/2026-06-12_bench_dense_vs_quant_qwen0.5b.log
"""

import argparse
import json
import os
import subprocess
import sys
import time

# Ensure we're in the right directory
if not os.path.exists("vllm_ascend"):
    print("ERROR: Must run from /data/ascend/vllm-ascend/ directory")
    sys.exit(1)


def get_npu_memory_mb():
    """Get current NPU memory usage via npu-smi (pre-run sanity check)."""
    try:
        result = subprocess.run(
            ["npu-smi", "info", "-t", "usages", "-i", "0"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.split("\n"):
            if "Used Capacity" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    mb_str = parts[1].strip().split("/")[0].strip().split()[0]
                    return float(mb_str)
    except Exception:
        pass
    return 0.0


def build_env():
    """Build subprocess env with LD_LIBRARY_PATH so vllm-ascend custom ops load.

    Without this, libvllm_ascend_kernels.so is not found and vllm disables all
    custom ops (RMSNorm/fused kernels) → performance collapses.
    """
    env = os.environ.copy()
    env["HF_HOME"] = "/data/huggingface_home"
    env["VLLM_USE_V1"] = "1"

    import torch
    import torch_npu
    lib_paths = [
        "/data/ascend/vllm-ascend/vllm_ascend",
        "/data/ascend/vllm-ascend/vllm_ascend/lib64",
        os.path.join(os.path.dirname(torch.__file__), "lib"),
        os.path.join(os.path.dirname(torch_npu.__file__), "lib"),
    ]
    env["LD_LIBRARY_PATH"] = ":".join(lib_paths) + ":" + env.get(
        "LD_LIBRARY_PATH", "")
    return env


def build_diverse_prompts(n):
    """Build n diverse prompts.

    Identical prompts let prefix caching serve 49/50 prefills for free, so the
    measured throughput mostly reflects decode bandwidth and is inflated vs a
    real mixed load. Diverse prompts force every request through a real prefill.
    """
    topics = ["machine learning", "climate change", "quantum computing",
              "renewable energy", "artificial intelligence", "space exploration",
              "genetic engineering", "blockchain", "neural networks",
              "cybersecurity", "deep learning", "data science"]
    templates = [
        "Explain {t} in simple terms.",
        "What are the main challenges in {t}?",
        "Write a short introduction about {t}.",
        "Describe the future of {t}.",
        "List three key concepts in {t}.",
    ]
    base = [fmt.format(t=t) for t in topics for fmt in templates]
    return (base * ((n // len(base)) + 1))[:n]


def run_single_model(model_name, quantization, dtype, label_name,
                     num_prompts, max_tokens, max_model_len, output_dir):
    """Run benchmark for a single model in a fresh subprocess.

    Each model gets its own process to avoid NPU state issues between loads.
    `quantization=None` runs the dense (unquantized) path.
    """
    prompts_list = build_diverse_prompts(num_prompts)
    script = f'''
import json, os, re, subprocess, sys, time
import torch

from vllm import LLM, SamplingParams


def npu_hbm_mb():
    """Read current NPU HBM usage (MB) via npu-smi — more reliable than
    torch.npu.max_memory_allocated() which returns 0 under
    expandable_segments mode."""
    try:
        r = subprocess.run(["npu-smi", "info"], capture_output=True,
                           text=True, timeout=5)
        for line in r.stdout.split("\\n"):
            m = re.search(r"(\\d+)\\s*/\\s*65536", line)
            if m:
                return float(m.group(1))
    except Exception:
        pass
    return 0.0


model_name = {model_name!r}
quantization = {quantization!r}
dtype = {dtype!r}
label_name = {label_name!r}
num_prompts = {num_prompts}
max_tokens = {max_tokens}
max_model_len = {max_model_len}
output_file = {output_dir!r} + "/single_result.json"
input_text = "The capital of France is"

print(f"\\n[BENCH] Loading {{label_name}} (model={{model_name}}, quant={{quantization}}, dtype={{dtype}})...", flush=True)
load_start = time.perf_counter()
llm_kwargs = dict(
    model=model_name,
    tensor_parallel_size=1,
    max_model_len=max_model_len,
    enforce_eager=True,
    gpu_memory_utilization=0.85,
    trust_remote_code=True,
    dtype=dtype,
)
if quantization is not None:
    llm_kwargs["quantization"] = quantization
llm = LLM(**llm_kwargs)
load_time = time.perf_counter() - load_start
print(f"[BENCH] Model loaded in {{load_time:.1f}}s", flush=True)

# Weight memory right after load (HBM includes weights + small overhead)
mem_after_load = npu_hbm_mb()
try:
    torch.npu.reset_peak_memory_allocated()
except Exception:
    pass

prompts = {prompts_list!r}  # diverse prompts — force real prefill every request
sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

# Warmup: full-batch runs to fully prime NPU kernels (stabilises throughput)
print(f"[BENCH] Warming up (2 full-batch runs)...", flush=True)
for _ in range(2):
    llm.generate(prompts, sampling_params, use_tqdm=False)

# Throughput benchmark (batch) — 3 rounds, median (NPU throughput varies run-to-run)
print(f"[BENCH] Throughput benchmark ({{num_prompts}} prompts x {{max_tokens}} tokens, 3 rounds)...", flush=True)
tp_rates = []
for _tpi in range(3):
    torch.npu.synchronize()
    _tp_t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    torch.npu.synchronize()
    _tp_dt = time.perf_counter() - _tp_t0
    _tp_tot = sum(len(o.outputs[0].token_ids) for o in outputs)
    tp_rates.append(_tp_tot / _tp_dt)
tp_rates.sort()
print(f"[BENCH] Throughput rounds: {{[round(r,1) for r in tp_rates]}} tok/s", flush=True)
tokens_per_sec = tp_rates[1]
total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
total_time = total_tokens / tokens_per_sec
requests_per_sec = num_prompts / total_time

# Peak HBM right after the throughput run (weights + KV cache + activations)
peak_hbm = npu_hbm_mb()
peak_alloc = 0.0
try:
    peak_alloc = torch.npu.max_memory_allocated() / (1024 * 1024)
except Exception:
    pass
peak_mem = max(peak_hbm, peak_alloc)

sample_text = outputs[0].outputs[0].text[:200] if outputs else ""
print(f"[BENCH] Sample prompt: {{prompts[0][:80]}}", flush=True)
print(f"[BENCH] Sample output: {{sample_text[:120]}}...", flush=True)

# TTFT approximation: time to first token (max_tokens=1, median of 5 runs)
print(f"[BENCH] Measuring TTFT (max_tokens=1, 5 runs)...", flush=True)
ttft_samples = []
sp_one = SamplingParams(max_tokens=1, temperature=0.0)
for _ in range(5):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    llm.generate([input_text], sp_one, use_tqdm=False)
    torch.npu.synchronize()
    ttft_samples.append(time.perf_counter() - t0)
ttft_samples.sort()
ttft_ms = ttft_samples[len(ttft_samples) // 2] * 1000

# Latency benchmark (sequential single requests) -> TPOT
print(f"[BENCH] Latency benchmark (20 sequential requests)...", flush=True)
latencies = []
gen_lens = []
for _ in range(20):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    out = llm.generate([input_text], sampling_params, use_tqdm=False)
    torch.npu.synchronize()
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000)
    gen_lens.append(len(out[0].outputs[0].token_ids))

latencies.sort()
avg_latency = sum(latencies) / len(latencies)
p50_latency = latencies[len(latencies) // 2]
p99_latency = latencies[int(len(latencies) * 0.99)]
mean_gen = sum(gen_lens) / len(gen_lens)
# TPOT = (total latency - TTFT) / (remaining generated tokens)
tpot_ms = (avg_latency - ttft_ms) / max(mean_gen - 1, 1)

result = {{
    "label_name": label_name,
    "model": model_name,
    "quantization": quantization,
    "dtype": dtype,
    "load_time_s": round(load_time, 2),
    "memory_after_load_hbm_mb": round(mem_after_load, 1),
    "memory_peak_hbm_mb": round(peak_mem, 1),
    "memory_peak_alloc_mb": round(peak_alloc, 1),
    "throughput": {{
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 3),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "requests_per_sec": round(requests_per_sec, 2),
        "num_prompts": num_prompts,
        "max_tokens": max_tokens,
    }},
    "latency": {{
        "ttft_ms": round(ttft_ms, 2),
        "tpot_ms": round(tpot_ms, 2),
        "avg_ms": round(avg_latency, 2),
        "p50_ms": round(p50_latency, 2),
        "p99_ms": round(p99_latency, 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "mean_gen_len": round(mean_gen, 1),
    }},
    "sample_output": sample_text,
}}

with open(output_file, "w") as f:
    json.dump(result, f, indent=2)

print(f"[BENCH] Done {{label_name}}: {{tokens_per_sec:.1f}} tok/s, "
      f"TTFT={{ttft_ms:.1f}}ms, TPOT={{tpot_ms:.2f}}ms, "
      f"HBM(load)={{mem_after_load:.0f}}MB, HBM(peak)={{peak_mem:.0f}}MB", flush=True)
'''

    env = build_env()

    output_file = os.path.join(output_dir, "single_result.json")
    # Remove stale result
    if os.path.exists(output_file):
        os.remove(output_file)

    print(f"\n{'='*70}")
    print(f" Starting: {label_name} ({model_name})")
    print(f"{'='*70}")

    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd="/data/ascend/vllm-ascend",
        timeout=900,
    )

    if os.path.exists(output_file):
        with open(output_file) as f:
            data = json.load(f)
        os.remove(output_file)
        return data
    print(f"  [ERROR] No result file produced for {label_name}")
    return None


def main():
    parser = argparse.ArgumentParser(description="AWQ/GPTQ Benchmark for Ascend NPU")
    parser.add_argument(
        "--label", default="default",
        help="Label for this run (e.g., baseline, optimized).",
    )
    parser.add_argument(
        "--output", default="benchmarks/results",
        help="Output directory for results.",
    )
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Dense vs Quantized Benchmark — {args.label}")
    print(f"Config: {args.num_prompts} prompts x {args.max_tokens} tokens, "
          f"max_model_len={args.max_model_len}, eager, greedy")
    print(f"NPU memory before: {get_npu_memory_mb():.0f} MB")

    # (model, quantization, dtype, label_name)
    # Each quantized model is benchmarked under BOTH float16 and bfloat16
    # activations, since T6 (BF16 activation support) must be demonstrated in
    # the paired comparison, not just asserted.
    models = [
        ("Qwen/Qwen2.5-0.5B-Instruct", None, "float16", "dense-fp16"),
        ("Qwen/Qwen2.5-0.5B-Instruct", None, "bfloat16", "dense-bf16"),
        ("Qwen/Qwen2.5-0.5B-Instruct-AWQ", "awq", "float16", "awq-fp16"),
        ("Qwen/Qwen2.5-0.5B-Instruct-AWQ", "awq", "bfloat16", "awq-bf16"),
        ("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", "gptq", "float16", "gptq-int4-fp16"),
        ("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", "gptq", "bfloat16", "gptq-int4-bf16"),
        ("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8", "gptq", "float16", "gptq-int8-fp16"),
        ("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8", "gptq", "bfloat16", "gptq-int8-bf16"),
    ]

    results = []
    for model_name, quant, dtype, label_name in models:
        r = run_single_model(
            model_name, quant, dtype, label_name,
            args.num_prompts, args.max_tokens,
            args.max_model_len, args.output,
        )
        if r is not None:
            results.append(r)

    # ---- Save aggregate results ----
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(
        args.output, f"python_bench_{args.label}_{timestamp}.json"
    )

    output = {
        "label": args.label,
        "timestamp": timestamp,
        "config": {
            "num_prompts": args.num_prompts,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
        },
        "models": results,
    }

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    # ---- Print comparison table ----
    print(f"\n{'='*86}")
    print(f"  BENCHMARK SUMMARY — {args.label}")
    print(f"  {args.num_prompts} prompts x {args.max_tokens} tokens, eager, greedy")
    print(f"{'='*86}")
    print(
        f"  {'Label':<16} {'Load(s)':>8} {'tok/s':>9} {'TTFT(ms)':>9} "
        f"{'TPOT(ms)':>9} {'p99(ms)':>8} {'HBMpk(MB)':>10}"
    )
    print(f"  {'-'*85}")
    for r in results:
        print(
            f"  {r['label_name']:<16} {r['load_time_s']:>8.1f} "
            f"{r['throughput']['tokens_per_sec']:>9.1f} "
            f"{r['latency']['ttft_ms']:>9.1f} {r['latency']['tpot_ms']:>9.2f} "
            f"{r['latency']['p99_ms']:>8.1f} {r['memory_peak_hbm_mb']:>10.0f}"
        )
    print(f"\n  Full results: {out_file}")


if __name__ == "__main__":
    main()
