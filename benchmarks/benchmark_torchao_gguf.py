#!/usr/bin/env python3
"""torchao / gguf Performance Benchmark for Ascend NPU — Python API version.

Paired comparison (required by the scoring rules): on the SAME hardware/config,
measure dense FP16 vs the torchao/gguf quantized paths:

  * dense-fp16                — baseline (Qwen2.5-0.5B-Instruct)
  * torchao-int8wo            — HIGH-PERF npu_weight_quant_batchmatmul (per-channel)
  * torchao-int4wo            — HIGH-PERF npu op (per-group128, self-impl RTN)
  * torchao-fp8wo             — dense fallback (CPU fp8 quant → dense → F.linear)
  * gguf-q4_0                 — HIGH-PERF gguf_repack → npu op
  * gguf-q8_0                 — HIGH-PERF gguf_repack → npu op
  * gguf-q4_k_m               — dense fallback (CPU dequant K-quants)

Measures throughput / TTFT / TPOT / peak HBM for each. Each model runs in a fresh
subprocess to avoid NPU state issues between loads.

MUST run from /data/ascend/vllm-ascend/ (CWD namespace conflict). Save stdout:

    cd /data/ascend/vllm-ascend
    source /data/ascend/.venv/bin/activate
    python benchmarks/benchmark_torchao_gguf.py --label torchao_gguf \
      2>&1 | tee /data/ascend/logs/bench/2026-06-17_torchao-gguf-bench.log
"""

import argparse
import json
import os
import subprocess
import sys
import time

if not os.path.exists("vllm_ascend"):
    print("ERROR: Must run from /data/ascend/vllm-ascend/ directory")
    sys.exit(1)


def get_npu_memory_mb():
    try:
        result = subprocess.run(
            ["npu-smi", "info", "-t", "usages", "-i", "0"],
            capture_output=True,
            text=True,
            timeout=5,
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
    """LD_LIBRARY_PATH so vllm-ascend custom ops (libvllm_ascend_kernels.so) load;
    without it all custom ops are disabled and performance collapses."""
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
    env["LD_LIBRARY_PATH"] = ":".join(lib_paths) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def build_diverse_prompts(n):
    """Diverse prompts — identical prompts let prefix caching serve prefills for
    free and inflate throughput vs a real mixed load."""
    topics = [
        "machine learning",
        "climate change",
        "quantum computing",
        "renewable energy",
        "artificial intelligence",
        "space exploration",
        "genetic engineering",
        "blockchain",
        "neural networks",
        "cybersecurity",
        "deep learning",
        "data science",
    ]
    templates = [
        "Explain {t} in simple terms.",
        "What are the main challenges in {t}?",
        "Write a short introduction about {t}.",
        "Describe the future of {t}.",
        "List three key concepts in {t}.",
    ]
    base = [fmt.format(t=t) for t in topics for fmt in templates]
    return (base * ((n // len(base)) + 1))[:n]


def run_single_model(
    model_name, quantization, dtype, label_name, load_format, num_prompts, max_tokens, max_model_len, output_dir
):
    """Run one config in a fresh subprocess. quantization=None → dense path.
    load_format=None unless gguf (which needs load_format='gguf')."""
    prompts_list = build_diverse_prompts(num_prompts)
    script = f'''
import json, os, re, subprocess, sys, time
import torch

from vllm import LLM, SamplingParams


def npu_hbm_mb():
    """Read NPU HBM (MB) via npu-smi — torch.npu.max_memory_allocated() returns
    0 under expandable_segments mode, so npu-smi is the reliable source."""
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
load_format = {load_format!r}
num_prompts = {num_prompts}
max_tokens = {max_tokens}
max_model_len = {max_model_len}
output_file = {output_dir!r} + "/single_result.json"
input_text = "The capital of France is"

print(f"\\n[BENCH] Loading {{label_name}} (model={{model_name}}, quant={{quantization}}, "
      f"load_format={{load_format}}, dtype={{dtype}})...", flush=True)
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
if load_format is not None:
    llm_kwargs["load_format"] = load_format
llm = LLM(**llm_kwargs)
load_time = time.perf_counter() - load_start
print(f"[BENCH] Model loaded in {{load_time:.1f}}s", flush=True)

mem_after_load = npu_hbm_mb()
try:
    torch.npu.reset_peak_memory_allocated()
except Exception:
    pass

prompts = {prompts_list!r}
sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

print(f"[BENCH] Warming up (2 full-batch runs)...", flush=True)
for _ in range(2):
    llm.generate(prompts, sampling_params, use_tqdm=False)

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

peak_hbm = npu_hbm_mb()
peak_alloc = 0.0
try:
    peak_alloc = torch.npu.max_memory_allocated() / (1024 * 1024)
except Exception:
    pass
peak_mem = max(peak_hbm, peak_alloc)

sample_text = outputs[0].outputs[0].text[:200] if outputs else ""
correct = "paris" in sample_text.lower()
print(f"[BENCH] Sample output: {{sample_text[:120]}}...", flush=True)

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
tpot_ms = (avg_latency - ttft_ms) / max(mean_gen - 1, 1)

result = {{
    "label_name": label_name,
    "model": model_name,
    "quantization": quantization,
    "load_format": load_format,
    "dtype": dtype,
    "correct_paris": correct,
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
      f"TTFT={{ttft_ms:.1f}}ms, TPOT={{tpot_ms:.2f}}ms, Paris={{correct}}, "
      f"HBM(load)={{mem_after_load:.0f}}MB, HBM(peak)={{peak_mem:.0f}}MB", flush=True)
'''

    env = build_env()
    output_file = os.path.join(output_dir, "single_result.json")
    if os.path.exists(output_file):
        os.remove(output_file)

    print(f"\n{'=' * 70}")
    print(f" Starting: {label_name}")
    print(f"{'=' * 70}")

    # Run from this worktree's root so `import vllm_ascend` resolves to THIS
    # branch's code (CWD shadows the editable install; the main repo may be on a
    # different branch). Portable across worktrees.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=repo_root,
        timeout=900,
    )

    if os.path.exists(output_file):
        with open(output_file) as f:
            data = json.load(f)
        os.remove(output_file)
        return data
    print(f"  [ERROR] No result file produced for {label_name}")
    return None


def _warmup_npu(model_name, num_prompts, max_model_len):
    """Throwaway NPU warmup before the measured loop (results discarded).

    The first model loaded in a run hits a cold NPU (driver/kernel lazy-init),
    depressing its throughput ~15-20%. fp8wo — identical fp16 compute to
    dense-fp16 — measured +17.8% vs the cold dense baseline, i.e. the cold start,
    not a real speedup. This primes the NPU so config #1 measures warm.
    """
    prompts = build_diverse_prompts(num_prompts)
    script = f"""
from vllm import LLM, SamplingParams

llm = LLM(model={model_name!r}, tensor_parallel_size=1, max_model_len={max_model_len},
          enforce_eager=True, gpu_memory_utilization=0.85, trust_remote_code=True,
          dtype="float16")
sp = SamplingParams(max_tokens=16, temperature=0.0)
for _ in range(2):
    llm.generate({prompts!r}, sp, use_tqdm=False)
print("[WARMUP] NPU primed", flush=True)
"""
    env = build_env()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"\n{'=' * 70}")
    print(" [WARMUP] Priming NPU (dense load + 2 batches, discarded)")
    print(f"{'=' * 70}")
    subprocess.run([sys.executable, "-c", script], env=env, cwd=repo_root, timeout=600)


def main():
    parser = argparse.ArgumentParser(description="torchao/gguf Benchmark for Ascend NPU")
    parser.add_argument("--label", default="torchao_gguf")
    parser.add_argument("--output", default="benchmarks/results")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    gguf_dir = "/data/ascend/torchao_gguf/e2e_models/qwen05b-gguf"
    torchao_dir = "/data/ascend/torchao_gguf/e2e_models"
    # (model, quantization, dtype, label, load_format)
    models = [
        ("Qwen/Qwen2.5-0.5B-Instruct", None, "float16", "dense-fp16", None),
        (f"{torchao_dir}/qwen05b-torchao-int8wo", "torchao", "float16", "torchao-int8wo", None),
        (f"{torchao_dir}/qwen05b-torchao-int4wo", "torchao", "float16", "torchao-int4wo", None),
        (f"{torchao_dir}/qwen05b-torchao-fp8wo", "torchao", "float16", "torchao-fp8wo", None),
        (f"{gguf_dir}/qwen2.5-0.5b-instruct-q4_0.gguf", "gguf", "float16", "gguf-q4_0-hp", "gguf"),
        (f"{gguf_dir}/qwen2.5-0.5b-instruct-q8_0.gguf", "gguf", "float16", "gguf-q8_0-hp", "gguf"),
        (f"{gguf_dir}/qwen2.5-0.5b-instruct-q4_k_m.gguf", "gguf", "float16", "gguf-q4_k_m-dense", "gguf"),
    ]

    print(f"torchao/gguf vs dense Benchmark — {args.label}")
    print(
        f"Config: {args.num_prompts} prompts x {args.max_tokens} tokens, "
        f"max_model_len={args.max_model_len}, eager, greedy"
    )
    print(f"NPU memory before: {get_npu_memory_mb():.0f} MB")

    _warmup_npu("Qwen/Qwen2.5-0.5B-Instruct", args.num_prompts, args.max_model_len)

    results = []
    for model_name, quant, dtype, label_name, load_fmt in models:
        r = run_single_model(
            model_name,
            quant,
            dtype,
            label_name,
            load_fmt,
            args.num_prompts,
            args.max_tokens,
            args.max_model_len,
            args.output,
        )
        if r is not None:
            results.append(r)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(args.output, f"python_bench_{args.label}_{timestamp}.json")

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

    # Paired comparison vs dense-fp16
    dense_tp = next((r["throughput"]["tokens_per_sec"] for r in results if r["label_name"] == "dense-fp16"), None)

    print(f"\n{'=' * 98}")
    print(f"  BENCHMARK SUMMARY — {args.label}")
    print(
        f"  {args.num_prompts} prompts x {args.max_tokens} tokens, eager, greedy  (dense-fp16 = {dense_tp:.1f} tok/s)"
        if dense_tp
        else ""
    )
    print(f"{'=' * 98}")
    print(
        f"  {'Label':<20} {'Load(s)':>7} {'tok/s':>8} {'vs dense':>9} "
        f"{'TTFT':>7} {'TPOT':>7} {'Paris':>6} {'HBMpk(MB)':>10}"
    )
    print(f"  {'-' * 97}")
    for r in results:
        tp = r["throughput"]["tokens_per_sec"]
        delta = f"{(tp / dense_tp - 1) * 100:+.1f}%" if dense_tp else "-"
        print(
            f"  {r['label_name']:<20} {r['load_time_s']:>7.1f} {tp:>8.1f} "
            f"{delta:>9} {r['latency']['ttft_ms']:>6.1f}m "
            f"{r['latency']['tpot_ms']:>6.2f}m "
            f"{str(r['correct_paris']):>6} {r['memory_peak_hbm_mb']:>10.0f}"
        )
    print(f"\n  Full results: {out_file}")


if __name__ == "__main__":
    main()
