#!/usr/bin/env python3
"""AWQ/GPTQ Performance Benchmark for Ascend NPU — Python API version.

Uses vLLM's Python API (LLM class) for controlled offline inference
measurements. Each model runs in a fresh process to avoid NPU state issues.

Must be run from /data/ascend/vllm-ascend/ directory:
    cd /data/ascend/vllm-ascend
    source /data/ascend/.venv/bin/activate
    HF_HOME=/data/huggingface_home python benchmarks/benchmark_awq_gptq.py --label baseline
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
    """Get current NPU memory usage via npu-smi."""
    try:
        result = subprocess.run(
            ["npu-smi", "info", "-t", "usages", "-i", "0"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.split("\n"):
            if "Used Capacity" in line:
                # Parse "Used Capacity : XXXX / 65536 MB"
                parts = line.split(":")
                if len(parts) >= 2:
                    mb_str = parts[1].strip().split("/")[0].strip().split()[0]
                    return float(mb_str)
    except Exception:
        pass
    return 0.0


def run_single_model(model_name, quantization, num_prompts, max_tokens,
                     max_model_len, label, output_dir):
    """Run benchmark for a single model in a subprocess.

    Each model gets its own process to avoid NPU state issues between loads.
    """
    script = f'''
import json, os, sys, time
import torch

from vllm import LLM, SamplingParams

model_name = "{model_name}"
quantization = "{quantization}"
num_prompts = {num_prompts}
max_tokens = {max_tokens}
max_model_len = {max_model_len}
output_file = "{output_dir}/single_result.json"
input_text = "The capital of France is"

print(f"\\n[BENCH] Loading {{model_name}} ({{quantization}})...", flush=True)
load_start = time.perf_counter()
llm = LLM(
    model=model_name,
    quantization=quantization,
    tensor_parallel_size=1,
    max_model_len=max_model_len,
    enforce_eager=True,
    gpu_memory_utilization=0.85,
    trust_remote_code=True,
)
load_time = time.perf_counter() - load_start
print(f"[BENCH] Model loaded in {{load_time:.1f}}s", flush=True)

# Get NPU memory after loading
mem_after_load = 0.0
try:
    torch.npu.synchronize()
    mem_after_load = torch.npu.memory_allocated() / (1024*1024)
except Exception:
    pass

prompts = [input_text] * num_prompts
sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

# Warmup
print(f"[BENCH] Warming up...", flush=True)
for _ in range(2):
    llm.generate(prompts[:1], sampling_params)

# Throughput benchmark
print(f"[BENCH] Running throughput benchmark ({{num_prompts}} prompts)...", flush=True)
torch.npu.synchronize()
start_time = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
torch.npu.synchronize()
total_time = time.perf_counter() - start_time

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
tokens_per_sec = total_tokens / total_time
requests_per_sec = num_prompts / total_time

# Peak memory
peak_mem = 0.0
try:
    peak_mem = torch.npu.max_memory_allocated() / (1024*1024)
except Exception:
    pass

# Sample output
sample_text = ""
if outputs:
    sample_text = outputs[0].outputs[0].text[:200]
    print(f"[BENCH] Sample: {{sample_text[:100]}}...", flush=True)

# Latency benchmark (sequential single requests)
print(f"[BENCH] Running latency benchmark (20 sequential requests)...", flush=True)
latencies = []
for i in range(20):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    out = llm.generate([input_text], sampling_params)
    torch.npu.synchronize()
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000)

latencies.sort()
avg_latency = sum(latencies) / len(latencies)
p50_latency = latencies[len(latencies)//2]
p99_latency = latencies[int(len(latencies)*0.99)]

result = {{
    "model": model_name,
    "quantization": quantization,
    "load_time_s": round(load_time, 2),
    "memory_weight_mb": round(mem_after_load, 1),
    "memory_peak_mb": round(peak_mem, 1),
    "throughput": {{
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 3),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "requests_per_sec": round(requests_per_sec, 2),
        "num_prompts": num_prompts,
        "max_tokens": max_tokens,
    }},
    "latency": {{
        "avg_ms": round(avg_latency, 2),
        "p50_ms": round(p50_latency, 2),
        "p99_ms": round(p99_latency, 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
    }},
    "sample_output": sample_text,
}}

with open(output_file, "w") as f:
    json.dump(result, f, indent=2)

print(f"[BENCH] Done. {{tokens_per_sec:.1f}} tok/s, {{avg_latency:.1f}} ms avg latency", flush=True)
'''

    env = os.environ.copy()
    env["HF_HOME"] = "/data/huggingface_home"
    env["VLLM_USE_V1"] = "1"

    output_file = os.path.join(output_dir, f"single_result.json")
    # Remove stale result
    if os.path.exists(output_file):
        os.remove(output_file)

    print(f"\n{'='*60}")
    print(f" Starting: {model_name} ({quantization})")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd="/data/ascend/vllm-ascend",
        timeout=600,
    )

    if os.path.exists(output_file):
        with open(output_file) as f:
            data = json.load(f)
        # Clean up temp file
        os.remove(output_file)
        return data
    else:
        print(f"  [ERROR] No result file produced for {model_name}")
        return None


def main():
    parser = argparse.ArgumentParser(description="AWQ/GPTQ Benchmark for Ascend NPU")
    parser.add_argument(
        "--label", default="default",
        help="Label for this run (e.g., baseline, optimized)",
    )
    parser.add_argument(
        "--output", default="benchmarks/results",
        help="Output directory for results",
    )
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"AUQ/GPTQ Performance Benchmark — {args.label}")
    print(f"Num prompts: {args.num_prompts}, Max tokens: {args.max_tokens}")

    # Check NPU
    mem_before = get_npu_memory_mb()
    print(f"NPU memory before: {mem_before:.0f} MB")

    models = [
        ("Qwen/Qwen2.5-0.5B-Instruct-AWQ", "awq"),
        ("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", "gptq"),
        ("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8", "gptq"),
    ]

    results = []
    for model_name, quant in models:
        r = run_single_model(
            model_name, quant,
            args.num_prompts, args.max_tokens,
            args.max_model_len, args.label, args.output,
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
    print(f"\n{'='*70}")
    print(f"  BENCHMARK SUMMARY — {args.label}")
    print(f"  {args.num_prompts} prompts × {args.max_tokens} max tokens")
    print(f"{'='*70}")
    print(
        f"  {'Model':<35} {'Load(s)':>8} {'tok/s':>10} "
        f"{'Lat(ms)':>8} {'Mem(MB)':>8}"
    )
    print(f"  {'-'*69}")
    for r in results:
        name = r["model"].replace("Qwen/", "").replace("-Instruct", "")
        tp = r["throughput"]["tokens_per_sec"]
        lt = r["latency"]["avg_ms"]
        ld = r["load_time_s"]
        mem = r.get("memory_weight_mb", 0)
        print(
            f"  {name:<35} {ld:>8.1f} {tp:>10.1f} "
            f"{lt:>8.1f} {mem:>8.1f}"
        )
    print(f"\n  Full results: {out_file}")


if __name__ == "__main__":
    main()
