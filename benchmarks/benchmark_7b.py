#!/usr/bin/env python3
"""7B paired benchmark: reuse benchmark_awq_gptq.run_single_model to compare
Qwen2.5-7B dense vs AWQ vs GPTQ-Int4 on the same 910B card — the same paired
methodology as the 0.5B table (diverse prompts, 3-round median, Paris probe,
TTFT/TPOT/peak HBM). Lifts the 0.5B-only scope to a real-size model where the
quantization HBM savings are observable.

Run::

    cd /data/ascend/vllm-ascend
    source /data/ascend/.venv/bin/activate
    python benchmarks/benchmark_7b.py \\
        2>&1 | tee /data/ascend/logs/bench/2026-07-01_7b-bench.log
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_awq_gptq import run_single_model  # noqa: E402

# (model, quantization, dtype, label) — paired: dense baseline + both quant methods.
MODELS = [
    ("Qwen/Qwen2.5-7B-Instruct", None, "float16", "dense-7b-fp16"),
    ("Qwen/Qwen2.5-7B-Instruct-AWQ", "awq", "float16", "awq-7b-int4-fp16"),
    ("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", "gptq", "float16", "gptq-7b-int4-fp16"),
]


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs",
        default=",".join(m[3] for m in MODELS),
        help="comma-separated labels to run (default: all). Useful for running each model as its download completes.",
    )
    args = ap.parse_args()
    wanted = {c.strip() for c in args.configs.split(",")}
    models = [m for m in MODELS if m[3] in wanted]

    out_dir = "benchmarks/results"
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for model_name, quant, dtype, label in models:
        r = run_single_model(
            model_name,
            quant,
            dtype,
            label,
            num_prompts=50,
            max_tokens=128,
            max_model_len=4096,
            output_dir=out_dir,
        )
        if r is not None:
            results.append(r)

    print(f"\n{'=' * 86}")
    print("  7B PAIRED BENCHMARK SUMMARY (dense vs AWQ vs GPTQ-Int4)")
    print("  50 diverse prompts x 128 tokens, eager, greedy")
    print(f"{'=' * 86}")
    print(f"  {'Label':<22} {'Load(s)':>8} {'tok/s':>9} {'TTFT(ms)':>9} {'TPOT(ms)':>9} {'HBMpk(MB)':>10}")
    print(f"  {'-' * 85}")
    for r in results:
        print(
            f"  {r['label_name']:<22} {r['load_time_s']:>8.1f} "
            f"{r['throughput']['tokens_per_sec']:>9.1f} "
            f"{r['latency']['ttft_ms']:>9.1f} {r['latency']['tpot_ms']:>9.2f} "
            f"{r['memory_peak_hbm_mb']:>10.0f}"
        )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"python_bench_7b_{timestamp}.json")
    with open(out_file, "w") as f:
        json.dump({"label": "7b-paired", "timestamp": timestamp, "models": results}, f, indent=2)
    print(f"\n  Full results: {out_file}")


if __name__ == "__main__":
    main()
