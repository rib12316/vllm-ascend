# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This file is part of the vllm-ascend project.
"""Throughput benchmark for real AWQ/GPTQ MoE models on Ascend NPU.

Reuses ``run_single_model`` from ``benchmark_awq_gptq.py`` (each model runs in
its own subprocess, so NPU memory is released between loads) to measure load
time / peak HBM / throughput (3-round median, diverse prompts) / TTFT / TPOT
for the real MoE models on disk.

⚠️ UNPAIRED: this is quantized-only. The project's paired-data rule (quantized
vs dense on the same machine) needs a dense MoE counterpart; none is on disk,
so these numbers show MoE inference is performant, NOT the quantized-vs-dense
delta. Re-run with dense models added to ``MODELS`` once available.

Usage:
    cd /data/ascend/vllm-ascend
    source /data/ascend/.venv/bin/activate
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export ASCEND_CUSTOM_OPP_PATH=$PWD/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
    export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$PWD/vllm_ascend:$LD_LIBRARY_PATH
    python benchmarks/benchmark_moe_throughput.py 2>&1 | tee logs/bench/<date>_moe-throughput.log
"""

import argparse
import json
import os
import time

from benchmark_awq_gptq import run_single_model  # same dir (benchmarks/)

# Quantized MoE models on disk. dense counterparts are NOT available locally.
MODELS = [
    # (model_path, quantization, dtype, label)
    ("/data/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4", "gptq", "float16", "moe-gptq-int4"),
    ("/data/DeepSeek-V2-Lite-Chat-AWQ", "awq", "float16", "moe-awq-int4"),
]


def main():
    parser = argparse.ArgumentParser(description="MoE throughput benchmark (Ascend NPU)")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--output", default="benchmarks/results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print("MoE Throughput Benchmark — quantized-only (NO dense pairing on disk)")
    print(
        f"Config: {args.num_prompts} diverse prompts x {args.max_tokens} tokens, "
        f"max_model_len={args.max_model_len}, eager, greedy, 3-round median\n"
    )

    results = []
    for model_path, quant, dtype, label in MODELS:
        r = run_single_model(
            model_path,
            quant,
            dtype,
            label,
            args.num_prompts,
            args.max_tokens,
            args.max_model_len,
            args.output,
        )
        if r is not None:
            results.append(r)

    if not results:
        print("[ERROR] No benchmark results collected.")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(args.output, f"python_bench_moe_{ts}.json")
    with open(out_file, "w") as f:
        json.dump(
            {
                "label": "moe-throughput",
                "timestamp": ts,
                "paired": False,
                "note": "quantized-only; no dense MoE baseline on disk",
                "config": {
                    "num_prompts": args.num_prompts,
                    "max_tokens": args.max_tokens,
                    "max_model_len": args.max_model_len,
                },
                "models": results,
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 92}")
    print(f"{'Label':<18}{'Load(s)':>9}{'tok/s':>9}{'TTFT(ms)':>10}{'TPOT(ms)':>10}{'HBMpk(MB)':>11}")
    print("-" * 92)
    for r in results:
        print(
            f"{r['label_name']:<18}{r['load_time_s']:>9.1f}"
            f"{r['throughput']['tokens_per_sec']:>9.1f}"
            f"{r['latency']['ttft_ms']:>10.1f}{r['latency']['tpot_ms']:>10.2f}"
            f"{r['memory_peak_hbm_mb']:>11.0f}"
        )
    print("=" * 92)
    print("NOTE: UNPAIRED — no dense MoE baseline on disk (violates paired-data rule).")
    print(f"Results JSON: {out_file}")


if __name__ == "__main__":
    main()
