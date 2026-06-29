#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This file is part of the vllm-ascend project.
"""One-click repro for the Ascend AWQ/GPTQ quant results in 项目任务书 §5.6.

Runs, in order: unit tests → real-MoE end-to-end → 0.5B paired benchmark →
real-MoE throughput → real-MoE accuracy. Each NPU step runs in its own
subprocess (the benchmark/eval scripts already do per-model subprocesses), so
HBM is released between models. Set SKIP_LONG=1 to skip the 0.5B + MoE
benchmarks (keep only tests + e2e).

Run from vllm-ascend/ with the venv active (this script sources set_env.sh):
    bash benchmarks/run_all_quant.sh 2>&1 | tee logs/$(date +%F)_run_all.log
"""
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$HERE"

# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
# shellcheck disable=SC1091
source /data/ascend/.venv/bin/activate
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-false}
export VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
export ASCEND_CUSTOM_OPP_PATH="$PWD/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
# Append (never overwrite) so CANN's libascendcl.so/libhccl.so stay on the path.
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$PWD/vllm_ascend:${LD_LIBRARY_PATH:-}"

step() { printf '\n========== %s ==========\n' "$1"; }

step "1/5 unit tests (109)"
python -m pytest tests/quantization/test_awq_gptq.py tests/quantization/test_moe_synthetic_npu.py \
                 tests/quantization/test_quant_routing.py tests/ut/quantization/test_method_adapters.py -q

step "2/5 real-MoE end-to-end (GPTQ ALL PASS expected)"
python benchmarks/verify_moe_e2e.py

if [ "${SKIP_LONG:-0}" = "1" ]; then
    echo "SKIP_LONG=1 → skipping 0.5B benchmark + MoE throughput/accuracy."
    exit 0
fi

step "3/5 0.5B paired benchmark (dense/AWQ/GPTQ-Int4/Int8)"
python benchmarks/benchmark_awq_gptq.py --label run_all

step "4/5 real-MoE throughput (unpaired)"
python benchmarks/benchmark_moe_throughput.py

step "5/5 real-MoE accuracy (lm_eval, GPTQ + AWQ via --max-num-seqs 4)"
python benchmarks/eval_moe_accuracy.py --limit 500 --max-model-len 1024 --max-num-seqs 4

echo "Done. See logs/ for each step's full output."
