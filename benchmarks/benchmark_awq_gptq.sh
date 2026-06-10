#!/usr/bin/env bash
# ==============================================================================
# AWQ/GPTQ Performance Benchmark for Ascend NPU
# ==============================================================================
# Measures throughput, latency, and memory for AWQ/GPTQ quantized models.
# Must be run from /data/ascend/vllm-ascend/ directory.
#
# Usage:
#   bash benchmarks/benchmark_awq_gptq.sh [--baseline|--optimized]
#   --baseline:   Run before optimization (for baseline comparison)
#   --optimized:  Run after P1 NZ format optimization
#   (default):    Run all benchmarks
# ==============================================================================

set -euo pipefail

# ---- Configuration ----
VENV_PATH="/data/ascend/.venv/bin/activate"
HF_HOME="/data/huggingface_home"
RESULTS_DIR="benchmarks/results"
SHAREGPT_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
SHAREGPT_PATH="/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
MAX_MODEL_LEN=4096
NUM_PROMPTS=100
LABEL="default"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline)  LABEL="baseline" ;;
        --optimized) LABEL="optimized" ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="${RESULTS_DIR}/benchmark_${LABEL}_${TIMESTAMP}.json"

# ---- Environment setup ----
echo "=============================================="
echo " AWQ/GPTQ Performance Benchmark — ${LABEL}"
echo " Timestamp: ${TIMESTAMP}"
echo "=============================================="

source "${VENV_PATH}"
export HF_HOME="${HF_HOME}"
export VLLM_USE_V1=0  # Use V0 for stability with quantized models

# Check NPU
echo "[INFO] Checking NPU device..."
if ! npu-smi info > /dev/null 2>&1; then
    echo "[ERROR] NPU not available. Aborting."
    exit 1
fi
npu-smi info | head -20

# Download ShareGPT dataset
if [[ ! -f "${SHAREGPT_PATH}" ]]; then
    echo "[INFO] Downloading ShareGPT dataset..."
    mkdir -p /data/datasets
    wget -q "${SHAREGPT_URL}" -O "${SHAREGPT_PATH}"
    echo "[INFO] Dataset downloaded: ${SHAREGPT_PATH}"
else
    echo "[INFO] ShareGPT dataset found: ${SHAREGPT_PATH}"
fi

# Create results directory
mkdir -p "${RESULTS_DIR}"

# ---- Model definitions ----
declare -a MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct-AWQ|awq"
    "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4|gptq"
    "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8|gptq"
)

# ---- Helper functions ----
run_throughput_bench() {
    local model=$1
    local quant=$2
    local model_safe=$(echo "${model}" | tr '/' '_')
    local out_file="${RESULTS_DIR}/throughput_${model_safe}_${LABEL}.json"

    echo ""
    echo "================================================"
    echo " THROUGHPUT: ${model} (quantization=${quant})"
    echo "================================================"

    vllm bench throughput \
        --model "${model}" \
        --quantization "${quant}" \
        --tensor-parallel-size 1 \
        --max-model-len "${MAX_MODEL_LEN}" \
        --enforce-eager \
        --dataset-name sharegpt \
        --dataset-path "${SHAREGPT_PATH}" \
        --num-prompts "${NUM_PROMPTS}" \
        --output-json "${out_file}" \
        2>&1 | tail -20

    echo "[INFO] Throughput result saved to: ${out_file}"
    if [[ -f "${out_file}" ]]; then
        echo "[INFO] Result preview:"
        python3 -c "
import json
with open('${out_file}') as f:
    d = json.load(f)
print(f'  Total tokens: {d.get(\"num_tokens_total\", \"N/A\")}')
print(f'  Requests/s:   {d.get(\"requests_per_sec\", \"N/A\")}')
print(f'  Tokens/s:     {d.get(\"tokens_per_sec\", \"N/A\")}')
print(f'  Elapsed (s):  {d.get(\"elapsed_time\", \"N/A\")}')
" 2>/dev/null || echo "  (Could not parse JSON result)"
    fi
}

run_latency_bench() {
    local model=$1
    local quant=$2
    local model_safe=$(echo "${model}" | tr '/' '_')
    local out_file="${RESULTS_DIR}/latency_${model_safe}_${LABEL}.json"

    echo ""
    echo "================================================"
    echo " LATENCY: ${model} (quantization=${quant})"
    echo "================================================"

    vllm bench latency \
        --model "${model}" \
        --quantization "${quant}" \
        --tensor-parallel-size 1 \
        --max-model-len "${MAX_MODEL_LEN}" \
        --enforce-eager \
        --batch-size 1 \
        --input-len 128 \
        --output-len 128 \
        --num-iters-warmup 3 \
        --num-iters 10 \
        --output-json "${out_file}" \
        2>&1 | tail -20

    echo "[INFO] Latency result saved to: ${out_file}"
    if [[ -f "${out_file}" ]]; then
        echo "[INFO] Result preview:"
        python3 -c "
import json
with open('${out_file}') as f:
    d = json.load(f)
print(f'  Avg latency (ms):     {d.get(\"avg_latency_ms\", \"N/A\")}')
print(f'  TTFT (ms):            {d.get(\"avg_ttft_ms\", \"N/A\")}')
print(f'  TPOT (ms):            {d.get(\"avg_tpot_ms\", \"N/A\")}')
print(f'  Throughput (tok/s):   {d.get(\"avg_throughput_tok_per_s\", \"N/A\")}')
" 2>/dev/null || echo "  (Could not parse JSON result)"
    fi
}

# ---- Run benchmarks ----
echo ""
echo "[INFO] Starting benchmarks..."

# Initialize aggregate results
echo "{ \"label\": \"${LABEL}\", \"timestamp\": \"${TIMESTAMP}\", \"results\": {} }" > "${RESULT_FILE}"

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model quant <<< "${entry}"

    # Throughput benchmark
    run_throughput_bench "${model}" "${quant}"

    # Latency benchmark
    run_latency_bench "${model}" "${quant}"

    echo ""
done

# ---- Aggregate results ----
echo ""
echo "=============================================="
echo " Aggregating results..."
echo "=============================================="

python3 << 'AGGREGATE_SCRIPT'
import json, glob, os

label = os.environ.get("LABEL", "default")
results_dir = "${RESULTS_DIR}"

aggregate = {
    "label": label,
    "timestamp": "${TIMESTAMP}",
    "models": []
}

for entry in ["Qwen_Qwen2.5-0.5B-Instruct-AWQ|awq",
              "Qwen_Qwen2.5-0.5B-Instruct-GPTQ-Int4|gptq",
              "Qwen_Qwen2.5-0.5B-Instruct-GPTQ-Int8|gptq"]:
    model_safe, quant = entry.split("|")

    model_result = {"model": model_safe, "quantization": quant}

    # Throughput
    tp_files = glob.glob(f"${RESULTS_DIR}/throughput_{model_safe}_{label}.json")
    if tp_files:
        with open(tp_files[-1]) as f:
            tp = json.load(f)
        model_result["throughput"] = {
            "tokens_per_sec": tp.get("tokens_per_sec"),
            "requests_per_sec": tp.get("requests_per_sec"),
            "elapsed_time": tp.get("elapsed_time"),
            "num_tokens_total": tp.get("num_tokens_total"),
        }

    # Latency
    lt_files = glob.glob(f"${RESULTS_DIR}/latency_{model_safe}_{label}.json")
    if lt_files:
        with open(lt_files[-1]) as f:
            lt = json.load(f)
        model_result["latency"] = {
            "avg_latency_ms": lt.get("avg_latency_ms"),
            "avg_ttft_ms": lt.get("avg_ttft_ms"),
            "avg_tpot_ms": lt.get("avg_tpot_ms"),
            "avg_throughput_tok_per_s": lt.get("avg_throughput_tok_per_s"),
        }

    aggregate["models"].append(model_result)

# Print summary
print("\n" + "=" * 60)
print(f"  BENCHMARK SUMMARY — {label}")
print("=" * 60)
print(f"{'Model':<45} {'tok/s':>8} {'Lat(ms)':>8}")
print("-" * 60)
for m in aggregate["models"]:
    tp = m.get("throughput", {}).get("tokens_per_sec", "N/A")
    lt = m.get("latency", {}).get("avg_latency_ms", "N/A")
    name = m["model"].replace("Qwen_", "").replace("_Instruct", "")
    print(f"  {name:<43} {str(tp):>8} {str(lt):>8}")

# Save
out_file = "${RESULT_FILE}"
with open(out_file, "w") as f:
    json.dump(aggregate, f, indent=2, default=str)
print(f"\n[INFO] Full results saved to: {out_file}")
AGGREGATE_SCRIPT

echo ""
echo "=============================================="
echo " Benchmark complete!"
echo "=============================================="
