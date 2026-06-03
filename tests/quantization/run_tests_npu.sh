#!/bin/bash
# ===========================================================================
# Ascend NPU AWQ/GPTQ 量化集成测试脚本
# ===========================================================================
# 用途：在 NPU 服务器上运行完整的量化推理测试
# 前置条件：
#   1. Docker 环境已加载（vllm-ascend 镜像）
#   2. torch_npu, vllm, vllm-ascend 已安装
#   3. NPU 设备可用（Ascend 910B）
#
# 使用方法：
#   chmod +x tests/quantization/run_tests_npu.sh
#   bash tests/quantization/run_tests_npu.sh
# ===========================================================================

set -euo pipefail

# ---- 配置区 ----
# 可根据实际环境修改以下变量
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-"0"}
export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-"false"}

# 测试模型列表（需要提前下载到本地或使用 HF repo ID）
# 格式: "模型名:量化方法:预期说明"
AWQ_MODELS=(
    "TheBloke/Llama-2-7B-AWQ:awq:AWQ 4-bit Linear"
    "casperhansen/mistral-7b-instruct-v0.1-awq:awq:AWQ 4-bit Linear (Mistral)"
    # MoE 模型（如果需要测试 MoE 路径）:
    # "TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ:awq:AWQ 4-bit MoE"
)

GPTQ_MODELS=(
    "TheBloke/Llama-2-7B-GPTQ:gptq:GPTQ 4-bit Linear v1"
    "TheBloke/Llama-2-7B-chat-GPTQ:gptq:GPTQ 4-bit Linear v1 (chat)"
    # v2 格式模型:
    # "ModelCloud/Qwen1.5-7B-Chat-GPTQ-Int4:gptq:GPTQ 4-bit v2"
    # 8-bit 模型:
    # "TheBloke/Llama-2-7B-GPTQ-gptq-8bit:gptq:GPTQ 8-bit"
)

# 测试输出目录
TEST_OUTPUT_DIR="/tmp/vllm_ascend_quant_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${TEST_OUTPUT_DIR}"

# ---- 颜色定义 ----
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# ---- 辅助函数 ----
log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_pass()  { echo -e "${GREEN}[PASS]${NC} $1"; }
log_fail()  { echo -e "${RED}[FAIL]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_sep()   { echo "================================================================"; }

# 检查 NPU 设备
check_npu() {
    log_info "检查 NPU 设备..."
    if python3 -c "import torch_npu; print(f'NPU 可用: {torch.npu.is_available()}, 设备数: {torch.npu.device_count()}')" 2>/dev/null; then
        log_pass "NPU 设备检查通过"
    else
        log_fail "NPU 设备不可用，请检查驱动和 CANN 安装"
        exit 1
    fi
}

# 运行单模型推理测试
# 参数: $1=模型路径, $2=量化方法, $3=说明, $4=输出文件前缀
run_inference_test() {
    local model="$1"
    local quant="$2"
    local desc="$3"
    local prefix="$4"
    local outfile="${TEST_OUTPUT_DIR}/${prefix}_output.txt"
    local errfile="${TEST_OUTPUT_DIR}/${prefix}_error.txt"

    log_sep
    log_info "测试: ${desc}"
    log_info "  模型: ${model}"
    log_info "  量化: --quantization ${quant}"
    log_info "  输出: ${outfile}"

    # 运行 vLLM 离线推理
    python3 -c "
import torch
from vllm import LLM, SamplingParams

print('加载模型...')
llm = LLM(
    model='${model}',
    quantization='${quant}',
    dtype='float16',
    tensor_parallel_size=1,
    trust_remote_code=True,
    enforce_eager=True,   # 使用 eager 模式（不编译图）
    gpu_memory_utilization=0.9,
)

print('模型加载成功！开始推理...')
prompts = [
    'Hello, my name is',
    'The capital of France is',
    'The largest planet in our solar system is',
    '1 + 1 = ',
]
sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(prompts, sampling_params)

print('\\n推理结果:')
for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    print(f'  提示: {prompt!r}')
    print(f'  生成: {generated!r}')
    print()

print('测试通过!')
" > "${outfile}" 2> "${errfile}"

    local exit_code=$?

    if [ ${exit_code} -eq 0 ]; then
        log_pass "${desc} — 成功"
        # 显示推理结果
        grep -A1 "生成:" "${outfile}" | head -8 || true
    else
        log_fail "${desc} — 失败 (exit code: ${exit_code})"
        echo "--- 错误信息 ---"
        tail -30 "${errfile}"
        echo "---"
    fi

    return ${exit_code}
}

# 运行 vLLM serve 测试（在线推理）
# 参数: $1=模型路径, $2=量化方法, $3=说明, $4=端口
run_serve_test() {
    local model="$1"
    local quant="$2"
    local desc="$3"
    local port="${4:-8000}"
    local outfile="${TEST_OUTPUT_DIR}/serve_${quant}_output.txt"

    log_sep
    log_info "Serve 测试: ${desc}"
    log_info "  模型: ${model}"
    log_info "  量化: --quantization ${quant}"
    log_info "  端口: ${port}"

    # 后台启动 vLLM server
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${model}" \
        --quantization "${quant}" \
        --dtype float16 \
        --port "${port}" \
        --tensor-parallel-size 1 \
        --trust-remote-code \
        --enforce-eager \
        > "${outfile}" 2>&1 &
    local server_pid=$!

    log_info "等待服务器启动 (PID: ${server_pid})..."
    local max_wait=300  # 最多等 5 分钟
    local waited=0
    while ! curl -s "http://localhost:${port}/health" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [ ${waited} -ge ${max_wait} ]; then
            log_fail "服务器启动超时 (${max_wait}s)"
            kill ${server_pid} 2>/dev/null || true
            return 1
        fi
        # 检查进程是否还在
        if ! kill -0 ${server_pid} 2>/dev/null; then
            log_fail "服务器进程已退出"
            tail -30 "${outfile}"
            return 1
        fi
    done

    log_info "服务器已启动，发送测试请求..."

    # 发送测试请求
    local response
    response=$(curl -s "http://localhost:${port}/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${model}"'",
            "prompt": "San Francisco is a",
            "max_tokens": 20,
            "temperature": 0
        }')

    if echo "${response}" | python3 -c "import sys,json; r=json.load(sys.stdin); print('生成:', r['choices'][0]['text'])" 2>/dev/null; then
        log_pass "${desc} — Serve 测试成功"
    else
        log_fail "${desc} — Serve 测试失败"
        echo "响应: ${response}"
    fi

    # 停止服务器
    kill ${server_pid} 2>/dev/null || true
    wait ${server_pid} 2>/dev/null || true
}

# ===========================================================================
# 主测试流程
# ===========================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Ascend NPU AWQ/GPTQ 量化集成测试                          ║"
echo "║   时间: $(date)                                             ║"
echo "║   输出目录: ${TEST_OUTPUT_DIR}                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ---- 阶段 1: 环境检查 ----
log_sep
log_info "阶段 1: 环境检查"
log_sep

check_npu

# 检查 vllm-ascend 版本
python3 -c "
try:
    import vllm_ascend
    print(f'vllm-ascend 版本: {vllm_ascend.__version__}')
except:
    print('vllm-ascend 未安装')
try:
    import vllm
    print(f'vllm 版本: {vllm.__version__}')
except:
    print('vllm 未安装')
" 2>/dev/null || log_warn "无法获取版本信息"

# ---- 阶段 2: 单元测试 ----
log_sep
log_info "阶段 2: 单元测试（不需要 NPU）"
log_sep

log_info "运行 pytest 单元测试..."
python3 -m pytest tests/quantization/test_awq_gptq.py -v \
    -k "not NPUOperator" \
    --tb=short \
    2>&1 | tee "${TEST_OUTPUT_DIR}/unit_test_results.txt" || true

# ---- 阶段 3: AWQ 推理测试 ----
log_sep
log_info "阶段 3: AWQ 推理测试"
log_sep

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

for entry in "${AWQ_MODELS[@]}"; do
    IFS=':' read -r model quant desc <<< "${entry}"

    # 检查模型是否可访问
    if [ -d "${model}" ] || python3 -c "from huggingface_hub import model_info; model_info('${model}')" 2>/dev/null; then
        if run_inference_test "${model}" "${quant}" "${desc}" "awq_$(echo ${model} | tr '/' '_')"; then
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        log_warn "跳过 ${model}（模型不可访问）"
        SKIP_COUNT=$((SKIP_COUNT + 1))
    fi
done

# ---- 阶段 4: GPTQ 推理测试 ----
log_sep
log_info "阶段 4: GPTQ 推理测试"
log_sep

for entry in "${GPTQ_MODELS[@]}"; do
    IFS=':' read -r model quant desc <<< "${entry}"

    if [ -d "${model}" ] || python3 -c "from huggingface_hub import model_info; model_info('${model}')" 2>/dev/null; then
        if run_inference_test "${model}" "${quant}" "${desc}" "gptq_$(echo ${model} | tr '/' '_')"; then
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        log_warn "跳过 ${model}（模型不可访问）"
        SKIP_COUNT=$((SKIP_COUNT + 1))
    fi
done

# ---- 阶段 5: NPU 算子测试 ----
log_sep
log_info "阶段 5: NPU 算子集成测试"
log_sep

python3 -m pytest tests/quantization/test_awq_gptq.py -v \
    -k "NPUOperator" \
    --tb=short \
    2>&1 | tee "${TEST_OUTPUT_DIR}/npu_operator_results.txt" || true

# ---- 汇总 ----
log_sep
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   测试结果汇总                                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo -e "  ${GREEN}通过: ${PASS_COUNT}${NC}"
echo -e "  ${RED}失败: ${FAIL_COUNT}${NC}"
echo -e "  ${YELLOW}跳过: ${SKIP_COUNT}${NC}"
echo ""
echo "详细输出目录: ${TEST_OUTPUT_DIR}"
echo ""

if [ ${FAIL_COUNT} -gt 0 ]; then
    log_fail "存在失败的测试，请检查错误日志"
    exit 1
else
    log_pass "所有测试通过！"
    exit 0
fi
