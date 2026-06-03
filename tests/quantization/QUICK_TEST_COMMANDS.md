# Ascend NPU AWQ/GPTQ 测试命令参考卡

> **用途**：在 NPU 服务器上逐步验证 AWQ/GPTQ 量化功能
> **环境要求**：vllm-ascend Docker 镜像 + Ascend 910B NPU

---

## 一、环境验证（30 秒）

```bash
# 1. 检查 NPU 设备
python3 -c "import torch_npu; print(f'NPU: {torch.npu.is_available()}, Count: {torch.npu.device_count()}')"

# 2. 检查 vllm 版本
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')"

# 3. 检查 vllm-ascend 是否识别 AWQ/GPTQ
python3 -c "
from vllm_ascend.quantization import AWQConfig, GPTQConfig
print('AWQConfig: OK')
print('GPTQConfig: OK')
"
```

---

## 二、单元测试（1 分钟）

```bash
# 在 vllm-ascend 根目录执行
cd /path/to/vllm-ascend

# 运行所有 AWQ/GPTQ 单元测试（不需要 NPU）
pytest tests/quantization/test_awq_gptq.py -v -k "not NPUOperator" --tb=short

# 仅运行 AWQ 测试
pytest tests/quantization/test_awq_gptq.py -v -k "awq and not NPUOperator"

# 仅运行 GPTQ 测试
pytest tests/quantization/test_awq_gptq.py -v -k "gptq and not NPUOperator"

# 运行 NPU 算子集成测试（需要 NPU）
pytest tests/quantization/test_awq_gptq.py -v -k "NPUOperator" --tb=short
```

---

## 三、AWQ 模型推理测试

### 3.1 离线推理（推荐首选）

```bash
# AWQ Llama-2-7B (最常见的 AWQ 测试模型)
python3 -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='TheBloke/Llama-2-7B-AWQ',
    quantization='awq',
    dtype='float16',
    tensor_parallel_size=1,
    enforce_eager=True,
)

prompts = ['Hello, my name is', 'The capital of France is']
outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=32))
for o in outputs:
    print(f'Prompt: {o.prompt!r} -> {o.outputs[0].text!r}')
print('AWQ test PASSED')
"
```

### 3.2 AWQ BF16 激活测试

```bash
python3 -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='TheBloke/Llama-2-7B-AWQ',
    quantization='awq',
    dtype='bfloat16',
    tensor_parallel_size=1,
    enforce_eager=True,
)

outputs = llm.generate(
    ['Hello, my name is'],
    SamplingParams(temperature=0.0, max_tokens=32),
)
print(f'BF16 result: {outputs[0].outputs[0].text!r}')
print('AWQ BF16 test PASSED')
"
```

### 3.3 AWQ MoE 模型测试（Mixtral）

```bash
python3 -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ',
    quantization='awq',
    dtype='float16',
    tensor_parallel_size=1,
    enforce_eager=True,
    max_model_len=512,
)

outputs = llm.generate(
    ['Explain quantum computing briefly.'],
    SamplingParams(temperature=0.0, max_tokens=64),
)
print(f'MoE result: {outputs[0].outputs[0].text!r}')
print('AWQ MoE test PASSED')
"
```

### 3.4 AWQ 在线服务测试

```bash
# 启动 vLLM server
python3 -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-7B-AWQ \
    --quantization awq \
    --dtype float16 \
    --port 8000 \
    --enforce-eager &

# 等待启动
sleep 60

# 发送请求
curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"TheBloke/Llama-2-7B-AWQ","prompt":"San Francisco is a","max_tokens":20,"temperature":0}'

# 停止服务器
kill %1
```

---

## 四、GPTQ 模型推理测试

### 4.1 GPTQ 4-bit 离线推理

```bash
python3 -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='TheBloke/Llama-2-7B-GPTQ',
    quantization='gptq',
    dtype='float16',
    tensor_parallel_size=1,
    enforce_eager=True,
)

prompts = ['Hello, my name is', 'The capital of France is']
outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=32))
for o in outputs:
    print(f'Prompt: {o.prompt!r} -> {o.outputs[0].text!r}')
print('GPTQ 4-bit test PASSED')
"
```

### 4.2 GPTQ Marlin 拦截验证

```bash
# 验证 GPTQ 模型不会自动升级到 gptq_marlin
python3 -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='TheBloke/Llama-2-7B-GPTQ',
    quantization='gptq',
    dtype='float16',
    enforce_eager=True,
)
# 如果没有报 'Marlin not supported' 错误，说明拦截成功
print('GPTQ Marlin interception test PASSED')
"
```

### 4.3 GPTQ v2 格式测试

```bash
python3 -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='ModelCloud/Qwen1.5-7B-Chat-GPTQ-Int4',
    quantization='gptq',
    dtype='float16',
    enforce_eager=True,
    trust_remote_code=True,
)

outputs = llm.generate(
    ['你好，请介绍一下你自己。'],
    SamplingParams(temperature=0.0, max_tokens=64),
)
print(f'GPTQ v2 result: {outputs[0].outputs[0].text!r}')
print('GPTQ v2 test PASSED')
"
```

### 4.4 GPTQ desc_act 测试

```bash
# desc_act=True 的模型需要 g_idx 排序处理
python3 -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='TheBloke/Llama-2-7B-Chat-GPTQ',
    quantization='gptq',
    dtype='float16',
    enforce_eager=True,
)

outputs = llm.generate(
    ['What is machine learning?'],
    SamplingParams(temperature=0.0, max_tokens=64),
)
print(f'desc_act result: {outputs[0].outputs[0].text!r}')
print('GPTQ desc_act test PASSED')
"
```

### 4.5 GPTQ 8-bit 测试

```bash
python3 -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='TheBloke/Llama-2-7B-GPTQ-gptq-8bit',
    quantization='gptq',
    dtype='float16',
    enforce_eager=True,
)

outputs = llm.generate(
    ['Hello, my name is'],
    SamplingParams(temperature=0.0, max_tokens=32),
)
print(f'8-bit result: {outputs[0].outputs[0].text!r}')
print('GPTQ 8-bit test PASSED')
"
```

### 4.6 GPTQ MoE 模型测试

```bash
python3 -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ',
    quantization='gptq',
    dtype='float16',
    tensor_parallel_size=1,
    enforce_eager=True,
    max_model_len=512,
)

outputs = llm.generate(
    ['Explain quantum computing briefly.'],
    SamplingParams(temperature=0.0, max_tokens=64),
)
print(f'GPTQ MoE result: {outputs[0].outputs[0].text!r}')
print('GPTQ MoE test PASSED')
"
```

---

## 五、自动化测试（一键运行）

```bash
# 运行完整自动化测试脚本
bash tests/quantization/run_tests_npu.sh
```

---

## 六、常见问题排查

### Q1: "GPTQMarlinConfig is not supported"
```
说明: GPTQConfig.override_quantization_method 未生效
排查: 确认 gptq_config.py 已被 platform.py 正确导入
```

### Q2: "npu_weight_quant_batchmatmul failed"
```
说明: NPU 算子参数错误
排查:
  1. 检查 antiquant_group_size 与 group_size 一致
  2. 检查 antiquant_scale/offset 的 shape
  3. 检查 qweight 的 dtype 是否为 int32
```

### Q3: 输出 shape 不对
```
AWQ: out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
GPTQ 4-bit: out_shape = x.shape[:-1] + (qweight.shape[-1],)  # 已重打包
GPTQ 8-bit: out_shape = x.shape[:-1] + (qweight.shape[-1],)
```

### Q4: "No such config file: quantize_config.json"
```
AWQ 使用 quant_config.json 或 quantize_config.json
GPTQ 仅使用 quantize_config.json
```

---

## 七、预期测试结果矩阵

| 测试项 | 预期结果 | 备注 |
|--------|---------|------|
| AWQ 4-bit Linear | ✅ 正常推理 | 核心路径 |
| AWQ BF16 | ✅ 正常推理 | NPU 支持 BF16 |
| AWQ MoE | ⚠️ 需验证 | 依赖 fused_experts |
| GPTQ 4-bit v1 | ✅ 正常推理 | qzeros + 1 |
| GPTQ 4-bit v2 | ✅ 正常推理 | qzeros 直接用 |
| GPTQ 8-bit | ⚠️ 需验证 | int8 路径 |
| GPTQ desc_act | ⚠️ 需验证 | g_idx 排序 |
| GPTQ MoE 4-bit | ⚠️ 需验证 | npu_convert_weight_to_int4pack |
| GPTQ MoE 8-bit | ⚠️ 需验证 | int8 MoE 路径 |
| Marlin 拦截 | ✅ 返回 "gptq" | 不走 Marlin |
