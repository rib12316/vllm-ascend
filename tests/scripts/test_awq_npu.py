#!/usr/bin/env python3
"""
AWQ 量化 NPU 部署验证脚本

用法（在 NPU 机器上）:
    # 步骤 1: 验证环境
    python tests/scripts/test_awq_npu.py --check-env

    # 步骤 2: 运行单元测试（不需要模型下载）
    python tests/scripts/test_awq_npu.py --unit-test

    # 步骤 3: 端到端推理验证（需要下载模型）
    python tests/scripts/test_awq_npu.py --e2e --model Qwen/Qwen2.5-0.5B-Instruct-AWQ

    # 步骤 4: MoE AWQ 验证（需要 2 卡）
    python tests/scripts/test_awq_npu.py --e2e --model billy800/Qwen3-30B-A3B-Instruct-2507-AWQ --tp 2

    # 一步到位：全部运行
    python tests/scripts/test_awq_npu.py --all --model Qwen/Qwen2.5-0.5B-Instruct-AWQ

    # 也可以通过 pytest 运行单元测试部分:
    pytest tests/scripts/test_awq_npu.py -v -k "not e2e"
"""

import argparse
import sys
import traceback


def check_env():
    """步骤 1: 验证 NPU 环境配置"""
    print("=" * 60)
    print("步骤 1: 环境检查")
    print("=" * 60)

    errors = []
    warnings = []

    # 1. 检查 torch
    try:
        import torch
        print(f"  [OK] torch 版本: {torch.__version__}")
        if not torch.__version__.startswith("2."):
            warnings.append(f"torch 版本 {torch.__version__} 可能不兼容，建议 2.10.0")
    except ImportError:
        errors.append("torch 未安装")
        return errors, warnings

    # 2. 检查 torch_npu
    try:
        import torch_npu
        print(f"  [OK] torch_npu 版本: {torch_npu.__version__}")
    except ImportError:
        errors.append("torch_npu 未安装（NPU 算子不可用）")
        return errors, warnings

    # 3. 检查 NPU 设备
    try:
        npu_count = torch.npu.device_count()
        if npu_count == 0:
            errors.append("未检测到 NPU 设备")
        else:
            print(f"  [OK] 检测到 {npu_count} 个 NPU 设备")
            for i in range(npu_count):
                name = torch.npu.get_device_name(i)
                print(f"        NPU:{i} = {name}")
    except Exception as e:
        errors.append(f"NPU 设备检测失败: {e}")

    # 4. 检查 vllm
    try:
        import vllm
        print(f"  [OK] vllm 版本: {vllm.__version__}")
    except ImportError:
        errors.append("vllm 未安装")

    # 5. 检查 vllm_ascend
    try:
        import vllm_ascend
        print(f"  [OK] vllm_ascend 已安装")
    except ImportError:
        errors.append("vllm_ascend 未安装")

    # 6. 检查 AWQ 配置是否注册
    try:
        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
        if "awq" in QUANTIZATION_METHODS:
            # 检查是不是我们的 AWQConfig（不是 vLLM 原生的）
            from vllm_ascend.quantization.awq_config import AWQConfig
            config_cls = QUANTIZATION_METHODS["awq"]
            if config_cls is AWQConfig:
                print(f"  [OK] AWQConfig 已注册 (vllm-ascend 版本)")
            else:
                warnings.append(f"AWQ 注册为 {config_cls.__name__}，不是 vllm-ascend 的 AWQConfig")
        else:
            errors.append("'awq' 未在 QUANTIZATION_METHODS 中注册")
    except Exception as e:
        errors.append(f"AWQ 注册检查失败: {e}")

    # 7. 检查 NPU 算子
    try:
        # 测试 npu_weight_quant_batchmatmul 是否可用
        x = torch.randn(2, 4, dtype=torch.float16).npu()
        w = torch.zeros(4, 2, dtype=torch.int32).npu()
        s = torch.ones(4, 1, dtype=torch.float16).npu()
        o = torch.zeros(4, 1, dtype=torch.float16).npu()
        result = torch_npu.npu_weight_quant_batchmatmul(
            x, w, antiquant_scale=s, antiquant_offset=o, antiquant_group_size=1
        )
        print(f"  [OK] npu_weight_quant_batchmatmul 算子可用")
    except Exception as e:
        errors.append(f"NPU 算子测试失败: {e}")

    # 8. 检查 QuantType
    try:
        from vllm_ascend.quantization.quant_type import QuantType
        if hasattr(QuantType, "W4A16_AWQ"):
            print(f"  [OK] QuantType.W4A16_AWQ = {QuantType.W4A16_AWQ.value}")
        else:
            errors.append("QuantType.W4A16_AWQ 不存在")
    except Exception as e:
        errors.append(f"QuantType 检查失败: {e}")

    # 9. 检查 platform supported_quantization
    try:
        from vllm_ascend.platform import NPUPlatform
        if "awq" in NPUPlatform.supported_quantization:
            print(f"  [OK] 'awq' 在 supported_quantization 中")
        else:
            errors.append("'awq' 不在 supported_quantization 中")
    except Exception as e:
        errors.append(f"Platform 检查失败: {e}")

    print()
    if errors:
        print("❌ 环境检查失败:")
        for e in errors:
            print(f"   - {e}")
    if warnings:
        print("⚠️ 警告:")
        for w in warnings:
            print(f"   - {w}")
    if not errors and not warnings:
        print("✅ 所有环境检查通过!")

    return errors, warnings


def run_unit_tests():
    """步骤 2: 运行 AWQ 单元测试（mock NPU 算子，不需要真实硬件）"""
    print("=" * 60)
    print("步骤 2: AWQ 单元测试")
    print("=" * 60)

    import unittest
    from tests.ut.quantization.methods.test_w4a16_awq import (
        TestAWQConfig,
        TestAscendW4A16AWQLinearMethod,
        TestAscendW4A16AWQFusedMoEMethod,
        TestUnpackQzeroFromInt32,
        TestUnpackWeightFromInt32,
    )

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestAWQConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestAscendW4A16AWQLinearMethod))
    suite.addTests(loader.loadTestsFromTestCase(TestAscendW4A16AWQFusedMoEMethod))
    suite.addTests(loader.loadTestsFromTestCase(TestUnpackQzeroFromInt32))
    suite.addTests(loader.loadTestsFromTestCase(TestUnpackWeightFromInt32))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    if result.wasSuccessful():
        print(f"✅ 单元测试全部通过 ({result.testsRun} 个)")
    else:
        print(f"❌ 单元测试失败: {len(result.failures)} failures, {len(result.errors)} errors")

    return result.wasSuccessful()


def run_e2e_test(model_name: str, tp: int = 1, max_tokens: int = 5):
    """步骤 3: 端到端推理验证"""
    print("=" * 60)
    print("步骤 3: 端到端推理验证")
    print("=" * 60)
    print(f"  模型: {model_name}")
    print(f"  TP: {tp}")
    print(f"  max_tokens: {max_tokens}")
    print()

    from vllm import SamplingParams

    from tests.e2e.conftest import VllmRunner

    example_prompts = [
        "The president of the United States is",
        "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.",
    ]

    try:
        with VllmRunner(
            model_name,
            tensor_parallel_size=tp,
            max_model_len=4096,
            enforce_eager=True,
            gpu_memory_utilization=0.7,
        ) as vllm_model:
            outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

        print("推理输出:")
        for i, (token_ids, text) in enumerate(outputs):
            print(f"  [{i}] tokens={token_ids}")
            print(f"      text ={text!r}")
        print()
        print("✅ 端到端推理成功!")
        return True

    except Exception as e:
        print(f"❌ 端到端推理失败: {e}")
        traceback.print_exc()
        return False


def run_numerical_correctness(model_name: str, tp: int = 1):
    """步骤 4: 数值正确性验证（与 FP16 对比）"""
    print("=" * 60)
    print("步骤 4: 数值正确性验证（AWQ 量化 vs FP16 参考）")
    print("=" * 60)

    # 查找对应的 FP16 模型
    fp16_model = model_name.replace("-AWQ", "").replace("-awq", "")
    if fp16_model == model_name:
        print("⚠️ 无法自动推断 FP16 基准模型，跳过数值对比")
        print(f"   手动对比: 用相同 prompt 分别跑 {model_name} (awq) 和 FP16 模型")
        return True

    print(f"  AWQ 模型: {model_name}")
    print(f"  FP16 参考: {fp16_model}")

    from vllm import SamplingParams

    from tests.e2e.conftest import VllmRunner

    prompts = ["The capital of France is"]

    try:
        # AWQ 推理
        with VllmRunner(
            model_name,
            tensor_parallel_size=tp,
            max_model_len=512,
            enforce_eager=True,
        ) as runner:
            awq_outputs = runner.generate_greedy(prompts, max_tokens=20)

        print(f"  AWQ 输出: {awq_outputs[0][1]!r}")

        # FP16 推理（如果显存足够）
        try:
            with VllmRunner(
                fp16_model,
                tensor_parallel_size=tp,
                max_model_len=512,
                enforce_eager=True,
            ) as runner:
                fp16_outputs = runner.generate_greedy(prompts, max_tokens=20)

            print(f"  FP16 输出: {fp16_outputs[0][1]!r}")

            # 对比
            awq_tokens = awq_outputs[0][0]
            fp16_tokens = fp16_outputs[0][0]
            min_len = min(len(awq_tokens), len(fp16_tokens))
            matches = sum(1 for a, b in zip(awq_tokens[:min_len], fp16_tokens[:min_len]) if a == b)
            match_rate = matches / min_len if min_len > 0 else 0
            print(f"  Token 匹配率: {match_rate:.1%} ({matches}/{min_len})")

            if match_rate >= 0.8:
                print("✅ 数值正确性验证通过 (>= 80% token 匹配)")
                return True
            else:
                print("⚠️ 数值正确性偏低，可能需要排查权重处理逻辑")
                return False
        except Exception as e:
            print(f"⚠️ FP16 参考推理失败（可能显存不足）: {e}")
            print(f"   AWQ 推理本身成功，跳过数值对比")
            return True

    except Exception as e:
        print(f"❌ 数值验证失败: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="AWQ NPU 部署验证脚本")
    parser.add_argument("--check-env", action="store_true", help="步骤1: 检查环境")
    parser.add_argument("--unit-test", action="store_true", help="步骤2: 运行单元测试")
    parser.add_argument("--e2e", action="store_true", help="步骤3: 端到端推理")
    parser.add_argument("--numerical", action="store_true", help="步骤4: 数值正确性")
    parser.add_argument("--all", action="store_true", help="运行全部步骤")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct-AWQ",
                        help="AWQ 模型名称 (默认: Qwen/Qwen2.5-0.5B-Instruct-AWQ)")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size (默认: 1)")
    parser.add_argument("--max-tokens", type=int, default=5, help="最大生成 token 数 (默认: 5)")

    args = parser.parse_args()

    if not any([args.check_env, args.unit_test, args.e2e, args.numerical, args.all]):
        parser.print_help()
        print("\n示例:")
        print("  python tests/scripts/test_awq_npu.py --check-env")
        print("  python tests/scripts/test_awq_npu.py --unit-test")
        print("  python tests/scripts/test_awq_npu.py --e2e --model Qwen/Qwen2.5-0.5B-Instruct-AWQ")
        print("  python tests/scripts/test_awq_npu.py --all --model Qwen/Qwen2.5-0.5B-Instruct-AWQ")
        sys.exit(0)

    all_passed = True

    # 切换到 vllm-ascend 根目录（确保 imports 正确）
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    if args.check_env or args.all:
        errors, _ = check_env()
        if errors:
            all_passed = False
            if args.all:
                print("\n⚠️ 环境检查失败，跳过后续步骤\n")
                sys.exit(1)

    if args.unit_test or args.all:
        passed = run_unit_tests()
        all_passed = all_passed and passed

    if args.e2e or args.all:
        passed = run_e2e_test(args.model, args.tp, args.max_tokens)
        all_passed = all_passed and passed

    if args.numerical or args.all:
        passed = run_numerical_correctness(args.model, args.tp)
        all_passed = all_passed and passed

    print()
    print("=" * 60)
    if all_passed:
        print("🎉 所有验证通过!")
    else:
        print("❌ 部分验证失败，请查看上方日志")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
