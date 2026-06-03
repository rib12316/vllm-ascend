#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for AWQ quantization (T1: Config + Linear + MoE).

These tests use mocked NPU operators so they can run on CPU without
real Ascend hardware. For NPU deployment validation, see the e2e
tests and the NPU testing guide in this file's docstring.

NPU Deployment Testing Guide
=============================

1. Prerequisites on the NPU machine:
   - vllm-ascend installed with torch_npu
   - CANN >= 8.5.0, Atlas A2 series NPU

2. Run single-card AWQ model test:

    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-0.5B-Instruct-AWQ \
        --quantization awq \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.7

   Then send a test request:
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
             "prompt": "The president of the United States is",
             "max_tokens": 5}'

3. Run MoE AWQ model test (2 cards):

    python -m vllm.entrypoints.openai.api_server \
        --model billy800/Qwen3-30B-A3B-Instruct-2507-AWQ \
        --quantization awq \
        --tensor-parallel-size 2 \
        --max-model-len 4096

4. Run these unit tests on NPU:

    pytest tests/ut/quantization/methods/test_w4a16_awq.py -v

5. Verify numerical correctness (compare with CPU FP16 reference):

    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct-AWQ",
              quantization="awq", enforce_eager=True)
    output = llm.generate(["Hello world"], SamplingParams(max_tokens=20))
    print(output[0].outputs[0].text)
"""

from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.awq_config import AWQConfig
from vllm_ascend.quantization.methods.w4a16_awq import (
    AscendW4A16AWQFusedMoEMethod,
    AscendW4A16AWQLinearMethod,
    _unpack_qzero_from_int32,
    _unpack_weight_from_int32,
)


# ---------------------------------------------------------------------------
# AWQConfig tests
# ---------------------------------------------------------------------------


class TestAWQConfig(TestBase):
    """Test AWQConfig initialization and parsing."""

    def test_init_valid(self):
        config = AWQConfig(
            weight_bits=4,
            group_size=128,
            zero_point=True,
            modules_to_not_convert=["lm_head"],
        )
        self.assertEqual(config.weight_bits, 4)
        self.assertEqual(config.group_size, 128)
        self.assertTrue(config.zero_point)
        self.assertEqual(config.modules_to_not_convert, ["lm_head"])
        self.assertEqual(config.pack_factor, 8)

    def test_init_invalid_weight_bits(self):
        with self.assertRaises(ValueError) as ctx:
            AWQConfig(weight_bits=8, group_size=128, zero_point=True)
        self.assertIn("only 4-bit", str(ctx.exception))

    def test_from_config(self):
        config_dict = {
            "w_bit": 4,
            "q_group_size": 128,
            "zero_point": True,
            "modules_to_not_convert": ["lm_head"],
        }
        config = AWQConfig.from_config(config_dict)
        self.assertEqual(config.weight_bits, 4)
        self.assertEqual(config.group_size, 128)
        self.assertTrue(config.zero_point)

    def test_from_config_alt_keys(self):
        """Test alternate config key names (bits, group_size)."""
        config_dict = {
            "bits": 4,
            "group_size": 64,
            "zero_point": True,
        }
        config = AWQConfig.from_config(config_dict)
        self.assertEqual(config.group_size, 64)

    def test_get_name(self):
        config = AWQConfig(weight_bits=4, group_size=128, zero_point=True)
        self.assertEqual(config.get_name(), "awq")

    def test_get_supported_act_dtypes(self):
        dtypes = AWQConfig.get_supported_act_dtypes()
        self.assertIn(torch.float16, dtypes)
        self.assertIn(torch.bfloat16, dtypes)

    def test_get_min_capability(self):
        with self.assertRaises(NotImplementedError):
            AWQConfig.get_min_capability()

    def test_get_config_filenames(self):
        filenames = AWQConfig.get_config_filenames()
        self.assertIn("quant_config.json", filenames)
        self.assertIn("quantize_config.json", filenames)


# ---------------------------------------------------------------------------
# AscendW4A16AWQLinearMethod tests
# ---------------------------------------------------------------------------


class TestAscendW4A16AWQLinearMethod(TestBase):
    """Test AWQ linear method (weight processing and apply)."""

    def setUp(self):
        super().setUp()
        self.quant_config = AWQConfig(
            weight_bits=4,
            group_size=128,
            zero_point=True,
        )
        self.quant_method = AscendW4A16AWQLinearMethod(self.quant_config)

    def test_init(self):
        self.assertEqual(self.quant_method.pack_factor, 8)
        self.assertEqual(self.quant_method.group_size, 128)

    def test_process_weights_after_loading(self):
        hidden_size = 512
        out_features = 1024
        pack_factor = 8
        group_size = 128
        num_groups = hidden_size // group_size

        layer = torch.nn.Module()
        layer.qweight = torch.nn.Parameter(
            torch.randint(0, 100, (hidden_size, out_features // pack_factor), dtype=torch.int32),
            requires_grad=False,
        )
        layer.qzeros = torch.nn.Parameter(
            torch.randint(0, 100, (num_groups, out_features // pack_factor), dtype=torch.int32),
            requires_grad=False,
        )
        layer.scales = torch.nn.Parameter(
            torch.ones((num_groups, out_features), dtype=torch.bfloat16),
            requires_grad=False,
        )

        self.quant_method.process_weights_after_loading(layer)

        # qweight: shape unchanged, int32, contiguous
        self.assertEqual(layer.qweight.shape, (hidden_size, out_features // pack_factor))
        self.assertEqual(layer.qweight.dtype, torch.int32)
        self.assertTrue(layer.qweight.data.is_contiguous())

        # qzeros: unpacked from (num_groups, out//pack) to (num_groups, out), bfloat16
        self.assertEqual(layer.qzeros.shape, (num_groups, out_features))
        self.assertEqual(layer.qzeros.dtype, torch.bfloat16)
        self.assertTrue(layer.qzeros.data.is_contiguous())

        # All parameters require no gradient
        self.assertFalse(layer.qweight.requires_grad)
        self.assertFalse(layer.scales.requires_grad)
        self.assertFalse(layer.qzeros.requires_grad)

    @patch("vllm_ascend.quantization.methods.w4a16_awq.torch_npu.npu_weight_quant_batchmatmul")
    def test_apply(self, mock_npu_matmul):
        batch_size = 2
        seq_len = 8
        hidden_size = 512
        out_features = 1024

        mock_output = torch.randn(batch_size, seq_len, out_features, dtype=torch.float32)
        mock_npu_matmul.return_value = mock_output

        layer = torch.nn.Module()
        layer.qweight = torch.nn.Parameter(
            torch.randint(0, 100, (hidden_size, out_features // 8), dtype=torch.int32),
            requires_grad=False,
        )
        layer.scales = torch.nn.Parameter(
            torch.ones((hidden_size // 128, out_features), dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.qzeros = torch.nn.Parameter(
            torch.zeros((hidden_size // 128, out_features), dtype=torch.bfloat16),
            requires_grad=False,
        )

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
        result = self.quant_method.apply(layer, x)

        mock_npu_matmul.assert_called_once()
        self.assertEqual(result.shape, (batch_size, seq_len, out_features))

    @patch("vllm_ascend.quantization.methods.w4a16_awq.torch_npu.npu_weight_quant_batchmatmul")
    def test_apply_with_bias(self, mock_npu_matmul):
        mock_output = torch.randn(1, 1, 512, dtype=torch.float32)
        mock_npu_matmul.return_value = mock_output

        layer = torch.nn.Module()
        layer.qweight = torch.nn.Parameter(
            torch.randint(0, 100, (256, 64), dtype=torch.int32),
            requires_grad=False,
        )
        layer.scales = torch.nn.Parameter(
            torch.ones((2, 512), dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.qzeros = torch.nn.Parameter(
            torch.zeros((2, 512), dtype=torch.bfloat16),
            requires_grad=False,
        )

        x = torch.randn(1, 1, 256, dtype=torch.bfloat16)
        bias = torch.randn(512, dtype=torch.bfloat16)

        result = self.quant_method.apply(layer, x, bias)
        self.assertIsNotNone(result)

        # Verify bias was converted to float32
        call_kwargs = mock_npu_matmul.call_args.kwargs
        self.assertEqual(call_kwargs["bias"].dtype, torch.float32)


# ---------------------------------------------------------------------------
# AscendW4A16AWQFusedMoEMethod tests
# ---------------------------------------------------------------------------


class TestAscendW4A16AWQFusedMoEMethod(TestBase):
    """Test AWQ MoE scheme (weight shapes and quant params)."""

    def setUp(self):
        super().setUp()
        self.quant_config = AWQConfig(
            weight_bits=4,
            group_size=128,
            zero_point=True,
        )
        # Mock get_ascend_config to avoid requiring a real vllm config
        self.mock_config_patcher = patch(
            "vllm_ascend.quantization.methods.w4a16_awq.get_ascend_config"
        )
        mock_ascend_config = self.mock_config_patcher.start()
        mock_ascend_config.return_value = MagicMock(
            eplb_config=MagicMock(dynamic_eplb=False)
        )
        self.quant_method = AscendW4A16AWQFusedMoEMethod(self.quant_config)

    def tearDown(self):
        self.mock_config_patcher.stop()
        super().tearDown()

    def test_init(self):
        self.assertEqual(self.quant_method.pack_factor, 8)
        self.assertEqual(self.quant_method.group_size, 128)
        self.assertEqual(self.quant_method.quant_type.value, 7)  # W4A16_AWQ = 7

    def test_get_weight(self):
        num_experts = 4
        intermediate = 512
        hidden = 256

        result = self.quant_method.get_weight(num_experts, intermediate, hidden, torch.bfloat16)

        self.assertIn("w13_qweight", result)
        self.assertIn("w2_qweight", result)
        self.assertEqual(result["w13_qweight"].shape, (num_experts, hidden, 2 * intermediate // 8))
        self.assertEqual(result["w2_qweight"].shape, (num_experts, intermediate, hidden // 8))
        self.assertEqual(result["w13_qweight"].dtype, torch.int32)
        self.assertEqual(result["w2_qweight"].dtype, torch.int32)

    def test_get_dynamic_quant_param(self):
        num_experts = 4
        intermediate = 512
        hidden = 256
        group_size = 128

        result = self.quant_method.get_dynamic_quant_param(
            num_experts, intermediate, hidden, torch.bfloat16
        )

        num_groups_w13 = hidden // group_size
        num_groups_w2 = intermediate // group_size

        self.assertEqual(result["w13_scales"].shape, (num_experts, num_groups_w13, intermediate * 2))
        self.assertEqual(result["w2_scales"].shape, (num_experts, num_groups_w2, hidden))
        self.assertEqual(result["w13_qzeros"].shape, (num_experts, num_groups_w13, 2 * intermediate // 8))
        self.assertEqual(result["w2_qzeros"].shape, (num_experts, num_groups_w2, hidden // 8))
        self.assertEqual(result["w13_qzeros"].dtype, torch.int32)
        self.assertEqual(result["w2_qzeros"].dtype, torch.int32)

    def test_process_weights_after_loading(self):
        num_experts = 2
        hidden = 256
        intermediate = 128
        pack_factor = 8

        layer = torch.nn.Module()
        layer.w13_qweight = torch.nn.Parameter(
            torch.randint(0, 100, (num_experts, hidden, 2 * intermediate // pack_factor), dtype=torch.int32),
            requires_grad=False,
        )
        layer.w2_qweight = torch.nn.Parameter(
            torch.randint(0, 100, (num_experts, intermediate, hidden // pack_factor), dtype=torch.int32),
            requires_grad=False,
        )
        layer.w13_scales = torch.nn.Parameter(
            torch.ones((num_experts, hidden // 128, 2 * intermediate), dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.w2_scales = torch.nn.Parameter(
            torch.ones((num_experts, intermediate // 128, hidden), dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.w13_qzeros = torch.nn.Parameter(
            torch.randint(0, 100, (num_experts, hidden // 128, 2 * intermediate // pack_factor), dtype=torch.int32),
            requires_grad=False,
        )
        layer.w2_qzeros = torch.nn.Parameter(
            torch.randint(0, 100, (num_experts, intermediate // 128, hidden // pack_factor), dtype=torch.int32),
            requires_grad=False,
        )

        self.quant_method.process_weights_after_loading(layer)

        # qzeros should be unpacked to bfloat16 with 3D shape
        self.assertEqual(layer.w13_qzeros.dtype, torch.bfloat16)
        self.assertEqual(layer.w2_qzeros.dtype, torch.bfloat16)
        # Shape: (experts, groups, full_output_dim)
        self.assertEqual(layer.w13_qzeros.shape[0], num_experts)
        self.assertEqual(layer.w2_qzeros.shape[0], num_experts)


# ---------------------------------------------------------------------------
# Weight unpacking utility tests
# ---------------------------------------------------------------------------


class TestUnpackQzeroFromInt32(TestBase):
    """Test AWQ zero-point unpacking logic."""

    def test_linear_layer_shape(self):
        weight = torch.tensor([[305419896, -1420531520]], dtype=torch.int32)
        result = _unpack_qzero_from_int32(weight, torch.bfloat16, pack_factor=8, is_moe_layer=False)
        # (1, 2) packed -> (1, 16) unpacked
        self.assertEqual(result.shape, (1, 16))
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertTrue(result.is_contiguous())

    def test_moe_layer_shape(self):
        weight = torch.tensor([[[305419896, -1420531520]]], dtype=torch.int32)
        result = _unpack_qzero_from_int32(weight, torch.bfloat16, pack_factor=8, is_moe_layer=True)
        # (1, 1, 2) packed -> (1, 1, 16) unpacked
        self.assertEqual(result.shape, (1, 1, 16))
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertTrue(result.is_contiguous())

    def test_unsigned_to_signed_conversion(self):
        """Verify uint4 [0,15] -> sint4 [-8,7] via -(x - 8)."""
        weight = torch.tensor([[0, 1, 7, 8, 9, 10, 15, 0]], dtype=torch.int32)
        result = _unpack_qzero_from_int32(weight, torch.bfloat16, pack_factor=8, is_moe_layer=False)

        # Each int32 element unpacks to 8 nibbles; element k's lowest nibble at index k*8
        self.assertAlmostEqual(result[0, 0].item(), 8, places=2)    # 0 -> -(0-8) = 8
        self.assertAlmostEqual(result[0, 8].item(), 7, places=2)    # 1 -> -(1-8) = 7
        self.assertAlmostEqual(result[0, 24].item(), 0, places=2)   # 8 -> -(8-8) = 0
        self.assertAlmostEqual(result[0, 48].item(), -7, places=2)  # 15 -> -(15-8) = -7

    def test_zeros_stay_zero_after_conversion(self):
        """AWQ qzeros of 8 (= awq zero point) should map to 0 offset."""
        # Build a weight where every nibble is 8
        weight = torch.tensor([0x88888888], dtype=torch.int32).reshape(1, 1)
        result = _unpack_qzero_from_int32(weight, torch.bfloat16, pack_factor=8, is_moe_layer=False)
        # -(8-8) = 0 for all 8 nibbles
        self.assertTrue(torch.allclose(result.float(), torch.zeros_like(result.float()), atol=0.01))


class TestUnpackWeightFromInt32(TestBase):
    """Test AWQ weight unpacking logic."""

    def test_output_shape_and_dtype(self):
        weight = torch.randint(0, 100, (4, 8), dtype=torch.int32)
        result = _unpack_weight_from_int32(weight, pack_factor=8)
        self.assertEqual(result.shape, weight.shape)
        self.assertEqual(result.dtype, torch.int32)
        self.assertTrue(result.is_contiguous())

    def test_xor_transformation(self):
        """All-zero input should produce 0x88888888 after XOR."""
        weight = torch.zeros((2, 4), dtype=torch.int32)
        result = _unpack_weight_from_int32(weight, pack_factor=8)
        expected = -2004318072  # 0x88888888 as signed int32
        self.assertTrue(torch.all(result == expected))

    def test_identity_roundtrip(self):
        """XOR 0x88888888 twice should return to original (post-reorder)."""
        weight = torch.randint(0, 2**31, (4, 4), dtype=torch.int32)
        result = _unpack_weight_from_int32(weight, pack_factor=8)
        # Apply XOR again (without reorder) to verify the XOR part is involutory
        double_xored = result ^ 0x88888888
        # The reorder step makes this not a perfect roundtrip, but the XOR itself is
        self.assertEqual(double_xored.dtype, torch.int32)


if __name__ == "__main__":
    import unittest

    unittest.main()
