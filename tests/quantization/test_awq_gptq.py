# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""
AWQ & GPTQ quantization unit tests for Ascend NPU.

These tests validate weight packing/unpacking, zero-point conversion, and
the NPU dequantization formula independently of real model loading.

Usage (on NPU machine):
    pytest tests/quantization/test_awq_gptq.py -v

    # Run only AWQ tests
    pytest tests/quantization/test_awq_gptq.py -v -k awq

    # Run only GPTQ tests
    pytest tests/quantization/test_awq_gptq.py -v -k gptq
"""

import struct

import pytest
import torch


def _to_int32_tensor(val):
    """Convert an unsigned packed int32 value to a torch int32 tensor element.

    Python ints are arbitrary precision, so a packed value like 0x80000000
    exceeds the signed int32 range. We use struct.pack('I', ...) to handle
    the unsigned→signed conversion correctly.
    """
    return torch.frombuffer(
        struct.pack('I', val & 0xFFFFFFFF), dtype=torch.int32
    ).reshape(())

# ---------------------------------------------------------------------------
# AWQ weight unpacking tests
# ---------------------------------------------------------------------------


class TestAWQWeightUnpack:
    """Tests for AWQ weight packing/unpacking logic."""

    def setup_method(self):
        """Import AWQ functions dynamically (avoids torch_npu import on non-NPU)."""
        from vllm_ascend.quantization.methods.w4a16_awq import (
            _unpack_qzero_from_int32,
            _unpack_weight_from_int32,
            REVERSE_AWQ_PACK_ORDER,
        )
        self.unpack_qzero = _unpack_qzero_from_int32
        self.unpack_weight = _unpack_weight_from_int32
        self.pack_order = REVERSE_AWQ_PACK_ORDER

    def test_awq_pack_order_values(self):
        """Verify the AWQ interleaved pack order is correct."""
        assert self.pack_order == [0, 4, 1, 5, 2, 6, 3, 7]

    def _pack_awq_weight(self, uint4_values):
        """Helper: pack 8 uint4 values into a single int32 using AWQ order."""
        # AWQ stores value at position i in bits [pack_order[i]*4, pack_order[i]*4+4]
        result = 0
        for i, val in enumerate(uint4_values):
            result |= (val & 0xF) << (self.pack_order[i] * 4)
        return result

    def test_unpack_weight_standard_values(self):
        """Test unpacking AWQ weights with known values."""
        # Pack values [1, 2, 3, 4, 5, 6, 7, 8] using AWQ interleaved order
        uint4_vals = [1, 2, 3, 4, 5, 6, 7, 8]
        packed_int32 = self._pack_awq_weight(uint4_vals)
        weight = _to_int32_tensor(packed_int32).reshape(1, 1)

        result = self.unpack_weight(weight, pack_factor=8)
        # Result should be a single int32 with values in standard order + XOR
        # Standard order: val0 at bits 0-3, val1 at bits 4-7, etc.
        # Then XOR with 0x88888888 converts uint4 -> sint4

        # Verify the result is an int32 tensor
        assert result.dtype == torch.int32

        # Extract the rearranged values (standard order, after XOR)
        for i in range(8):
            val = (result.item() >> (4 * i)) & 0xF
            # After XOR: uint4 XOR 8 = sint4 raw bits
            expected = uint4_vals[i] ^ 8
            assert val == expected, f"Position {i}: got {val}, expected {expected}"

    def test_unpack_weight_zero_values(self):
        """Test that zeros remain consistent through packing/unpacking."""
        uint4_vals = [0] * 8
        packed_int32 = self._pack_awq_weight(uint4_vals)
        weight = torch.tensor([[packed_int32]], dtype=torch.int32)

        result = self.unpack_weight(weight, pack_factor=8)

        # 0 XOR 8 = 8 for each position
        for i in range(8):
            val = (result.item() >> (4 * i)) & 0xF
            assert val == 8, f"Position {i}: got {val}, expected 8 (0 XOR 8)"

    def test_unpack_qzero_conversion(self):
        """Test AWQ zero-point unpacking and uint4->sint4 conversion.

        Formula: offset = -(uint4 - 8)
        This maps: 0 -> 8, 1 -> 7, ..., 7 -> 1, 8 -> 0, ..., 15 -> -7
        """
        # Create a packed int32 with value 7 in all 8 positions
        uint4_val = 7
        packed = 0
        for i in range(8):
            packed |= (uint4_val & 0xF) << (self.pack_order[i] * 4)

        weight = torch.tensor([[packed]], dtype=torch.int32)
        result = self.unpack_qzero(weight, torch.float32, pack_factor=8)

        # -(7 - 8) = 1.0 for all positions
        assert result.dtype == torch.float32
        expected = torch.tensor([[1.0] * 8], dtype=torch.float32)
        # shape: (1, 8) because input was (1, 1) int32 -> (1, 8) unpacked
        assert result.shape == (1, 8)
        torch.testing.assert_close(result, expected)

    def test_unpack_qzero_boundary_values(self):
        """Test zero-point conversion at boundary values (0 and 15)."""
        # Value 0: -(0 - 8) = 8
        packed_0 = 0  # all zeros
        weight_0 = torch.tensor([[packed_0]], dtype=torch.int32)
        result_0 = self.unpack_qzero(weight_0, torch.float32, pack_factor=8)
        assert torch.all(result_0 == 8.0), f"Expected 8.0 for zero-point 0, got {result_0}"

        # Value 15: -(15 - 8) = -7
        packed_15 = 0
        for i in range(8):
            packed_15 |= (15 & 0xF) << (self.pack_order[i] * 4)
        weight_15 = _to_int32_tensor(packed_15).reshape(1, 1)
        result_15 = self.unpack_qzero(weight_15, torch.float32, pack_factor=8)
        assert torch.all(result_15 == -7.0), f"Expected -7.0 for zero-point 15, got {result_15}"

    def test_unpack_qzero_moe_layer(self):
        """Test MoE layer zero-point unpacking (is_moe_layer=True)."""
        packed = 0
        for i in range(8):
            packed |= (8 & 0xF) << (self.pack_order[i] * 4)

        # MoE shape: (E, G, N_packed) e.g. (2, 3, 1)
        weight = _to_int32_tensor(packed).reshape(1, 1).expand(2, 3, 1).contiguous().clone()
        result = self.unpack_qzero(weight, torch.float32, pack_factor=8, is_moe_layer=True)

        # -(8 - 8) = 0 for all positions
        assert result.shape == (2, 3, 8)
        assert torch.all(result == 0.0)


# ---------------------------------------------------------------------------
# GPTQ weight unpacking tests
# ---------------------------------------------------------------------------


class TestGPTQWeightUnpack:
    """Tests for GPTQ weight packing/unpacking logic."""

    def setup_method(self):
        from vllm_ascend.quantization.methods.gptq import (
            _unpack_qweight_from_int32,
            _unpack_qzeros_from_int32,
        )
        self.unpack_qweight = _unpack_qweight_from_int32
        self.unpack_qzeros = _unpack_qzeros_from_int32

    def _pack_gptq_weight(self, values_4bit, num_bits=4):
        """Helper: pack values using GPTQ standard sequential order.

        GPTQ packs along dim=0: row i of the packed tensor contains values
        at positions [i, i+pack_factor, i+2*pack_factor, ...].
        """
        pack_factor = 32 // num_bits
        result = 0
        for i, val in enumerate(values_4bit):
            result |= (val & ((1 << num_bits) - 1)) << (num_bits * i)
        return result

    def test_unpack_qweight_4bit(self):
        """Test 4-bit GPTQ weight unpacking with known values."""
        # Pack values [0, 1, 2, 3, 4, 5, 6, 7] in standard order
        values = [0, 1, 2, 3, 4, 5, 6, 7]
        packed = self._pack_gptq_weight(values, num_bits=4)

        # Shape: (1, 1) — 1 packed row, 1 output column
        weight = torch.tensor([[packed]], dtype=torch.int32)

        result = self.unpack_qweight(weight, num_bits=4)

        # Result shape: (8, 1) — unpacked to 8 rows
        assert result.shape == (8, 1)
        assert result.dtype == torch.int8

        # Values are centered: val - 8
        expected = torch.tensor(
            [[-8], [-7], [-6], [-5], [-4], [-3], [-2], [-1]],
            dtype=torch.int8,
        )
        torch.testing.assert_close(result, expected)

    def test_unpack_qweight_8bit(self):
        """Test 8-bit GPTQ weight unpacking with known values."""
        values = [100, 200, 50, 155]
        packed = self._pack_gptq_weight(values, num_bits=8)

        weight = _to_int32_tensor(packed).reshape(1, 1)
        result = self.unpack_qweight(weight, num_bits=8)

        # Result shape: (4, 1)
        assert result.shape == (4, 1)
        assert result.dtype == torch.int8

        # Values are centered: val - 128
        expected = torch.tensor(
            [[-28], [72], [-78], [27]],
            dtype=torch.int8,
        )
        torch.testing.assert_close(result, expected)

    def test_unpack_qzeros_v1_format(self):
        """Test GPTQ v1 zero-point unpacking (qzeros += 1)."""
        values = [0, 1, 2, 3, 4, 5, 6, 7]
        packed = self._pack_gptq_weight(values, num_bits=4)

        # Shape: (1, 1) — 1 group, 1 packed output
        weight = torch.tensor([[packed]], dtype=torch.int32)
        result = self.unpack_qzeros(weight, num_bits=4, use_v2_format=False)

        # v1: values + 1 = [1, 2, 3, 4, 5, 6, 7, 8]
        # Shape: (1, 8)
        assert result.shape == (1, 8)
        expected = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int32)
        torch.testing.assert_close(result, expected)

    def test_unpack_qzeros_v2_format(self):
        """Test GPTQ v2 zero-point unpacking (no adjustment)."""
        values = [0, 1, 2, 3, 4, 5, 6, 7]
        packed = self._pack_gptq_weight(values, num_bits=4)

        weight = torch.tensor([[packed]], dtype=torch.int32)
        result = self.unpack_qzeros(weight, num_bits=4, use_v2_format=True)

        # v2: values as-is
        assert result.shape == (1, 8)
        expected = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32)
        torch.testing.assert_close(result, expected)

    def test_unpack_qzeros_8bit_v1_no_overflow(self):
        """Test that 8-bit v1 zeros don't overflow (was a bug).

        8-bit values can be 0-255, after v1 +1 = 1-256.
        Must NOT convert to int8 (max 127) — keep as int32.
        """
        # Value 255: after +1 = 256, must not overflow
        values = [255, 255, 255, 255]
        packed = self._pack_gptq_weight(values, num_bits=8)

        weight = _to_int32_tensor(packed).reshape(1, 1)
        result = self.unpack_qzeros(weight, num_bits=8, use_v2_format=False)

        assert result.dtype == torch.int32, f"Expected int32, got {result.dtype}"
        assert result.shape == (1, 4)
        # 255 + 1 = 256
        assert torch.all(result == 256), f"Expected 256, got {result}"

    def test_unpack_qzeros_8bit_v2(self):
        """Test 8-bit v2 zeros (no adjustment needed)."""
        values = [128, 200, 50, 0]
        packed = self._pack_gptq_weight(values, num_bits=8)

        weight = torch.tensor([[packed]], dtype=torch.int32)
        result = self.unpack_qzeros(weight, num_bits=8, use_v2_format=True)

        assert result.dtype == torch.int32
        expected = torch.tensor([[128, 200, 50, 0]], dtype=torch.int32)
        torch.testing.assert_close(result, expected)

    def test_qweight_large_shape(self):
        """Test unpacking with realistic tensor shapes."""
        K, N = 512, 256
        pack_factor = 8  # 4-bit

        # Create random packed weights
        weight_packed = torch.randint(
            0, 2**31 - 1, (K // pack_factor, N), dtype=torch.int32
        )
        result = self.unpack_qweight(weight_packed, num_bits=4)

        assert result.shape == (K, N), f"Expected ({K}, {N}), got {result.shape}"
        assert result.dtype == torch.int8

        # Verify centering: all values should be in [-8, 7]
        assert result.min() >= -8
        assert result.max() <= 7


# ---------------------------------------------------------------------------
# Config creation tests
# ---------------------------------------------------------------------------


class TestAWQConfig:
    """Tests for AWQConfig creation and validation."""

    def test_basic_creation(self):
        from vllm_ascend.quantization.awq_config import AWQConfig

        config = AWQConfig(
            weight_bits=4,
            group_size=128,
            zero_point=True,
            quant_config={"w_bit": 4, "group_size": 128, "zero_point": True},
        )
        assert config.weight_bits == 4
        assert config.group_size == 128
        assert config.zero_point is True
        assert config.pack_factor == 8
        assert config.get_name() == "awq"

    def test_supported_dtypes(self):
        from vllm_ascend.quantization.awq_config import AWQConfig

        dtypes = AWQConfig.get_supported_act_dtypes()
        assert torch.float16 in dtypes
        assert torch.bfloat16 in dtypes

    def test_invalid_bits(self):
        from vllm_ascend.quantization.awq_config import AWQConfig

        with pytest.raises(ValueError, match="4-bit"):
            AWQConfig(weight_bits=8, group_size=128, zero_point=True)

    def test_config_filenames(self):
        from vllm_ascend.quantization.awq_config import AWQConfig

        filenames = AWQConfig.get_config_filenames()
        assert "quant_config.json" in filenames
        assert "quantize_config.json" in filenames

    def test_from_config(self):
        from vllm_ascend.quantization.awq_config import AWQConfig

        hf_config = {
            "w_bit": 4,
            "q_group_size": 128,
            "zero_point": True,
        }
        config = AWQConfig.from_config(hf_config)
        assert config.weight_bits == 4
        assert config.group_size == 128
        assert config.zero_point is True

    def test_from_config_alternate_keys(self):
        """Test that alternate config keys work (bits vs w_bit)."""
        from vllm_ascend.quantization.awq_config import AWQConfig

        hf_config = {
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
        }
        config = AWQConfig.from_config(hf_config)
        assert config.weight_bits == 4


class TestGPTQConfig:
    """Tests for GPTQConfig creation and validation."""

    def test_basic_creation_4bit(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        config = GPTQConfig(
            weight_bits=4,
            group_size=128,
            desc_act=False,
            checkpoint_format="",
        )
        assert config.weight_bits == 4
        assert config.group_size == 128
        assert config.desc_act is False
        assert config.pack_factor == 8
        assert config.get_name() == "gptq"

    def test_basic_creation_8bit(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        config = GPTQConfig(
            weight_bits=8,
            group_size=128,
            desc_act=False,
        )
        assert config.weight_bits == 8
        assert config.pack_factor == 4

    def test_v2_format_detection(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        config = GPTQConfig(
            weight_bits=4,
            group_size=128,
            desc_act=False,
            checkpoint_format="gptq_v2",
        )
        assert config.use_v2_format is True

    def test_v1_format_default(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        config = GPTQConfig(
            weight_bits=4,
            group_size=128,
            desc_act=False,
            checkpoint_format="",
        )
        assert config.use_v2_format is False

    def test_supported_dtypes(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        dtypes = GPTQConfig.get_supported_act_dtypes()
        assert torch.float16 in dtypes
        assert torch.bfloat16 in dtypes

    def test_invalid_bits(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        with pytest.raises(ValueError, match="2/3/4/8-bit"):
            GPTQConfig(weight_bits=16, group_size=128, desc_act=False)

    def test_config_filenames(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        filenames = GPTQConfig.get_config_filenames()
        assert "quantize_config.json" in filenames

    def test_from_config(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        hf_config = {
            "bits": 4,
            "group_size": 128,
            "desc_act": False,
            "checkpoint_format": "gptq_v2",
        }
        config = GPTQConfig.from_config(hf_config)
        assert config.weight_bits == 4
        assert config.group_size == 128
        assert config.desc_act is False
        assert config.use_v2_format is True

    def test_override_quantization_method_not_defined(self):
        """Verify we do NOT override quantization method (Bug #4 fix).

        We deleted override_quantization_method to avoid vLLM v0.20.2's
        override validation error. The upstream GPTQMarlinConfig already
        handles the case where user_quant == "gptq" by returning None,
        so Ascend-side interception is unnecessary.
        """
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        # The method should not exist on GPTQConfig itself
        assert "override_quantization_method" not in GPTQConfig.__dict__
        # Inherited from parent QuantizationConfig, returns None by default
        result = GPTQConfig.override_quantization_method(
            {"quant_method": "gptq"}, "gptq")
        assert result is None


# ---------------------------------------------------------------------------
# QuantType registry tests
# ---------------------------------------------------------------------------


class TestQuantTypeRegistry:
    """Tests for QuantType enum and scheme registry."""

    def test_quant_type_values(self):
        from vllm_ascend.quantization.quant_type import QuantType

        assert QuantType.W4A16_AWQ.value == 7
        assert QuantType.W4A16_GPTQ.value == 8
        assert QuantType.W8A16_GPTQ.value == 9

    def test_awq_moe_scheme_registered(self):
        from vllm_ascend.quantization.methods.registry import get_scheme_class

        cls = get_scheme_class("W4A16_AWQ", "moe")
        assert cls is not None

    def test_gptq_w4_moe_scheme_registered(self):
        from vllm_ascend.quantization.methods.registry import get_scheme_class

        cls = get_scheme_class("W4A16_GPTQ", "moe")
        assert cls is not None

    def test_gptq_w8_moe_scheme_registered(self):
        from vllm_ascend.quantization.methods.registry import get_scheme_class

        cls = get_scheme_class("W8A16_GPTQ", "moe")
        assert cls is not None

    def test_awq_linear_scheme_registered(self):
        """AWQ Linear scheme is registered via Pattern A."""
        from vllm_ascend.quantization.methods.registry import get_scheme_class

        cls = get_scheme_class("W4A16_AWQ", "linear")
        assert cls is not None

    def test_gptq_w4_linear_scheme_registered(self):
        """GPTQ W4A16 Linear scheme is registered via Pattern A."""
        from vllm_ascend.quantization.methods.registry import get_scheme_class

        cls = get_scheme_class("W4A16_GPTQ", "linear")
        assert cls is not None

    def test_gptq_w8_linear_scheme_registered(self):
        """GPTQ W8A16 Linear scheme is registered via Pattern A."""
        from vllm_ascend.quantization.methods.registry import get_scheme_class

        cls = get_scheme_class("W8A16_GPTQ", "linear")
        assert cls is not None


# ---------------------------------------------------------------------------
# NPU dequantization formula verification
# ---------------------------------------------------------------------------


class TestNPUFormulaEquivalence:
    """Verify the mathematical equivalence of AWQ/GPTQ → NPU formula mapping.

    AWQ formula: output = (uint4_weight - zeros) * scale
    NPU formula: output = (sint4_weight + offset) * scale

    Therefore: sint4_weight = uint4_weight - 8
               offset = -(zeros - 8) = 8 - zeros
    """

    def test_awq_formula_equivalence(self):
        """Verify AWQ → NPU formula produces same results."""
        # Simulate a simple case
        uint4_weights = torch.tensor([0, 4, 8, 12, 15], dtype=torch.float32)
        zeros = torch.tensor([7, 7, 7, 7, 7], dtype=torch.float32)  # typical AWQ zero-point
        scale = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01], dtype=torch.float32)

        # AWQ formula
        awq_output = (uint4_weights - zeros) * scale

        # NPU formula
        sint4_weights = uint4_weights - 8  # XOR conversion
        offset = -(zeros - 8)  # zero-point conversion
        npu_output = (sint4_weights + offset) * scale

        torch.testing.assert_close(awq_output, npu_output)

    def test_gptq_formula_equivalence_4bit(self):
        """Verify GPTQ 4-bit → NPU formula produces same results."""
        uint4_weights = torch.tensor([0, 4, 8, 12, 15], dtype=torch.float32)
        # GPTQ v2 zeros (already correct)
        zeros_v2 = torch.tensor([8, 8, 8, 8, 8], dtype=torch.float32)
        scale = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01], dtype=torch.float32)

        # GPTQ formula: (weight - zeros) * scale
        gptq_output = (uint4_weights - zeros_v2) * scale

        # NPU formula: (sint4 + offset) * scale
        sint4 = uint4_weights - 8
        offset = -(zeros_v2 - 8)
        npu_output = (sint4 + offset) * scale

        torch.testing.assert_close(gptq_output, npu_output)

    def test_gptq_v1_zeropoint_adjustment(self):
        """Verify GPTQ v1 zero-point: stored_value + 1 = actual_zero."""
        stored_zeros = torch.tensor([0, 7, 14], dtype=torch.int32)
        actual_zeros = stored_zeros + 1  # v1 adjustment

        # Expected: [1, 8, 15]
        expected = torch.tensor([1, 8, 15], dtype=torch.int32)
        torch.testing.assert_close(actual_zeros, expected)


# ---------------------------------------------------------------------------
# NPU operator integration tests (require torch_npu)
# ---------------------------------------------------------------------------


class TestNPUOperatorIntegration:
    """Tests that require torch_npu and NPU hardware.

    These tests will be skipped if torch_npu is not available.
    """

    @pytest.fixture(autouse=True)
    def skip_without_npu(self):
        pytest.importorskip("torch_npu", reason="torch_npu not available")
        if not torch.npu.is_available():
            pytest.skip("No NPU device available")

    def test_npu_weight_quant_batchmatmul_runs(self):
        """Test that npu_weight_quant_batchmatmul runs without error."""
        import torch_npu

        # Weight layout for AWQ/GPTQ: (K, N_packed) where N_packed = N / 8.
        # The operator checks x.shape[-1] (K) == weight.shape[0] (K).
        # Constraint: group_size must be a multiple of 32 in [32, K-1].
        M, K, N = 4, 256, 32
        group_size = 64
        N_packed = N // 8  # = 4

        x = torch.randn(M, K, dtype=torch.float16).npu()
        qweight = torch.randint(0, 100, (K, N_packed), dtype=torch.int32).npu()
        scale = torch.randn(K // group_size, N, dtype=torch.float16).npu()
        offset = torch.randn(K // group_size, N, dtype=torch.float16).npu()

        out = torch_npu.npu_weight_quant_batchmatmul(
            x, qweight,
            antiquant_scale=scale,
            antiquant_offset=offset,
            antiquant_group_size=group_size,
        )

        assert out.shape == (M, N), f"Expected ({M}, {N}), got {out.shape}"

    def test_npu_convert_weight_to_int4pack_runs(self):
        """Test that npu_convert_weight_to_int4pack runs without error."""
        import torch_npu

        K, N = 32, 16
        weight = torch.randint(-8, 7, (K, N), dtype=torch.int8)
        weight_int32 = weight.to(torch.int32).npu()

        packed = torch_npu.npu_convert_weight_to_int4pack(weight_int32)
        assert packed is not None
        assert packed.is_npu


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
