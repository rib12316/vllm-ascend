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

"""NPU operator smoke tests for AWQ/GPTQ weight-quant kernels.

These verify that the two ``torch_npu`` kernels AWQ/GPTQ depend on
(``npu_weight_quant_batchmatmul`` and ``npu_convert_weight_to_int4pack``)
execute on real NPU hardware and return the expected shape. They do NOT
verify numerical correctness against a dense oracle — that is covered by
``test_quant_moe_synthetic.py`` (per-expert dequant vs dense reference) and
by the real-model e2e probes (Paris / MoE ALL PASS).

Requires torch_npu + NPU hardware; skipped otherwise.

Usage:
    pytest tests/e2e/singlecard/test_quant_npu_operator_smoke.py -v
"""

import pytest
import torch


class TestNPUOperatorIntegration:
    """Smoke tests that require torch_npu and NPU hardware.

    Skipped if torch_npu or an NPU device is unavailable.
    """

    @pytest.fixture(autouse=True)
    def skip_without_npu(self):
        pytest.importorskip("torch_npu", reason="torch_npu not available")
        # Probe the actual tensor method: under TORCH_DEVICE_BACKEND_AUTOLOAD=0
        # the torch_npu package imports and torch.npu.is_available() can return
        # True, but the ``.npu()`` tensor method isn't registered — so the test
        # would AttributeError instead of skip. Require both the namespace and
        # the tensor method.
        if not hasattr(torch, "npu") or not hasattr(torch.Tensor, "npu"):
            pytest.skip("No NPU device available")
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
            x,
            qweight,
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
