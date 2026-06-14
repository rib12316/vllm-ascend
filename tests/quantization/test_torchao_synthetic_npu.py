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

"""Synthetic NPU round-trip tests for torchao Linear schemes (int4wo / int8wo).

Builds a fake layer from the scheme's weight-processing output and compares
``scheme.apply(layer, x)`` (which calls ``npu_weight_quant_batchmatmul``) against
a pure-torch reference reconstructed from the quantized params:

- int8wo: reference = x @ (qweight * scales); the quantization itself is verified
  to *exactly* match torchao ``Int8WeightOnlyConfig`` on CPU (see tests in the
  code-review of methods/torchao.py), so this validates the NPU forward path.
- int4wo: reference = x @ (q_flat * group-expanded scales); validates the
  self-implemented per-group symmetric int4 path end-to-end on the NPU op.

Requires Ascend NPU hardware; skipped otherwise.

Usage:
    pytest tests/quantization/test_torchao_synthetic_npu.py -v
"""

import pytest
import torch


def _npu_available() -> bool:
    try:
        return torch.npu.device_count() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _npu_available(), reason="Ascend NPU not available")


def _make_layer(scheme, weight_nk, dtype):
    """Build a fake layer with a dense ``weight`` and run process_weights."""
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight_nk.to(dtype), requires_grad=False)
    scheme.process_weights_after_loading(layer)
    return layer


def test_int8_synthetic_matches_reference():
    from vllm_ascend.quantization.methods.torchao import (
        AscendW8A16TorchAOLinearScheme,
    )
    from vllm_ascend.quantization.torchao_config import TorchAOConfig

    torch.manual_seed(0)
    N, K = 64, 128
    dtype = torch.float16
    W = torch.randn(N, K, dtype=dtype) * 0.1
    cfg = TorchAOConfig.from_config({"quant_method": "torchao", "quant_type": {"default": "int8wo"}})
    scheme = AscendW8A16TorchAOLinearScheme(cfg)
    layer = _make_layer(scheme, W, dtype)

    x = (torch.randn(8, K, dtype=dtype) * 0.1).npu()
    out = scheme.apply(layer, x)
    # Reference: the NPU op dequants qweight with (offset=0)*scales, then x @ dequant.
    ref = x.float() @ (layer.qweight.float() * layer.scales.float())
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_int4_synthetic_matches_reference():
    from vllm_ascend.quantization.methods.torchao import (
        AscendW4A16TorchAOLinearScheme,
        _int4_symmetric_quant,
    )
    from vllm_ascend.quantization.torchao_config import TorchAOConfig

    torch.manual_seed(0)
    group_size = 128
    N, K = 64, group_size  # K == group_size ⇒ one group per output channel
    dtype = torch.float16
    W = torch.randn(N, K, dtype=dtype) * 0.1

    cfg = TorchAOConfig.from_config({"quant_type": {"default": "int4wo"}})
    scheme = AscendW4A16TorchAOLinearScheme(cfg)
    layer = _make_layer(scheme, W, dtype)

    # Reference reconstructed from the (unpacked) math, not the int4-packed tensor.
    q_flat, scales = _int4_symmetric_quant(W, group_size)  # [K,N], [G,N]
    scales_expanded = scales.float().repeat_interleave(group_size, dim=0)  # [K,N]
    ref_weight = q_flat.float() * scales_expanded

    x = (torch.randn(8, K, dtype=dtype) * 0.1).npu()
    out = scheme.apply(layer, x)
    ref = x.float() @ ref_weight
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
