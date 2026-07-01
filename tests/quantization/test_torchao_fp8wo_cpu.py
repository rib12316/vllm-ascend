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

"""CPU tests for the torchao fp8wo scheme (dense-compute fallback).

fp8wo quantizes the dense weight with torchao ``Float8WeightOnlyConfig`` on CPU
(no ``mslk`` needed, unlike int4) and dequantizes back to dense at load time, so
the whole scheme is CPU-runnable (no NPU). These tests verify the load-time
fp8 round-trip and that ``apply`` matches a dense reference matmul.

Usage:
    pytest tests/quantization/test_torchao_fp8wo_cpu.py -v
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from vllm.config import set_current_vllm_config

from vllm_ascend.quantization.methods.torchao import (
    AscendFP8WTorchAOLinearScheme,
    _dequantize_fp8_weight,
)
from vllm_ascend.quantization.torchao_config import TorchAOConfig


@pytest.fixture(autouse=True)
def default_vllm_config():
    # AscendLinearMethod.__init__ reads the live vLLM config; provide a mock.
    mock_config = MagicMock()
    mock_config.compilation_config.custom_ops = ["all"]
    with set_current_vllm_config(mock_config):
        yield mock_config


def _make_layer(scheme, weight_nk, dtype):
    layer = nn.Module()
    layer.weight = nn.Parameter(weight_nk.to(dtype), requires_grad=False)
    scheme.process_weights_after_loading(layer)
    return layer


def test_fp8_round_trip_bounded_error():
    # fp8 (e4m3) round-trip of a small-magnitude weight has bounded relative error.
    torch.manual_seed(0)
    W = torch.randn(64, 128).float() * 0.1
    deq = _dequantize_fp8_weight(W, torch.float32)
    rel = (deq - W).abs().mean() / W.abs().mean()
    assert rel < 0.05, f"fp8 round-trip relative error too high: {rel}"


def test_fp8wo_apply_matches_dense():
    torch.manual_seed(0)
    N, K = 64, 128
    dtype = torch.float32
    W = torch.randn(N, K, dtype=dtype) * 0.1
    cfg = TorchAOConfig.from_config({"quant_type": {"default": "fp8wo"}})
    scheme = AscendFP8WTorchAOLinearScheme(cfg)
    layer = _make_layer(scheme, W, dtype)

    x = torch.randn(8, K, dtype=dtype) * 0.1
    out = scheme.apply(layer, x)
    # apply is dense F.linear over the fp8-dequantized weight → exact match.
    ref = torch.nn.functional.linear(x, layer.weight)
    torch.testing.assert_close(out, ref)


def test_fp8wo_routing():
    from unittest.mock import MagicMock

    from vllm.model_executor.layers.linear import LinearBase

    from vllm_ascend.quantization.method_adapters import AscendLinearMethod

    cfg = TorchAOConfig.from_config({"quant_type": {"default": "fp8wo"}})
    method = cfg.get_quant_method(MagicMock(spec=LinearBase), prefix="model.layers.0.mlp.gate_proj")
    assert isinstance(method, AscendLinearMethod)
    assert isinstance(method.quant_method, AscendFP8WTorchAOLinearScheme)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
