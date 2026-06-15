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

"""Bit-exact unit tests for the pure-torch GGUF dequant kernels.

Each test constructs a GGUF block from known quantized values (so the expected
dequant output is known exactly), runs ``dequantize``, and asserts equality.
Pure torch — runs on CPU without NPU hardware, so it gates CI.

Block layouts from vllm/csrc/quantization/gguf/ggml-common.h:
- Q8_0: {fp16 d; int8 qs[32]}              → y = qs * d
- Q4_0: {fp16 d; uint8 qs[16]}             → y = (nib - 8) * d  (low nibs 0..15, high 16..31)
- Q4_1: {fp16 d, fp16 m, uint8 qs[16]}     → y = nib * d + m

Usage:
    pytest tests/ut/quantization/methods/test_gguf_dequant.py -v
"""

import torch
from gguf import GGMLQuantizationType as WT

from vllm_ascend.quantization.methods.gguf_dequant import dequantize


def _q8_0_block(d_val: float, qs_vals: torch.Tensor) -> torch.Tensor:
    d = torch.tensor([d_val], dtype=torch.float16)
    return torch.cat([d.view(torch.uint8), qs_vals.view(torch.uint8)]).unsqueeze(0)


def _q4_0_block(d_val: float, low_nib: int, high_nib: int) -> torch.Tensor:
    d = torch.tensor([d_val], dtype=torch.float16)
    qs = torch.full((16,), (high_nib << 4) | low_nib, dtype=torch.uint8)
    return torch.cat([d.view(torch.uint8), qs]).unsqueeze(0)


def _q4_1_block(d_val: float, m_val: float, low_nib: int, high_nib: int) -> torch.Tensor:
    dm = torch.tensor([d_val, m_val], dtype=torch.float16)
    qs = torch.full((16,), (high_nib << 4) | low_nib, dtype=torch.uint8)
    return torch.cat([dm.view(torch.uint8), qs]).unsqueeze(0)


def test_q8_0_dequant():
    qs = torch.arange(1, 33, dtype=torch.int8)
    out = dequantize(_q8_0_block(2.5, qs), WT.Q8_0, torch.float32)
    torch.testing.assert_close(out[0], qs.float() * 2.5)


def test_q4_0_dequant_low_high_ordering():
    # low nibble 9 -> (9-8)*2 = 2.0 ; high nibble 0 -> (0-8)*2 = -16.0
    out = dequantize(_q4_0_block(2.0, low_nib=9, high_nib=0), WT.Q4_0, torch.float32)
    expected = torch.cat([torch.full((16,), 2.0), torch.full((16,), -16.0)])
    torch.testing.assert_close(out[0], expected)


def test_q4_1_dequant_asymmetric():
    # nib*1.0 + 0.5 ; low=3 -> 3.5, high=5 -> 5.5
    out = dequantize(_q4_1_block(1.0, 0.5, low_nib=3, high_nib=5), WT.Q4_1, torch.float32)
    expected = torch.cat([torch.full((16,), 3.5), torch.full((16,), 5.5)])
    torch.testing.assert_close(out[0], expected)


def test_unquantized_f16_passthrough():
    w = torch.randn(4, 8, dtype=torch.float16)
    out = dequantize(w.view(torch.uint8), WT.F16, torch.float32)
    torch.testing.assert_close(out, w.float())


def test_unsupported_block_type_raises():
    block = _q8_0_block(1.0, torch.zeros(32, dtype=torch.int8))
    import pytest

    # Q5_K is not yet implemented (only Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/Q4_K are).
    with pytest.raises(NotImplementedError, match="Q5_K"):
        dequantize(block, WT.Q5_K, torch.float32)


def test_multi_block_row():
    # Two Q8_0 blocks in one row: [d0, qs0(32), d1, qs1(32)] -> 68 bytes.
    d0 = torch.tensor([1.0], dtype=torch.float16)
    d1 = torch.tensor([3.0], dtype=torch.float16)
    qs0 = torch.full((32,), 2, dtype=torch.int8)
    qs1 = torch.full((32,), 1, dtype=torch.int8)
    row = torch.cat([d0.view(torch.uint8), qs0.view(torch.uint8), d1.view(torch.uint8), qs1.view(torch.uint8)])
    out = dequantize(row.unsqueeze(0), WT.Q8_0, torch.float32)
    expected = torch.cat([torch.full((32,), 2.0), torch.full((32,), 3.0)])
    torch.testing.assert_close(out[0], expected)


def test_q5_0_dequant():
    # d=2, qh=0, qs low=3 high=5 -> (3-16)*2=-26 ; (5-16)*2=-22
    d = torch.tensor([2.0], dtype=torch.float16)
    qs = torch.full((16,), (5 << 4) | 3, dtype=torch.uint8)
    out = dequantize(
        torch.cat([d.view(torch.uint8), torch.zeros(4, dtype=torch.uint8), qs]).unsqueeze(0),
        WT.Q5_0,
        torch.float32,
    )
    expected = torch.cat([torch.full((16,), -26.0), torch.full((16,), -22.0)])
    torch.testing.assert_close(out[0], expected)


def test_q5_0_5th_bit():
    # qh bit0=1 -> low[0] 5-bit value = 3|16 = 19 -> (19-16)*2 = 6
    d = torch.tensor([2.0], dtype=torch.float16)
    qs = torch.full((16,), (5 << 4) | 3, dtype=torch.uint8)
    qw = torch.cat([d.view(torch.uint8), torch.tensor([1, 0, 0, 0], dtype=torch.uint8), qs]).unsqueeze(0)
    out = dequantize(qw, WT.Q5_0, torch.float32)
    assert out[0, 0].item() == 6.0
    assert out[0, 1].item() == -26.0


def test_q5_1_dequant():
    # d=1, m=0.5, qs low=3 high=5 -> 3.5 ; 5.5
    dm = torch.tensor([1.0, 0.5], dtype=torch.float16)
    qs = torch.full((16,), (5 << 4) | 3, dtype=torch.uint8)
    out = dequantize(
        torch.cat([dm.view(torch.uint8), torch.zeros(4, dtype=torch.uint8), qs]).unsqueeze(0),
        WT.Q5_1,
        torch.float32,
    )
    expected = torch.cat([torch.full((16,), 3.5), torch.full((16,), 5.5)])
    torch.testing.assert_close(out[0], expected)


def test_q4_k_dequant_sections():
    # Degenerate: all 6-bit scales=1, mins=0; dall=2, dmin=0; qs byte=0x13 (low=3, high=1).
    # Each low section = dall*1*3 = 6, each high section = dall*1*1 = 2.
    dm = torch.tensor([2.0, 0.0], dtype=torch.float16)
    scales = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.uint8)
    qs = torch.full((128,), (1 << 4) | 3, dtype=torch.uint8)
    out = dequantize(
        torch.cat([dm.view(torch.uint8), scales, qs]).unsqueeze(0),
        WT.Q4_K,
        torch.float32,
    )
    section_heads = [round(out[0, 32 * s].item()) for s in range(8)]
    assert section_heads == [6, 2, 6, 2, 6, 2, 6, 2]


def test_q4_k_scale_unpacking():
    # sc[0]=2 (q[0]=2) -> section 0 (low, il0) = dall*sc0*nib = 2*2*3 = 12.
    dm = torch.tensor([2.0, 0.0], dtype=torch.float16)
    scales = torch.tensor([2, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.uint8)
    qs = torch.full((128,), (1 << 4) | 3, dtype=torch.uint8)
    out = dequantize(
        torch.cat([dm.view(torch.uint8), scales, qs]).unsqueeze(0),
        WT.Q4_K,
        torch.float32,
    )
    assert out[0, 0].item() == 12.0  # section 0: dall * sc0 * nib = 2 * 2 * 3
    assert out[0, 32].item() == 2.0  # section 1 (high, il0): sc1=1 -> 2 * 1 * 1


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
