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
from vllm_ascend.quantization.methods.gguf_repack import repack_to_npu


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

    # IQ2_XXS is not implemented (the k-quant family Q2_K/Q3_K/Q4_K/Q5_K/Q6_K
    # now is; IQ variants remain stretch).
    with pytest.raises(NotImplementedError, match="IQ2_XXS"):
        dequantize(block, WT.IQ2_XXS, torch.float32)


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


def test_q5_0_repack_matches_dequant():
    # High-perf repack (int8 path) must reconstruct the same weights as dequant.
    # Same block as test_q5_0_dequant: d=2, qs low=3 high=5 -> (-26, -22).
    d = torch.tensor([2.0], dtype=torch.float16)
    qs = torch.full((16,), (5 << 4) | 3, dtype=torch.uint8)
    block = torch.cat([d.view(torch.uint8), torch.zeros(4, dtype=torch.uint8), qs]).unsqueeze(0)
    ref = dequantize(block, WT.Q5_0, torch.float32)  # [1, 32]
    qw, scale, offset, _group = repack_to_npu(block, WT.Q5_0, torch.float16)
    # NPU op antiquant per group: (qw + offset) * scale; qw is [K,N]=[32,1].
    recon = (qw.to(torch.float32) + offset.to(torch.float32)) * scale.to(torch.float32)
    torch.testing.assert_close(recon.t().contiguous(), ref, rtol=1e-3, atol=1e-3)


def test_q5_1_repack_matches_dequant():
    # Same block as test_q5_1_dequant: d=1, m=0.5 -> (3.5, 5.5).
    dm = torch.tensor([1.0, 0.5], dtype=torch.float16)
    qs = torch.full((16,), (5 << 4) | 3, dtype=torch.uint8)
    block = torch.cat([dm.view(torch.uint8), torch.zeros(4, dtype=torch.uint8), qs]).unsqueeze(0)
    ref = dequantize(block, WT.Q5_1, torch.float32)
    qw, scale, offset, _group = repack_to_npu(block, WT.Q5_1, torch.float16)
    recon = (qw.to(torch.float32) + offset.to(torch.float32)) * scale.to(torch.float32)
    torch.testing.assert_close(recon.t().contiguous(), ref, rtol=1e-3, atol=1e-3)


def _mock_int4pack_passthrough(monkeypatch):
    """npu_convert_weight_to_int4pack needs NPU hardware; for a CPU bit-exact
    test of the repack MATH (nibble-unpack + scale/offset derivation), pass the
    pre-pack signed int4 values through unchanged. The int4pack packing format
    itself is validated by the Q4_0 real-model e2e ("Paris")."""
    import torch_npu

    monkeypatch.setattr(torch_npu, "npu_convert_weight_to_int4pack", lambda w: w.to(torch.int8))


def test_q4_0_repack_matches_dequant(monkeypatch):
    # Q4_0 high-perf repack (symmetric int4: nibble-8, scale=d, offset=0) must
    # reconstruct dequant. Same block as test_q4_0_dequant_low_high_ordering:
    # d=2, low=3 high=5 -> ((3-8)*2, (5-8)*2) = (-10, -6).
    _mock_int4pack_passthrough(monkeypatch)
    block = _q4_0_block(2.0, low_nib=3, high_nib=5)
    ref = dequantize(block, WT.Q4_0, torch.float32)  # [1, 32]
    qw, scale, offset, _group = repack_to_npu(block, WT.Q4_0, torch.float16)
    recon = (qw.to(torch.float32) + offset.to(torch.float32)) * scale.to(torch.float32)
    torch.testing.assert_close(recon.t().contiguous(), ref, rtol=1e-3, atol=1e-3)


def test_q4_1_repack_matches_dequant(monkeypatch):
    # Q4_1 high-perf repack (asymmetric int4: nibble, scale=d, offset=m/d+8)
    # must reconstruct dequant — closes the Q4_1 gap (no real q4_1.gguf available
    # in any repo, so the high-perf path is validated here at the unit level).
    # Same block as test_q4_1_dequant_asymmetric: d=1, m=0.5 -> (3.5, 5.5).
    _mock_int4pack_passthrough(monkeypatch)
    block = _q4_1_block(1.0, 0.5, low_nib=3, high_nib=5)
    ref = dequantize(block, WT.Q4_1, torch.float32)  # [1, 32]
    qw, scale, offset, _group = repack_to_npu(block, WT.Q4_1, torch.float16)
    recon = (qw.to(torch.float32) + offset.to(torch.float32)) * scale.to(torch.float32)
    torch.testing.assert_close(recon.t().contiguous(), ref, rtol=1e-3, atol=1e-3)


def test_q5_repack_forces_cpu_no_npu_bitop():
    # Regression guard: repack must run on CPU (the tensor-broadcast right-shift
    # can't run on NPU: aclnnRightShift 161002). Confirms CPU tensors out + the
    # Q5_0 symmetric layout (offset=0, int8 qw, group_size=32).
    d = torch.tensor([2.0], dtype=torch.float16)
    qs = torch.full((16,), (5 << 4) | 3, dtype=torch.uint8)
    block = torch.cat([d.view(torch.uint8), torch.zeros(4, dtype=torch.uint8), qs]).unsqueeze(0)
    qw, _scale, offset, group = repack_to_npu(block, WT.Q5_0, torch.float16)
    assert qw.device.type == "cpu"
    assert qw.dtype == torch.int8
    assert group == 32
    assert offset.abs().sum().item() == 0.0  # Q5_0 is symmetric


def test_q8_0_repack_matches_dequant():
    # Q8_0 high-perf repack (int8 path) must reconstruct dequant — closes the
    # Q8_0 gap (previously only its dequant was tested, not the repack).
    block = _q8_0_block(2.0, torch.randint(0, 256, (32,)).to(torch.uint8))
    ref = dequantize(block, WT.Q8_0, torch.float32)  # [1, 32]
    qw, scale, offset, _group = repack_to_npu(block, WT.Q8_0, torch.float16)
    recon = (qw.to(torch.float32) + offset.to(torch.float32)) * scale.to(torch.float32)
    torch.testing.assert_close(recon.t().contiguous(), ref, rtol=1e-3, atol=1e-3)


def test_partial_block_bytes_rejected():
    # H3: bytes/row not divisible by type_size must raise a clear ValueError
    # instead of silently dropping data in the kernel's reshape.
    import pytest

    block = _q8_0_block(2.0, torch.arange(32, dtype=torch.uint8))  # [1, 34]
    truncated = block[:, :-1]  # [1, 33]; 33 % 34 != 0
    with pytest.raises(ValueError, match="divisible by the block type_size"):
        dequantize(truncated, WT.Q8_0, torch.float32)


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


def test_q6_k_dequant():
    # block_q6_K = {ql[128]; qh[64]; int8 scales[16]; fp16 d}. Degenerate:
    # ql=5 (low nibble 5, high 0), qh=0, scales=1, d=1 → sub 0,1 (low nibble)
    # → (5-32)=-27; sub 2,3 (high nibble) → (0-32)=-32.
    ql = torch.full((128,), 5, dtype=torch.uint8)
    qh = torch.zeros(64, dtype=torch.uint8)
    scales = torch.full((16,), 1, dtype=torch.int8).view(torch.uint8)
    d = torch.tensor([1.0], dtype=torch.float16)
    out = dequantize(
        torch.cat([ql, qh, scales, d.view(torch.uint8)]).unsqueeze(0),
        WT.Q6_K,
        torch.float32,
    )
    # sub = (p%128)//32: sub0,1 when p%128<64 → -27; sub2,3 → -32
    expected = torch.tensor([-27.0 if (p % 128) < 64 else -32.0 for p in range(256)])
    torch.testing.assert_close(out[0], expected)


def test_q6_k_scale_per_group():
    # scales[0]=2 → group 0 (output 0..15) = d*scales[0]*(5-32) = 2*-27 = -54
    ql = torch.full((128,), 5, dtype=torch.uint8)
    qh = torch.zeros(64, dtype=torch.uint8)
    scales = torch.full((16,), 1, dtype=torch.int8).view(torch.uint8)
    scales[0] = 2
    d = torch.tensor([1.0], dtype=torch.float16)
    out = dequantize(
        torch.cat([ql, qh, scales, d.view(torch.uint8)]).unsqueeze(0),
        WT.Q6_K,
        torch.float32,
    )
    assert out[0, 0].item() == -54.0  # group 0: scales[0]=2
    assert out[0, 16].item() == -27.0  # group 1: scales[1]=1


def test_q5_k_dequant():
    # block_q5_K = {dm; scales[12]; qh[32]; qs[128]}. Degenerate: scales=1, mn=0,
    # dall=1, dmin=0, qs byte=0x15 (low=5, high=1), qh=0 → even sections(low)=5,
    # odd sections(high)=1.
    dm = torch.tensor([1.0, 0.0], dtype=torch.float16)
    scales = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.uint8)
    qh = torch.zeros(32, dtype=torch.uint8)
    qs = torch.full((128,), (1 << 4) | 5, dtype=torch.uint8)
    out = dequantize(
        torch.cat([dm.view(torch.uint8), scales, qh, qs]).unsqueeze(0),
        WT.Q5_K,
        torch.float32,
    )
    heads = [round(out[0, 32 * s].item()) for s in range(8)]
    assert heads == [5, 1, 5, 1, 5, 1, 5, 1]


def test_q5_k_5th_bit():
    # qh[0]=1 → section 0 reads qh bit 0 → position 0 gets +16 → 5+16=21.
    dm = torch.tensor([1.0, 0.0], dtype=torch.float16)
    scales = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.uint8)
    qh = torch.zeros(32, dtype=torch.uint8)
    qh[0] = 1
    qs = torch.full((128,), (1 << 4) | 5, dtype=torch.uint8)
    out = dequantize(
        torch.cat([dm.view(torch.uint8), scales, qh, qs]).unsqueeze(0),
        WT.Q5_K,
        torch.float32,
    )
    assert out[0, 0].item() == 21.0  # section 0, pos 0: 5 + 16
    assert out[0, 1].item() == 5.0  # pos 1: no 5th bit


def test_q2_k_dequant():
    # Q2_K: scales[16] | qs[64] | dm(d,dmin). Each scale byte = (min4<<4)|sc4.
    # qs byte 0xD8 -> 2-bit values per shift: s0=0, s2=2, s4=1, s6=3.
    # d=2, dmin=1, sc4=3, min4=1 (byte 0x13) -> per sub-block y = 6*q - 1.
    scales = torch.full((16,), 0x13, dtype=torch.uint8)
    qs = torch.full((64,), 0xD8, dtype=torch.uint8)
    dm = torch.tensor([2.0, 1.0], dtype=torch.float16).view(torch.uint8)
    block = torch.cat([scales, qs, dm]).unsqueeze(0)
    out = dequantize(block, WT.Q2_K, torch.float32)
    # Sub-block is (0..15): j=(is%8)//2, q=[0,2,1,3][j], y=6*q-1 -> [-1,11,5,17].
    q_per_j = torch.tensor([0, 2, 1, 3])
    y_per_is = (6 * q_per_j[(torch.arange(16) % 8) // 2] - 1).repeat_interleave(16)
    torch.testing.assert_close(out[0], y_per_is.to(torch.float32))


def _q2_k_scalar(raw):
    """Scalar dequant mirroring llama.cpp dequantize_row_q2_K line-for-line."""
    import struct

    scales, qs = raw[0:16], raw[16:80]
    d, dmin = struct.unpack("<ee", bytes(raw[80:84]))
    y, is_, q = [], 0, 0
    for _ in range(2):
        shift = 0
        for _j in range(4):
            sc = scales[is_]
            is_ += 1
            dl, ml = d * (sc & 0xF), dmin * (sc >> 4)
            y += [dl * ((qs[q + li] >> shift) & 3) - ml for li in range(16)]
            sc = scales[is_]
            is_ += 1
            dl, ml = d * (sc & 0xF), dmin * (sc >> 4)
            y += [dl * ((qs[q + 16 + li] >> shift) & 3) - ml for li in range(16)]
            shift += 2
        q += 32
    return y


def _q3_k_scalar(raw):
    """Scalar dequant mirroring llama.cpp dequantize_row_q3_K line-for-line."""
    import struct

    hm, qs, sb = raw[0:32], raw[32:96], raw[96:108]
    d = struct.unpack("<e", bytes(raw[108:110]))[0]
    a = list(struct.unpack("<3I", bytes(sb))) + [0]
    tmp = a[2]
    k1, k2 = 0x03030303, 0x0F0F0F0F
    na0 = (a[0] & k2) | (((tmp >> 0) & k1) << 4)
    na1 = (a[1] & k2) | (((tmp >> 2) & k1) << 4)
    na2 = ((a[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4)
    na3 = ((a[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4)
    sc16 = [((na >> (8 * bi)) & 0xFF) for na in (na0, na1, na2, na3) for bi in range(4)]
    y, is_, q, m = [], 0, 0, 1
    for _ in range(2):
        shift = 0
        for _j in range(4):
            dl = d * (sc16[is_] - 32)
            is_ += 1
            y += [dl * (((qs[q + li] >> shift) & 3) - (0 if (hm[li] & m) else 4)) for li in range(16)]
            dl = d * (sc16[is_] - 32)
            is_ += 1
            y += [dl * (((qs[q + 16 + li] >> shift) & 3) - (0 if (hm[li + 16] & m) else 4)) for li in range(16)]
            shift += 2
            m <<= 1
        q += 32
    return y


def test_q2_k_matches_scalar_reference():
    # Vectorized _q2_k vs a line-for-line scalar port of the llama.cpp reference,
    # on a random block with finite d/dmin (catches vectorization/indexing bugs).
    import struct

    torch.manual_seed(0)
    raw = torch.randint(0, 256, (84,), dtype=torch.uint8).tolist()
    raw[80:84] = list(struct.pack("<ee", 1.5, 0.25))
    vec = dequantize(torch.tensor(raw, dtype=torch.uint8).unsqueeze(0), WT.Q2_K, torch.float32)[0]
    ref = torch.tensor(_q2_k_scalar(raw), dtype=torch.float32)
    torch.testing.assert_close(vec, ref, rtol=1e-3, atol=1e-3)


def test_q3_k_matches_scalar_reference():
    # Vectorized _q3_k vs scalar reference (authoritative on the 6-bit scale
    # unpack + hmask bit indexing m=n*4+j, which a hand-computed test easily
    # gets wrong — this is how the n*4+j hbit bug was caught).
    import struct

    torch.manual_seed(1)
    raw = torch.randint(0, 256, (110,), dtype=torch.uint8).tolist()
    raw[108:110] = list(struct.pack("<e", 1.0))
    vec = dequantize(torch.tensor(raw, dtype=torch.uint8).unsqueeze(0), WT.Q3_K, torch.float32)[0]
    ref = torch.tensor(_q3_k_scalar(raw), dtype=torch.float32)
    torch.testing.assert_close(vec, ref, rtol=1e-3, atol=1e-3)


def test_q4_k_repack_matches_dequant(monkeypatch):
    # Q4_K high-perf repack (int4-pack, per-32-section scale/offset, group_size=32)
    # must reconstruct dequant. Random super-block, finite d/dmin, sc forced >=1
    # (sc==0 is a theoretical edge case absent in real weights: 0/19456 on Qwen Q4_K,
    # log logs/bench/2026-07-06_q4k-repack-real.log).
    import struct

    _mock_int4pack_passthrough(monkeypatch)
    torch.manual_seed(2)
    raw = torch.randint(0, 256, (144,), dtype=torch.uint8).tolist()
    raw[0:4] = list(struct.pack("<ee", 1.5, 0.25))  # dall=1.5, dmin=0.25
    raw[4:8] = [b | 1 for b in raw[4:8]]  # force sc_lo >= 1
    raw[12:16] = [b | 1 for b in raw[12:16]]  # force sc_hi >= 1
    block = torch.tensor(raw, dtype=torch.uint8).unsqueeze(0)
    ref = dequantize(block, WT.Q4_K, torch.float32)  # [1, 256]
    qw, scale, offset, group = repack_to_npu(block, WT.Q4_K, torch.float16)
    assert group == 32
    # scale/offset are [G, N] (G=K/32); expand to [K, N] for elementwise compare.
    exp_off = offset.repeat_interleave(32, dim=0)
    exp_sc = scale.repeat_interleave(32, dim=0)
    recon = (qw.to(torch.float32) + exp_off.to(torch.float32)) * exp_sc.to(torch.float32)
    # scale/offset are returned as fp16 (what the op consumes); dequant ref is float32,
    # so allow fp16 precision (the algebra is bit-exact in float32: 5.96e-8 on real Q4_K,
    # log logs/bench/2026-07-06_q4k-repack-real.log). A real bug is O(1), caught here.
    torch.testing.assert_close(recon.t().contiguous(), ref, rtol=3e-2, atol=3e-2)


def test_q5_k_repack_matches_dequant():
    # Q5_K high-perf repack (int8 path, 5-bit, per-32-section scale/offset).
    # Exercises the 5th-bit extraction + the offset=16-dmin*mn/(dall*sc) derivation.
    import struct

    torch.manual_seed(3)
    raw = torch.randint(0, 256, (176,), dtype=torch.uint8).tolist()
    raw[0:4] = list(struct.pack("<ee", 1.5, 0.25))  # dall=1.5, dmin=0.25
    raw[4:8] = [b | 1 for b in raw[4:8]]
    raw[12:16] = [b | 1 for b in raw[12:16]]
    block = torch.tensor(raw, dtype=torch.uint8).unsqueeze(0)
    ref = dequantize(block, WT.Q5_K, torch.float32)
    qw, scale, offset, group = repack_to_npu(block, WT.Q5_K, torch.float16)
    assert group == 32
    assert qw.dtype == torch.int8  # 5-bit → int8 path (no int4pack)
    exp_off = offset.repeat_interleave(32, dim=0)
    exp_sc = scale.repeat_interleave(32, dim=0)
    recon = (qw.to(torch.float32) + exp_off.to(torch.float32)) * exp_sc.to(torch.float32)
    torch.testing.assert_close(recon.t().contiguous(), ref, rtol=3e-2, atol=3e-2)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
