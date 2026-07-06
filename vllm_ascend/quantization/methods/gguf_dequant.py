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
"""Pure-torch GGUF (llama.cpp k-quant) dequantization for Ascend NPU.

Replaces vLLM's CUDA ``ops.ggml_dequantize`` so a GGUF model's quantized
weights can be dequantized to dense fp16/bf16 **once at load time** on the NPU
(the universal MVP path for ``--quantization gguf``). Block layouts and the
dequant formulas are ported bit-exactly from
``vllm/csrc/quantization/gguf/ggml-common.h`` and the ggml reference
``dequantize_row_*`` functions.

Supported block types: Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, and the full k-quant
family Q2_K/Q3_K/Q4_K/Q5_K/Q6_K. TODO: IQ variants (stretch).
"""

import torch
from gguf import GGML_QUANT_SIZES
from gguf import GGMLQuantizationType as WT

# Quantization types dequantized to dense by this module.
GGUF_DEQUANT_TYPES = {WT.Q8_0, WT.Q4_0, WT.Q4_1, WT.Q5_0, WT.Q5_1, WT.Q4_K, WT.Q5_K, WT.Q6_K, WT.Q2_K, WT.Q3_K}
# Unquantized GGUF dtypes stored verbatim in the qweight bytes.
GGUF_UNQUANTIZED_TYPES = {WT.F16, WT.BF16, WT.F32}


def _block_layout(qweight_type: int) -> tuple[int, int]:
    """Return ``(block_size, type_size)`` for a GGML quant type."""
    return GGML_QUANT_SIZES[qweight_type]


def _bytes_to_le_u32(b4: torch.Tensor) -> torch.Tensor:
    """Read 4 uint8 bytes as a little-endian uint32 (returned as int64).

    Avoids ``view(int32)`` which needs 4-byte storage alignment (slices like
    ``b[..., 2:6]`` can have a non-divisible offset).
    """
    q = b4.to(torch.int64)
    return q[..., 0] | (q[..., 1] << 8) | (q[..., 2] << 16) | (q[..., 3] << 24)


def _q8_0(qweight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """block_q8_0 = {fp16 d; int8 qs[32]} → y = qs * d."""
    block_size, type_size = _block_layout(WT.Q8_0)  # 32, 34
    n_rows, n_bytes = qweight.shape
    n_blocks = n_bytes // type_size
    b = qweight.reshape(n_rows, n_blocks, type_size)
    d = b[..., :2].contiguous().view(torch.float16).to(dtype)  # [r, nb, 1] (2 bytes → 1 fp16)
    qs = b[..., 2:].contiguous().view(torch.int8).to(torch.float32)  # [r, nb, 32]
    return (qs * d).reshape(n_rows, n_blocks * block_size).to(dtype)


def _q4_0(qweight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """block_q4_0 = {fp16 d; uint8 qs[16]} → y = (nib - 8) * d.

    ggml ordering: the 16 low nibbles fill values 0..15, the 16 high nibbles
    fill values 16..31 (see dequantize_row_q4_0 reference).
    """
    block_size, type_size = _block_layout(WT.Q4_0)  # 32, 18
    n_rows, n_bytes = qweight.shape
    n_blocks = n_bytes // type_size
    b = qweight.reshape(n_rows, n_blocks, type_size)
    d = b[..., :2].contiguous().view(torch.float16).to(dtype)  # [r, nb, 1] (2 bytes → 1 fp16)
    qs = b[..., 2:].to(torch.int32)  # [r, nb, 16] uint8 byte values
    low = qs & 0xF  # values 0..15
    high = qs >> 4  # values 0..15
    vals = torch.cat([low, high], dim=-1).to(torch.float32) - 8.0  # [r, nb, 32]
    return (vals * d).reshape(n_rows, n_blocks * block_size).to(dtype)


def _q4_1(qweight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """block_q4_1 = {fp16 d, fp16 m, uint8 qs[16]} → y = nib * d + m."""
    block_size, type_size = _block_layout(WT.Q4_1)  # 32, 20
    n_rows, n_bytes = qweight.shape
    n_blocks = n_bytes // type_size
    b = qweight.reshape(n_rows, n_blocks, type_size)
    dm = b[..., :4].contiguous().view(torch.float16).to(torch.float32)  # [r, nb, 2]
    d = dm[..., 0:1]  # [r, nb, 1]
    m = dm[..., 1:2]  # [r, nb, 1]
    qs = b[..., 4:].to(torch.int32)  # [r, nb, 16]
    low = qs & 0xF
    high = qs >> 4
    vals = torch.cat([low, high], dim=-1).to(torch.float32)  # [r, nb, 32]
    return (vals * d + m).reshape(n_rows, n_blocks * block_size).to(dtype)


def _q5_0(qweight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """block_q5_0 = {fp16 d; uint8 qh[4]; uint8 qs[16]} → y = (5bit - 16) * d.

    5-bit value = 4-bit nibble | (qh bit << 4). ggml low/high ordering:
    low nibbles (+ qh bits 0..15) → 0..15, high nibbles (+ qh bits 16..31) → 16..31.
    """
    block_size, type_size = _block_layout(WT.Q5_0)  # 32, 22
    n_rows, n_bytes = qweight.shape
    n_blocks = n_bytes // type_size
    b = qweight.reshape(n_rows, n_blocks, type_size)
    d = b[..., :2].contiguous().view(torch.float16).to(dtype)  # [r, nb, 1]
    # qh[4] read as a little-endian uint32 (avoids int32-view alignment issues).
    qh = _bytes_to_le_u32(b[..., 2:6])  # [r, nb]
    qs = b[..., 6:].to(torch.int32)  # [r, nb, 16]
    idx = torch.arange(16, device=qweight.device)
    bit_low = ((qh.unsqueeze(-1) >> idx) & 1) << 4  # 5th bit for low nibbles
    bit_high = ((qh.unsqueeze(-1) >> (idx + 16)) & 1) << 4  # 5th bit for high nibbles
    low = (qs & 0xF) | bit_low
    high = (qs >> 4) | bit_high
    vals = torch.cat([low, high], dim=-1).to(torch.float32) - 16.0  # [r, nb, 32]
    return (vals * d).reshape(n_rows, n_blocks * block_size).to(dtype)


def _q5_1(qweight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """block_q5_1 = {fp16 d, fp16 m, uint8 qh[4], uint8 qs[16]} → y = 5bit * d + m."""
    block_size, type_size = _block_layout(WT.Q5_1)  # 32, 24
    n_rows, n_bytes = qweight.shape
    n_blocks = n_bytes // type_size
    b = qweight.reshape(n_rows, n_blocks, type_size)
    dm = b[..., :4].contiguous().view(torch.float16).to(torch.float32)  # [r, nb, 2]
    d = dm[..., 0:1]  # [r, nb, 1]
    m = dm[..., 1:2]  # [r, nb, 1]
    qh = _bytes_to_le_u32(b[..., 4:8])  # [r, nb]
    qs = b[..., 8:].to(torch.int32)  # [r, nb, 16]
    idx = torch.arange(16, device=qweight.device)
    bit_low = ((qh.unsqueeze(-1) >> idx) & 1) << 4
    bit_high = ((qh.unsqueeze(-1) >> (idx + 16)) & 1) << 4
    low = (qs & 0xF) | bit_low
    high = (qs >> 4) | bit_high
    vals = torch.cat([low, high], dim=-1).to(torch.float32)  # [r, nb, 32]
    return (vals * d + m).reshape(n_rows, n_blocks * block_size).to(dtype)


def _unpack_k4_scales(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack 8 (scale, min) 6-bit pairs from 12 scale bytes (get_scale_min_k4).

    Mirrors ``dequantize.cuh::get_scale_min_k4``: returns ``(sc, mn)`` each of
    shape ``[..., 8]`` with values in 0..63.
    """
    q = scales.to(torch.int32)
    sc_lo = q[..., 0:4] & 63
    mn_lo = q[..., 4:8] & 63
    sc_hi = (q[..., 8:12] & 0xF) | ((q[..., 0:4] >> 6) << 4)
    mn_hi = (q[..., 8:12] >> 4) | ((q[..., 4:8] >> 6) << 4)
    return torch.cat([sc_lo, sc_hi], dim=-1), torch.cat([mn_lo, mn_hi], dim=-1)


def _q4_k(qweight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    r"""block_q4_K super-block {half2 dm (dall,dmin); scales[12]; qs[128]} → 256 values.

    The super-block has 8 sections of 32 values, each with its own 6-bit scale
    and min (unpacked from ``scales[12]``). Section 2*il holds the low nibbles
    of qs bytes [32*il : 32*il+32], section 2*il+1 the high nibbles. Per section:
    ``y = dall * sc * nibble - dmin * mn``.
    """
    block_size, type_size = _block_layout(WT.Q4_K)  # 256, 144
    n_rows, n_bytes = qweight.shape
    n_blocks = n_bytes // type_size
    b = qweight.reshape(n_rows, n_blocks, type_size)
    dm = b[..., :4].contiguous().view(torch.float16).to(torch.float32)  # [r, nb, 2]
    dall = dm[..., 0:1]  # [r, nb, 1]
    dmin = dm[..., 1:2]
    scales = b[..., 4:16]  # [r, nb, 12]
    qs = b[..., 16:].to(torch.int32)  # [r, nb, 128]
    sc, mn = _unpack_k4_scales(scales)  # [r, nb, 8] each
    d_sections = (dall * sc.to(torch.float32))[..., None]  # [r, nb, 8, 1]
    m_sections = (dmin * mn.to(torch.float32))[..., None]  # [r, nb, 8, 1]
    qs_groups = qs.reshape(n_rows, n_blocks, 4, 32)  # 4 il-groups of 32 bytes
    low = qs_groups & 0xF  # [r, nb, 4, 32]
    high = qs_groups >> 4  # [r, nb, 4, 32]
    # Interleave low/high per il-group → 8 sections of 32 (s=2*il low, 2*il+1 high).
    sections = torch.stack([low, high], dim=3).reshape(n_rows, n_blocks, 8, 32)
    y = d_sections * sections.to(torch.float32) - m_sections  # [r, nb, 8, 32]
    return y.reshape(n_rows, n_blocks * block_size).to(dtype)


def _q6_k(qweight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    r"""block_q6_K super-block {uint8 ql[128]; uint8 qh[64]; int8 scales[16]; half d} → 256 values.

    6-bit value = (ql nibble) | (qh 2-bit << 4). Ported from
    ``dequantize.cuh::dequantize_block_q6_K`` via per-output-position index
    patterns: output p → ip=p//128, sub=(p%128)//32, il=p%32, with
    ``y = d * scales[8*ip + 2*sub + il//16] * (quant - 32)``.
    """
    block_size, type_size = _block_layout(WT.Q6_K)  # 256, 210
    n_rows, n_bytes = qweight.shape
    n_blocks = n_bytes // type_size
    b = qweight.reshape(n_rows, n_blocks, type_size)
    ql = b[..., 0:128].to(torch.int64)  # [r, nb, 128]
    qh = b[..., 128:192].to(torch.int64)  # [r, nb, 64]
    sc = b[..., 192:208].contiguous().view(torch.int8).to(torch.float32)  # [r, nb, 16]
    d = b[..., 208:210].contiguous().view(torch.float16).to(torch.float32)  # [r, nb, 1]

    # Per-output-position index patterns (block_size=256), computed once.
    p = torch.arange(block_size)
    ip = p // 128
    local = p % 128
    sub = local // 32  # 0..3 (offset 0,32,64,96)
    il = local % 32
    ql_idx = 64 * ip + il + 32 * (sub % 2)  # sub 0,2 → +0; sub 1,3 → +32
    qh_idx = 32 * ip + il
    scale_idx = 8 * ip + 2 * sub + (il // 16)
    qh_shift = 2 * sub
    low = sub < 2  # sub 0,1 → low nibble; sub 2,3 → high nibble

    ql_g = ql[..., ql_idx]  # [r, nb, 256]
    ql_nib = torch.where(low, ql_g & 0xF, ql_g >> 4)
    qh_g = qh[..., qh_idx]
    quant = ql_nib | (((qh_g >> qh_shift) & 3) << 4)  # 6-bit [r, nb, 256]
    scale = sc[..., scale_idx]  # [r, nb, 256]
    y = d * scale * (quant.to(torch.float32) - 32.0)
    return y.reshape(n_rows, n_blocks * block_size).to(dtype)


def _q5_k(qweight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    r"""block_q5_K super-block {half2 dm (dall,dmin); scales[12]; qh[32]; qs[128]} → 256 values.

    Like Q4_K (8 sections of 32, low/high nibbles, per-section 6-bit scale/min via
    ``get_scale_min_k4``) but each value is 5-bit = nibble + (qh bit << 4). Section
    ``s`` reads qh bit ``s``. ``y = dall*sc*(nib + bit5*16) - dmin*mn``.
    Ported from ``dequantize.cuh::dequantize_block_q5_K``.
    """
    block_size, type_size = _block_layout(WT.Q5_K)  # 256, 176
    n_rows, n_bytes = qweight.shape
    n_blocks = n_bytes // type_size
    b = qweight.reshape(n_rows, n_blocks, type_size)
    dm = b[..., :4].contiguous().view(torch.float16).to(torch.float32)  # [r, nb, 2]
    dall = dm[..., 0:1]  # [r, nb, 1]
    dmin = dm[..., 1:2]  # [r, nb, 1]
    scales = b[..., 4:16]  # [r, nb, 12]
    qh = b[..., 16:48].to(torch.int64)  # [r, nb, 32]
    qs = b[..., 48:].to(torch.int32)  # [r, nb, 128]
    sc, mn = _unpack_k4_scales(scales)  # [r, nb, 8] each
    d_sections = (dall * sc.to(torch.float32))[..., None]  # [r, nb, 8, 1]
    m_sections = (dmin * mn.to(torch.float32))[..., None]  # [r, nb, 8, 1]
    qs_groups = qs.reshape(n_rows, n_blocks, 4, 32)  # 4 il-groups of 32 bytes
    low = qs_groups & 0xF  # [r, nb, 4, 32]
    high = qs_groups >> 4  # [r, nb, 4, 32]
    sections = torch.stack([low, high], dim=3).reshape(n_rows, n_blocks, 8, 32)
    # 5th bit: section s reads qh bit s, at byte position j → [r, nb, 8, 32].
    bit5 = ((qh.unsqueeze(2) >> torch.arange(8, device=qweight.device).reshape(1, 1, 8, 1)) & 1) << 4
    sections_5bit = sections.to(torch.int64) + bit5  # 5-bit values 0..31
    y = d_sections * sections_5bit.to(torch.float32) - m_sections  # [r, nb, 8, 32]
    return y.reshape(n_rows, n_blocks * block_size).to(dtype)


def _q2_k(qweight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    r"""block_q2_K {scales[16]; qs[64]; half2 dm (d, dmin)} → 256 values.

    16 sub-blocks of 16 weights. Each sub-block's scale byte holds a 4-bit
    scale (low nibble) and a 4-bit min-scale (high nibble). Weights are 2-bit
    (0..3) packed into ``qs`` with a per-j bit ``shift`` (0, 2, 4, 6). Ported
    from llama.cpp ``dequantize_row_q2_K``. Per sub-block:
    ``y = d*(sc&0xF)*q2 - dmin*(sc>>4)``.
    """
    block_size, type_size = _block_layout(WT.Q2_K)  # 256, 84
    n_rows, n_bytes = qweight.shape
    n_blocks = n_bytes // type_size
    b = qweight.reshape(n_rows, n_blocks, type_size)
    scales = b[..., 0:16].to(torch.int32)  # [r, nb, 16]
    qs = b[..., 16:80].to(torch.int32)  # [r, nb, 64]
    dm = b[..., 80:84].contiguous().view(torch.float16).to(torch.float32)  # [r, nb, 2]
    dall = dm[..., 0:1]  # [r, nb, 1]
    dmin = dm[..., 1:2]  # [r, nb, 1]
    dl = (dall * (scales & 0xF).to(torch.float32))[..., None]  # [r, nb, 16, 1]
    ml = (dmin * (scales >> 4).to(torch.float32))[..., None]  # [r, nb, 16, 1]
    # Sub-block is (0..15) -> q_base=(is//8)*32+(is%2)*16, shift=((is%8)//2)*2.
    is_idx = torch.arange(16, device=qweight.device)
    q_base = (is_idx // 8) * 32 + (is_idx % 2) * 16
    shift = ((is_idx % 8) // 2) * 2
    cols = q_base[:, None] + torch.arange(16, device=qweight.device)[None, :]  # [16, 16]
    qsub = qs[..., cols]  # [r, nb, 16, 16]
    w = ((qsub >> shift[:, None]) & 3).to(torch.float32)  # [r, nb, 16, 16]
    y = dl * w - ml  # [r, nb, 16, 16]
    return y.reshape(n_rows, n_blocks * block_size).to(dtype)


def _unpack_q3k_scales(scales: torch.Tensor) -> torch.Tensor:
    """Unpack 16 six-bit scales (0..63) from Q3_K's 12 scale bytes.

    Ported from the ``aux`` / ``kmask`` rearrangement in llama.cpp
    ``dequantize_row_q3_K`` (the 12 raw bytes hold the 16 scales' low nibbles;
    the high 2 bits are borrowed from ``aux[2]``'s bit-pairs).
    """
    s = scales.to(torch.int32)  # [..., 12]
    a0 = s[..., 0] | (s[..., 1] << 8) | (s[..., 2] << 16) | (s[..., 3] << 24)
    a1 = s[..., 4] | (s[..., 5] << 8) | (s[..., 6] << 16) | (s[..., 7] << 24)
    a2 = s[..., 8] | (s[..., 9] << 8) | (s[..., 10] << 16) | (s[..., 11] << 24)
    tmp = a2
    kmask1, kmask2 = 0x03030303, 0x0F0F0F0F
    na0 = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4)
    na1 = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4)
    na2 = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
    na3 = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
    na = torch.stack([na0, na1, na2, na3], dim=-1)  # [..., 4] uint32
    bytes16 = [((na[..., i] >> (8 * byte)) & 0xFF) for i in range(4) for byte in range(4)]
    return torch.stack(bytes16, dim=-1)  # [..., 16]


def _q3_k(qweight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    r"""block_q3_K {hmask[32]; qs[64]; scales[12]; half d} → 256 values.

    16 sub-blocks of 16 weights. Each weight is 3-bit: the low 2 bits come from
    ``qs`` (per-j ``shift`` 0/2/4/6), the sign bit from ``hmask`` (bit ``j``):
    value = ``qbits`` if the hmask bit is set, else ``qbits - 4``. Per sub-block
    scale is 6-bit (unpacked from ``scales[12]``): ``y = d*(sc-32)*value``.
    Ported from llama.cpp ``dequantize_row_q3_K``.
    """
    block_size, type_size = _block_layout(WT.Q3_K)  # 256, 110
    n_rows, n_bytes = qweight.shape
    n_blocks = n_bytes // type_size
    b = qweight.reshape(n_rows, n_blocks, type_size)
    hmask = b[..., 0:32].to(torch.int32)  # [r, nb, 32]
    qs = b[..., 32:96].to(torch.int32)  # [r, nb, 64]
    d_all = b[..., 108:110].contiguous().view(torch.float16).to(torch.float32)  # [r, nb, 1]
    scale6 = _unpack_q3k_scales(b[..., 96:108]).to(torch.float32)  # [r, nb, 16]
    dl = (d_all * (scale6 - 32))[..., None]  # [r, nb, 16, 1]
    is_idx = torch.arange(16, device=qweight.device)
    q_base = (is_idx // 8) * 32 + (is_idx % 2) * 16
    j = (is_idx % 8) // 2
    shift = j * 2
    cols = q_base[:, None] + torch.arange(16, device=qweight.device)[None, :]  # [16, 16]
    # hmask does NOT advance with n (only q does in the C reference): its byte
    # index depends on sub only (A→hm[0:16], B→hm[16:32]), reusing hm[0:32] both
    # n-halves. The bit WITHIN the byte is m, which the C reference left-shifts
    # once per j across BOTH n-halves (8 shifts total) -> bit = n*4 + j (0..7).
    hm_cols = (is_idx % 2)[:, None] * 16 + torch.arange(16, device=qweight.device)[None, :]
    bit_idx = ((is_idx // 8) * 4 + j)[:, None]  # [16, 1]
    qsub = qs[..., cols]  # [r, nb, 16, 16]
    hsub = hmask[..., hm_cols]  # [r, nb, 16, 16]
    qbits = (qsub >> shift[:, None]) & 3  # [r, nb, 16, 16]
    hbit = (hsub >> bit_idx) & 1  # [r, nb, 16, 16]
    value = (qbits - 4 * (1 - hbit)).to(torch.float32)  # hbit=1→qbits; hbit=0→qbits-4
    y = dl * value  # [r, nb, 16, 16]
    return y.reshape(n_rows, n_blocks * block_size).to(dtype)


_DEQUANT_KERNELS = {
    WT.Q8_0: _q8_0,
    WT.Q4_0: _q4_0,
    WT.Q4_1: _q4_1,
    WT.Q5_0: _q5_0,
    WT.Q5_1: _q5_1,
    WT.Q4_K: _q4_k,
    WT.Q5_K: _q5_k,
    WT.Q6_K: _q6_k,
    WT.Q2_K: _q2_k,
    WT.Q3_K: _q3_k,
}


def dequantize(
    qweight: torch.Tensor,
    qweight_type: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize a GGUF qweight (uint8 bytes [N, K_bytes]) to dense ``[N, K]``.

    Args:
        qweight: Raw quantized weight bytes, shape ``[output, K // block_size *
            type_size]``.
        qweight_type: ``GGMLQuantizationType`` int (e.g. Q8_0=8).
        dtype: Output dtype (fp16 / bf16).

    Returns:
        Dense weight tensor ``[output, K]`` in ``dtype``.
    """
    if qweight_type in GGUF_UNQUANTIZED_TYPES:
        # Stored verbatim; reinterpret bytes as the matching dtype.
        torch_dtype = {
            WT.F16: torch.float16,
            WT.BF16: torch.bfloat16,
            WT.F32: torch.float32,
        }[qweight_type]
        return qweight.view(torch_dtype).to(dtype)

    kernel = _DEQUANT_KERNELS.get(qweight_type)
    if kernel is None:
        supported = ", ".join(sorted(k.name for k in _DEQUANT_KERNELS))
        raise NotImplementedError(
            f"GGUF dequant for {WT(qweight_type).name} is not yet implemented on Ascend NPU (supported: {supported})."
        )
    # Validate byte alignment: each row must hold a whole number of blocks,
    # else the reshape inside the kernel would silently drop data. Raises a
    # clear error on truncated/corrupted checkpoints instead.
    _, type_size = _block_layout(qweight_type)
    n_bytes = qweight.shape[-1]
    if n_bytes % type_size != 0:
        raise ValueError(
            f"GGUF {WT(qweight_type).name} weight has {n_bytes} bytes/row, not "
            f"divisible by the block type_size {type_size}; the checkpoint may "
            f"be truncated or corrupted."
        )
    return kernel(qweight, dtype)
