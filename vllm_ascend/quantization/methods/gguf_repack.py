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
"""Repack simple GGUF block types to ``npu_weight_quant_batchmatmul`` format.

The high-performance gguf path: instead of dequantizing to dense fp16 (which
forfeits the quantization memory saving), repack the simple symmetric /
asymmetric block types into the per-group (group_size=32) int8/int4 format that
``npu_weight_quant_batchmatmul`` consumes, so weights stay quantized at runtime
(real memory saving + the fused dequant+matmul).

Supported (block_size=32, maps cleanly to the stock op at group_size=32):
- **Q8_0** (symmetric int8): weight int8, scale per-block, offset 0.
- **Q4_0** (symmetric int4): weight (nibble-8) int4-packed, scale per-block, offset 0.
- **Q4_1** (asymmetric int4): weight (nibble-8) int4-packed, scale=d, offset=m/d+8.
- **Q5_0** (symmetric int5→int8): weight (5bit-16) int8, scale per-block, offset 0.
- **Q5_1** (asymmetric int5→int8): weight 5bit int8, scale=d, offset=m/d.

k-quants whose natural group is 32 also map cleanly (super-block sections of 32 are
flattened to per-32-group scale/offset; the 6-bit scales unpack to fp16 at repack time
and the stock op does not care about the original bit-width):
- **Q4_K** (4-bit, 8 sections of 32 per 256-super-block): int4-pack + per-section scale=dall*sc,
  offset=8-dmin*mn/(dall*sc).
- **Q5_K** (5-bit, same section layout): int8 + per-section scale, offset=16-dmin*mn/(dall*sc).

Q5_0/Q5_1/Q5_K are 5-bit and don't fit the int4 path, but map to the int8 path (values in
[-16,15] / [0,31] fit int8). k-quants whose natural group is **16** (Q6_K, Q2_K, Q3_K) and
IQ variants CANNOT map (the stock op rejects group_size<32; see probe log
``logs/bench/2026-07-06_group-size-probe.log``) and keep the dense-dequant fallback.
int4 packing uses ``npu_convert_weight_to_int4pack`` (an NPU op); Q8_0/Q5_0/Q5_1/Q5_K repack
is pure torch.

Note: Q4_K/Q5_K sections with a zero 6-bit scale (sc==0) are unrepresentable by the stock
op's ``(w+offset)*scale`` (scale=0 forces output 0, losing the section's ``-min`` constant).
Empirically sc==0 does not occur in real LLM weights (Q4_K Qwen-FFN: 0/19456 sections) —
the quantizer never produces flat sub-blocks — so this is a theoretical edge case only.
"""

import torch
from gguf import GGML_QUANT_SIZES
from gguf import GGMLQuantizationType as WT

from .gguf_dequant import _bytes_to_le_u32, _unpack_k4_scales

# Block types repacked to the NPU op (high-perf). Others dequant→dense.
GGUF_NPU_REPACK_TYPES = {
    WT.Q8_0,
    WT.Q4_0,
    WT.Q4_1,
    WT.Q5_0,
    WT.Q5_1,
    WT.Q4_K,
    WT.Q5_K,  # k-quants with natural group=32 → map to stock op at gs=32
}


def _repack_q8_0(qweight: torch.Tensor, dtype: torch.dtype):
    """Q8_0 {fp16 d; int8 qs[32]} → int8 [K,N] + scale [G,N] + offset 0, group=32."""
    block_size, type_size = GGML_QUANT_SIZES[WT.Q8_0]  # 32, 34
    n_rows, n_bytes = qweight.shape
    n_blocks = n_bytes // type_size
    b = qweight.reshape(n_rows, n_blocks, type_size)
    d = b[..., :2].contiguous().view(torch.float16).to(dtype).squeeze(-1)  # [N, G]
    qs = b[..., 2:].contiguous().view(torch.int8)  # [N, G, 32]
    qweight_out = qs.reshape(n_rows, n_blocks * block_size).t().contiguous()  # [K, N]
    scale = d.t().contiguous()  # [G, N]
    offset = torch.zeros_like(scale)  # symmetric
    return qweight_out, scale, offset, block_size


def _repack_q4_0(qweight: torch.Tensor, dtype: torch.dtype):
    """Q4_0 {fp16 d; uint8 qs[16]} → int4-pack(nibble-8) [K,N] + scale [G,N] + offset 0."""
    block_size, type_size = GGML_QUANT_SIZES[WT.Q4_0]  # 32, 18
    n_rows, n_bytes = qweight.shape
    n_blocks = n_bytes // type_size
    b = qweight.reshape(n_rows, n_blocks, type_size)
    d = b[..., :2].contiguous().view(torch.float16).to(dtype).squeeze(-1)  # [N, G]
    qs = b[..., 2:].to(torch.int32)  # [N, G, 16]
    low = qs & 0xF
    high = qs >> 4
    nib = torch.cat([low, high], dim=-1).reshape(n_rows, n_blocks * block_size)  # [N, K]
    signed = (nib - 8).to(torch.int8)  # signed int4 values [-8, 7]
    import torch_npu  # lazy: int4pack needs the NPU runtime; keeps the module CPU-importable

    qweight_packed = torch_npu.npu_convert_weight_to_int4pack(signed.t().contiguous().to(torch.int32))
    scale = d.t().contiguous()  # [G, N]
    offset = torch.zeros_like(scale)
    return qweight_packed, scale, offset, block_size


def _repack_q4_1(qweight: torch.Tensor, dtype: torch.dtype):
    """Q4_1 {fp16 d, fp16 m, qs[16]} → int4-pack(nibble-8) + scale=d + offset=m/d+8."""
    block_size, type_size = GGML_QUANT_SIZES[WT.Q4_1]  # 32, 20
    n_rows, n_bytes = qweight.shape
    n_blocks = n_bytes // type_size
    b = qweight.reshape(n_rows, n_blocks, type_size)
    dm = b[..., :4].contiguous().view(torch.float16).to(torch.float32)  # [N, G, 2]
    d = dm[..., 0]  # [N, G]
    m = dm[..., 1]  # [N, G]
    qs = b[..., 4:].to(torch.int32)  # [N, G, 16]
    low = qs & 0xF
    high = qs >> 4
    nib = torch.cat([low, high], dim=-1).reshape(n_rows, n_blocks * block_size)  # [N, K]
    signed = (nib - 8).to(torch.int8)
    import torch_npu  # lazy: int4pack needs the NPU runtime; keeps the module CPU-importable

    qweight_packed = torch_npu.npu_convert_weight_to_int4pack(signed.t().contiguous().to(torch.int32))
    scale = d.t().contiguous()  # [G, N]
    # dequant = (weight + offset)*scale = (nib-8 + m/d+8)*d = nib*d + m
    offset = (m / d.clamp(min=1e-8) + 8.0).to(dtype).t().contiguous()  # [G, N]
    return qweight_packed, scale, offset, block_size


def _repack_q5_0(qweight: torch.Tensor, dtype: torch.dtype):
    """Q5_0 {fp16 d; uint8 qh[4]; uint8 qs[16]} → int8 [K,N] + scale [G,N] + offset 0.

    5-bit value (0..31) → signed (5bit-16) ∈ [-16,15], stored as int8. The NPU int8
    op computes (qw + 0) * scale = (5bit-16) * d = the Q5_0 dequant value. 5-bit
    doesn't fit the int4 path, but fits int8. Bit extraction mirrors gguf_dequant._q5_0.

    Forced to CPU: the 5th-bit extraction uses a tensor-broadcast right-shift
    (``qh.unsqueeze(-1) >> idx``) which the NPU ``aclnnRightShift`` kernel rejects
    (error 161002, broadcast shape quirk). Pure-torch, no NPU op needed; the caller
    moves the result to the NPU device.
    """
    qweight = qweight.cpu()
    block_size, type_size = GGML_QUANT_SIZES[WT.Q5_0]  # 32, 22
    n_rows, n_bytes = qweight.shape
    n_blocks = n_bytes // type_size
    b = qweight.reshape(n_rows, n_blocks, type_size)
    d = b[..., :2].contiguous().view(torch.float16).to(dtype).squeeze(-1)  # [N, G]
    qh = _bytes_to_le_u32(b[..., 2:6])  # [N, G]
    qs = b[..., 6:].to(torch.int32)  # [N, G, 16]
    idx = torch.arange(16, device=qweight.device)
    bit_low = ((qh.unsqueeze(-1) >> idx) & 1) << 4  # 5th bit, low nibbles (vals 0..15)
    bit_high = ((qh.unsqueeze(-1) >> (idx + 16)) & 1) << 4  # 5th bit, high nibbles (16..31)
    low = (qs & 0xF) | bit_low
    high = (qs >> 4) | bit_high
    vals = torch.cat([low, high], dim=-1).to(torch.int32) - 16  # [N, G, 32] in [-16,15]
    qweight_out = vals.reshape(n_rows, n_blocks * block_size).t().contiguous().to(torch.int8)
    scale = d.t().contiguous()  # [G, N]
    offset = torch.zeros_like(scale)  # symmetric
    return qweight_out, scale, offset, block_size


def _repack_q5_1(qweight: torch.Tensor, dtype: torch.dtype):
    """Q5_1 {fp16 d, fp16 m, uint8 qh[4], uint8 qs[16]} → int8 [K,N] + scale=d + offset=m/d.

    NPU int8 op: (qw + m/d) * d = 5bit*d + m = the Q5_1 dequant value. 5-bit values
    (0..31) stored as int8. Bit extraction mirrors gguf_dequant._q5_1.

    Forced to CPU (see ``_repack_q5_0``): the tensor-broadcast right-shift fails on
    NPU. Pure-torch; caller moves the result to the NPU device.
    """
    qweight = qweight.cpu()
    block_size, type_size = GGML_QUANT_SIZES[WT.Q5_1]  # 32, 24
    n_rows, n_bytes = qweight.shape
    n_blocks = n_bytes // type_size
    b = qweight.reshape(n_rows, n_blocks, type_size)
    d = b[..., :2].contiguous().view(torch.float16).to(torch.float32).squeeze(-1)  # [N, G]
    m = b[..., 2:4].contiguous().view(torch.float16).to(torch.float32).squeeze(-1)  # [N, G]
    qh = _bytes_to_le_u32(b[..., 4:8])  # [N, G]
    qs = b[..., 8:].to(torch.int32)  # [N, G, 16]
    idx = torch.arange(16, device=qweight.device)
    bit_low = ((qh.unsqueeze(-1) >> idx) & 1) << 4
    bit_high = ((qh.unsqueeze(-1) >> (idx + 16)) & 1) << 4
    low = (qs & 0xF) | bit_low
    high = (qs >> 4) | bit_high
    vals = torch.cat([low, high], dim=-1).to(torch.int32)  # [N, G, 32] in [0,31]
    qweight_out = vals.reshape(n_rows, n_blocks * block_size).t().contiguous().to(torch.int8)
    scale = d.to(dtype).t().contiguous()  # [G, N]
    # (qw + offset) * scale = (5bit + m/d) * d = 5bit*d + m
    offset = (m / d.clamp(min=1e-8)).to(dtype).t().contiguous()  # [G, N]
    return qweight_out, scale, offset, block_size


def _repack_q4_k(qweight: torch.Tensor, dtype: torch.dtype):
    """Q4_K super-block → int4-pack + per-32-section scale/offset, group_size=32.

    block_q4_K ``{half2 dm (dall,dmin); scales[12]; qs[128]}`` → 256 values in 8 sections
    of 32, each with its own 6-bit scale/min (unpacked via ``_unpack_k4_scales``).
    Per section the stock op must compute ``dall*sc*nib - dmin*mn``; setting
    ``scale = dall*sc`` and ``offset = 8 - dmin*mn/(dall*sc)`` makes
    ``(sint4 + offset)*scale == dall*sc*nib - dmin*mn`` (sint4 = nibble-8). The section
    scale/min is constant over 32 weights, matching the op's group_size=32.
    """
    block_size, type_size = GGML_QUANT_SIZES[WT.Q4_K]  # 256, 144
    n_rows, n_bytes = qweight.shape
    n_blocks = n_bytes // type_size
    b = qweight.reshape(n_rows, n_blocks, type_size)
    dm = b[..., :4].contiguous().view(torch.float16).to(torch.float32)  # [N, nb, 2]
    dall = dm[..., 0:1]
    dmin = dm[..., 1:2]
    sc, mn = _unpack_k4_scales(b[..., 4:16])  # [N, nb, 8] each
    d_sections = (dall * sc.to(torch.float32))[..., None]  # [N, nb, 8, 1]
    m_sections = (dmin * mn.to(torch.float32))[..., None]
    qs = b[..., 16:].to(torch.int32).reshape(n_rows, n_blocks, 4, 32)
    low = qs & 0xF
    high = qs >> 4
    nib = (
        torch.stack([low, high], dim=3).reshape(n_rows, n_blocks, 8, 32).reshape(n_rows, n_blocks * block_size)
    )  # uint4 [N, K]

    scale_NK = d_sections.expand(-1, -1, -1, 32).reshape(n_rows, n_blocks * block_size)
    offset_NK = (
        (8.0 - m_sections / d_sections.clamp(min=1e-8)).expand(-1, -1, -1, 32).reshape(n_rows, n_blocks * block_size)
    )
    offset_NK = offset_NK.clamp(-65504.0, 65504.0)  # fp16-safe; avoids inf→nan if sc==0

    groups = (n_blocks * block_size) // 32
    scale = scale_NK.reshape(n_rows, groups, 32)[..., 0].to(dtype).t().contiguous()  # [G, N]
    offset = offset_NK.reshape(n_rows, groups, 32)[..., 0].to(dtype).t().contiguous()  # [G, N]
    signed = (nib - 8).to(torch.int8)  # sint4 [N, K]
    import torch_npu  # lazy: int4pack needs the NPU runtime; keeps the module CPU-importable

    qweight_packed = torch_npu.npu_convert_weight_to_int4pack(signed.t().contiguous().to(torch.int32))
    return qweight_packed, scale, offset, 32


def _repack_q5_k(qweight: torch.Tensor, dtype: torch.dtype):
    """Q5_K super-block → int8 + per-32-section scale/offset, group_size=32.

    Like Q4_K (8 sections of 32, 6-bit scale/min) but each value is 5-bit = nibble +
    (qh bit << 4). Maps to the int8 path (values 0..31 fit int8) with ``scale = dall*sc``
    and ``offset = 16 - dmin*mn/(dall*sc)`` so ``(signed + offset)*scale == dall*sc*5bit
    - dmin*mn`` (signed = 5bit-16).

    Forced to CPU: the 5th-bit extraction uses a tensor-broadcast right-shift
    (``qh.unsqueeze(-1) >> arange``) which NPU ``aclnnRightShift`` rejects (error 161002).
    Pure-torch; caller moves the result to the NPU device.
    """
    qweight = qweight.cpu()
    block_size, type_size = GGML_QUANT_SIZES[WT.Q5_K]  # 256, 176
    n_rows, n_bytes = qweight.shape
    n_blocks = n_bytes // type_size
    b = qweight.reshape(n_rows, n_blocks, type_size)
    dm = b[..., :4].contiguous().view(torch.float16).to(torch.float32)
    dall = dm[..., 0:1]
    dmin = dm[..., 1:2]
    sc, mn = _unpack_k4_scales(b[..., 4:16])
    d_sections = (dall * sc.to(torch.float32))[..., None]
    m_sections = (dmin * mn.to(torch.float32))[..., None]
    qh = b[..., 16:48].to(torch.int64)  # [N, nb, 32]
    qs = b[..., 48:].to(torch.int32).reshape(n_rows, n_blocks, 4, 32)
    low = qs & 0xF
    high = qs >> 4
    sections = torch.stack([low, high], dim=3).reshape(n_rows, n_blocks, 8, 32)
    bit5 = ((qh.unsqueeze(2) >> torch.arange(8).reshape(1, 1, 8, 1)) & 1) << 4
    nib = (sections.to(torch.int64) + bit5).reshape(n_rows, n_blocks * block_size)  # 5bit [N, K]

    scale_NK = d_sections.expand(-1, -1, -1, 32).reshape(n_rows, n_blocks * block_size)
    offset_NK = (
        (16.0 - m_sections / d_sections.clamp(min=1e-8)).expand(-1, -1, -1, 32).reshape(n_rows, n_blocks * block_size)
    )
    offset_NK = offset_NK.clamp(-65504.0, 65504.0)

    groups = (n_blocks * block_size) // 32
    scale = scale_NK.reshape(n_rows, groups, 32)[..., 0].to(dtype).t().contiguous()
    offset = offset_NK.reshape(n_rows, groups, 32)[..., 0].to(dtype).t().contiguous()
    signed = (nib - 16).to(torch.int8)  # [N, K] int8 [-16,15]
    qweight_out = signed.t().contiguous()  # [K, N] int8
    return qweight_out, scale, offset, 32


_REPACK_KERNELS = {
    WT.Q8_0: _repack_q8_0,
    WT.Q4_0: _repack_q4_0,
    WT.Q4_1: _repack_q4_1,
    WT.Q5_0: _repack_q5_0,
    WT.Q5_1: _repack_q5_1,
    WT.Q4_K: _repack_q4_k,
    WT.Q5_K: _repack_q5_k,
}


def repack_to_npu(qweight: torch.Tensor, qweight_type: int, dtype: torch.dtype):
    """Repack GGUF bytes [N, K_bytes] → (qweight, scale, offset, group_size) for the NPU op.

    Returns None if the type can't be repacked (caller falls back to dense dequant).
    """
    kernel = _REPACK_KERNELS.get(qweight_type)
    if kernel is None:
        return None
    return kernel(qweight, dtype)
