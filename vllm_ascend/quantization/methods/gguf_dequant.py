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

Supported block types: Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K (standard + the most
common k-quant). TODO: Q5_K, Q6_K, IQ variants.
"""

import torch
from gguf import GGML_QUANT_SIZES
from gguf import GGMLQuantizationType as WT

# Quantization types dequantized to dense by this module.
GGUF_DEQUANT_TYPES = {WT.Q8_0, WT.Q4_0, WT.Q4_1, WT.Q5_0, WT.Q5_1, WT.Q4_K}
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


_DEQUANT_KERNELS = {
    WT.Q8_0: _q8_0,
    WT.Q4_0: _q4_0,
    WT.Q4_1: _q4_1,
    WT.Q5_0: _q5_0,
    WT.Q5_1: _q5_1,
    WT.Q4_K: _q4_k,
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
    return kernel(qweight, dtype)
