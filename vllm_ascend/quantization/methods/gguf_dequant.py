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

MVP block types: Q8_0, Q4_0, Q4_1 (standard symmetric / asymmetric quants).
TODO (G-4 expansion): Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, IQ variants.
"""

import torch
from gguf import GGML_QUANT_SIZES
from gguf import GGMLQuantizationType as WT

# Quantization types dequantized to dense by this module (MVP subset).
GGUF_DEQUANT_TYPES = {WT.Q8_0, WT.Q4_0, WT.Q4_1}
# Unquantized GGUF dtypes stored verbatim in the qweight bytes.
GGUF_UNQUANTIZED_TYPES = {WT.F16, WT.BF16, WT.F32}


def _block_layout(qweight_type: int) -> tuple[int, int]:
    """Return ``(block_size, type_size)`` for a GGML quant type."""
    return GGML_QUANT_SIZES[qweight_type]


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


_DEQUANT_KERNELS = {
    WT.Q8_0: _q8_0,
    WT.Q4_0: _q4_0,
    WT.Q4_1: _q4_1,
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
        raise NotImplementedError(
            f"GGUF dequant for {WT(qweight_type).name} is not yet implemented on "
            f"Ascend NPU (MVP supports Q8_0/Q4_0/Q4_1). Please use a model with "
            f"one of the supported block types."
        )
    return kernel(qweight, dtype)
