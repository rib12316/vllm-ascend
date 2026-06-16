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

Supported (block_size=32, maps cleanly):
- **Q8_0** (symmetric int8): weight int8, scale per-block, offset 0.
- **Q4_0** (symmetric int4): weight (nibble-8) int4-packed, scale per-block, offset 0.
- **Q4_1** (asymmetric int4): weight (nibble-8) int4-packed, scale=d, offset=m/d+8.

k-quants (Q4_K/Q5_K/Q6_K) and others CANNOT map (6-bit super-block scales) and
keep the dense-dequant fallback. int4 packing uses ``npu_convert_weight_to_int4pack``
(an NPU op); Q8_0 repack is pure torch (CPU-verifiable).
"""

import torch
import torch_npu
from gguf import GGML_QUANT_SIZES
from gguf import GGMLQuantizationType as WT

# Block types repacked to the NPU op (high-perf). Others dequant→dense.
GGUF_NPU_REPACK_TYPES = {WT.Q8_0, WT.Q4_0, WT.Q4_1}


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
    qweight_packed = torch_npu.npu_convert_weight_to_int4pack(signed.t().contiguous().to(torch.int32))
    scale = d.t().contiguous()  # [G, N]
    # dequant = (weight + offset)*scale = (nib-8 + m/d+8)*d = nib*d + m
    offset = (m / d.clamp(min=1e-8) + 8.0).to(dtype).t().contiguous()  # [G, N]
    return qweight_packed, scale, offset, block_size


_REPACK_KERNELS = {
    WT.Q8_0: _repack_q8_0,
    WT.Q4_0: _repack_q4_0,
    WT.Q4_1: _repack_q4_1,
}


def repack_to_npu(qweight: torch.Tensor, qweight_type: int, dtype: torch.dtype):
    """Repack GGUF bytes [N, K_bytes] → (qweight, scale, offset, group_size) for the NPU op.

    Returns None if the type can't be repacked (caller falls back to dense dequant).
    """
    kernel = _REPACK_KERNELS.get(qweight_type)
    if kernel is None:
        return None
    return kernel(qweight, dtype)
