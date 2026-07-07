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
"""Ascend GGUF MoE method (torchao-gguf-moe project).

Architecture (A): inherit upstream ``GGUFMoEMethod`` and override only
``apply`` (+ add ``process_weights_after_loading``). Upstream ``create_weights``
is reused verbatim so the ``is_gguf_weight`` sideload contract stays intact.

Two runtime paths, chosen per gguf block type at load:
- **Optimized (row 3/5)** for the repackable per-group(32) types (Q4_K/Q5_K/
  Q4_0/Q4_1/Q5_0/Q5_1/Q8_0): repack each expert to the NPU per-group antiquant
  format (packed int4/int8 weight + scale ``[G, N]`` per expert), so experts
  stay quantized at runtime — memory saving. Repack reuses the verified Linear
  ``gguf_repack.repack_to_npu`` kernels, extended to 3D by looping over experts.
- **Dense fallback (row 4)** for everything else (K-quants Q6_K/Q2_K/Q3_K at
  group=16, IQ types): dequantize each expert to dense fp16 at load, run a plain
  per-token / per-expert MoE loop. Correctness milestone + permanent safety net
  (Fallback-first); NOT memory-saving.

Weight orientation (gguf storage, R7 — SETTLED at real e2e 2026-07-07 against
SzymonOzog/test-gguf-moe-sample): BOTH are ``[out, in]``, so both use F.linear.
- ``w13 = [E, 2*inter, H]`` → gate_up = ``F.linear(x, w13[e])`` (H → 2*inter).
- ``w2  = [E, H, inter]``   → down    = ``F.linear(act, w2[e])`` (inter → H).
"""

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.quantization.gguf import GGUFMoEMethod

from .gguf_dequant import dequantize
from .gguf_repack import GGUF_NPU_REPACK_TYPES, repack_to_npu


def repack_moe_to_npu(qweight: torch.Tensor, qweight_type: int, dtype: torch.dtype):
    """Repack a 3D per-expert gguf weight ``[E, N, K_bytes]`` for the NPU op.

    Extends the (verified, Linear) 2D :func:`gguf_repack.repack_to_npu` to MoE by
    looping over experts and stacking. Returns ``(qweight_packed, scale, offset,
    group_size)`` where the per-expert packed weight / ``[G, N]`` scale / ``[G, N]``
    offset are stacked along a new expert dim 0, or ``None`` if the block type is
    not repackable (caller uses dense fallback).
    """
    if qweight_type not in GGUF_NPU_REPACK_TYPES:
        return None
    num_experts = qweight.shape[0]
    packed, scales, offsets, gs = [], [], [], None
    for e in range(num_experts):
        res = repack_to_npu(qweight[e], qweight_type, dtype)
        if res is None:
            return None
        qw_e, scale_e, offset_e, gs = res
        packed.append(qw_e)
        scales.append(scale_e)
        offsets.append(offset_e)
    return (
        torch.stack(packed),  # [E, N, K//pack]
        torch.stack(scales),  # [E, G, N]
        torch.stack(offsets),  # [E, G, N]
        gs,
    )


class AscendGGUFMoEMethod(GGUFMoEMethod):
    """Ascend GGUF MoE: reuse upstream sideload create_weights; replace apply."""

    def process_weights_after_loading(self, layer: FusedMoE) -> None:
        """Repack (optimized) or dequantize-to-dense (fallback) each expert at load."""
        dtype = torch.float16
        w13_q = layer.w13_qweight  # [E, 2*inter, H_bytes]
        w2_q = layer.w2_qweight  # [E, H, inter_bytes]
        qt13 = int(layer.w13_qweight_type.weight_type)
        qt2 = int(layer.w2_qweight_type.weight_type)

        # Optimized path: both projections repackable to the per-group(32) NPU
        # antiquant format → experts stay quantized (memory saving).
        r13 = repack_moe_to_npu(w13_q, qt13, dtype)
        r2 = repack_moe_to_npu(w2_q, qt2, dtype)
        if r13 is not None and r2 is not None:
            layer.w13_weight_packed, layer.w13_scale, layer.w13_offset, layer.group_size = r13
            layer.w2_weight_packed, layer.w2_scale, layer.w2_offset, _ = r2
            layer.gguf_moe_optimized = True
            return

        # Dense fallback: dequantize every expert to dense fp16.
        num_experts = w13_q.shape[0]
        layer.w13_weight = torch.stack(
            [dequantize(w13_q[e], qt13, dtype) for e in range(num_experts)]
        )  # [E, 2*inter, H]  ([out, in])
        layer.w2_weight = torch.stack(
            [dequantize(w2_q[e], qt2, dtype) for e in range(num_experts)]
        )  # [E, H, inter]  ([out, in])
        layer.gguf_moe_optimized = False

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dense fallback MoE forward (per-token, per-selected-expert).

        Activation is silu/swiglu (common case; TODO generalize via
        ``layer.activation``). ``shared_experts_input`` ignored (milestone).
        """
        w13 = layer.w13_weight  # [E, 2*inter, H] ([out, in])
        w2 = layer.w2_weight  # [E, H, inter] ([out, in])
        x = x.to(w13.dtype)
        inter = w13.shape[1] // 2
        out = torch.empty_like(x)
        for tok, (w_row, idx_row) in enumerate(zip(topk_weights, topk_ids)):
            inp = x[tok]  # [H]
            cur = None
            for ww, ii in zip(w_row, idx_row):
                gate_up = F.linear(inp, w13[int(ii)])  # [2*inter]
                gate, up = gate_up[:inter], gate_up[inter:]
                act_out = F.silu(gate) * up  # swiglu → [inter]
                down = F.linear(act_out, w2[int(ii)]) * float(ww)  # [H]
                cur = down if cur is None else cur + down
            if cur is None:
                cur = torch.zeros_like(x[tok])
            out[tok] = cur
        return out
