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

Architecture A′ (hybrid — settled at row 5a after reading the real Ascend MoE
forward path):
- **create_weights**: inherited verbatim from upstream ``GGUFMoEMethod`` so the
  ``is_gguf_weight`` / ``is_gguf_weight_type`` sideload contract stays intact
  (``GGUFModelLoader`` fills ``w13_qweight`` / ``w2_qweight`` + their types).
  Self-building via ``AscendFusedMoEMethod`` would re-wrap params with
  ``torch.nn.Parameter`` and break the lazy ``GGUFUninitializedParameter``.
- **apply**: must match the ASCEND signature — ``AscendFusedMoE.forward_impl``
  calls ``quant_method.apply(layer, x, router_logits, top_k, renormalize, ...)``
  and expects a ``FusedExpertsResult`` (NOT the upstream ``(x, topk_weights,
  topk_ids) -> Tensor`` signature). So apply is overridden with the Ascend
  signature (R8).

Two runtime paths, chosen per gguf block type at load:
- **Dense fallback** (K-quants Q6_K/Q2_K/Q3_K at group=16, IQ types, and the
  current default): dequantize every expert to dense fp16 at load, then delegate
  the whole forward to ``AscendUnquantizedFusedMoEMethod`` — which already
  integrates correctly with the Ascend dispatch / ``moe_comm_method`` / finalize
  pipeline. (A hand-rolled per-token loop is numerically correct — see
  :func:`dense_moe_reference` / the row-4 tests — but does NOT fit that pipeline,
  which is why we delegate.) Correctness milestone + safety net; NOT memory-saving.
- **Optimized** for the repackable per-group(32) types (Q4_K/Q5_K/Q4_0/Q4_1/
  Q5_0/Q5_1/Q8_0): repack each expert to packed int4/int8 + per-group scale
  ``[E, G, N]`` (row 3, verified vs 2D repack), then run the quantized
  ``moe_comm_method.fused_experts`` (mirrors AWQ/GPTQ MoE apply). Experts stay
  quantized — memory saving. NOTE: end-to-end numeric validation of this path is
  pending a real Q4_K/Q5_K MoE model (the Q3_K test model exercises dense only).

Weight orientation (gguf storage, R7 — SETTLED at real e2e vs
SzymonOzog/test-gguf-moe-sample): BOTH ``w13=[E,2*inter,H]`` and ``w2=[E,H,inter]``
are ``[out, in]``.
"""

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.quantization.gguf import GGUFMoEMethod

from .gguf_dequant import dequantize
from .gguf_repack import GGUF_NPU_REPACK_TYPES, repack_to_npu


def repack_moe_to_npu(qweight: torch.Tensor, qweight_type: int, dtype: torch.dtype):
    """Repack a 3D per-expert gguf weight ``[E, N, K_bytes]`` for the NPU op.

    Extends the (verified, Linear) 2D :func:`gguf_repack.repack_to_npu` to MoE by
    looping over experts and stacking. Returns ``(qweight_packed, scale, offset,
    group_size)`` (each stacked along a new expert dim 0), or ``None`` if the
    block type is not repackable (caller uses dense fallback).
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


def dense_moe_reference(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Pure-torch swiglu MoE reference (both weights ``[out, in]`` → F.linear).

    Numerically-verified dense compute (row-4 tests + real-bytes e2e). Kept as a
    standalone helper (the production dense path delegates to the unquantized MoE
    method for correct pipeline integration; this stays as the correctness oracle).
    """
    inter = w13.shape[1] // 2
    x = x.to(w13.dtype)
    out = torch.empty_like(x)
    for tok, (w_row, idx_row) in enumerate(zip(topk_weights, topk_ids)):
        cur = None
        for ww, ii in zip(w_row, idx_row):
            gate_up = F.linear(x[tok], w13[int(ii)])
            gate, up = gate_up[: inter], gate_up[inter:]
            down = F.linear(F.silu(gate) * up, w2[int(ii)]) * float(ww)
            cur = down if cur is None else cur + down
        out[tok] = cur if cur is not None else torch.zeros_like(x[tok])
    return out


class AscendGGUFMoEMethod(GGUFMoEMethod):
    """Ascend GGUF MoE: inherit upstream sideload create_weights; Ascend apply.

    Dense fallback delegates to ``AscendUnquantizedFusedMoEMethod`` for correct
    Ascend MoE pipeline integration.
    """

    def __init__(self, quant_config, moe: FusedMoEConfig) -> None:
        super().__init__(quant_config, moe)
        self._moe_config = moe
        self._dense_impl_cached = None

    @property
    def _dense_impl(self):
        """Lazily build the unquantized MoE delegate (needs Ascend runtime config).

        Deferred so constructing this method (e.g. in routing tests / on CPU)
        doesn't require ``init_ascend_config``; only the real forward path builds it.
        """
        if self._dense_impl_cached is None:
            # Lazy import: fused_moe pulls torch_npu / runtime.
            from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod

            self._dense_impl_cached = AscendUnquantizedFusedMoEMethod(self._moe_config)
        return self._dense_impl_cached

    def process_weights_after_loading(self, layer: FusedMoE) -> None:
        """Repack (optimized) or dequantize-to-dense (fallback) each expert at load."""
        dtype = torch.float16
        w13_q = layer.w13_qweight  # [E, 2*inter, H_bytes]
        w2_q = layer.w2_qweight  # [E, H, inter_bytes]
        qt13 = int(layer.w13_qweight_type.weight_type)
        qt2 = int(layer.w2_qweight_type.weight_type)

        # Optimized path: both projections repackable to per-group(32) NPU format.
        r13 = repack_moe_to_npu(w13_q, qt13, dtype)
        r2 = repack_moe_to_npu(w2_q, qt2, dtype)
        if r13 is not None and r2 is not None:
            layer.w13_weight_packed, layer.w13_scale, layer.w13_offset, layer.group_size = r13
            layer.w2_weight_packed, layer.w2_scale, layer.w2_offset, _ = r2
            layer.gguf_moe_optimized = True
            return

        # Dense fallback: dequantize every expert to dense fp16, then let the
        # unquantized MoE method finish weight prep (transpose + NZ cast).
        num_experts = w13_q.shape[0]
        layer.w13_weight = torch.nn.Parameter(
            torch.stack([dequantize(w13_q[e], qt13, dtype) for e in range(num_experts)]),
            requires_grad=False,
        )  # [E, 2*inter, H] ([out, in])
        layer.w2_weight = torch.nn.Parameter(
            torch.stack([dequantize(w2_q[e], qt2, dtype) for e in range(num_experts)]),
            requires_grad=False,
        )  # [E, H, inter] ([out, in])
        layer.gguf_moe_optimized = False
        self._dense_impl.process_weights_after_loading(layer)

    def apply(self, layer: FusedMoE, x: torch.Tensor, **kwargs):
        """Ascend-signature apply (R8). Delegates dense path to the unquantized MoE.

        ``AscendFusedMoE.forward_impl`` calls this with the Ascend kwargs
        (``router_logits``, ``top_k``, ``renormalize``, ...) and expects a
        ``FusedExpertsResult``. For the dense fallback we delegate to the
        unquantized MoE method, which produces exactly that.
        """
        if not getattr(layer, "gguf_moe_optimized", False):
            return self._dense_impl.apply(layer, x, **kwargs)
        # Optimized path (Q4_K/Q5_K/...): pending validation on a real Q4_K MoE
        # model. Route the packed weights + per-group scales through the quantized
        # fused_experts pipeline (mirrors AWQ/GPTQ MoE apply).
        raise NotImplementedError(
            "AscendGGUFMoEMethod optimized apply: pending real Q4_K/Q5_K MoE model "
            "for end-to-end validation (row 5b)."
        )
