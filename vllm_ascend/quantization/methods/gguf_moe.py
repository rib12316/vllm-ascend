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

Row 4 (this file): **dense fallback** — dequantize each expert's gguf-packed
weights to dense fp16 at load (:func:`gguf_dequant.dequantize`, per expert per
``qweight_type``); run a plain per-token / per-expert MoE loop at runtime
(mirrors upstream ``_fused_moe_gguf``'s slow path, but dense ``F.linear`` / ``@``).
Correctness milestone + permanent safety net (Fallback-first); NOT memory-saving.

Weight orientation (gguf storage, R7 — verify at real e2e):
- ``w13 = [E, 2*inter, K]`` is ``[out, in]`` → gate_up = ``F.linear(x, w13[e])``.
- ``w2  = [E, inter, K]``   is ``[in, out]`` → down    = ``act @ w2[e]``.
"""

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.quantization.gguf import GGUFMoEMethod

from .gguf_dequant import dequantize


class AscendGGUFMoEMethod(GGUFMoEMethod):
    """Ascend GGUF MoE: reuse upstream sideload create_weights; replace apply."""

    def process_weights_after_loading(self, layer: FusedMoE) -> None:
        """Dequantize each expert's gguf-packed weights to dense fp16 at load."""
        dtype = torch.float16
        w13_q = layer.w13_qweight  # [E, 2*inter, K_bytes]
        w2_q = layer.w2_qweight  # [E, inter, K_bytes]
        qt13 = int(layer.w13_qweight_type.weight_type)
        qt2 = int(layer.w2_qweight_type.weight_type)
        num_experts = w13_q.shape[0]
        layer.w13_weight = torch.stack(
            [dequantize(w13_q[e], qt13, dtype) for e in range(num_experts)]
        )  # [E, 2*inter, K]  ([out, in])
        layer.w2_weight = torch.stack(
            [dequantize(w2_q[e], qt2, dtype) for e in range(num_experts)]
        )  # [E, inter, K]  ([in, out])

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
        w13 = layer.w13_weight  # [E, 2*inter, K] ([out, in])
        w2 = layer.w2_weight  # [E, inter, K] ([in, out])
        x = x.to(w13.dtype)
        inter = w13.shape[1] // 2
        out = torch.empty_like(x)
        for tok, (w_row, idx_row) in enumerate(zip(topk_weights, topk_ids)):
            inp = x[tok]  # [K]
            cur = None
            for ww, ii in zip(w_row, idx_row):
                gate_up = F.linear(inp, w13[int(ii)])  # [2*inter] (x @ w13[e].T)
                gate, up = gate_up[:inter], gate_up[inter:]
                act_out = F.silu(gate) * up  # swiglu → [inter]
                down = (act_out @ w2[int(ii)]) * float(ww)  # [K] (act @ w2[e])
                cur = down if cur is None else cur + down
            if cur is None:
                cur = torch.zeros_like(x[tok])
            out[tok] = cur
        return out
