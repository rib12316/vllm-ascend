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
"""Ascend torchao quantization schemes for Linear layers.

torchao is used **only at load time** to produce integer weight tensors; the
forward pass runs ``npu_weight_quant_batchmatmul`` (Pattern A, same operator as
AWQ/GPTQ). torchao is never on the inference hot path.

Supported quant types:

- **int8wo** (``AscendW8A16TorchAOLinearScheme``): uses torchao's
  ``Int8WeightOnlyConfig`` to quantize the dense weight → per-output-channel
  *symmetric* int8 (``get_plain()`` → data ``[N,K]`` int8, scale ``[N]``,
  zero-point all-zero). On the NPU: ``antiquant_offset=0``,
  ``antiquant_group_size=K`` (one group per output channel).

- **int4wo** (``AscendW4A16TorchAOLinearScheme``): *self-implemented* per-group
  symmetric int4 RTN (group_size=128), because torchao 0.17's standard int4 path
  requires ``mslk`` (a CUDA/H100-only kernel) unavailable on NPU. Its numerics
  match torchao int4wo's per-group symmetric formulation and are validated by
  round-trip error vs the dense weight and end-to-end output quality. On the NPU
  it maps to the *same proven configuration* as GPTQ/AWQ W4
  (``antiquant_group_size=128``, ``antiquant_offset=0``).
"""

from typing import TYPE_CHECKING, Any

import torch
import torch_npu

from .base import AscendLinearScheme
from .registry import register_scheme

if TYPE_CHECKING:
    from vllm_ascend.quantization.torchao_config import TorchAOConfig

# int4 symmetric range is [-8, 7]; divisor for the scale.
_INT4_SYMMETRIC_MAX = 7


def _apply_torchao_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    """Shared forward for torchao Linear schemes via ``npu_weight_quant_batchmatmul``.

    Reads ``layer.qweight``, ``layer.scales``, ``layer.torchao_offset`` (zeros
    for symmetric) and ``layer.torchao_output_size`` (saved before int4 repack).
    """
    qweight = layer.qweight
    if bias is not None and bias.dtype == torch.bfloat16:
        bias = bias.float()

    reshaped_x = x.reshape(-1, x.shape[-1])
    out = torch_npu.npu_weight_quant_batchmatmul(
        reshaped_x,
        qweight,
        antiquant_scale=layer.scales,
        antiquant_offset=layer.torchao_offset,
        antiquant_group_size=group_size,
        bias=bias,
    )
    return out.reshape(x.shape[:-1] + (layer.torchao_output_size,))


def _quantize_dense_int8_per_channel(
    weight_nk: torch.Tensor,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a dense ``[N, K]`` weight with torchao ``Int8WeightOnlyConfig``.

    Returns ``(qweight[K, N] int8, scales[1, N])``. torchao int8wo is
    per-output-channel symmetric: ``dequant = data * scale`` (zero-point 0).
    """
    # Imported lazily: torchao is a load-time-only dependency.
    import torch.nn as nn
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    dummy = nn.Sequential(nn.Linear(weight_nk.shape[1], weight_nk.shape[0], bias=False))
    dummy[0].weight = torch.nn.Parameter(weight_nk.detach().clone())
    quantize_(dummy, Int8WeightOnlyConfig())

    data, scale, _zp = dummy[0].weight.tensor_impl.get_plain()
    # data: [N, K] int8 (centered, range -128..127); scale: [N] fp32 (symmetric).
    qweight = data.to(torch.int8).t().contiguous()  # [K, N] for the NPU op
    scales = scale.to(out_dtype).unsqueeze(0).contiguous()  # [1, N]
    return qweight, scales


def _int4_symmetric_quant(
    weight_nk: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group symmetric int4 RTN on a dense ``[N, K]`` weight (math only).

    ``scale = max(|w_group|) / 7``, ``q = round(w / scale)`` clipped to ``[-8, 7]``.

    Returns ``(q_flat[K, N] int8 (values in [-8,7]), scales[G, N] float32)``.
    Split from packing so the quantization math is CPU-testable without an NPU.
    """
    N, K = weight_nk.shape
    if K % group_size != 0:
        raise ValueError(f"int4wo input_size ({K}) must be divisible by group_size ({group_size}).")
    num_groups = K // group_size

    w = weight_nk.detach().to(torch.float32).reshape(N, num_groups, group_size)
    max_abs = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # [N, G, 1]
    scale = max_abs / _INT4_SYMMETRIC_MAX  # [N, G, 1]
    q = torch.round(w / scale).clamp(-8, 7).to(torch.int8)  # [N, G, group] in [-8,7]

    q_flat = q.reshape(N, K).t().contiguous()  # [K, N], values in [-8, 7]
    scales = scale.squeeze(-1).t().contiguous()  # [G, N] fp32
    return q_flat, scales


def _quantize_dense_int4_symmetric(
    weight_nk: torch.Tensor,
    group_size: int,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Self-implemented per-group symmetric int4 RTN, packed for the NPU op.

    Wraps :func:`_int4_symmetric_quant` with ``npu_convert_weight_to_int4pack``
    (the int32→int4pack step, identical to GPTQ W4). Returns
    ``(qweight_packed, scales[G, N], offset[G, N])`` with a zero offset
    (symmetric) — the same NPU configuration as GPTQ W4.
    """
    q_flat, scales = _int4_symmetric_quant(weight_nk, group_size)
    # npu_convert_weight_to_int4pack expects an int32 source (see methods/gptq.py).
    qweight_packed = torch_npu.npu_convert_weight_to_int4pack(q_flat.to(torch.int32))
    scales = scales.to(out_dtype)
    return qweight_packed, scales, torch.zeros_like(scales)


@register_scheme("W8A16_TORCHAO", "linear")
class AscendW8A16TorchAOLinearScheme(AscendLinearScheme):
    """Linear scheme for torchao int8wo (per-channel symmetric int8, Pattern A).

    The dense weight is loaded from the checkpoint (online quantization), then
    quantized with torchao ``Int8WeightOnlyConfig`` at load time. Forward runs
    ``npu_weight_quant_batchmatmul`` with per-channel scale and zero offset.
    """

    def __init__(self, quant_config: "TorchAOConfig"):
        self.group_size = quant_config.group_size
        self.is_checkpoint_torchao_serialized = quant_config.is_checkpoint_torchao_serialized

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        # Online path: the dense weight is loaded from the checkpoint and
        # quantized in process_weights_after_loading.
        return {
            "weight": torch.empty(output_size, input_size, dtype=params_dtype),
            "_param_dims": {"weight": {"input_dim": 1, "output_dim": 0}},
        }

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        out_dtype = layer.weight.dtype
        input_size = layer.weight.shape[1]
        output_size = layer.weight.shape[0]

        qweight, scales = _quantize_dense_int8_per_channel(layer.weight.data, out_dtype)
        layer.qweight = torch.nn.Parameter(qweight, requires_grad=False)
        layer.scales = torch.nn.Parameter(scales, requires_grad=False)
        # Symmetric: dequant = data * scale ⟹ antiquant_offset = 0.
        layer.torchao_offset = torch.nn.Parameter(torch.zeros(1, output_size, dtype=out_dtype), requires_grad=False)
        layer.torchao_output_size = output_size
        layer.torchao_group_size = input_size  # per-channel: one group spans K
        # Free the dense weight (the whole point of quantization).
        layer.weight = torch.nn.Parameter(torch.empty(0, dtype=out_dtype), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        return _apply_torchao_linear(layer, x, bias, layer.torchao_group_size)


@register_scheme("W4A16_TORCHAO", "linear")
class AscendW4A16TorchAOLinearScheme(AscendLinearScheme):
    """Linear scheme for torchao int4wo (self-implemented per-group symmetric int4).

    torchao 0.17's standard int4 path requires ``mslk`` (CUDA/H100-only), so the
    quantization step is reimplemented here as per-group symmetric int4 RTN
    (group_size=128). It maps to the same proven NPU configuration as GPTQ/AWQ
    W4 (antiquant_group_size=128, antiquant_offset=0).
    """

    def __init__(self, quant_config: "TorchAOConfig"):
        self.group_size = quant_config.group_size

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return {
            "weight": torch.empty(output_size, input_size, dtype=params_dtype),
            "_param_dims": {"weight": {"input_dim": 1, "output_dim": 0}},
        }

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        out_dtype = layer.weight.dtype
        output_size = layer.weight.shape[0]

        qweight, scales, offset = _quantize_dense_int4_symmetric(layer.weight.data, self.group_size, out_dtype)
        layer.qweight = torch.nn.Parameter(qweight, requires_grad=False)
        layer.scales = torch.nn.Parameter(scales, requires_grad=False)
        layer.torchao_offset = torch.nn.Parameter(offset, requires_grad=False)
        layer.torchao_output_size = output_size
        layer.torchao_group_size = self.group_size
        layer.weight = torch.nn.Parameter(torch.empty(0, dtype=out_dtype), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        return _apply_torchao_linear(layer, x, bias, layer.torchao_group_size)
