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
forward pass runs ``npu_weight_quant_batchmatmul`` (Ascend Scheme 框架, same operator as
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

    Returns ``(qweight[K, N] int8, scales[N])``. torchao int8wo is per-output-channel
    symmetric: ``dequant = data * scale`` (zero-point 0). The NPU op uses
    ``antiquant_group_size=0`` for per-channel (group_size==K is unsupported).
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
    scales = scale.to(out_dtype).contiguous()  # [N] (per-output-channel)
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
    """Linear scheme for torchao int8wo (per-channel symmetric int8, Ascend Scheme 框架).

    The dense weight is loaded from the checkpoint (online quantization), then
    quantized with torchao ``Int8WeightOnlyConfig`` at load time. Forward runs
    ``npu_weight_quant_batchmatmul`` with per-channel scale and zero offset.
    """

    def __init__(self, quant_config: "TorchAOConfig"):
        self.group_size = quant_config.group_size
        # Flat-tensor pre-quantized checkpoint (T-10): int8 data + scale loaded
        # directly via the DEFAULT weight loader (NOT torchao's native AQTensor
        # loader — that is triggered by is_checkpoint_torchao_serialized and
        # expects a different on-disk format).
        self.is_prequant_checkpoint = getattr(quant_config, "is_prequant_checkpoint", False)

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        if self.is_prequant_checkpoint:
            # Pre-quantized checkpoint (T-10): the int8 data [K,N] + per-channel
            # scale [N] are stored as flat tensors (AWQ/GPTQ-style) and loaded
            # directly — no online re-quantization. Mirrors the online path's
            # post-quant layout so apply() is identical.
            return {
                "qweight": torch.empty(input_size, output_size, dtype=torch.int8),
                "scales": torch.empty(output_size, dtype=params_dtype),
                "_param_dims": {
                    "qweight": {"input_dim": 0, "output_dim": 1},
                    "scales": {"output_dim": 0},
                },
            }
        # Online path: the dense weight is loaded from the checkpoint and
        # quantized in process_weights_after_loading.
        return {
            "weight": torch.empty(output_size, input_size, dtype=params_dtype),
            "_param_dims": {"weight": {"input_dim": 1, "output_dim": 0}},
        }

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.is_prequant_checkpoint:
            # Pre-quantized: qweight [K,N] int8 + scales [N] already loaded from
            # the checkpoint. Just place on device + set up the (symmetric,
            # per-channel) NPU-op params — no quantization step.
            out_dtype = layer.scales.dtype
            output_size = layer.scales.shape[0]
            device = layer.qweight.device
            layer.qweight = torch.nn.Parameter(layer.qweight.data.to(device), requires_grad=False)
            layer.scales = torch.nn.Parameter(layer.scales.data.to(out_dtype).to(device), requires_grad=False)
            layer.torchao_offset = torch.nn.Parameter(
                torch.zeros(output_size, dtype=out_dtype, device=device), requires_grad=False
            )
            layer.torchao_output_size = output_size
            layer.torchao_group_size = 0  # per-channel (NPU op rejects group_size == K)
            return

        out_dtype = layer.weight.dtype
        output_size = layer.weight.shape[0]
        device = layer.weight.device

        qweight, scales = _quantize_dense_int8_per_channel(layer.weight.data, out_dtype)
        layer.qweight = torch.nn.Parameter(qweight, requires_grad=False)
        layer.scales = torch.nn.Parameter(scales, requires_grad=False)
        # Symmetric: dequant = data * scale ⟹ antiquant_offset = 0 (kept on-device).
        layer.torchao_offset = torch.nn.Parameter(
            torch.zeros(output_size, dtype=out_dtype, device=device), requires_grad=False
        )
        layer.torchao_output_size = output_size
        # per-channel via group_size=0 (the NPU op rejects group_size == K).
        layer.torchao_group_size = 0
        # Free the dense weight (the whole point of quantization).
        layer.weight = torch.nn.Parameter(torch.empty(0, dtype=out_dtype, device=device), requires_grad=False)

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
        # Flat-tensor pre-quantized checkpoint (T-10): int4 values (as int8
        # [-8,7]) + per-group scale loaded directly, packed for the NPU op at
        # load time (no online re-quantization). See AscendW8A16TorchAOLinearScheme.
        self.is_prequant_checkpoint = getattr(quant_config, "is_prequant_checkpoint", False)

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        if self.is_prequant_checkpoint:
            # Pre-quantized checkpoint: int4 values stored as int8 [K,N] (range
            # [-8,7]) + per-group(128) scale [G,N], loaded directly. Packing to
            # the NPU int4 format happens in process_weights_after_loading.
            num_groups = input_size // self.group_size
            return {
                "qweight": torch.empty(input_size, output_size, dtype=torch.int8),
                "scales": torch.empty(num_groups, output_size, dtype=params_dtype),
                "_param_dims": {
                    "qweight": {"input_dim": 0, "output_dim": 1},
                    "scales": {"input_dim": 0, "output_dim": 1},
                },
            }
        return {
            "weight": torch.empty(output_size, input_size, dtype=params_dtype),
            "_param_dims": {"weight": {"input_dim": 1, "output_dim": 0}},
        }

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.is_prequant_checkpoint:
            # qweight [K,N] int8 (int4 values [-8,7]) + scales [G,N] loaded
            # directly. Pack to the NPU int4 format (same as the online path's
            # post-quant step) — no quantization.
            import torch_npu

            out_dtype = layer.scales.dtype
            output_size = layer.scales.shape[1]
            device = layer.qweight.device
            qweight_packed = torch_npu.npu_convert_weight_to_int4pack(layer.qweight.data.to(device).to(torch.int32))
            scales = layer.scales.data.to(out_dtype).to(device)
            layer.qweight = torch.nn.Parameter(qweight_packed, requires_grad=False)
            layer.scales = torch.nn.Parameter(scales, requires_grad=False)
            layer.torchao_offset = torch.nn.Parameter(torch.zeros_like(scales), requires_grad=False)
            layer.torchao_output_size = output_size
            layer.torchao_group_size = self.group_size
            return

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


def _dequantize_fp8_weight(weight_nk: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Quantize a dense ``[N, K]`` weight with torchao ``Float8WeightOnlyConfig``
    and dequantize back to dense at load time.

    The fp8 (e4m3) round-trip yields fp8-quality weights; compute then runs as a
    standard dense linear. This is the CPU-verifiable fallback used until an NPU
    fp8 matmul op is verified and wired (T-S0 spike) — it carries real torchao fp8
    quantization precision while running dense.
    """
    import torch.nn as nn
    from torchao.quantization import Float8WeightOnlyConfig, quantize_

    # fp8 quantize/dequant runs on CPU: on NPU the fp8 path hits CANN error
    # 561103 (missing prebuilt kernel). The fp8 round-trip is device-independent,
    # so doing it on CPU is numerically identical; the caller moves the dense
    # result back to the NPU device.
    w_cpu = weight_nk.detach().cpu()
    dummy = nn.Sequential(nn.Linear(w_cpu.shape[1], w_cpu.shape[0], bias=False))
    dummy[0].weight = torch.nn.Parameter(w_cpu.clone())
    quantize_(dummy, Float8WeightOnlyConfig())
    # Float8Tensor.dequantize() → dense (Float8Tensor has no tensor_impl.get_plain()).
    return dummy[0].weight.dequantize().to(out_dtype)


@register_scheme("FP8W_TORCHAO", "linear")
class AscendFP8WTorchAOLinearScheme(AscendLinearScheme):
    """Linear scheme for torchao fp8wo (Float8 weight-only), dense-compute fallback.

    torchao ``Float8WeightOnlyConfig`` quantizes the dense weight to fp8 (e4m3)
    on CPU at load time (unlike int4, this needs no ``mslk``); the weight is then
    dequantized to dense fp16/bf16 and computed via a standard dense linear. The
    weights thus carry real torchao fp8 quantization precision, while compute
    runs dense until an NPU fp8 matmul op is verified and wired (T-S0 spike).
    """

    def __init__(self, quant_config: "TorchAOConfig"):
        self.is_checkpoint_torchao_serialized = quant_config.is_checkpoint_torchao_serialized

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return {
            "weight": torch.empty(output_size, input_size, dtype=params_dtype),
            "_param_dims": {"weight": {"input_dim": 1, "output_dim": 0}},
        }

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        out_dtype = layer.weight.dtype
        device = layer.weight.device
        deq = _dequantize_fp8_weight(layer.weight.data, out_dtype)
        # fp8 quant/dequant ran on CPU; move the dense result to the NPU device.
        layer.weight = torch.nn.Parameter(deq.to(device).contiguous(), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        # Dense path: layer.weight holds the fp8-dequantized dense weight.
        return torch.nn.functional.linear(x, layer.weight, bias)
