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
"""Ascend GPTQ quantization scheme for Linear and MoE layers.

This module provides GPTQ (Generalized Post-Training Quantization) support on
Ascend NPU, using ``npu_weight_quant_batchmatmul`` for linear layers and
``npu_grouped_matmul`` (via fused_experts) for MoE layers.

Key differences from AWQ:
- GPTQ packs weights along the **input dimension** (dim=0), not output dim
- GPTQ uses **standard sequential bit order** (not AWQ's interleaved order)
- GPTQ has ``desc_act`` (g_idx) for activation ordering
- GPTQ has v1/v2 checkpoint format (affects zero-point handling)
- GPTQ supports both 4-bit and 8-bit weights

Weight processing pipeline:
  4-bit: unpack (standard order) → subtract 8 → npu_convert_weight_to_int4pack
  8-bit: unpack (standard order) → subtract 128 → int8 direct use

Architecture (Ascend Scheme 框架):
  Linear schemes use autonomous weight registration via ``get_weight()`` /
  ``get_pergroup_param()``. GPTQ's qweight is packed along dim=0 while qzeros
  is packed along dim=1, so they must go in different ``get_*()`` methods.

  MoE schemes share a single base class (``_AscendGPTQFusedMoEMethodBase``):
  the 4-bit and 8-bit variants differ ONLY in the weight repack step
  (``npu_convert_weight_to_int4pack`` vs ``int32`` storage view).
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch_npu

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input

from .base import AscendLinearScheme, AscendMoEScheme, QuantType
from .registry import register_scheme

if TYPE_CHECKING:
    from vllm_ascend.quantization.gptq_config import GPTQConfig


def _unpack_qweight_from_int32(
    weight: torch.Tensor,
    num_bits: int,
) -> torch.Tensor:
    """Unpack GPTQ weights from packed int32 to individual values.

    GPTQ uses **standard sequential packing** along dim=0 (input_dim).
    For 4-bit: 8 values per int32, for 8-bit: 4 values per int32.

    After unpacking, values are centered:
      4-bit: subtract 8  (uint4 [0,15] → sint4 [-8,7])
      8-bit: subtract 128 (uint8 [0,255] → int8 [-128,127])

    Args:
        weight: Packed int32 tensor of shape ``(K // pack_factor, N)``.
        num_bits: Bits per weight element (4 or 8).

    Returns:
        Unpacked tensor of shape ``(K, N)`` in int8 dtype.
    """
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    K_packed, N = weight.shape
    K = K_packed * pack_factor

    unpacked = torch.zeros((K, N), device=weight.device, dtype=torch.int32)
    for i in range(pack_factor):
        unpacked[i::pack_factor, :] = (weight >> (num_bits * i)) & mask

    # Center the values: uint → signed
    offset = 1 << (num_bits - 1)  # 8 for 4-bit, 128 for 8-bit
    unpacked = (unpacked - offset).to(torch.int8)

    return unpacked


def _unpack_qzeros_from_int32(
    weight: torch.Tensor,
    num_bits: int,
    use_v2_format: bool = False,
) -> torch.Tensor:
    """Unpack GPTQ zero-points (qzeros) from packed int32.

    GPTQ qzeros are packed along the LAST dim (output_dim) with standard
    order. Supports any number of leading dims: a 2-D ``(G, N // pf)``
    tensor (Linear) yields ``(G, N)``; a 3-D ``(E, G, N // pf)`` tensor
    (MoE) yields ``(E, G, N)``.

    For v1 format: unpacked values need ``+1`` adjustment.
    For v2 format: use as-is.

    Args:
        weight: Packed int32 tensor whose last dim is ``N // pack_factor``.
        num_bits: Bits per element (4 or 8).
        use_v2_format: True if checkpoint_format == "gptq_v2".

    Returns:
        Unpacked zero-points tensor in int32 dtype (leading dims preserved).
    """
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    lead_shape = weight.shape[:-1]
    N_packed = weight.shape[-1]
    N = N_packed * pack_factor

    unpacked = torch.zeros((*lead_shape, N), device=weight.device, dtype=torch.int32)
    for i in range(pack_factor):
        unpacked[..., i::pack_factor] = (weight >> (num_bits * i)) & mask

    # v1 format: qzeros were stored with an implicit +1 offset
    if not use_v2_format:
        unpacked = unpacked + 1

    # Keep as int32 to avoid overflow for 8-bit (values can reach 256).
    # The caller will convert to the appropriate dtype.
    return unpacked


def _get_gptq_linear_weight_spec(
    input_size: int,
    output_size: int,
    pack_factor: int,
) -> dict[str, Any]:
    """Shared weight spec for both W4A16 and W8A16 GPTQ linear schemes.

    GPTQ qweight is packed along dim=0 (input dimension). g_idx is always
    registered (even when desc_act=False) because GPTQ checkpoints always
    contain g_idx tensors, and the weight_loader needs the parameter to
    load them into.
    """
    return {
        "qweight": torch.empty(input_size // pack_factor, output_size, dtype=torch.int32),
        "g_idx": torch.empty(input_size, dtype=torch.int32),
        "_packed_dim": 0,
        "_packed_factor": pack_factor,
        "_param_dims": {
            "qweight": {"input_dim": 0, "output_dim": 1},
            "g_idx": {"input_dim": 0},
        },
        # g_idx is NOT packed — exclude it from receiving packed_dim/packed_factor
        "_unpacked_params": {"g_idx"},
    }


def _get_gptq_linear_pergroup_spec(
    input_size: int,
    output_size: int,
    group_size: int,
    pack_factor: int,
    params_dtype: torch.dtype,
) -> dict[str, Any]:
    """Shared pergroup param spec for both W4A16 and W8A16 GPTQ linear schemes.

    GPTQ qzeros are packed along dim=1 (output dimension), which is different
    from qweight's packing along dim=0. This is why qzeros must go in
    ``get_pergroup_param()`` instead of ``get_weight()``.
    """
    if input_size % group_size != 0:
        raise ValueError(f"GPTQ input_size ({input_size}) must be divisible by group_size ({group_size}).")
    num_groups = input_size // group_size
    return {
        "scales": torch.empty(num_groups, output_size, dtype=params_dtype),
        "qzeros": torch.empty(num_groups, output_size // pack_factor, dtype=torch.int32),
        "_param_dims": {
            "scales": {"input_dim": 0, "output_dim": 1},
            "qzeros": {"input_dim": 0, "output_dim": 1},
        },
        "_packed_params": {
            "qzeros": {"packed_dim": 1, "packed_factor": pack_factor},
        },
    }


def _process_gptq_weights_after_loading(
    layer: torch.nn.Module,
    weight_bits: int,
    desc_act: bool,
    use_v2_format: bool,
) -> None:
    """Shared weight processing for both W4A16 and W8A16 GPTQ linear schemes.

    Steps:
    1. Handle desc_act: sort g_idx, shuffle qweight if needed
    2. Unpack qweight from int32 to int8
    3. Unpack qzeros, adjust for v1/v2, compute antiquant_offset
    4. For 4-bit: repack via npu_convert_weight_to_int4pack
    5. For 8-bit: use int8 directly
    """
    # Save original output size before any transformation.
    # GPTQ qweight is packed along dim=0: shape (K/pack_factor, N).
    # The output dim N is qweight.shape[-1].
    layer.gptq_output_size = layer.qweight.data.shape[-1]

    # --- desc_act handling ---
    if desc_act and hasattr(layer, "g_idx"):
        # Sort g_idx to get the permutation that orders weights by group
        g_idx = layer.g_idx.data
        perm = torch.argsort(g_idx).to(torch.int32)
        layer.g_idx = torch.nn.Parameter(perm, requires_grad=False)

        # Unpack first, then shuffle by permutation, then repack later.
        # qweight shape: (K // pack_factor, N) — pack along dim=0
        unpacked_qweight = _unpack_qweight_from_int32(layer.qweight.data, weight_bits)
        # Apply permutation to the unpacked weight (dim=0 is the input dim)
        unpacked_qweight = unpacked_qweight[perm]
        layer.qweight.data = unpacked_qweight
    else:
        # No desc_act — just unpack
        layer.qweight.data = _unpack_qweight_from_int32(layer.qweight.data, weight_bits)
        if hasattr(layer, "g_idx"):
            layer.g_idx = torch.nn.Parameter(
                torch.empty((0,), dtype=torch.int32),
                requires_grad=False,
            )

    # --- Repack weight for NPU ---
    if weight_bits == 4:
        # 4-bit: need npu_convert_weight_to_int4pack
        # Weight is currently int8 (K, N), convert to int32 for packing
        qweight_int32 = layer.qweight.data.to(torch.int32)
        packed_qweight = torch_npu.npu_convert_weight_to_int4pack(qweight_int32)
        layer.qweight = torch.nn.Parameter(packed_qweight.contiguous(), requires_grad=False)
    else:
        # 8-bit: int8 directly, view as int32 for batchmatmul
        layer.qweight = torch.nn.Parameter(layer.qweight.data.contiguous(), requires_grad=False)

    # --- Process qzeros → antiquant_offset ---
    if hasattr(layer, "qzeros") and hasattr(layer, "scales"):
        qzeros_int8 = _unpack_qzeros_from_int32(
            layer.qzeros.data,
            weight_bits,
            use_v2_format,
        )
        # Convert qzeros to antiquant_offset in target dtype
        # NPU formula: output = (weight + offset) * scale
        # GPTQ formula: output = (weight - zeros) * scale
        # Therefore: offset = -zeros (negated)
        # But weight is already centered (uint→signed), so:
        # The unpacked qweight has been centered by subtracting offset (8 or 128)
        # The qzeros represent the zero-point in uint space.
        # After centering: antiquant_offset = -(qzeros - center_offset)
        center_offset = 1 << (weight_bits - 1)  # 8 for 4-bit, 128 for 8-bit
        antiquant_offset = -(qzeros_int8.to(torch.float32) - center_offset)

        layer.qzeros = torch.nn.Parameter(
            antiquant_offset.to(layer.scales.data.dtype).contiguous(),
            requires_grad=False,
        )
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)


def _apply_gptq_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    """Shared apply for both W4A16 and W8A16 GPTQ linear schemes."""
    qweight = layer.qweight
    if bias is not None and bias.dtype == torch.bfloat16:
        bias = bias.float()

    reshaped_x = x.reshape(-1, x.shape[-1])

    out = torch_npu.npu_weight_quant_batchmatmul(
        reshaped_x,
        qweight,
        antiquant_scale=layer.scales,
        antiquant_offset=layer.qzeros,
        antiquant_group_size=group_size,
        bias=bias,
    )
    # Output size is the original N dimension (saved before repacking).
    # After int4pack, qweight.shape[-1] is N/8 for 4-bit, so we must
    # use the saved gptq_output_size instead.
    out_shape = x.shape[:-1] + (layer.gptq_output_size,)
    return out.reshape(out_shape)


@register_scheme("W4A16_GPTQ", "linear")
class AscendW4A16GPTQLinearScheme(AscendLinearScheme):
    """Linear scheme for Ascend W4A16 GPTQ quantization (4-bit, Ascend Scheme 框架).

    Uses autonomous weight registration. GPTQ packs weights along dim=0
    (input dimension) with standard sequential bit order. qweight is packed
    along dim=0 but qzeros is packed along dim=1, so they go in different
    ``get_*()`` methods.
    """

    def __init__(self, quant_config: "GPTQConfig"):
        self.weight_bits = 4
        self.pack_factor = 32 // self.weight_bits  # 8
        self.group_size = quant_config.group_size
        self.desc_act = quant_config.desc_act
        self.use_v2_format = quant_config.use_v2_format

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        """Return qweight and g_idx specifications."""
        return _get_gptq_linear_weight_spec(input_size, output_size, self.pack_factor)

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        """Return scales and qzeros specifications.

        qzeros is packed along dim=1 (different from qweight's dim=0).
        """
        return _get_gptq_linear_pergroup_spec(input_size, output_size, self.group_size, self.pack_factor, params_dtype)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert GPTQ 4-bit weights to NPU-compatible format."""
        _process_gptq_weights_after_loading(layer, self.weight_bits, self.desc_act, self.use_v2_format)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        """Forward pass using npu_weight_quant_batchmatmul."""
        return _apply_gptq_linear(layer, x, bias, self.group_size)


@register_scheme("W8A16_GPTQ", "linear")
class AscendW8A16GPTQLinearScheme(AscendLinearScheme):
    """Linear scheme for Ascend W8A16 GPTQ quantization (8-bit, Ascend Scheme 框架).

    8-bit GPTQ uses int8 weights directly without additional repacking.
    Same structure as 4-bit but with pack_factor=4.
    """

    def __init__(self, quant_config: "GPTQConfig"):
        self.weight_bits = 8
        self.pack_factor = 32 // self.weight_bits  # 4
        self.group_size = quant_config.group_size
        self.desc_act = quant_config.desc_act
        self.use_v2_format = quant_config.use_v2_format

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        """Return qweight and g_idx specifications."""
        return _get_gptq_linear_weight_spec(input_size, output_size, self.pack_factor)

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        """Return scales and qzeros specifications."""
        return _get_gptq_linear_pergroup_spec(input_size, output_size, self.group_size, self.pack_factor, params_dtype)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert GPTQ 8-bit weights to NPU-compatible format."""
        _process_gptq_weights_after_loading(layer, self.weight_bits, self.desc_act, self.use_v2_format)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        """Forward pass using npu_weight_quant_batchmatmul."""
        return _apply_gptq_linear(layer, x, bias, self.group_size)


# ---------------------------------------------------------------------------
# Shared MoE helpers — W4A16 and W8A16 GPTQ MoE differ ONLY in the weight
# repack step (_repack_gptq_moe_qweight); every other concern is identical.
# ---------------------------------------------------------------------------


def _get_gptq_moe_weight_spec(
    num_experts: int,
    intermediate_size_per_partition: int,
    hidden_sizes: int,
    pack_factor: int,
) -> dict[str, Any]:
    """Shared weight spec for W4A16 and W8A16 GPTQ MoE schemes.

    GPTQ MoE ``qweight`` is packed along the input dim (dim=1 of the 3-D
    expert tensor), identical for both bit-widths:
      - w13 (gate_up): ``(E, H // pack_factor, 2 * IN)``
      - w2  (down_proj): ``(E, IN // pack_factor, H)``
    """
    if intermediate_size_per_partition % pack_factor != 0:
        raise ValueError(
            f"Expecting `intermediate_size_per_partition` "
            f"{intermediate_size_per_partition} can be divided by "
            f"`pack_factor` {pack_factor}"
        )
    if hidden_sizes % pack_factor != 0:
        raise ValueError(f"Expecting `hidden_sizes` {hidden_sizes} can be divided by `pack_factor` {pack_factor}")
    return {
        "w13_qweight": torch.empty(
            num_experts,
            hidden_sizes // pack_factor,
            2 * intermediate_size_per_partition,
            dtype=torch.int32,
        ),
        "w2_qweight": torch.empty(
            num_experts,
            intermediate_size_per_partition // pack_factor,
            hidden_sizes,
            dtype=torch.int32,
        ),
        # g_idx: per-input-element group index (activation reorder, desc_act).
        # Registered even when desc_act=False because GPTQ checkpoints always
        # store per-expert g_idx tensors, and the model loader (layer.py
        # ``_load_g_idx``) expects matching ``w13_g_idx``/``w2_g_idx`` params to
        # exist or it raises KeyError. MoE desc_act is unsupported (see
        # ``_process_gptq_moe_weights_after_loading``), so these are loaded and
        # left unused. Shapes are per-expert, one entry per input element:
        # w13 input = hidden_sizes, w2 input = intermediate_size_per_partition.
        "w13_g_idx": torch.empty(num_experts, hidden_sizes, dtype=torch.int32),
        "w2_g_idx": torch.empty(num_experts, intermediate_size_per_partition, dtype=torch.int32),
    }


def _get_gptq_moe_quant_param(
    num_experts: int,
    intermediate_size_per_partition: int,
    hidden_sizes: int,
    group_size: int,
    pack_factor: int,
    params_dtype: torch.dtype,
) -> dict[str, Any]:
    """Shared per-group quant param spec (scales + packed qzeros) for MoE.

    qzeros are packed along the output dim (last dim), same as the linear path.
    """
    if intermediate_size_per_partition % group_size != 0:
        raise ValueError(
            f"GPTQ MoE intermediate_size_per_partition "
            f"({intermediate_size_per_partition}) must be divisible by "
            f"group_size ({group_size})."
        )
    if hidden_sizes % group_size != 0:
        raise ValueError(f"GPTQ MoE hidden_sizes ({hidden_sizes}) must be divisible by group_size ({group_size}).")
    num_groups_w13 = hidden_sizes // group_size
    num_groups_w2 = intermediate_size_per_partition // group_size
    return {
        "w13_scales": torch.empty(
            num_experts,
            num_groups_w13,
            2 * intermediate_size_per_partition,
            dtype=params_dtype,
        ),
        "w2_scales": torch.empty(num_experts, num_groups_w2, hidden_sizes, dtype=params_dtype),
        "w13_qzeros": torch.empty(
            num_experts,
            num_groups_w13,
            2 * intermediate_size_per_partition // pack_factor,
            dtype=torch.int32,
        ),
        "w2_qzeros": torch.empty(
            num_experts,
            num_groups_w2,
            hidden_sizes // pack_factor,
            dtype=torch.int32,
        ),
    }


def _repack_gptq_moe_qweight(
    qweight_data: torch.Tensor,
    weight_bits: int,
    pack_factor: int,
) -> torch.Tensor:
    """Unpack a GPTQ MoE qweight (packed along the input dim) and repack it into
    the NPU MoE weight layout consumed by ``fused_experts``.

    GPTQ stores ``{w13,w2}_qweight`` packed along the input dim. Unpacking lands
    directly on the NPU's expected input-first layout ``(E, K, N)`` — no
    transpose is needed (unlike AWQ, which stores output-first and transposes in
    post-processing).

    The two bit-widths differ ONLY in how the *output* dim is re-stored:
      - 4-bit: repack via ``npu_convert_weight_to_int4pack`` → ``(E, K, N // 8)``
      - 8-bit: view 4 consecutive int8 as one int32 → ``(E, K, N // 4)``

    This is the single branch where W4A16 and W8A16 diverge.
    """
    unpacked = _unpack_qweight_from_int32(qweight_data.flatten(0, 1), weight_bits).view(
        qweight_data.shape[0], -1, qweight_data.shape[2]
    )
    if weight_bits == 4:
        packed = torch_npu.npu_convert_weight_to_int4pack(unpacked.flatten(0, 1).int())
        return packed.view(
            qweight_data.shape[0],
            qweight_data.shape[1] * pack_factor,
            -1,
        )
    # 8-bit: keep int8, view as int32 for grouped_matmul storage.
    return unpacked.contiguous().view(torch.int32)


def _process_gptq_moe_weights_after_loading(
    layer: torch.nn.Module,
    weight_bits: int,
    pack_factor: int,
    desc_act: bool,
    use_v2_format: bool,
) -> None:
    """Shared weight processing for W4A16 and W8A16 GPTQ MoE schemes.

    desc_act (activation ordering) is not supported for MoE: the MoE weight
    registration does not load per-expert g_idx, so there is no permutation to
    apply. Fail loud rather than silently producing wrong output. See the
    Linear scheme (``_process_gptq_weights_after_loading``) for the desc_act
    implementation that MoE would need to mirror.

    For each expert weight (w13 = gate_up, w2 = down_proj):
      1. Repack qweight for the NPU (see ``_repack_gptq_moe_qweight``).
      2. Convert qzeros → antiquant_offset = -(zp - center_offset).
    """
    if desc_act:
        raise NotImplementedError(
            "GPTQ MoE with desc_act=True is not yet supported on Ascend "
            "NPU: per-expert g_idx reordering is not implemented. Please "
            "use a GPTQ MoE model with desc_act=False."
        )

    center_offset = 1 << (weight_bits - 1)  # 8 for 4-bit, 128 for 8-bit
    for prefix in ("w13", "w2"):
        # 1. Repack weight for the NPU (int4pack for 4-bit, int32 view for 8-bit).
        repacked = _repack_gptq_moe_qweight(
            getattr(layer, f"{prefix}_qweight").data,
            weight_bits,
            pack_factor,
        )
        layer.register_parameter(
            f"{prefix}_qweight",
            torch.nn.Parameter(repacked, requires_grad=False),
        )
        # 2. qzeros → antiquant_offset.
        #    NPU: out = (w + offset) * scale;  GPTQ: out = (w - zeros) * scale
        #    ⟹ offset = -(zeros - center_offset). center cancels the uint→signed
        #    shift already applied to the weight in _unpack_qweight_from_int32.
        qzeros = _unpack_qzeros_from_int32(
            getattr(layer, f"{prefix}_qzeros").data,
            weight_bits,
            use_v2_format,
        )
        offset = -(qzeros.to(torch.float32) - center_offset)
        scales_dtype = getattr(layer, f"{prefix}_scales").data.dtype
        layer.register_parameter(
            f"{prefix}_qzeros",
            torch.nn.Parameter(offset.to(scales_dtype).contiguous(), requires_grad=False),
        )


class _AscendGPTQFusedMoEMethodBase(AscendMoEScheme):
    """Shared GPTQ MoE implementation for Ascend NPU (4-bit and 8-bit).

    Subclasses set two class attributes and inherit the full weight-spec /
    weight-processing / ``apply`` pipeline:
      - ``quant_type``: the :class:`QuantType` used to route through
        ``fused_experts``.
      - ``weight_bits``: 4 or 8 — drives ``pack_factor`` and the single
        bit-width branch in ``_repack_gptq_moe_qweight``.

    The ``apply`` method delegates to the unified ``fused_experts`` pipeline,
    passing the GPTQ-specific scale (``{w13,w2}_scales``) and antiquant_offset
    (``{w13,w2}_qzeros``) tensors.
    """

    quant_type: QuantType = QuantType.NONE
    weight_bits: int = 0  # set by subclass
    weight_attrs: dict = {"is_transposed": True}

    def __init__(self, quant_config: "GPTQConfig"):
        self.quant_config = quant_config
        self.pack_factor = 32 // self.weight_bits
        self.group_size = quant_config.group_size
        self.desc_act = quant_config.desc_act
        self.use_v2_format = quant_config.use_v2_format
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        return _get_gptq_moe_weight_spec(
            num_experts,
            intermediate_size_per_partition,
            hidden_sizes,
            self.pack_factor,
        )

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        return _get_gptq_moe_quant_param(
            num_experts,
            intermediate_size_per_partition,
            hidden_sizes,
            self.group_size,
            self.pack_factor,
            params_dtype,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert GPTQ MoE weights to NPU-compatible format."""
        _process_gptq_moe_weights_after_loading(
            layer,
            self.weight_bits,
            self.pack_factor,
            self.desc_act,
            self.use_v2_format,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: Any | None = None,
    ) -> torch.Tensor:
        # vLLM passes ``activation`` as a MoEActivation enum (e.g.
        # MoEActivation.SILU). Normalize to its string value so the guard and
        # the downstream fused_experts path (build_fused_experts_input expects a
        # str) both work, whether a str or enum is supplied.
        if hasattr(activation, "value"):
            activation = activation.value
        if activation != "silu":
            raise ValueError("Only SiLU activation is supported for Ascend GPTQ MoE.")

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_experts,
        )

        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_qweight,
                w2=layer.w2_qweight,
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                w1_scale=layer.w13_scales,
                w2_scale=layer.w2_scales,
                w1_offset=layer.w13_qzeros,
                w2_offset=layer.w2_qzeros,
            )
        )


@register_scheme("W4A16_GPTQ", "moe")
class AscendW4A16GPTQFusedMoEMethod(_AscendGPTQFusedMoEMethodBase):
    """FusedMoE method for Ascend W4A16 GPTQ quantization (4-bit).

    Inherits the full pipeline from ``_AscendGPTQFusedMoEMethodBase``. The
    4-bit weight repack (``npu_convert_weight_to_int4pack`` along the output
    dim) is the sole bit-width-specific behavior, isolated in
    ``_repack_gptq_moe_qweight``.
    """

    quant_type: QuantType = QuantType.W4A16_GPTQ
    weight_bits: int = 4


@register_scheme("W8A16_GPTQ", "moe")
class AscendW8A16GPTQFusedMoEMethod(_AscendGPTQFusedMoEMethodBase):
    """FusedMoE method for Ascend W8A16 GPTQ quantization (8-bit).

    Inherits the full pipeline from ``_AscendGPTQFusedMoEMethodBase``. The
    8-bit weight repack (view 4 consecutive int8 as one int32) is the sole
    bit-width-specific behavior, isolated in ``_repack_gptq_moe_qweight``.
    """

    quant_type: QuantType = QuantType.W8A16_GPTQ
    weight_bits: int = 8
