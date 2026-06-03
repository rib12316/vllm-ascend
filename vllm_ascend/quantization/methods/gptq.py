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
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch_npu

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input

from .base import AscendMoEScheme, QuantType
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

    unpacked = torch.zeros(
        (K, N), device=weight.device, dtype=torch.int32
    )
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

    GPTQ qzeros are packed along dim=1 (output_dim) with standard order.

    For v1 format: unpacked values need ``+1`` adjustment.
    For v2 format: use as-is.

    Args:
        weight: Packed int32 tensor of shape ``(G, N // pack_factor)``.
        num_bits: Bits per element (4 or 8).
        use_v2_format: True if checkpoint_format == "gptq_v2".

    Returns:
        Unpacked zero-points tensor in int8 dtype.
    """
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    G, N_packed = weight.shape
    N = N_packed * pack_factor

    unpacked = torch.zeros(
        (G, N), device=weight.device, dtype=torch.int32
    )
    for i in range(pack_factor):
        unpacked[:, i::pack_factor] = (weight >> (num_bits * i)) & mask

    # v1 format: qzeros were stored with an implicit +1 offset
    if not use_v2_format:
        unpacked = unpacked + 1

    # Keep as int32 to avoid overflow for 8-bit (values can reach 256).
    # The caller will convert to the appropriate dtype.
    return unpacked


class AscendGPTQLinearMethod:
    """Linear method for Ascend GPTQ quantization.

    This class delegates ``create_weights`` to vLLM's ``GPTQLinearMethod``
    (which creates qweight, g_idx, qzeros, scales parameters with correct
    PackedvLLMParameter types). It implements its own
    ``process_weights_after_loading`` and ``apply`` to use Ascend NPU operators.

    Weight processing:
      - 4-bit: unpack → subtract 8 → ``npu_convert_weight_to_int4pack``
      - 8-bit: unpack → subtract 128 → int8 direct use
      - qzeros: unpack, adjust for v1/v2, convert to antiquant_offset
      - desc_act: sort g_idx, shuffle qweight accordingly
    """

    def __init__(self, quant_config: "GPTQConfig"):
        self.quant_config = quant_config
        self.pack_factor = quant_config.pack_factor
        self.group_size = quant_config.group_size
        self.weight_bits = quant_config.weight_bits
        self.desc_act = quant_config.desc_act
        self.use_v2_format = quant_config.use_v2_format

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Delegate weight creation to vLLM's GPTQLinearMethod.

        This reuses vLLM's parameter creation (qweight, g_idx, qzeros, scales)
        which handles PackedvLLMParameter correctly.
        """
        from vllm.model_executor.layers.quantization.gptq import (
            GPTQLinearMethod,
        )

        vllm_method = GPTQLinearMethod(self.quant_config)
        vllm_method.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert GPTQ weights to NPU-compatible format after loading.

        Steps:
        1. Handle desc_act: sort g_idx, shuffle qweight if needed
        2. Unpack qweight from int32 to int8
        3. Unpack qzeros, adjust for v1/v2, compute antiquant_offset
        4. For 4-bit: repack via npu_convert_weight_to_int4pack
        5. For 8-bit: use int8 directly
        """
        # --- desc_act handling ---
        if self.desc_act and hasattr(layer, "g_idx"):
            # Sort g_idx to get the permutation that orders weights by group
            g_idx = layer.g_idx.data
            perm = torch.argsort(g_idx).to(torch.int32)
            layer.g_idx = torch.nn.Parameter(perm, requires_grad=False)

            # Unpack first, then shuffle by permutation, then repack later.
            # qweight shape: (K // pack_factor, N) — pack along dim=0
            unpacked_qweight = _unpack_qweight_from_int32(
                layer.qweight.data, self.weight_bits
            )
            # Apply permutation to the unpacked weight (dim=0 is the input dim)
            unpacked_qweight = unpacked_qweight[perm]
            layer.qweight.data = unpacked_qweight
        else:
            # No desc_act — just unpack
            layer.qweight.data = _unpack_qweight_from_int32(
                layer.qweight.data, self.weight_bits
            )
            if hasattr(layer, "g_idx"):
                layer.g_idx = torch.nn.Parameter(
                    torch.empty((0,), dtype=torch.int32),
                    requires_grad=False,
                )

        # --- Repack weight for NPU ---
        if self.weight_bits == 4:
            # 4-bit: need npu_convert_weight_to_int4pack
            # Weight is currently int8 (K, N), convert to int32 for packing
            qweight_int32 = layer.qweight.data.to(torch.int32)
            packed_qweight = torch_npu.npu_convert_weight_to_int4pack(
                qweight_int32
            )
            layer.qweight = torch.nn.Parameter(
                packed_qweight.contiguous(), requires_grad=False
            )
        else:
            # 8-bit: int8 directly, view as int32 for batchmatmul
            layer.qweight = torch.nn.Parameter(
                layer.qweight.data.contiguous(), requires_grad=False
            )

        # --- Process qzeros → antiquant_offset ---
        if hasattr(layer, "qzeros") and hasattr(layer, "scales"):
            qzeros_int8 = _unpack_qzeros_from_int32(
                layer.qzeros.data,
                self.weight_bits,
                self.use_v2_format,
            )
            # Convert qzeros to antiquant_offset in target dtype
            # NPU formula: output = (weight + offset) * scale
            # GPTQ formula: output = (weight - zeros) * scale
            # Therefore: offset = -zeros (negated)
            # But weight is already centered (uint→signed), so:
            # The unpacked qweight has been centered by subtracting offset (8 or 128)
            # The qzeros represent the zero-point in uint space.
            # After centering: antiquant_offset = -(qzeros - center_offset)
            center_offset = 1 << (self.weight_bits - 1)  # 8 for 4-bit, 128 for 8-bit
            antiquant_offset = -(qzeros_int8.to(torch.float32) - center_offset)

            layer.qzeros = torch.nn.Parameter(
                antiquant_offset.to(layer.scales.data.dtype).contiguous(),
                requires_grad=False,
            )
            layer.scales = torch.nn.Parameter(
                layer.scales.data, requires_grad=False
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass using npu_weight_quant_batchmatmul.

        Dequantization is fused into the matrix multiplication:
        ``output = (weight + offset) * scale @ x``
        """
        qweight = layer.qweight
        if bias is not None and bias.dtype == torch.bfloat16:
            bias = bias.float()

        reshaped_x = x.reshape(-1, x.shape[-1])

        out = torch_npu.npu_weight_quant_batchmatmul(
            reshaped_x,
            qweight,
            antiquant_scale=layer.scales,
            antiquant_offset=layer.qzeros,
            antiquant_group_size=self.group_size,
            bias=bias,
        )
        # Output size is the second dimension of the original qweight
        out_shape = x.shape[:-1] + (qweight.shape[-1],)
        return out.reshape(out_shape)


@register_scheme("W4A16_GPTQ", "moe")
class AscendW4A16GPTQFusedMoEMethod(AscendMoEScheme):
    """FusedMoE method for Ascend W4A16 GPTQ quantization (4-bit).

    GPTQ MoE weights use **standard sequential packing** along the output
    dimension (same as the storage format for MoE). The ``apply`` method
    delegates to the unified ``moe_comm_method.fused_experts`` pipeline,
    passing the GPTQ-specific scale and offset tensors.
    """

    quant_type: QuantType = QuantType.W4A16_GPTQ
    weight_attrs: dict = {"is_transposed": True}

    def __init__(self, quant_config: "GPTQConfig"):
        self.quant_config = quant_config
        self.weight_bits = 4
        self.pack_factor = 32 // self.weight_bits  # 8
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
        assert intermediate_size_per_partition % self.pack_factor == 0, (
            f"Expecting `intermediate_size_per_partition` {intermediate_size_per_partition} "
            f"can be divided by `pack_factor` {self.pack_factor}"
        )
        assert hidden_sizes % self.pack_factor == 0, (
            f"Expecting `hidden_sizes` {hidden_sizes} can be divided by `pack_factor` {self.pack_factor}"
        )

        param_dict = {}
        # GPTQ MoE: qweight packed along input_dim (dim=0)
        # w13: gate_up, shape (E, K // pack_factor, 2*IN)
        # After unpack: (E, K, 2*IN), after transpose: (E, 2*IN, K // pack_factor)
        param_dict["w13_qweight"] = torch.empty(
            num_experts,
            hidden_sizes // self.pack_factor,
            2 * intermediate_size_per_partition,
            dtype=torch.int32,
        )
        param_dict["w2_qweight"] = torch.empty(
            num_experts,
            intermediate_size_per_partition // self.pack_factor,
            hidden_sizes,
            dtype=torch.int32,
        )
        return param_dict

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        assert intermediate_size_per_partition % self.group_size == 0, (
            f"Expecting `intermediate_size_per_partition` {intermediate_size_per_partition} "
            f"can be divided by `group_size` {self.group_size}"
        )
        assert hidden_sizes % self.group_size == 0, (
            f"Expecting `hidden_sizes` {hidden_sizes} can be divided by `group_size` {self.group_size}"
        )

        param_dict = {}
        num_groups_w13 = hidden_sizes // self.group_size
        num_groups_w2 = intermediate_size_per_partition // self.group_size

        # Scales
        param_dict["w13_scales"] = torch.empty(
            num_experts,
            num_groups_w13,
            2 * intermediate_size_per_partition,
            dtype=params_dtype,
        )
        param_dict["w2_scales"] = torch.empty(
            num_experts,
            num_groups_w2,
            hidden_sizes,
            dtype=params_dtype,
        )

        # Zero-points (packed int32)
        param_dict["w13_qzeros"] = torch.empty(
            num_experts,
            num_groups_w13,
            2 * intermediate_size_per_partition // self.pack_factor,
            dtype=torch.int32,
        )
        param_dict["w2_qzeros"] = torch.empty(
            num_experts,
            num_groups_w2,
            hidden_sizes // self.pack_factor,
            dtype=torch.int32,
        )
        return param_dict

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert GPTQ MoE weights to NPU-compatible format."""
        # Process w13 (gate_up)
        w13_qweight_unpacked = _unpack_qweight_from_int32(
            layer.w13_qweight.data.flatten(0, 1),
            self.weight_bits,
        ).view(layer.w13_qweight.data.shape[0], -1, layer.w13_qweight.data.shape[2])
        # Transpose: (E, K, 2*IN) → (E, 2*IN, K) for NPU MoE format
        w13_qweight_transposed = w13_qweight_unpacked.transpose(1, 2).contiguous().int()
        # Repack for NPU: npu_convert_weight_to_int4pack expects (E*2*IN, K) int32
        w13_packed = torch_npu.npu_convert_weight_to_int4pack(
            w13_qweight_transposed.flatten(0, 1)
        )
        layer.register_parameter(
            "w13_qweight",
            torch.nn.Parameter(
                w13_packed.view(
                    layer.w13_qweight.data.shape[0],
                    2 * (layer.w13_qweight.data.shape[1]),
                    -1,
                ),
                requires_grad=False,
            ),
        )

        # Process w2 (down_proj)
        w2_qweight_unpacked = _unpack_qweight_from_int32(
            layer.w2_qweight.data.flatten(0, 1),
            self.weight_bits,
        ).view(layer.w2_qweight.data.shape[0], -1, layer.w2_qweight.data.shape[2])
        # Transpose: (E, IN, H) → (E, H, IN) for NPU MoE format
        w2_qweight_transposed = w2_qweight_unpacked.transpose(1, 2).contiguous().int()
        w2_packed = torch_npu.npu_convert_weight_to_int4pack(
            w2_qweight_transposed.flatten(0, 1)
        )
        layer.register_parameter(
            "w2_qweight",
            torch.nn.Parameter(
                w2_packed.view(
                    layer.w2_qweight.data.shape[0],
                    layer.w2_qweight.data.shape[2],
                    -1,
                ),
                requires_grad=False,
            ),
        )

        # Process qzeros → antiquant_offset
        center_offset = 1 << (self.weight_bits - 1)  # 8 for 4-bit

        w13_qzeros_unpacked = _unpack_qzeros_from_int32(
            layer.w13_qzeros.data, self.weight_bits, self.use_v2_format
        )
        w13_offset = -(w13_qzeros_unpacked.to(torch.float32) - center_offset)
        layer.register_parameter(
            "w13_qzeros",
            torch.nn.Parameter(
                w13_offset.to(layer.w13_scales.data.dtype).contiguous(),
                requires_grad=False,
            ),
        )

        w2_qzeros_unpacked = _unpack_qzeros_from_int32(
            layer.w2_qzeros.data, self.weight_bits, self.use_v2_format
        )
        w2_offset = -(w2_qzeros_unpacked.to(torch.float32) - center_offset)
        layer.register_parameter(
            "w2_qzeros",
            torch.nn.Parameter(
                w2_offset.to(layer.w2_scales.data.dtype).contiguous(),
                requires_grad=False,
            ),
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
        assert activation == "silu", "Only SiLU activation is supported."

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


@register_scheme("W8A16_GPTQ", "moe")
class AscendW8A16GPTQFusedMoEMethod(AscendMoEScheme):
    """FusedMoE method for Ascend W8A16 GPTQ quantization (8-bit).

    8-bit GPTQ uses int8 weights directly without additional repacking.
    """

    quant_type: QuantType = QuantType.W8A16_GPTQ
    weight_attrs: dict = {"is_transposed": True}

    def __init__(self, quant_config: "GPTQConfig"):
        self.quant_config = quant_config
        self.weight_bits = 8
        self.pack_factor = 32 // self.weight_bits  # 4
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
        assert intermediate_size_per_partition % self.pack_factor == 0, (
            f"Expecting `intermediate_size_per_partition` {intermediate_size_per_partition} "
            f"can be divided by `pack_factor` {self.pack_factor}"
        )
        assert hidden_sizes % self.pack_factor == 0, (
            f"Expecting `hidden_sizes` {hidden_sizes} can be divided by `pack_factor` {self.pack_factor}"
        )

        param_dict = {}
        param_dict["w13_qweight"] = torch.empty(
            num_experts,
            hidden_sizes // self.pack_factor,
            2 * intermediate_size_per_partition,
            dtype=torch.int32,
        )
        param_dict["w2_qweight"] = torch.empty(
            num_experts,
            intermediate_size_per_partition // self.pack_factor,
            hidden_sizes,
            dtype=torch.int32,
        )
        return param_dict

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        assert intermediate_size_per_partition % self.group_size == 0, (
            f"Expecting `intermediate_size_per_partition` {intermediate_size_per_partition} "
            f"can be divided by `group_size` {self.group_size}"
        )
        assert hidden_sizes % self.group_size == 0, (
            f"Expecting `hidden_sizes` {hidden_sizes} can be divided by `group_size` {self.group_size}"
        )

        param_dict = {}
        num_groups_w13 = hidden_sizes // self.group_size
        num_groups_w2 = intermediate_size_per_partition // self.group_size

        param_dict["w13_scales"] = torch.empty(
            num_experts,
            num_groups_w13,
            2 * intermediate_size_per_partition,
            dtype=params_dtype,
        )
        param_dict["w2_scales"] = torch.empty(
            num_experts,
            num_groups_w2,
            hidden_sizes,
            dtype=params_dtype,
        )

        param_dict["w13_qzeros"] = torch.empty(
            num_experts,
            num_groups_w13,
            2 * intermediate_size_per_partition // self.pack_factor,
            dtype=torch.int32,
        )
        param_dict["w2_qzeros"] = torch.empty(
            num_experts,
            num_groups_w2,
            hidden_sizes // self.pack_factor,
            dtype=torch.int32,
        )
        return param_dict

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert 8-bit GPTQ MoE weights to NPU-compatible format."""
        # Process w13 (gate_up)
        w13_qweight_unpacked = _unpack_qweight_from_int32(
            layer.w13_qweight.data.flatten(0, 1),
            self.weight_bits,
        ).view(layer.w13_qweight.data.shape[0], -1, layer.w13_qweight.data.shape[2])
        # Transpose: (E, K, 2*IN) → (E, 2*IN, K) for NPU MoE format
        # 8-bit: view as int32 directly (4 int8 per int32)
        w13_transposed = w13_qweight_unpacked.transpose(1, 2).contiguous()
        layer.register_parameter(
            "w13_qweight",
            torch.nn.Parameter(
                w13_transposed.view(
                    layer.w13_qweight.data.shape[0],
                    2 * (layer.w13_qweight.data.shape[1]),
                    -1,
                ).view(torch.int32).contiguous(),
                requires_grad=False,
            ),
        )

        # Process w2 (down_proj)
        w2_qweight_unpacked = _unpack_qweight_from_int32(
            layer.w2_qweight.data.flatten(0, 1),
            self.weight_bits,
        ).view(layer.w2_qweight.data.shape[0], -1, layer.w2_qweight.data.shape[2])
        w2_transposed = w2_qweight_unpacked.transpose(1, 2).contiguous()
        layer.register_parameter(
            "w2_qweight",
            torch.nn.Parameter(
                w2_transposed.view(
                    layer.w2_qweight.data.shape[0],
                    layer.w2_qweight.data.shape[2],
                    -1,
                ).view(torch.int32).contiguous(),
                requires_grad=False,
            ),
        )

        # Process qzeros → antiquant_offset
        center_offset = 1 << (self.weight_bits - 1)  # 128 for 8-bit

        w13_qzeros_unpacked = _unpack_qzeros_from_int32(
            layer.w13_qzeros.data, self.weight_bits, self.use_v2_format
        )
        w13_offset = -(w13_qzeros_unpacked.to(torch.float32) - center_offset)
        layer.register_parameter(
            "w13_qzeros",
            torch.nn.Parameter(
                w13_offset.to(layer.w13_scales.data.dtype).contiguous(),
                requires_grad=False,
            ),
        )

        w2_qzeros_unpacked = _unpack_qzeros_from_int32(
            layer.w2_qzeros.data, self.weight_bits, self.use_v2_format
        )
        w2_offset = -(w2_qzeros_unpacked.to(torch.float32) - center_offset)
        layer.register_parameter(
            "w2_qzeros",
            torch.nn.Parameter(
                w2_offset.to(layer.w2_scales.data.dtype).contiguous(),
                requires_grad=False,
            ),
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
        assert activation == "silu", "Only SiLU activation is supported."

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
