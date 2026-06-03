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
"""Ascend W4A16 AWQ quantization scheme for Linear and MoE layers.

This module provides AWQ (Activation-aware Weight Quantization) support on
Ascend NPU, using ``npu_weight_quant_batchmatmul`` for linear layers and
``npu_grouped_matmul`` (via fused_experts) for MoE layers.

AWQ packs 4-bit weights along the **output dimension** (dim=1) using a
non-standard interleaved bit order ``[0, 4, 1, 5, 2, 6, 3, 7]``. The NPU
requires signed int4 representation, so weights are converted via
XOR ``0x88888888`` and zero-points are remapped as ``-(uint4 - 8)``.
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
    from vllm_ascend.quantization.awq_config import AWQConfig

# Bit shift pattern for unpacking 4-bit values from int32.
# AWQ uses a non-standard interleaved packing order.
# See: https://github.com/casper-hansen/AutoAWQ/blob/v0.2.8/awq/utils/quant_utils.py
REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def _unpack_qzero_from_int32(
    weight: torch.Tensor,
    param_dtype: torch.dtype,
    pack_factor: int = 8,
    is_moe_layer: bool = False,
) -> torch.Tensor:
    """Unpack and convert AWQ zero-points (qzeros) from int32 to target dtype.

    AWQ zero-points are stored as packed uint4 values in int32 using the
    interleaved bit order. This function unpacks them and converts from
    unsigned int4 [0, 15] to signed int4 [-8, 7] via ``-(uint4 - 8)``.

    Args:
        weight: Packed int32 tensor containing zero-points.
        param_dtype: Target dtype (e.g., bfloat16) for the output.
        pack_factor: Number of 4-bit values per int32 (default: 8).
        is_moe_layer: Whether this is for MoE layer (affects reshape).

    Returns:
        Unpacked and converted zero-points tensor in param_dtype.
    """
    weight_list = []

    for i in range(pack_factor):
        shift_num = REVERSE_AWQ_PACK_ORDER[i] * 4
        weight_list.append((weight.reshape(-1, 1) >> shift_num) & 0xF)

    if is_moe_layer:
        weight = torch.cat(weight_list, dim=-1).reshape(weight.shape[0], weight.shape[1], -1)
    else:
        weight = torch.cat(weight_list, dim=-1).reshape(weight.shape[0], -1)

    # Convert unsigned int4 [0,15] to signed int4 [-8,7]
    weight = -(weight - 8)
    return weight.to(param_dtype).contiguous()


def _unpack_weight_from_int32(
    weight: torch.Tensor,
    pack_factor: int = 8,
) -> torch.Tensor:
    """Unpack and convert AWQ weights (qweight) from int32 to NPU format.

    AWQ weights are stored as packed uint4 values using the interleaved bit
    order. This function rearranges them to standard sequential order and
    applies XOR ``0x88888888`` to convert from uint4 to sint4 representation
    expected by the NPU hardware.

    Args:
        weight: Packed int32 tensor containing quantized weights.
        pack_factor: Number of 4-bit values per int32 (default: 8).

    Returns:
        Repacked and signed-converted weight tensor (same shape, int32).
    """
    weight_tmp = torch.zeros_like(weight)
    for i in range(pack_factor):
        shift_num = REVERSE_AWQ_PACK_ORDER[i] * 4
        weight_tmp.bitwise_or_(((weight >> shift_num) * (2 ** (4 * i))) & (0xF << (4 * i)))
    weight_tmp.bitwise_xor_(0x88888888)
    return weight_tmp.contiguous()


class AscendW4A16AWQLinearMethod:
    """Linear method for Ascend W4A16 AWQ quantization.

    This class delegates ``create_weights`` to vLLM's ``AWQLinearMethod``
    (which creates qweight, qzeros, scales parameters with correct
    PackedvLLMParameter types). It implements its own
    ``process_weights_after_loading`` and ``apply`` to use Ascend NPU operators.

    The weight processing converts AWQ's uint4 packed weights into the signed
    int4 format expected by ``npu_weight_quant_batchmatmul``, and unpacks
    zero-points from int32 to bfloat16 for the same operator.
    """

    def __init__(self, quant_config: "AWQConfig"):
        # Avoid importing vLLM's AWQLinearMethod at module level to prevent
        # circular imports. We inherit from it lazily.
        self.quant_config = quant_config
        self.pack_factor = self.quant_config.pack_factor
        self.group_size = self.quant_config.group_size

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
        """Delegate weight creation to vLLM's AWQLinearMethod.

        This reuses vLLM's parameter creation (qweight, qzeros, scales) which
        handles PackedvLLMParameter and GroupQuantScaleParameter correctly.
        """
        from vllm.model_executor.layers.quantization.awq import AWQLinearMethod

        # Create a vLLM AWQConfig-compatible shim that has the same attributes
        # needed by AWQLinearMethod.create_weights.
        vllm_method = AWQLinearMethod(self.quant_config)
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
        """Convert AWQ weights to NPU-compatible format after loading.

        - qzeros: Unpack from int32 to bfloat16, convert uint4 -> sint4
        - qweight: Reorder interleaved bits, apply XOR for signed conversion
        - scales: Wrap as Parameter
        """
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(
            _unpack_qzero_from_int32(
                weight=layer.qzeros.data,
                param_dtype=layer.scales.data.dtype,
                pack_factor=self.pack_factor,
            ),
            requires_grad=False,
        )
        layer.qweight = torch.nn.Parameter(
            _unpack_weight_from_int32(
                weight=layer.qweight.data,
                pack_factor=self.pack_factor,
            ),
            requires_grad=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass using npu_weight_quant_batchmatmul.

        Dequantization is fused into the matrix multiplication:
        ``output = (sint4_weight + offset) * scale @ x``
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
        out_shape = x.shape[:-1] + (qweight.shape[-1] * self.pack_factor,)
        return out.reshape(out_shape)


@register_scheme("W4A16_AWQ", "moe")
class AscendW4A16AWQFusedMoEMethod(AscendMoEScheme):
    """FusedMoE method for Ascend W4A16 AWQ quantization.

    AWQ MoE weights follow the same packing convention as AWQ linear weights
    (interleaved bit order, packed along output dim). The ``apply`` method
    delegates to the unified ``moe_comm_method.fused_experts`` pipeline,
    passing the AWQ-specific scale and offset tensors.
    """

    quant_type: QuantType = QuantType.W4A16_AWQ
    weight_attrs: dict = {"is_transposed": True}

    def __init__(self, quant_config: "AWQConfig"):
        self.quant_config = quant_config
        self.pack_factor = self.quant_config.pack_factor
        self.group_size = self.quant_config.group_size
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
            hidden_sizes,
            2 * intermediate_size_per_partition // self.pack_factor,
            dtype=torch.int32,
        )
        param_dict["w2_qweight"] = torch.empty(
            num_experts,
            intermediate_size_per_partition,
            hidden_sizes // self.pack_factor,
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

        # WEIGHT_SCALES
        # Allocate combined scales for w1 and w3.
        param_dict["w13_scales"] = torch.empty(
            num_experts,
            num_groups_w13,
            intermediate_size_per_partition * 2,
            dtype=params_dtype,
        )
        param_dict["w2_scales"] = torch.empty(
            num_experts,
            num_groups_w2,
            hidden_sizes,
            dtype=params_dtype,
        )

        # WEIGHT_ZERO_POINTS (packed int32)
        # Allocate combined zero points for w1 and w3.
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
        """Convert AWQ MoE weights to NPU-compatible format."""
        w13_qzeros = torch.nn.Parameter(
            _unpack_qzero_from_int32(
                weight=layer.w13_qzeros.data,
                param_dtype=layer.w13_scales.data.dtype,
                pack_factor=self.pack_factor,
                is_moe_layer=True,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        w13_qweight = torch.nn.Parameter(
            _unpack_weight_from_int32(
                weight=layer.w13_qweight.data,
                pack_factor=self.pack_factor,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)

        w2_qzeros = torch.nn.Parameter(
            _unpack_qzero_from_int32(
                weight=layer.w2_qzeros.data,
                param_dtype=layer.w2_scales.data.dtype,
                pack_factor=self.pack_factor,
                is_moe_layer=True,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        w2_qweight = torch.nn.Parameter(
            _unpack_weight_from_int32(
                weight=layer.w2_qweight.data,
                pack_factor=self.pack_factor,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)

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
