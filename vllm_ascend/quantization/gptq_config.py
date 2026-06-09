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
"""GPTQ quantization config for Ascend NPU.

This config replaces vLLM's native ``GPTQConfig`` to route linear and MoE
layers through Ascend-specific scheme implementations (Pattern A):

- **Linear layers** → ``AscendW4A16GPTQLinearScheme`` or
  ``AscendW8A16GPTQLinearScheme`` (registered via ``@register_scheme``,
  dispatched through ``AscendLinearMethod`` adapter)
- **MoE layers** → ``AscendW4A16GPTQFusedMoEMethod`` or
  ``AscendW8A16GPTQFusedMoEMethod`` (registered schemes)
- **Skipped layers** (e.g. lm_head) → ``AscendUnquantizedLinearMethod``

Key differences from AWQ:
- GPTQ packs along **input_dim** (dim=0), not output_dim
- GPTQ uses standard sequential bit order, not AWQ interleaved
- GPTQ has ``desc_act`` (g_idx) for activation ordering
- GPTQ has v1/v2 checkpoint format (affects zero-point handling)
- GPTQ supports both 4-bit and 8-bit weights
"""

from typing import Any, Union

import torch
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.utils import GPTQ_QUANTIZATION_METHOD

from .method_adapters import AscendFusedMoEMethod, AscendLinearMethod
from .methods import get_scheme_class


@register_quantization_config(GPTQ_QUANTIZATION_METHOD)
class GPTQConfig(QuantizationConfig):
    """GPTQ quantization config for Ascend NPU.

    Registered as ``"gptq"``, this replaces vLLM's native GPTQ config so that
    GPTQ models are automatically routed through Ascend NPU operators.

    This also prevents vLLM from auto-upgrading GPTQ to GPTQ-Marlin, which
    is NVIDIA GPU-specific and not supported on Ascend NPU.
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        checkpoint_format: str = "",
        dynamic: dict[str, dict[str, int | bool]] | None = None,
        lm_head_quantized: bool = False,
        modules_in_block_to_quantize: list[str] | None = None,
        autoround_version: str = "",
        quant_config: dict[str, Any] | None = None,
    ):
        self.quant_description = quant_config if quant_config is not None else {}
        super().__init__()

        if weight_bits not in [2, 3, 4, 8]:
            raise ValueError(
                f"Currently, only 2/3/4/8-bit weight quantization is "
                f"supported for GPTQ on Ascend, but got {weight_bits} bits."
            )
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.checkpoint_format = checkpoint_format
        self.dynamic = dynamic or {}
        self.lm_head_quantized = lm_head_quantized
        # Flatten nested list[list[str]] → list[str] (upstream does this in
        # maybe_update_config which we haven't implemented; some models like
        # TheBloke store modules_in_block_to_quantize as nested lists).
        raw_modules = modules_in_block_to_quantize or []
        if raw_modules and isinstance(raw_modules[0], list):
            self.modules_in_block_to_quantize = [
                item for sublist in raw_modules for item in sublist
            ]
        else:
            self.modules_in_block_to_quantize = raw_modules
        self.autoround_version = autoround_version

        self.pack_factor = 32 // weight_bits

        # v2 format flag
        self.use_v2_format = checkpoint_format == "gptq_v2"

    def get_name(self) -> str:
        return GPTQ_QUANTIZATION_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "Ascend hardware does not support 'get_min_capability' feature."
        )

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GPTQConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        checkpoint_format = cls.get_from_keys_or(
            config, ["checkpoint_format"], default=""
        )
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        dynamic = {} if dynamic is None else dynamic
        lm_head_quantized = cls.get_from_keys_or(
            config, ["lm_head"], default=False
        )
        autoround_version = cls.get_from_keys_or(
            config, ["autoround_version"], default=""
        )
        modules_in_block_to_quantize = cls.get_from_keys_or(
            config, ["modules_in_block_to_quantize"], default=None
        )
        return cls(
            weight_bits=weight_bits,
            group_size=group_size,
            desc_act=desc_act,
            checkpoint_format=checkpoint_format,
            dynamic=dynamic,
            lm_head_quantized=lm_head_quantized,
            modules_in_block_to_quantize=modules_in_block_to_quantize,
            autoround_version=autoround_version,
            quant_config=config,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Union["LinearMethodBase", "QuantizeMethodBase"] | None:
        if isinstance(layer, LinearBase):
            # Only check skip when modules_in_block_to_quantize is populated
            # (requires maybe_update_config to have run). When empty, assume
            # all Linear layers are quantized — matching upstream GPTQ behavior.
            # Only skip when modules_in_block_to_quantize is populated AND
            # this layer is NOT in the list (i.e., NOT quantized).
            # Note: is_layer_skipped returns True when layer IS in the list,
            # so we invert it: skip when the layer is NOT in the list.
            if self.modules_in_block_to_quantize and not is_layer_skipped(
                prefix,
                self.modules_in_block_to_quantize,
                self.packed_modules_mapping,
                skip_with_substr=True,
            ):
                return AscendUnquantizedLinearMethod()
            # Pattern A: lookup scheme from registry and wrap with adapter
            if self.weight_bits == 4:
                scheme_name = "W4A16_GPTQ"
            elif self.weight_bits == 8:
                scheme_name = "W8A16_GPTQ"
            else:
                raise NotImplementedError(
                    f"GPTQ with {self.weight_bits}-bit weights is not "
                    f"supported on Ascend NPU."
                )
            scheme_cls = get_scheme_class(scheme_name, "linear")
            if scheme_cls is None:
                raise NotImplementedError(
                    f"{scheme_name} linear scheme not found for layer {prefix}"
                )
            return AscendLinearMethod(scheme_cls(self))

        elif isinstance(layer, FusedMoE):
            if self.modules_in_block_to_quantize and is_layer_skipped(
                prefix,
                self.modules_in_block_to_quantize,
                skip_with_substr=True,
            ):
                return AscendUnquantizedFusedMoEMethod(layer.moe_config)
            # Determine quant_type based on weight_bits
            if self.weight_bits == 4:
                scheme_name = "W4A16_GPTQ"
            elif self.weight_bits == 8:
                scheme_name = "W8A16_GPTQ"
            else:
                raise NotImplementedError(
                    f"GPTQ MoE with {self.weight_bits}-bit weights is not "
                    f"supported on Ascend NPU."
                )
            scheme_cls = get_scheme_class(scheme_name, "moe")
            if scheme_cls is None:
                raise NotImplementedError(
                    f"{scheme_name} moe scheme not found for layer {prefix}"
                )
            return AscendFusedMoEMethod(scheme_cls(self), layer.moe_config)

        return None
