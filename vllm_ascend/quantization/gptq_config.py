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
layers through Ascend-specific scheme implementations:

- **Linear layers** → ``AscendGPTQLinearMethod`` (delegates weight creation to
  vLLM's ``GPTQLinearMethod``, overrides process/apply for NPU)
- **MoE layers** → ``AscendW4A16GPTQFusedMoEMethod`` (registered scheme)
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

from .method_adapters import AscendFusedMoEMethod
from .methods import get_scheme_class
from .methods.gptq import AscendGPTQLinearMethod


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
        self.modules_in_block_to_quantize = modules_in_block_to_quantize or []
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

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None = None,
        hf_config: Any = None,
    ) -> str | None:
        """Prevent vLLM from auto-upgrading GPTQ to GPTQ-Marlin.

        The upstream ``GPTQMarlinConfig.override_quantization_method`` would
        return ``"gptq_marlin"`` for compatible models. Since Ascend NPU does
        not support Marlin, we intercept and keep ``"gptq"``.

        Args:
            hf_quant_cfg: The checkpoint's quantization config dict.
            user_quant: The user-specified quantization method string.
            hf_config: The HuggingFace model config object (may be None).
        """
        return GPTQ_QUANTIZATION_METHOD

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Union["LinearMethodBase", "QuantizeMethodBase"] | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix,
                self.modules_in_block_to_quantize,
                self.packed_modules_mapping,
                skip_with_substr=True,
            ):
                return AscendUnquantizedLinearMethod()
            return AscendGPTQLinearMethod(self)

        elif isinstance(layer, FusedMoE):
            if is_layer_skipped(
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
