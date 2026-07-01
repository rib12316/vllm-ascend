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
layers through Ascend-specific scheme implementations (Ascend Scheme 框架):

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

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Union

import regex as re
import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.transformers_utils.config import get_safetensors_params_metadata
from vllm.utils.collection_utils import is_list_of

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.model_executor.models.utils import WeightsMapper
else:
    PretrainedConfig = None
    WeightsMapper = None

from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.utils import GPTQ_QUANTIZATION_METHOD

from .method_adapters import AscendFusedMoEMethod, AscendLinearMethod
from .methods import get_scheme_class


def get_dynamic_override(
    config: "GPTQConfig",
    layer_name: str,
    key: str | None = None,
    default_value: int | bool | None = None,
) -> dict | int | bool | None:
    """Return the per-module dynamic override for ``layer_name``.

    Ported from upstream ``vllm/model_executor/layers/quantization/utils/
    gptq_utils.py:get_dynamic_override``. ``config.dynamic`` maps a regex
    pattern (optionally prefixed with ``+:`` or ``-:``) to an override dict:

    - ``-:<regex>`` (negative match): returns ``False`` — the matched module
      is excluded from quantization entirely.
    - ``+:<regex>`` or an unprefixed ``<regex>`` (positive match, the
      default): returns the override dict (or a single field via ``key``),
      which overrides the base quant config for that module.

    With ``key=None`` the return distinguishes the three cases:
    ``False`` (skip) / ``dict`` (positive) / ``default_value`` (no match,
    ``None`` by default). When ``config.dynamic`` is empty the loop never
    runs and ``default_value`` is returned, so this is a no-op for the
    common case of standard GPTQ checkpoints that do not set ``dynamic``.
    """
    for pattern, pattern_dict in config.dynamic.items():
        # Negative match: matched modules are excluded from quantized init.
        if pattern.startswith("-:"):
            if re.match(pattern.removeprefix("-:"), layer_name):
                return False
        # Positive match (explicit "+:" or unprefixed): matched modules have
        # quant properties that override the base quant config.
        elif re.match(pattern.removeprefix("+:"), layer_name):
            if key is None:
                return pattern_dict
            return pattern_dict.get(key, default_value)
    return default_value


def _override_config(config: "GPTQConfig", prefix: str) -> None:
    """Apply ``+:`` positive dynamic overrides to ``config`` in place.

    Ported from upstream ``override_config`` (the non-Marlin ``gptq`` branch).
    Mutates ``weight_bits`` / ``group_size`` / ``desc_act`` / ``pack_factor``
    only when the corresponding field is present in the matched rule, then
    re-validates ``weight_bits`` with the same rules as
    ``GPTQConfig.__init__`` (2/3-bit -> NotImplementedError; other
    unsupported widths -> ValueError), so a bad per-layer override fails
    fast instead of producing a malformed scheme.
    """
    weight_bits = get_dynamic_override(config, prefix, "bits", config.weight_bits)
    if isinstance(weight_bits, int):
        config.weight_bits = weight_bits
    group_size = get_dynamic_override(config, prefix, "group_size", config.group_size)
    if isinstance(group_size, int):
        config.group_size = group_size
    desc_act = get_dynamic_override(config, prefix, "desc_act", config.desc_act)
    if isinstance(desc_act, bool):
        config.desc_act = desc_act

    config.pack_factor = 32 // config.weight_bits  # packed into int32
    if config.weight_bits not in (2, 3, 4, 8):
        raise ValueError(
            f"Only 2/3/4/8-bit weight quantization is supported for GPTQ on "
            f"Ascend, but the dynamic override on '{prefix}' set "
            f"bits={config.weight_bits}."
        )
    if config.weight_bits in (2, 3):
        raise NotImplementedError(
            f"GPTQ with {config.weight_bits}-bit weights is not yet supported "
            f"on Ascend NPU (dynamic override on '{prefix}'). The NPU "
            f"quantization kernels only support 4-bit and 8-bit packing."
        )


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
        if weight_bits in (2, 3):
            raise NotImplementedError(
                f"GPTQ with {weight_bits}-bit weights is not yet supported on "
                f"Ascend NPU. The NPU quantization kernels "
                f"(npu_weight_quant_batchmatmul) only support 4-bit and 8-bit "
                f"weight packing. Please use a 4-bit or 8-bit GPTQ model instead."
            )
        self.weight_bits = weight_bits
        if group_size <= 0:
            raise ValueError(
                f"GPTQ group_size must be a positive integer on Ascend NPU, "
                f"but got {group_size}. Per-channel quantization "
                f"(group_size=-1) is not supported because the NPU operator "
                f"npu_weight_quant_batchmatmul requires a positive group_size."
            )
        self.group_size = group_size
        self.desc_act = desc_act
        self.checkpoint_format = checkpoint_format
        self.dynamic = dynamic or {}
        self.lm_head_quantized = lm_head_quantized
        # Stored as-is; flattening and auto-detection happen in
        # maybe_update_config (port of upstream GPTQConfig.maybe_update_config).
        # Some models (e.g. TheBloke) store nested list[list[str]] which
        # maybe_update_config will flatten.
        self.modules_in_block_to_quantize = modules_in_block_to_quantize or []
        self.autoround_version = autoround_version

        self.pack_factor = 32 // weight_bits

        # v2 format flag
        self.use_v2_format = checkpoint_format == "gptq_v2"

    def __repr__(self) -> str:
        return (
            f"GPTQConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act}, "
            f"lm_head_quantized={self.lm_head_quantized}, "
            f"dynamic={self.dynamic}, "
            f"modules_in_block_to_quantize={self.modules_in_block_to_quantize}, "
            f"checkpoint_format={self.checkpoint_format})"
        )

    def get_name(self) -> str:
        return GPTQ_QUANTIZATION_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError("Ascend hardware does not support 'get_min_capability' feature.")

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GPTQConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        checkpoint_format = cls.get_from_keys_or(config, ["checkpoint_format"], default="")
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        dynamic = {} if dynamic is None else dynamic
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        autoround_version = cls.get_from_keys_or(config, ["autoround_version"], default="")
        modules_in_block_to_quantize = cls.get_from_keys_or(config, ["modules_in_block_to_quantize"], default=None)
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

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        """Translate HF module names in modules_in_block_to_quantize to vLLM names.

        Called by the model loader after the HF→vLLM name mapping is known,
        so that quantize-list entries match the vLLM parameter names used in
        ``get_quant_method`` prefix checks.
        """
        if self.modules_in_block_to_quantize is not None:
            self.modules_in_block_to_quantize = hf_to_vllm_mapper.apply_list(self.modules_in_block_to_quantize)

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: "PretrainedConfig | None" = None,
        revision: str | None = None,
    ):
        """Flatten nested modules_in_block_to_quantize and auto-detect quantized layers.

        This is the Ascend port of upstream ``GPTQConfig.maybe_update_config``
        (vllm/vllm/.../quantization/gptq.py:197-222).

        Two-phase logic (matching upstream):
        1. If ``modules_in_block_to_quantize`` is already populated, flatten
           any nested ``list[list[str]]`` → ``list[str]`` (some models like
           TheBloke store nested lists) and return.
        2. If empty, auto-detect quantized layers by inspecting safetensors
           metadata: any parameter whose dtype is NOT fp16/bf16/fp32 is
           considered quantized.
        """
        if self.modules_in_block_to_quantize:
            if is_list_of(self.modules_in_block_to_quantize, list):
                # original modules_in_block_to_quantize: list[list[str]]
                # flatten to list[str]
                self.modules_in_block_to_quantize = [
                    item for sublist in self.modules_in_block_to_quantize for item in sublist
                ]
            return

        unquant_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        metadata = get_safetensors_params_metadata(model_name, revision=revision)
        quant_layers: set[str] = {
            param_name.rsplit(".", 1)[0]
            for param_name, info in metadata.items()
            if (dtype := info.get("dtype", None)) and _SAFETENSORS_TO_TORCH_DTYPE[dtype] not in unquant_dtypes
        }
        self.modules_in_block_to_quantize = list(quant_layers)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str, tid2eid: dict[int, int] | None = None
    ) -> Union["LinearMethodBase", "QuantizeMethodBase"] | None:
        # Handle lm_head: ParallelLMHead is NOT a LinearBase subclass,
        # so we must explicitly check it when lm_head_quantized=True.
        # Upstream does this in get_linear_quant_method (gptq_utils.py).
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            UnquantizedEmbeddingMethod,
        )

        parallel_lm_head_quantized = isinstance(layer, ParallelLMHead) and self.lm_head_quantized

        if isinstance(layer, LinearBase) or parallel_lm_head_quantized:
            # Only check skip when modules_in_block_to_quantize is populated
            # (requires maybe_update_config to have run). When empty, assume
            # all Linear layers are quantized — matching upstream GPTQ behavior.
            # Only skip when modules_in_block_to_quantize is populated AND
            # this layer is NOT in the list (i.e., NOT quantized).
            # Note: is_layer_skipped returns True when layer IS in the list,
            # so we invert it: skip when the layer is NOT in the list.
            not_in_quant_list = self.modules_in_block_to_quantize and not is_layer_skipped(
                prefix,
                self.modules_in_block_to_quantize,
                self.packed_modules_mapping,
                skip_with_substr=True,
            )
            # GPTQ dynamic config (R7/G18): a "-:<regex>" rule forces a module
            # OUT of quantization even when it appears in the quantize list;
            # a "+:<regex>" rule (handled below) overrides the base config per
            # module. When ``dynamic`` is empty — the default for all standard
            # GPTQ checkpoints — ``get_dynamic_override`` is not even called,
            # so this whole block is a no-op and routing is unchanged.
            # Ported from upstream get_dynamic_override / get_linear_quant_method.
            dyn = get_dynamic_override(self, prefix) if self.dynamic else None
            # dyn: None = no rule matched; False = negative match (skip);
            # dict = positive match (override).
            if not_in_quant_list or dyn is False:
                if parallel_lm_head_quantized:
                    return UnquantizedEmbeddingMethod()
                return AscendUnquantizedLinearMethod()
            # A positive "+:" rule overrides the base config for this module
            # (e.g. mixing 4-bit and 8-bit across layers). Apply on a deep copy
            # so the shared base config (used by every other layer) is untouched.
            if dyn:
                quant_config = deepcopy(self)
                _override_config(quant_config, prefix)
            else:
                quant_config = self
            # Ascend Scheme 框架: lookup scheme from registry and wrap with adapter.
            # Scheme selection honors an overridden weight_bits (4 vs 8).
            if quant_config.weight_bits == 4:
                scheme_name = "W4A16_GPTQ"
            elif quant_config.weight_bits == 8:
                scheme_name = "W8A16_GPTQ"
            else:
                raise NotImplementedError(
                    f"GPTQ with {quant_config.weight_bits}-bit weights is not supported on Ascend NPU."
                )
            scheme_cls = get_scheme_class(scheme_name, "linear")
            if scheme_cls is None:
                raise NotImplementedError(f"{scheme_name} linear scheme not found for layer {prefix}")
            return AscendLinearMethod(scheme_cls(quant_config))

        elif isinstance(layer, FusedMoE):
            # Decide whether this FusedMoE is quantized. GPTQ quantizes expert
            # weights uniformly across MoE layers, so the MoE is quantized iff
            # the quantize list contains ANY expert submodule (e.g.
            # "mlp.experts.0.up_proj"). We must NOT use is_layer_skipped() with
            # skip_with_substr here: it does one-directional substring matching
            # (entry-in-prefix), but the MoE layer name ("mlp.experts") is the
            # PARENT of the expert entries, so it never matches and the layer is
            # wrongly returned as unquantized (upstream GPTQ in fact quantizes
            # every FusedMoE unconditionally).
            has_quantized_experts = any("experts" in q for q in self.modules_in_block_to_quantize)
            if self.modules_in_block_to_quantize and not has_quantized_experts:
                return AscendUnquantizedFusedMoEMethod(layer.moe_config)
            # Determine quant_type based on weight_bits
            if self.weight_bits == 4:
                scheme_name = "W4A16_GPTQ"
            elif self.weight_bits == 8:
                scheme_name = "W8A16_GPTQ"
            else:
                raise NotImplementedError(
                    f"GPTQ MoE with {self.weight_bits}-bit weights is not supported on Ascend NPU."
                )
            scheme_cls = get_scheme_class(scheme_name, "moe")
            if scheme_cls is None:
                raise NotImplementedError(f"{scheme_name} moe scheme not found for layer {prefix}")
            return AscendFusedMoEMethod(scheme_cls(self), layer.moe_config, tid2eid)

        return None
