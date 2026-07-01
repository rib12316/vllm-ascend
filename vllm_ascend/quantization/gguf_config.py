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
"""GGUF quantization config for Ascend NPU.

This config overrides vLLM's native ``GGUFConfig`` so ``--quantization gguf``
loads ``.gguf`` models on Ascend NPU via a dedicated ``AscendGGUFLinearMethod``
that takes one of two paths per block type:

- **High-perf path** (Q8_0/Q4_0/Q4_1): ``methods/gguf_repack.py`` repacks the
  block into ``npu_weight_quant_batchmatmul``'s per-group(32) int8/int4 format,
  so weights stay quantized at runtime (memory saving + fused dequant+matmul).
- **Dense fallback** (K-quants Q4_K/Q5_K/Q6_K and others): pure-torch dequant to
  dense fp16/bf16 at load time (``methods/gguf_dequant.py``), then ``F.linear``.
  Correctness-first; the 6-bit super-block scales cannot map to the NPU op.

Unlike AWQ/GPTQ, GGUF uses the ``is_gguf_weight`` / ``is_gguf_weight_type``
loader contract, so it does NOT route through the ``AscendLinearScheme``
registry; the config returns ``AscendGGUFLinearMethod`` directly.
"""

from typing import Any

import torch
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.gguf import is_layer_skipped_gguf
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

from vllm_ascend.utils import GGUF_QUANTIZATION_METHOD

from .methods.gguf import AscendGGUFEmbeddingMethod, AscendGGUFLinearMethod


@register_quantization_config(GGUF_QUANTIZATION_METHOD)
class GGUFConfig(QuantizationConfig):
    """GGUF quantization config for Ascend NPU.

    Registered as ``"gguf"``, overriding vLLM's native config. ``from_config``
    takes no parameters (the per-tensor block type is carried by each weight's
    ``qweight_type``, resolved at load time by the dequant kernels).
    """

    def __init__(self, unquantized_modules: list[str] | None = None) -> None:
        super().__init__()
        self.quant_description = {}
        self.unquantized_modules = unquantized_modules or []

    def __repr__(self) -> str:
        return "GGUFConfig()"

    def get_name(self) -> str:
        return GGUF_QUANTIZATION_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        # Dequant kernels produce fp16/bf16; the dense linear runs in half precision.
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError("Ascend hardware does not support 'get_min_capability' feature.")

    @staticmethod
    def get_config_filenames() -> list[str]:
        # GGUF has no separate quant config file; the block type is per-tensor.
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GGUFConfig":
        return cls()

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg: dict[str, Any], user_quant: str | None, hf_config=None
    ) -> str | None:
        # When the user explicitly passes --quantization gguf, take precedence
        # over whatever quant method is declared in the HF model config.
        if user_quant == GGUF_QUANTIZATION_METHOD:
            return GGUF_QUANTIZATION_METHOD
        return None

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        if self.unquantized_modules:
            self.unquantized_modules = hf_to_vllm_mapper.apply_list(self.unquantized_modules)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped_gguf(prefix, self.unquantized_modules, self.packed_modules_mapping):
                from vllm.model_executor.layers.linear import UnquantizedLinearMethod

                return UnquantizedLinearMethod()
            return AscendGGUFLinearMethod(self)
        elif isinstance(layer, VocabParallelEmbedding):
            # lm_head / embed_tokens: dequant to dense at load (same as linear),
            # then a plain embedding lookup. GGUF always pairs qweight with a
            # qweight_type, so the method must create both params.
            if is_layer_skipped_gguf(prefix, self.unquantized_modules, self.packed_modules_mapping):
                return UnquantizedEmbeddingMethod()
            return AscendGGUFEmbeddingMethod(self)
        elif isinstance(layer, FusedMoE):
            # MoE dequant-at-load is G-10; not in the MVP.
            raise NotImplementedError(
                "GGUF MoE on Ascend NPU is not yet supported (planned: dequant "
                "experts → fused_experts). Please use a non-MoE GGUF model."
            )
        return None
