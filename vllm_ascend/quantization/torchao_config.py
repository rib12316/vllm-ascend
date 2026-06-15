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
"""torchao quantization config for Ascend NPU.

This config replaces vLLM's native ``TorchAOConfig`` so that
``--quantization torchao`` is routed through Ascend NPU operators (Pattern A):

- **Linear layers** → ``AscendW4A16TorchAOLinearScheme`` /
  ``AscendW8A16TorchAOLinearScheme`` (registered via ``@register_scheme``,
  dispatched through the ``AscendLinearMethod`` adapter)
- **Skipped layers** (e.g. lm_head) → ``AscendUnquantizedLinearMethod``

Key design point: torchao is used **only at load time** to produce the integer
weight tensors (``tensor_impl.get_plain()`` → data/scale/zero_point); at forward
time the weights run through ``npu_weight_quant_batchmatmul`` exactly like
AWQ/GPTQ — torchao is never on the hot path.

int4wo is *self-implemented* (per-group symmetric int4 RTN) because torchao 0.17's
standard int4 path requires ``mslk`` (a CUDA/H100-only kernel) unavailable on NPU;
its numerics match torchao int4wo and are validated against ``.dequantize()``.
"""

from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.utils import TORCHAO_QUANTIZATION_METHOD

from .method_adapters import AscendLinearMethod
from .methods import get_scheme_class

# torchao quant-type shorthand → Ascend scheme registry key.
_TORCHAO_SCHEME_KEY = {
    "int4wo": "W4A16_TORCHAO",
    "int8wo": "W8A16_TORCHAO",
    "fp8wo": "FP8W_TORCHAO",
}

_DEFAULT_TORCHAO_GROUP_SIZE = 128


@register_quantization_config(TORCHAO_QUANTIZATION_METHOD)
class TorchAOConfig(QuantizationConfig):
    """torchao quantization config for Ascend NPU.

    Registered as ``"torchao"``, this overrides vLLM's native torchao config so
    that torchao models are routed through Ascend NPU operators instead of
    torchao's CUDA (mslk/tinygemm) kernels.
    """

    def __init__(
        self,
        torchao_quant_type: str,
        group_size: int = _DEFAULT_TORCHAO_GROUP_SIZE,
        is_checkpoint_torchao_serialized: bool = False,
        modules_to_not_convert: list[str] | None = None,
        quant_config: dict[str, Any] | None = None,
    ):
        self.quant_description = quant_config if quant_config is not None else {}
        super().__init__()

        self.torchao_quant_type = torchao_quant_type  # "int4wo"/"int8wo"/"fp8wo"
        self.group_size = group_size
        self.is_checkpoint_torchao_serialized = is_checkpoint_torchao_serialized
        self.modules_to_not_convert = modules_to_not_convert or []

        if torchao_quant_type not in _TORCHAO_SCHEME_KEY:
            raise ValueError(
                f"Unsupported torchao quant type on Ascend NPU: {torchao_quant_type!r}. "
                f"Supported: {sorted(_TORCHAO_SCHEME_KEY)}."
            )
        if self.group_size <= 0:
            raise ValueError(
                f"torchao group_size must be a positive integer on Ascend NPU, "
                f"got {self.group_size} (npu_weight_quant_batchmatmul requires it)."
            )

    def get_name(self) -> str:
        return TORCHAO_QUANTIZATION_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError("Ascend hardware does not support 'get_min_capability' feature.")

    @staticmethod
    def get_config_filenames() -> list[str]:
        # torchao reads the HF config.json quantization_config; no extra file.
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TorchAOConfig":
        """Build the config from an HF model ``quantization_config`` dict.

        Mirrors upstream ``TorchAOConfig.from_config``: reads ``quant_method``
        (to detect a torchao-serialized checkpoint) and ``quant_type.default``
        (the torchao quant type, e.g. ``"int8wo"`` / ``"int4wo-g128"``).
        """
        # Online quantization of a dense checkpoint is the default. A torchao-
        # serialized (pre-quantized) checkpoint is opt-in via an explicit field:
        # "torchao" in quant_method is set for online quant too, so it cannot
        # distinguish the two (and would wrongly trigger vLLM's torchao
        # safetensors strategy on a dense checkpoint → "No tensors found").
        is_checkpoint_torchao_serialized = bool(config.get("is_checkpoint_torchao_serialized", False))

        hf_config = cls.get_from_keys_or(config, ["quant_type"], None)
        if hf_config is None:
            raise ValueError(
                "torchao quant_type must be specified in the model config "
                "(quantization_config.quant_type.default), e.g. 'int4wo'/'int8wo'/'fp8wo'."
            )
        if isinstance(hf_config, dict):
            assert len(hf_config) == 1 and "default" in hf_config, (
                "Expected only one key 'default' in quant_type dictionary"
            )
            quant_type = hf_config["default"]
        else:
            quant_type = hf_config

        torchao_quant_type, group_size = _parse_torchao_quant_type(quant_type)

        modules_to_not_convert = config.get("modules_to_not_convert", []) or []
        return cls(
            torchao_quant_type,
            group_size=group_size,
            is_checkpoint_torchao_serialized=is_checkpoint_torchao_serialized,
            modules_to_not_convert=list(modules_to_not_convert),
            quant_config=config,
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> LinearMethodBase | QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if _is_layer_skipped(prefix, self.modules_to_not_convert):
            return AscendUnquantizedLinearMethod()

        scheme_key = _TORCHAO_SCHEME_KEY[self.torchao_quant_type]
        scheme_cls = get_scheme_class(scheme_key, "linear")
        if scheme_cls is None:
            raise NotImplementedError(f"{scheme_key} linear scheme not registered for layer {prefix}")
        return AscendLinearMethod(scheme_cls(self))


def _parse_torchao_quant_type(quant_type: Any) -> tuple[str, int]:
    """Map a torchao quant_type to ``(ascend_key, group_size)``.

    Accepts the common shorthand strings ``int4wo`` / ``int8wo`` / ``fp8wo``,
    optionally suffixed with ``-g<group_size>`` (e.g. ``int4wo-g128``), and the
    dict payload form emitted by ``torchao.config_to_dict``.
    """
    if isinstance(quant_type, str):
        qt = quant_type.strip().lower()
        group_size = _DEFAULT_TORCHAO_GROUP_SIZE
        if "-g" in qt:
            base, gs = qt.split("-g", 1)
            group_size = int(gs)
            qt = base
        if qt not in _TORCHAO_SCHEME_KEY:
            raise ValueError(f"Unsupported torchao quant type string: {quant_type!r}")
        return qt, group_size

    if isinstance(quant_type, dict):
        name = str(quant_type.get("name") or quant_type.get("_type") or "").lower()
        group_size = int(quant_type.get("group_size", _DEFAULT_TORCHAO_GROUP_SIZE))
        if "int4" in name:
            return "int4wo", group_size
        if "int8" in name:
            return "int8wo", group_size
        if "float8" in name or "fp8" in name:
            return "fp8wo", group_size

    raise ValueError(
        f"Cannot determine torchao quant type from: {quant_type!r}. "
        f"Use one of {sorted(_TORCHAO_SCHEME_KEY)} (optionally '-g<group_size>')."
    )


def _is_layer_skipped(prefix: str, skip_modules: list[str]) -> bool:
    """True if a layer prefix matches a skip entry (exact or path-segment match)."""
    dotted = f".{prefix}."
    return any(prefix == s or f".{s}." in dotted for s in skip_modules)
