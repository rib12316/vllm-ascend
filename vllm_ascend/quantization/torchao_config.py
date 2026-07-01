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
``--quantization torchao`` is routed through Ascend NPU operators (Ascend Scheme 框架):

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

Why override the native config (same Ascend Scheme 框架 as AWQ/GPTQ): torchao is an external
PyTorch library whose native vLLM path delegates quant+matmul to the torchao
library's CUDA kernels (tinygemm/mslk). On NPU those kernels are unavailable, and
the native config is a thin shell with nothing NPU-adjustable inside it — so
replacing the config is the natural granularity, exactly as for AWQ/GPTQ.

Scope: int8wo/fp8wo/int4wo are *online* quantization of a dense checkpoint. Loading
a *pre-quantized* torchao int4 checkpoint is **out of MVP scope** (T-10): its packed
serialization format is not yet handled. int8/fp8 pre-quantized checkpoints are
tractable via torchao ``unflatten_tensor_state_dict``.

Per-layer overrides (T-12): ``module_fqn_to_config`` (under
``quant_type._data``) maps specific layer fqns — or ``re:`` regex patterns, with a
``_default`` fallback — to different quant types; ``None``/unmatched layers stay
dense, mirroring upstream ``ModuleFqnToConfig``. ``autoquant`` is rejected up front
(it needs runtime CUDA-kernel benchmarking, unavailable on NPU).
"""

import copy
import re
from typing import Any

import torch
from vllm.model_executor.layers.fused_moe import FusedMoE
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

# Sentinel for "no ``_default`` key in module_fqn_to_config" (distinct from an
# explicit ``None`` entry, which also means dense).
_UNSET: Any = object()


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
        is_prequant_checkpoint: bool = False,
        modules_to_not_convert: list[str] | None = None,
        module_fqn_to_config: dict[str, Any] | None = None,
        quant_config: dict[str, Any] | None = None,
    ):
        self.quant_description = quant_config if quant_config is not None else {}
        super().__init__()

        self.torchao_quant_type = torchao_quant_type  # "int4wo"/"int8wo"/"fp8wo"
        self.group_size = group_size
        self.is_checkpoint_torchao_serialized = is_checkpoint_torchao_serialized
        # Flat-tensor pre-quantized checkpoint (T-10): int8 data + scale are
        # loaded directly (no online re-quant). Distinct from
        # is_checkpoint_torchao_serialized (which triggers vLLM's native torchao
        # AQTensor loader). Read from the checkpoint's quantization_config.
        self.is_prequant_checkpoint = is_prequant_checkpoint
        self.modules_to_not_convert = modules_to_not_convert or []
        # Per-layer override map (T-12): ``{fqn_or_regex: torchao_cfg_dict | None}``.
        # Mirrors upstream ``ModuleFqnToConfig``: an entry value of ``None`` (or an
        # unmatched layer when no ``_default`` key is present) leaves that layer
        # dense. Resolution order in ``get_quant_method``: exact fqn → first
        # ``re:``-prefixed regex full-match → ``_default`` → global default.
        self.module_fqn_to_config: dict[str, Any] = module_fqn_to_config or {}

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
        # Flat-tensor pre-quantized checkpoint (T-10): the model's config.json
        # quantization_config sets ``"prequant": true``. Loaded with vLLM's
        # default loader (NOT the native torchao AQTensor loader).
        is_prequant_checkpoint = bool(config.get("prequant", False))

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
        # Per-layer overrides live inside the torchao quant_type payload, under
        # ``_data.module_fqn_to_config`` (the same place upstream's
        # ``config_from_dict`` sources ``ModuleFqnToConfig``).
        _data = quant_type.get("_data", {}) if isinstance(quant_type, dict) else {}
        module_fqn_to_config = _data.get("module_fqn_to_config", {}) if isinstance(_data, dict) else {}
        return cls(
            torchao_quant_type,
            group_size=group_size,
            is_checkpoint_torchao_serialized=is_checkpoint_torchao_serialized,
            is_prequant_checkpoint=is_prequant_checkpoint,
            modules_to_not_convert=list(modules_to_not_convert),
            module_fqn_to_config=dict(module_fqn_to_config),
            quant_config=config,
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> LinearMethodBase | QuantizeMethodBase | None:
        if isinstance(layer, FusedMoE):
            # MoE expert quantization is out of MVP scope (T-11): it would need
            # per-expert dequant/repack into ``fused_experts`` (like AWQ/GPTQ MoE).
            # Raise explicitly rather than returning ``None`` so a torchao-quantized
            # MoE model fails loudly instead of silently falling back to dense
            # experts (mirrors gguf_config.py's GGUF MoE guard).
            raise NotImplementedError(
                "torchao MoE on Ascend NPU is not yet supported (planned: dequant/"
                "repack experts -> fused_experts). Please use a non-MoE model, or use "
                "--quantization gptq/awq which support MoE."
            )
        if not isinstance(layer, LinearBase):
            return None
        if _is_layer_skipped(prefix, self.modules_to_not_convert):
            return AscendUnquantizedLinearMethod()

        resolved = self._resolve_fqn(prefix)
        if resolved is None:
            # Per-layer override says dense: explicit ``None`` entry, or an
            # unmatched layer with no ``_default`` (mirrors upstream's
            # ``UnquantizedLinearMethod`` fallback inside ModuleFqnToConfig).
            return AscendUnquantizedLinearMethod()

        torchao_quant_type, group_size = resolved
        scheme_key = _TORCHAO_SCHEME_KEY[torchao_quant_type]
        scheme_cls = get_scheme_class(scheme_key, "linear")
        if scheme_cls is None:
            raise NotImplementedError(f"{scheme_key} linear scheme not registered for layer {prefix}")
        # Schemes read only ``quant_config.group_size``; when a per-layer
        # override changes it, build the scheme against a shallow copy carrying
        # the resolved value (mirrors upstream's per-layer TorchAOConfig).
        if group_size != self.group_size or torchao_quant_type != self.torchao_quant_type:
            per_layer = copy.copy(self)
            per_layer.torchao_quant_type = torchao_quant_type
            per_layer.group_size = group_size
            return AscendLinearMethod(scheme_cls(per_layer))
        return AscendLinearMethod(scheme_cls(self))

    def _resolve_fqn(self, prefix: str) -> tuple[str, int] | None:
        """Resolve ``(quant_type, group_size)`` for ``prefix``.

        Returns ``None`` when the layer should stay dense: an explicit ``None``
        entry in ``module_fqn_to_config``, or an unmatched layer when no
        ``_default`` key is present. When ``module_fqn_to_config`` is empty the
        global default applies to every layer.
        """
        m = self.module_fqn_to_config
        if not m:
            return self.torchao_quant_type, self.group_size

        if prefix in m:
            cfg = m[prefix]
        else:
            for pattern, value in m.items():
                if pattern.startswith("re:") and re.fullmatch(pattern[3:], prefix):
                    cfg = value
                    break
            else:
                cfg = m.get("_default", _UNSET)

        if cfg is _UNSET or cfg is None:
            return None
        return _parse_torchao_quant_type(cfg)


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
        if "autoquant" in name or "auto_quant" in name:
            # torchao autoquant benchmarks candidate configs against CUDA kernels
            # at runtime; it has no NPU equivalent and cannot map to the fixed
            # int4/int8/fp8 Ascend schemes. Reject up front with a clear message
            # (consistent with the GPTQ 2/3-bit前置拒绝, T15).
            raise NotImplementedError(
                "torchao 'autoquant' is not supported on Ascend NPU: it requires "
                "runtime CUDA-kernel benchmarking. Use an explicit quant_type "
                "('int4wo'/'int8wo'/'fp8wo'), optionally per-layer via "
                "module_fqn_to_config."
            )
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
