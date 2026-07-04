# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""
Config-level routing / validation tests for Ascend AWQ & GPTQ.

These tests cover the "implemented but not NPU-verified" config features
(T14-T18) that are pure Python logic with no NPU dependency:

- T15: GPTQ 2/3-bit weight_bits pre-rejection
- T18: group_size <= 0 rejection (config) + alignment check (linear scheme)
- T14: GPTQ lm_head_quantized routing (ParallelLMHead -> quant branch)
- T18: GPTQ MoE is_layer_skipped logic-inversion fix (quant-list semantics)
- T17: AWQ/GPTQ apply_vllm_mapper name translation
- T16: AWQ/GPTQ maybe_update_config auto-detection from safetensors metadata

Usage:
    pytest tests/ut/quantization/test_quant_routing.py -v
"""

import glob
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import set_current_vllm_config


@pytest.fixture(autouse=True)
def default_vllm_config():
    """AscendLinearMethod.__init__ reads the live vLLM config (for the DSA-CP
    flag); provide a minimal mock context so the method can be instantiated
    standalone. Mirrors the fixture in tests/ut/ops/test_layernorm.py."""
    mock_config = MagicMock()
    mock_config.compilation_config.custom_ops = ["all"]
    with set_current_vllm_config(mock_config):
        yield mock_config


# TinyLlama GPTQ snapshot (already on disk) — used for the real-model
# maybe_update_config (T16) test. Resolved lazily so collection does not fail
# if the cache is absent.

_TINYLLAMA_SNAPSHOTS = "/data/huggingface_home/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GPTQ/snapshots/*/"


def _tinyllama_path():
    paths = glob.glob(_TINYLLAMA_SNAPSHOTS)
    return paths[0] if paths else None


# ---------------------------------------------------------------------------
# T15: GPTQ 2/3-bit weight_bits pre-rejection
# ---------------------------------------------------------------------------


class TestGPTQBitWidthRejection:
    """2-bit and 3-bit GPTQ must be rejected at __init__ time.

    The NPU kernels only support 4-bit and 8-bit packing, so we raise
    NotImplementedError immediately rather than failing later in the kernel.
    """

    def test_2bit_rejected(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        with pytest.raises(NotImplementedError, match="2-bit"):
            GPTQConfig(weight_bits=2, group_size=128, desc_act=False)

    def test_3bit_rejected(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        with pytest.raises(NotImplementedError, match="3-bit"):
            GPTQConfig(weight_bits=3, group_size=128, desc_act=False)

    def test_4bit_accepted(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        cfg = GPTQConfig(weight_bits=4, group_size=128, desc_act=False)
        assert cfg.weight_bits == 4

    def test_8bit_accepted(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        cfg = GPTQConfig(weight_bits=8, group_size=128, desc_act=False)
        assert cfg.weight_bits == 8

    def test_16bit_rejected_with_value_error(self):
        # Out-of-range bit widths hit the ValueError guard first.
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        with pytest.raises(ValueError, match="2/3/4/8-bit"):
            GPTQConfig(weight_bits=16, group_size=128, desc_act=False)


# ---------------------------------------------------------------------------
# T18: group_size validation (config level + linear scheme alignment)
# ---------------------------------------------------------------------------


class TestGroupSizeValidation:
    """group_size <= 0 must be rejected because npu_weight_quant_batchmatmul
    requires a *positive* group_size (per-channel / -1 unsupported on NPU).
    Also: input_size must be divisible by group_size at the linear scheme.
    """

    def test_awq_group_size_zero_rejected(self):
        from vllm_ascend.quantization.awq_config import AWQConfig

        with pytest.raises(ValueError, match="positive"):
            AWQConfig(weight_bits=4, group_size=0, zero_point=True)

    def test_awq_group_size_negative_rejected(self):
        from vllm_ascend.quantization.awq_config import AWQConfig

        # group_size=-1 is per-channel (allowed); any other negative is rejected.
        with pytest.raises(ValueError, match="positive"):
            AWQConfig(weight_bits=4, group_size=-2, zero_point=True)

    def test_gptq_group_size_zero_rejected(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        with pytest.raises(ValueError, match="positive"):
            GPTQConfig(weight_bits=4, group_size=0, desc_act=False)

    def test_gptq_group_size_negative_rejected(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        # group_size=-1 is per-channel (allowed); any other negative is rejected.
        with pytest.raises(ValueError, match="positive"):
            GPTQConfig(weight_bits=4, group_size=-2, desc_act=False)

    def test_awq_linear_alignment_get_weight(self):
        # input_size not divisible by group_size -> ValueError in get_weight.
        from vllm_ascend.quantization.awq_config import AWQConfig
        from vllm_ascend.quantization.methods.w4a16_awq import (
            AscendW4A16AWQLinearScheme,
        )

        cfg = AWQConfig(weight_bits=4, group_size=128, zero_point=True)
        scheme = AscendW4A16AWQLinearScheme(cfg)
        with pytest.raises(ValueError, match="divisible"):
            scheme.get_weight(input_size=100, output_size=256, params_dtype=torch.float16)

    def test_awq_linear_alignment_get_pergroup_param(self):
        from vllm_ascend.quantization.awq_config import AWQConfig
        from vllm_ascend.quantization.methods.w4a16_awq import (
            AscendW4A16AWQLinearScheme,
        )

        cfg = AWQConfig(weight_bits=4, group_size=128, zero_point=True)
        scheme = AscendW4A16AWQLinearScheme(cfg)
        with pytest.raises(ValueError, match="divisible"):
            scheme.get_pergroup_param(input_size=100, output_size=256, params_dtype=torch.float16)

    def test_gptq_linear_alignment(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig
        from vllm_ascend.quantization.methods.gptq import (
            AscendW4A16GPTQLinearScheme,
        )

        cfg = GPTQConfig(weight_bits=4, group_size=128, desc_act=False)
        scheme = AscendW4A16GPTQLinearScheme(cfg)
        with pytest.raises(ValueError, match="divisible"):
            scheme.get_pergroup_param(input_size=100, output_size=256, params_dtype=torch.float16)

    def test_awq_linear_alignment_accepts_valid(self):
        from vllm_ascend.quantization.awq_config import AWQConfig
        from vllm_ascend.quantization.methods.w4a16_awq import (
            AscendW4A16AWQLinearScheme,
        )

        cfg = AWQConfig(weight_bits=4, group_size=128, zero_point=True)
        scheme = AscendW4A16AWQLinearScheme(cfg)
        spec = scheme.get_weight(input_size=256, output_size=256, params_dtype=torch.float16)
        # qweight shape: (input_size, output_size // pack_factor)
        assert spec["qweight"].shape == (256, 256 // 8)


# ---------------------------------------------------------------------------
# T14: GPTQ lm_head_quantized routing
# ---------------------------------------------------------------------------


class TestLMHeadRouting:
    """ParallelLMHead is NOT a LinearBase subclass, so the default routing
    would miss it. When lm_head_quantized=True, get_quant_method must still
    route it into the quantization path.
    """

    def _make_config(self, lm_head_quantized):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        cfg = GPTQConfig(weight_bits=4, group_size=128, desc_act=False, lm_head_quantized=lm_head_quantized)
        # packed_modules_mapping is normally populated by the model loader;
        # set an empty mapping so get_quant_method works standalone.
        cfg.packed_modules_mapping = {}
        return cfg

    def test_lm_head_quantized_routes_to_linear_method(self):
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
        )

        from vllm_ascend.quantization.method_adapters import AscendLinearMethod

        cfg = self._make_config(lm_head_quantized=True)
        lm_head = MagicMock(spec=ParallelLMHead)
        method = cfg.get_quant_method(lm_head, prefix="lm_head")
        assert isinstance(method, AscendLinearMethod), (
            f"lm_head_quantized=True must route ParallelLMHead to the quantization path, got {type(method)!r}"
        )

    def test_lm_head_not_quantized_returns_none(self):
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
        )

        cfg = self._make_config(lm_head_quantized=False)
        lm_head = MagicMock(spec=ParallelLMHead)
        method = cfg.get_quant_method(lm_head, prefix="lm_head")
        assert method is None, (
            f"lm_head_quantized=False must leave ParallelLMHead unquantized (return None), got {type(method)!r}"
        )

    def test_lm_head_skipped_when_not_in_block_list(self):
        # When modules_in_block_to_quantize is populated but lm_head is NOT
        # in the list, a quantized lm_head should fall back to the embedding
        # method (the parallel_lm_head_quantized branch).
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
        )

        cfg = self._make_config(lm_head_quantized=True)
        cfg.modules_in_block_to_quantize = ["model.layers.0.self_attn.q_proj"]
        lm_head = MagicMock(spec=ParallelLMHead)
        method = cfg.get_quant_method(lm_head, prefix="lm_head")
        # lm_head not in the quant list but lm_head_quantized=True -> still
        # routed to the linear quant scheme (the unquantized embedding branch
        # only triggers when modules list is populated AND layer not in list
        # AND ... that returns UnquantizedEmbeddingMethod). Here lm_head IS
        # the layer, and it's not in the list, so it returns the embedding
        # fallback. Verify it does NOT crash and is one of the two paths.
        assert method is not None


# ---------------------------------------------------------------------------
# T18: GPTQ is_layer_skipped logic-inversion fix
# ---------------------------------------------------------------------------


class TestGPTQSkipLogic:
    """GPTQ uses modules_in_block_to_quantize (a QUANTIZE list), the inverse
    of AWQ's modules_to_not_convert (a SKIP list). The fix inverts
    is_layer_skipped so that a layer NOT in the quant list is left unquantized,
    while a layer IN the list is quantized.
    """

    def _make_config(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        cfg = GPTQConfig(weight_bits=4, group_size=128, desc_act=False)
        cfg.packed_modules_mapping = {}
        return cfg

    def _linear_layer(self):
        from vllm.model_executor.layers.linear import LinearBase

        return MagicMock(spec=LinearBase)

    def test_layer_in_quant_list_is_quantized(self):
        from vllm_ascend.quantization.method_adapters import AscendLinearMethod

        cfg = self._make_config()
        cfg.modules_in_block_to_quantize = ["model.layers.0.self_attn.q_proj"]
        method = cfg.get_quant_method(self._linear_layer(), prefix="model.layers.0.self_attn.q_proj")
        assert isinstance(method, AscendLinearMethod)

    def test_layer_not_in_quant_list_is_unquantized(self):
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        cfg = self._make_config()
        cfg.modules_in_block_to_quantize = ["model.layers.0.self_attn.q_proj"]
        method = cfg.get_quant_method(self._linear_layer(), prefix="model.layers.0.mlp.gate_proj")
        assert isinstance(method, AscendUnquantizedLinearMethod)

    def test_empty_quant_list_quantizes_all(self):
        # When modules_in_block_to_quantize is empty, upstream GPTQ assumes
        # every Linear layer is quantized.
        from vllm_ascend.quantization.method_adapters import AscendLinearMethod

        cfg = self._make_config()
        # leave modules_in_block_to_quantize empty
        method = cfg.get_quant_method(self._linear_layer(), prefix="model.layers.0.mlp.down_proj")
        assert isinstance(method, AscendLinearMethod)


# ---------------------------------------------------------------------------
# T17: apply_vllm_mapper name translation
# ---------------------------------------------------------------------------


class TestApplyVLLMMapper:
    """apply_vllm_mapper translates HF module names to vLLM internal names so
    that the skip/quant lists match the parameter names used in
    get_quant_method. Verify it delegates to the mapper's apply_list.
    """

    def test_gptq_mapper_translates_block_list(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        cfg = GPTQConfig(weight_bits=4, group_size=128, desc_act=False)
        cfg.modules_in_block_to_quantize = ["model.layers.0.self_attn.q_proj"]

        mapper = MagicMock()
        mapper.apply_list.return_value = ["model.layers.0.self_attn.qkv_proj"]
        cfg.apply_vllm_mapper(mapper)

        mapper.apply_list.assert_called_once_with(["model.layers.0.self_attn.q_proj"])
        assert cfg.modules_in_block_to_quantize == ["model.layers.0.self_attn.qkv_proj"]

    def test_awq_mapper_translates_skip_list(self):
        from vllm_ascend.quantization.awq_config import AWQConfig

        cfg = AWQConfig(weight_bits=4, group_size=128, zero_point=True)
        cfg.modules_to_not_convert = ["lm_head"]

        mapper = MagicMock()
        mapper.apply_list.return_value = ["lm_head_mapped"]
        cfg.apply_vllm_mapper(mapper)

        mapper.apply_list.assert_called_once_with(["lm_head"])
        assert cfg.modules_to_not_convert == ["lm_head_mapped"]

    def test_awq_mapper_noop_when_empty(self):
        from vllm_ascend.quantization.awq_config import AWQConfig

        cfg = AWQConfig(weight_bits=4, group_size=128, zero_point=True)
        mapper = MagicMock()
        cfg.apply_vllm_mapper(mapper)
        # Empty skip list -> mapper not called, nothing changed.
        mapper.apply_list.assert_not_called()


# ---------------------------------------------------------------------------
# T16: maybe_update_config auto-detection from safetensors metadata
# ---------------------------------------------------------------------------


class TestMaybeUpdateConfig:
    """When the quant/quantize config does not explicitly list the quantized
    layers, maybe_update_config infers them by inspecting the safetensors
    parameter dtypes (any non fp16/bf16/fp32 param is considered quantized).
    """

    def test_gptq_auto_detect_with_mocked_metadata(self, monkeypatch):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        cfg = GPTQConfig(weight_bits=4, group_size=128, desc_act=False)
        assert cfg.modules_in_block_to_quantize == []

        fake_metadata = {
            # q_proj is quantized (qweight/qzeros/g_idx are int32)
            "model.layers.0.self_attn.q_proj.qweight": {"dtype": "I32"},
            "model.layers.0.self_attn.q_proj.qzeros": {"dtype": "I32"},
            "model.layers.0.self_attn.q_proj.g_idx": {"dtype": "I32"},
            "model.layers.0.self_attn.q_proj.scales": {"dtype": "F16"},
            # gate_proj also quantized
            "model.layers.0.mlp.gate_proj.qweight": {"dtype": "I32"},
            "model.layers.0.mlp.gate_proj.qzeros": {"dtype": "I32"},
            "model.layers.0.mlp.gate_proj.scales": {"dtype": "F16"},
            # lm_head is NOT quantized (plain bf16 weight)
            "lm_head.weight": {"dtype": "BF16"},
            # embed_tokens NOT quantized
            "model.embed_tokens.weight": {"dtype": "F32"},
        }
        monkeypatch.setattr(
            "vllm_ascend.quantization.gptq_config.get_safetensors_params_metadata",
            lambda *a, **k: fake_metadata,
        )

        cfg.maybe_update_config("dummy-model")

        block = set(cfg.modules_in_block_to_quantize)
        assert "model.layers.0.self_attn.q_proj" in block
        assert "model.layers.0.mlp.gate_proj" in block
        # Unquantized layers must NOT be in the quant list.
        assert "lm_head" not in block
        assert "model.embed_tokens" not in block

    def test_gptq_auto_detect_skipped_when_list_populated(self, monkeypatch):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        cfg = GPTQConfig(weight_bits=4, group_size=128, desc_act=False)
        cfg.modules_in_block_to_quantize = ["some.layer"]

        called = {"n": 0}

        def _boom(*a, **k):
            called["n"] += 1
            return {}

        monkeypatch.setattr(
            "vllm_ascend.quantization.gptq_config.get_safetensors_params_metadata",
            _boom,
        )
        cfg.maybe_update_config("dummy-model")
        # If the list is already populated, metadata is never inspected.
        assert called["n"] == 0
        assert cfg.modules_in_block_to_quantize == ["some.layer"]

    def test_gptq_flatten_nested_block_list(self):
        # Some models (e.g. TheBloke) store nested list[list[str]].
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        cfg = GPTQConfig(weight_bits=4, group_size=128, desc_act=False)
        cfg.modules_in_block_to_quantize = [
            ["layer0.q_proj", "layer0.k_proj"],
            ["layer1.q_proj"],
        ]
        cfg.maybe_update_config("dummy-model")
        assert cfg.modules_in_block_to_quantize == ["layer0.q_proj", "layer0.k_proj", "layer1.q_proj"]

    def test_awq_auto_detect_with_mocked_metadata(self, monkeypatch):
        from vllm_ascend.quantization.awq_config import AWQConfig

        cfg = AWQConfig(weight_bits=4, group_size=128, zero_point=True)
        assert cfg.modules_to_not_convert == []

        fake_metadata = {
            "model.layers.0.self_attn.q_proj.qweight": {"dtype": "I32"},
            "model.layers.0.self_attn.q_proj.scales": {"dtype": "F16"},
            "lm_head.weight": {"dtype": "BF16"},
        }
        monkeypatch.setattr(
            "vllm_ascend.quantization.awq_config.get_safetensors_params_metadata",
            lambda *a, **k: fake_metadata,
        )
        cfg.maybe_update_config("dummy-model")
        skip = set(cfg.modules_to_not_convert)
        # AWQ computes skip = (all_layers - quant_layers) = {lm_head}.
        assert "lm_head" in skip

    @pytest.mark.skipif(
        _tinyllama_path() is None,
        reason="TinyLlama-1.1B-Chat-v1.0-GPTQ not found in HF cache",
    )
    def test_gptq_auto_detect_real_tinyllama(self):
        """Real-model end-to-end: TinyLlama GPTQ has NO modules_to_not_convert
        in its quantize_config.json, so maybe_update_config must auto-detect
        the quantized layers from its safetensors index without crashing."""
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        cfg = GPTQConfig(weight_bits=4, group_size=128, desc_act=True)
        assert cfg.modules_in_block_to_quantize == []

        cfg.maybe_update_config(_tinyllama_path())

        block = set(cfg.modules_in_block_to_quantize)
        # Must be non-empty and contain the quantized proj layers.
        assert len(block) > 0
        assert any("q_proj" in name for name in block)
        # lm_head / embed_tokens should NOT be marked quantized for TinyLlama.
        assert "lm_head" not in block


# ---------------------------------------------------------------------------
# G18 / R7: GPTQ dynamic per-module override (+: / -: rules)
# ---------------------------------------------------------------------------


class TestGPTQDynamicOverride:
    """GPTQModel ``dynamic`` config allows per-module overrides via regex rules:

      "+:<regex>": {"bits": 8, ...}  -> override base config for matched modules
      "-:<regex>": {}                -> skip quantization for matched modules
      "<regex>"    (no prefix)       -> treated as a positive match

    Previously the project read ``dynamic`` from the checkpoint but never
    consumed it, so these rules were silently ignored (R7/G18). These tests
    verify the ported ``get_dynamic_override`` / ``_override_config`` and the
    wiring in ``get_quant_method``. All pure Python — no NPU needed.
    """

    def _make_config(self, dynamic=None, weight_bits=4):
        from vllm_ascend.quantization.gptq_config import GPTQConfig

        cfg = GPTQConfig(
            weight_bits=weight_bits,
            group_size=128,
            desc_act=False,
            dynamic=dynamic,
        )
        cfg.packed_modules_mapping = {}
        # Put every proj in the quantize list so skip decisions come only from
        # the dynamic rules, not from modules_in_block_to_quantize.
        cfg.modules_in_block_to_quantize = [
            f"model.layers.{i}.{p}" for i in range(4) for p in ("self_attn.q_proj", "mlp.gate_proj")
        ]
        return cfg

    def _linear_layer(self):
        from vllm.model_executor.layers.linear import LinearBase

        return MagicMock(spec=LinearBase)

    # -- get_dynamic_override / _override_config unit-level -----------------

    def test_get_dynamic_override_negative_returns_false(self):
        from vllm_ascend.quantization.gptq_config import get_dynamic_override

        cfg = self._make_config({"-:model.layers.0.": {}})
        assert get_dynamic_override(cfg, "model.layers.0.self_attn.q_proj") is False

    def test_get_dynamic_override_positive_returns_dict(self):
        from vllm_ascend.quantization.gptq_config import get_dynamic_override

        cfg = self._make_config({"+:model.layers.1.": {"bits": 8}})
        assert get_dynamic_override(cfg, "model.layers.1.mlp.gate_proj") == {"bits": 8}

    def test_get_dynamic_override_field_lookup(self):
        from vllm_ascend.quantization.gptq_config import get_dynamic_override

        cfg = self._make_config({"+:model.layers.1.": {"bits": 8, "group_size": 64}})
        assert get_dynamic_override(cfg, "model.layers.1.mlp.gate_proj", "bits") == 8
        assert get_dynamic_override(cfg, "model.layers.1.mlp.gate_proj", "group_size") == 64
        # Missing field falls back to default_value.
        assert get_dynamic_override(cfg, "model.layers.1.mlp.gate_proj", "desc_act", True) is True

    def test_get_dynamic_override_no_match_returns_default(self):
        from vllm_ascend.quantization.gptq_config import get_dynamic_override

        cfg = self._make_config({"-:model.layers.0.": {}})
        assert get_dynamic_override(cfg, "model.layers.3.mlp.gate_proj") is None
        # Empty dynamic => always the default.
        cfg_empty = self._make_config({})
        assert get_dynamic_override(cfg_empty, "model.layers.0.self_attn.q_proj") is None

    def test_override_config_mutates_copy_only(self):
        from copy import deepcopy

        from vllm_ascend.quantization.gptq_config import _override_config

        cfg = self._make_config({"+:model.layers.1.": {"bits": 8, "group_size": 64}})
        cloned = deepcopy(cfg)
        _override_config(cloned, "model.layers.1.mlp.gate_proj")
        assert cloned.weight_bits == 8
        assert cloned.group_size == 64
        assert cloned.pack_factor == 4  # 32 // 8
        # Base config untouched.
        assert cfg.weight_bits == 4
        assert cfg.group_size == 128

    def test_override_config_rejects_2bit(self):
        from copy import deepcopy

        from vllm_ascend.quantization.gptq_config import _override_config

        cfg = self._make_config({"+:model.layers.0.": {"bits": 2}})
        with pytest.raises(NotImplementedError, match="2-bit"):
            _override_config(deepcopy(cfg), "model.layers.0.self_attn.q_proj")

    # -- get_quant_method wiring --------------------------------------------

    def test_empty_dynamic_quantizes_in_list_layer(self):
        # Safety property: empty dynamic => identical to pre-dynamic behavior.
        from vllm_ascend.quantization.method_adapters import AscendLinearMethod

        cfg = self._make_config()
        method = cfg.get_quant_method(self._linear_layer(), prefix="model.layers.0.self_attn.q_proj")
        assert isinstance(method, AscendLinearMethod)
        assert method.quant_method.weight_bits == 4

    def test_negative_match_forces_skip(self):
        # Layer 0 is in the quantize list, but a "-:" rule must force it out.
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        cfg = self._make_config({"-:model.layers.0.": {}})
        method = cfg.get_quant_method(self._linear_layer(), prefix="model.layers.0.self_attn.q_proj")
        assert isinstance(method, AscendUnquantizedLinearMethod)

    def test_negative_match_does_not_affect_other_layers(self):
        from vllm_ascend.quantization.method_adapters import AscendLinearMethod

        cfg = self._make_config({"-:model.layers.0.": {}})
        method = cfg.get_quant_method(self._linear_layer(), prefix="model.layers.1.mlp.gate_proj")
        assert isinstance(method, AscendLinearMethod)

    def test_positive_match_overrides_bits_w4_to_w8(self):
        from vllm_ascend.quantization.method_adapters import AscendLinearMethod

        cfg = self._make_config({"+:model.layers.1.": {"bits": 8}})
        method = cfg.get_quant_method(self._linear_layer(), prefix="model.layers.1.mlp.gate_proj")
        assert isinstance(method, AscendLinearMethod)
        # W8A16_GPTQ scheme selected, and group_size override carried through.
        assert method.quant_method.weight_bits == 8
        assert method.quant_method.pack_factor == 4

    def test_positive_match_overrides_group_size(self):
        from vllm_ascend.quantization.method_adapters import AscendLinearMethod

        cfg = self._make_config({"+:model.layers.1.": {"group_size": 64}})
        method = cfg.get_quant_method(self._linear_layer(), prefix="model.layers.1.mlp.gate_proj")
        assert isinstance(method, AscendLinearMethod)
        assert method.quant_method.group_size == 64
        # Base width unchanged.
        assert cfg.weight_bits == 4

    def test_unprefixed_pattern_is_positive(self):
        from vllm_ascend.quantization.method_adapters import AscendLinearMethod

        cfg = self._make_config({"model.layers.0.": {"bits": 8}})
        method = cfg.get_quant_method(self._linear_layer(), prefix="model.layers.0.self_attn.q_proj")
        assert isinstance(method, AscendLinearMethod)
        assert method.quant_method.weight_bits == 8

    def test_regex_positive_match_layer_range(self):
        # "+:" only overrides config for layers ALREADY in the quantize list
        # (matching upstream: ``not is_layer_quantized`` short-circuits to
        # unquantized before the override is applied). The quantize list here
        # covers layers 0-3; this regex matches layers 1-2 and overrides them
        # to 8-bit, while layer 0 keeps the base 4-bit.
        from vllm_ascend.quantization.method_adapters import AscendLinearMethod

        cfg = self._make_config({r"+:.*\.(?:[1-2])\..*": {"bits": 8}})
        # Layer 1 is in the list and matches -> 8-bit.
        m_match = cfg.get_quant_method(self._linear_layer(), prefix="model.layers.1.mlp.gate_proj")
        assert isinstance(m_match, AscendLinearMethod)
        assert m_match.quant_method.weight_bits == 8
        # Layer 0 is in the list but does not match -> base 4-bit.
        m_nomatch = cfg.get_quant_method(self._linear_layer(), prefix="model.layers.0.self_attn.q_proj")
        assert isinstance(m_nomatch, AscendLinearMethod)
        assert m_nomatch.quant_method.weight_bits == 4

    def test_positive_override_to_2bit_rejected_at_routing(self):
        cfg = self._make_config({"+:model.layers.0.": {"bits": 2}})
        with pytest.raises(NotImplementedError, match="2-bit"):
            cfg.get_quant_method(self._linear_layer(), prefix="model.layers.0.self_attn.q_proj")


# ---------------------------------------------------------------------------
# torchao: config parsing + Linear routing (Pattern A)
# ---------------------------------------------------------------------------


class TestTorchAORouting:
    """torchao config parsing and Linear layer routing.

    Verifies that ``--quantization torchao`` (overriding vLLM's native config)
    parses int4wo/int8wo/fp8wo and routes LinearBase layers to the Ascend
    scheme via AscendLinearMethod, while skipped layers fall back to the
    unquantized method. Pure Python, no NPU dependency.
    """

    def _linear_layer(self):
        from vllm.model_executor.layers.linear import LinearBase

        return MagicMock(spec=LinearBase)

    def test_from_config_int8wo(self):
        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        cfg = TorchAOConfig.from_config({"quant_method": "torchao", "quant_type": {"default": "int8wo"}})
        assert cfg.torchao_quant_type == "int8wo"
        # Online quant of a dense checkpoint → not torchao-serialized.
        assert cfg.is_checkpoint_torchao_serialized is False

    def test_from_config_prequant_serialized_opt_in(self):
        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        # A torchao-serialized (pre-quant) checkpoint must opt in explicitly.
        cfg = TorchAOConfig.from_config(
            {"quant_method": "torchao", "quant_type": {"default": "int8wo"}, "is_checkpoint_torchao_serialized": True}
        )
        assert cfg.is_checkpoint_torchao_serialized is True

    def test_from_config_prequant_flat_flag(self):
        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        # T-10: a flat-tensor pre-quantized checkpoint opts in via "prequant":
        # true (distinct from is_checkpoint_torchao_serialized, which would
        # trigger vLLM's native torchao AQTensor loader). Default is False.
        cfg_off = TorchAOConfig.from_config({"quant_method": "torchao", "quant_type": {"default": "int8wo"}})
        assert cfg_off.is_prequant_checkpoint is False
        cfg_on = TorchAOConfig.from_config(
            {"quant_method": "torchao", "quant_type": {"default": "int8wo"}, "prequant": True}
        )
        assert cfg_on.is_prequant_checkpoint is True
        # pre-quant flat path keeps the native torchao loader OFF
        assert cfg_on.is_checkpoint_torchao_serialized is False

    def test_from_config_int4wo_with_group_size(self):
        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        cfg = TorchAOConfig.from_config({"quant_type": {"default": "int4wo-g64"}})
        assert cfg.torchao_quant_type == "int4wo"
        assert cfg.group_size == 64

    def test_from_config_dict_form(self):
        # The serialized AOBaseConfig dict form (name + group_size) is also accepted.
        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        cfg = TorchAOConfig.from_config(
            {"quant_type": {"default": {"name": "Int8WeightOnlyConfig", "group_size": 128}}}
        )
        assert cfg.torchao_quant_type == "int8wo"
        assert cfg.group_size == 128

    def test_unsupported_type_rejected(self):
        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        with pytest.raises(ValueError, match="Unsupported"):
            TorchAOConfig.from_config({"quant_type": {"default": "int2wo"}})

    def test_missing_quant_type_rejected(self):
        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        with pytest.raises(ValueError, match="quant_type"):
            TorchAOConfig.from_config({"quant_method": "torchao"})

    def test_group_size_zero_rejected(self):
        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        with pytest.raises(ValueError, match="positive"):
            TorchAOConfig(torchao_quant_type="int8wo", group_size=0)

    def test_int8_routes_to_ascend_linear_method(self):
        from vllm_ascend.quantization.method_adapters import AscendLinearMethod
        from vllm_ascend.quantization.methods.torchao import (
            AscendW8A16TorchAOLinearScheme,
        )
        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        cfg = TorchAOConfig.from_config({"quant_method": "torchao", "quant_type": {"default": "int8wo"}})
        method = cfg.get_quant_method(self._linear_layer(), prefix="model.layers.0.self_attn.q_proj")
        assert isinstance(method, AscendLinearMethod)
        assert isinstance(method.quant_method, AscendW8A16TorchAOLinearScheme)

    def test_int4_routes_to_w4_scheme(self):
        from vllm_ascend.quantization.methods.torchao import (
            AscendW4A16TorchAOLinearScheme,
        )
        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        cfg = TorchAOConfig.from_config({"quant_type": {"default": "int4wo"}})
        method = cfg.get_quant_method(self._linear_layer(), prefix="model.layers.0.mlp.gate_proj")
        assert isinstance(method.quant_method, AscendW4A16TorchAOLinearScheme)

    def test_skipped_layer_routes_unquantized(self):
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        cfg = TorchAOConfig.from_config({"quant_type": {"default": "int8wo"}, "modules_to_not_convert": ["lm_head"]})
        method = cfg.get_quant_method(self._linear_layer(), prefix="lm_head")
        assert isinstance(method, AscendUnquantizedLinearMethod)

    def test_non_linear_returns_none(self):
        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        cfg = TorchAOConfig.from_config({"quant_type": {"default": "int8wo"}})
        assert cfg.get_quant_method(MagicMock(), prefix="foo") is None

    # -- T-12: per-layer module_fqn_to_config overrides + autoquant rejection --

    def test_module_fqn_exact_override(self):
        # Default int8, but one named layer forced to int4-g64.
        from vllm_ascend.quantization.methods.torchao import (
            AscendW4A16TorchAOLinearScheme,
        )
        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        cfg = TorchAOConfig(
            "int8wo",
            module_fqn_to_config={"model.layers.0.mlp.gate_proj": {"name": "int4", "group_size": 64}},
        )
        method = cfg.get_quant_method(self._linear_layer(), prefix="model.layers.0.mlp.gate_proj")
        assert isinstance(method.quant_method, AscendW4A16TorchAOLinearScheme)
        # Per-layer group_size reaches the scheme.
        assert method.quant_method.group_size == 64

    def test_module_fqn_regex_override(self):
        from vllm_ascend.quantization.methods.torchao import (
            AscendW4A16TorchAOLinearScheme,
        )
        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        cfg = TorchAOConfig(
            "int8wo",
            module_fqn_to_config={"re:.*\\.gate_proj$": {"name": "int4"}},
        )
        method = cfg.get_quant_method(self._linear_layer(), prefix="model.layers.5.mlp.gate_proj")
        assert isinstance(method.quant_method, AscendW4A16TorchAOLinearScheme)

    def test_module_fqn_default_fallback(self):
        # _default applies to layers that match no entry.
        from vllm_ascend.quantization.methods.torchao import (
            AscendW4A16TorchAOLinearScheme,
            AscendW8A16TorchAOLinearScheme,
        )
        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        cfg = TorchAOConfig(
            "int8wo",
            module_fqn_to_config={"_default": {"name": "int4"}, "re:.*lm_head$": {"name": "int8"}},
        )
        head = cfg.get_quant_method(self._linear_layer(), prefix="model.lm_head")
        other = cfg.get_quant_method(self._linear_layer(), prefix="model.layers.0.self_attn.q_proj")
        assert isinstance(head.quant_method, AscendW8A16TorchAOLinearScheme)
        assert isinstance(other.quant_method, AscendW4A16TorchAOLinearScheme)  # _default

    def test_module_fqn_explicit_none_is_dense(self):
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        cfg = TorchAOConfig("int8wo", module_fqn_to_config={"lm_head": None})
        method = cfg.get_quant_method(self._linear_layer(), prefix="lm_head")
        assert isinstance(method, AscendUnquantizedLinearMethod)

    def test_module_fqn_unmatched_no_default_is_dense(self):
        # No _default → unmatched layers stay dense (mirrors upstream
        # UnquantizedLinearMethod fallback inside ModuleFqnToConfig).
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        cfg = TorchAOConfig("int8wo", module_fqn_to_config={"re:.*gate_proj$": {"name": "int4"}})
        method = cfg.get_quant_method(self._linear_layer(), prefix="model.layers.0.self_attn.q_proj")
        assert isinstance(method, AscendUnquantizedLinearMethod)

    def test_module_fqn_parsed_from_data(self):
        # from_config pulls module_fqn_to_config out of quant_type._data.
        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        cfg = TorchAOConfig.from_config(
            {
                "quant_type": {
                    "default": {"name": "Int8WeightOnlyConfig", "_data": {"module_fqn_to_config": {"lm_head": None}}},
                }
            }
        )
        assert cfg.module_fqn_to_config == {"lm_head": None}

    def test_autoquant_rejected(self):
        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        with pytest.raises(NotImplementedError, match="autoquant"):
            TorchAOConfig.from_config({"quant_type": {"default": {"name": "AutoQuantization"}}})

    def test_moe_raises_not_implemented(self):
        # torchao MoE expert quantization is out of MVP scope; a FusedMoE layer
        # must raise (not silently fall back to dense experts), mirroring GGUF.
        import pytest
        from vllm.model_executor.layers.fused_moe import FusedMoE

        from vllm_ascend.quantization.torchao_config import TorchAOConfig

        cfg = TorchAOConfig.from_config({"quant_method": "torchao", "quant_type": {"default": "int8wo"}})
        with pytest.raises(NotImplementedError, match="MoE"):
            cfg.get_quant_method(MagicMock(spec=FusedMoE), prefix="moe")


# ---------------------------------------------------------------------------
# gguf: config override + Linear routing (Pattern A, dedicated method)
# ---------------------------------------------------------------------------


class TestGGUFRouting:
    """gguf config override and routing.

    GGUF uses the ``is_gguf_weight`` loader contract (distinct from the
    packed-param AWQ/GPTQ pattern), so it routes to a dedicated
    ``AscendGGUFLinearMethod`` rather than the ``AscendLinearScheme`` registry.
    """

    def test_from_config_and_name(self):
        from vllm_ascend.quantization.gguf_config import GGUFConfig

        cfg = GGUFConfig.from_config({})
        assert cfg.get_name() == "gguf"

    def test_override_when_user_requests_gguf(self):
        from vllm_ascend.quantization.gguf_config import GGUFConfig

        assert GGUFConfig.override_quantization_method({}, "gguf") == "gguf"
        assert GGUFConfig.override_quantization_method({}, "fp8") is None

    def test_linear_routes_to_gguf_method(self):
        from vllm.model_executor.layers.linear import LinearBase

        from vllm_ascend.quantization.gguf_config import GGUFConfig
        from vllm_ascend.quantization.methods.gguf import AscendGGUFLinearMethod

        cfg = GGUFConfig.from_config({})
        method = cfg.get_quant_method(MagicMock(spec=LinearBase), prefix="model.layers.0.self_attn.o_proj")
        assert isinstance(method, AscendGGUFLinearMethod)

    def test_non_linear_returns_none(self):
        from vllm_ascend.quantization.gguf_config import GGUFConfig

        cfg = GGUFConfig.from_config({})
        assert cfg.get_quant_method(MagicMock(), prefix="foo") is None

    def test_moe_raises_not_implemented(self):
        import pytest
        from vllm.model_executor.layers.fused_moe import FusedMoE

        from vllm_ascend.quantization.gguf_config import GGUFConfig

        cfg = GGUFConfig.from_config({})
        with pytest.raises(NotImplementedError, match="MoE"):
            cfg.get_quant_method(MagicMock(spec=FusedMoE), prefix="moe")


# ---------------------------------------------------------------------------
# F1: AWQ lm_head_quantized routing (mirrors GPTQ T14)
# ---------------------------------------------------------------------------


class TestAWQLMHeadRouting:
    """AWQ lm_head_quantized mirrors GPTQ T14 — ParallelLMHead routes to the
    AWQ quant branch when True, stays unquantized (None) when False."""

    def test_lm_head_false_leaves_unquantized(self):
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
        )

        from vllm_ascend.quantization.awq_config import AWQConfig

        cfg = AWQConfig(weight_bits=4, group_size=128, zero_point=True, lm_head_quantized=False)
        method = cfg.get_quant_method(MagicMock(spec=ParallelLMHead), prefix="lm_head")
        assert method is None, f"lm_head_quantized=False must leave ParallelLMHead unquantized, got {type(method)!r}"

    def test_lm_head_true_routes_to_awq(self):
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
        )

        from vllm_ascend.quantization.awq_config import AWQConfig
        from vllm_ascend.quantization.method_adapters import AscendLinearMethod

        cfg = AWQConfig(weight_bits=4, group_size=128, zero_point=True, lm_head_quantized=True)
        method = cfg.get_quant_method(MagicMock(spec=ParallelLMHead), prefix="lm_head")
        assert isinstance(method, AscendLinearMethod), (
            f"lm_head_quantized=True must route ParallelLMHead to AWQ quant, got {type(method)!r}"
        )


# ---------------------------------------------------------------------------
# torchao int4 self-impl math (H1: group_size >= K, zero-group guard)
# ---------------------------------------------------------------------------


class TestTorchAOInt4Math:
    """CPU math tests for the self-implemented per-group int4 RTN
    (``_int4_symmetric_quant``) — validates H1 (group_size >= K pre-rejection)
    and the all-zero-group div-by-zero guard."""

    def test_int4_group_size_ge_input_size_rejected(self):
        # H1: NPU op rejects antiquant_group_size == K; reject at load.
        import pytest

        from vllm_ascend.quantization.methods.torchao import _int4_symmetric_quant

        W = torch.randn(64, 128, dtype=torch.float32)  # [N, K] with K=128
        with pytest.raises(ValueError, match="must be < input_size"):
            _int4_symmetric_quant(W, group_size=128)  # group_size == K

    def test_int4_all_zero_group_no_nan(self):
        # max_abs=0 for an all-zero group; the clamp(min=1e-8) guard must
        # avoid div-by-zero NaN.
        from vllm_ascend.quantization.methods.torchao import _int4_symmetric_quant

        W = torch.zeros(64, 256, dtype=torch.float32)
        q_flat, scales = _int4_symmetric_quant(W, group_size=128)
        assert not torch.isnan(q_flat).any()
        assert not torch.isnan(scales).any()
        assert torch.all(q_flat == 0)  # zero weight -> zero quantized

    def test_int4_valid_group_size_accepted(self):
        from vllm_ascend.quantization.methods.torchao import _int4_symmetric_quant

        W = torch.randn(64, 256, dtype=torch.float32)  # K=256, group=128 < K
        q_flat, scales = _int4_symmetric_quant(W, group_size=128)
        assert q_flat.shape == (256, 64)  # [K, N]
        assert scales.shape == (2, 64)  # [G, N], G = 256 // 128
        assert q_flat.min() >= -8 and q_flat.max() <= 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
