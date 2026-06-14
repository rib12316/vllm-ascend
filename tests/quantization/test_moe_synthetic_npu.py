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
NPU synthetic-weight correctness tests for Ascend AWQ/GPTQ MoE (T3/T4).

There is no small AWQ/GPTQ MoE model on HuggingFace (smallest ~30B, exceeds
our disk/HBM budget), so the MoE ``apply`` -> ``fused_experts`` path cannot be
exercised end-to-end on a real model. Instead we verify the *weight-processing
pipeline* — the new, previously-unverified code — by:

1. Packing known uint4/uint8 values into the exact raw checkpoint format that
   ``get_weight`` / ``get_dynamic_quant_param`` define (so the oracle q / zp
   are known exactly — no quantization-error ambiguity).
2. Building a layer, running ``process_weights_after_loading`` (the code under
   test: unpack -> center -> int4pack / int32-view -> offset).
3. Dequantizing per-expert via ``npu_weight_quant_batchmatmul`` and comparing
   to an independent dense reference ``x @ ((q - zp) * scale)``.

Why per-expert npu_weight_quant_batchmatmul is faithful: the shipping
AscendW4A16FusedMoEMethod documents that fused_experts consumes an INPUT-FIRST
weight layout (E, K, N//pf) with scale (E, K//gs, N); AWQ/GPTQ MoE produce
exactly that layout after processing (see test_*_layout_*), and the dequant
formula (weight + offset) * scale is identical to the linear path that is
already NPU-verified. The token-dispatch inside grouped_matmul is existing
vllm-ascend infra shared with W4A16 and is not new code.

NOTE on per-expert slicing: npu_weight_quant_batchmatmul misreads a strided
NPU *view* of a larger (E,K,N) tensor for non-first experts, so each expert
slice is materialized to a fresh contiguous NPU tensor via ``_clean_npu``
before the op call. The real fused_experts path never Python-slices experts.

Requires torch_npu + NPU hardware.
"""

import pytest
import torch
import torch_npu  # noqa: F401
from unittest.mock import patch

from vllm.config import VllmConfig, set_current_vllm_config


@pytest.fixture(autouse=True)
def default_vllm_config():
    """The MoE scheme __init__ reads the live vLLM config (DSA-CP flag) and
    the Ascend config (eplb). Build a real VllmConfig, init Ascend config."""
    from vllm_ascend.ascend_config import (
        clear_ascend_config,
        init_ascend_config,
    )
    cfg = VllmConfig()
    with patch("vllm_ascend.platform.NPUPlatform.check_and_update_config"):
        init_ascend_config(cfg)
    cfg.compilation_config.custom_ops = ["all"]
    with set_current_vllm_config(cfg):
        yield cfg
    clear_ascend_config()


# Geometry
E = 2
H = 256       # hidden (w13 input / w2 output)
IN = 128      # intermediate (w13 half-output / w2 input)
GS = 64       # group_size (must be multiple of 32 for the NPU op)
M = 8         # tokens
DTYPE = torch.float16


def _clean_npu(t):
    """Materialize a per-expert slice into a fresh contiguous NPU tensor.

    Avoids the NPU op misreading a strided view of the (E, K, N) tensor for
    non-first experts. Pure test-harness concern; production never slices."""
    return t.detach().cpu().contiguous().npu()


# ---------------------------------------------------------------------------
# Packing helpers (inverse of production _unpack_*). Pack KNOWN uint values so
# the oracle is exact.
# ---------------------------------------------------------------------------

def pack_gptq_weight(q_uint, num_bits):
    """Pack along INPUT dim (dim=-2), GPTQ standard order.

    q_uint: (E, K, N) -> (E, K // pf, N) int32. Inverse of
    _unpack_qweight_from_int32 (unpacked[i::pf] = (w >> nb*i) & mask)."""
    pf = 32 // num_bits
    lead, K, N = q_uint.shape[:-2], q_uint.shape[-2], q_uint.shape[-1]
    packed = torch.zeros((*lead, K // pf, N), dtype=torch.int32)
    for i in range(pf):
        packed |= (q_uint[..., i::pf, :].to(torch.int32) << (num_bits * i))
    return packed


def pack_gptq_qzeros(zp_uint, num_bits):
    """Pack along OUTPUT dim (dim=-1), standard order.

    zp_uint: (..., G, N) -> (..., G, N // pf) int32."""
    pf = 32 // num_bits
    lead, N = zp_uint.shape[:-1], zp_uint.shape[-1]
    packed = torch.zeros((*lead, N // pf), dtype=torch.int32)
    for i in range(pf):
        packed |= (zp_uint[..., i::pf].to(torch.int32) << (num_bits * i))
    return packed


_AWQ_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def pack_awq(q_uint, num_bits):
    """Pack along OUTPUT dim (dim=-1), AWQ interleaved order.

    Inverse of _unpack_qzero_from_int32 / _unpack_weight_from_int32."""
    pf = 32 // num_bits
    order = _AWQ_ORDER[:pf]
    lead, N = q_uint.shape[:-1], q_uint.shape[-1]
    packed = torch.zeros((*lead, N // pf), dtype=torch.int32)
    for i in range(pf):
        packed |= (q_uint[..., i::pf].to(torch.int32) << (order[i] * num_bits))
    return packed


def reference_output(x, q_uint, zp_uint, scale, gs):
    """Dense reference per expert: out[e] = x @ W_ref[e],
    W_ref[e][k,n] = (q[e,k,n] - zp[e,k//gs,n]) * scale[e,k//gs,n]."""
    zp_exp = zp_uint.repeat_interleave(gs, dim=1)        # (E, K, N)
    scale_exp = scale.repeat_interleave(gs, dim=1)       # (E, K, N)
    w_ref = (q_uint.to(torch.float32) - zp_exp.to(torch.float32)) * scale_exp.to(torch.float32)
    return torch.matmul(x.to(torch.float32), w_ref)     # (E, M, N)


def _assert_rel(out_npu, out_ref, tag, e, tol=0.06):
    a, b = out_npu.float().cpu(), out_ref
    max_abs = b.abs().max().item()
    rel = (a - b).abs().max().item() / max(max_abs, 1e-6)
    assert rel < tol, (
        f"[{tag} expert {e}] max_diff={(a-b).abs().max().item():.4f} "
        f"max_abs={max_abs:.4f} rel_err={rel:.4f} exceeds {tol*100:.0f}%")


def _build_layer(scheme, w_specs, pq_specs, raw_weight, raw_zp, raw_scale,
                 device):
    """Fill the get_weight/get_dynamic_quant_param tensors with packed known
    values, build an nn.Module, and run process_weights_after_loading."""
    w_specs[raw_weight["qw"]].copy_(raw_weight["packed"])
    pq_specs[raw_scale["sc"]].copy_(raw_scale["val"].to(DTYPE))
    pq_specs[raw_zp["qz"]].copy_(raw_zp["packed"])
    layer = torch.nn.Module()
    for name, t in {**w_specs, **pq_specs}.items():
        layer.register_parameter(
            name, torch.nn.Parameter(t.clone(), requires_grad=False))
    if device == "npu":
        layer = layer.npu()
    scheme.process_weights_after_loading(layer)
    return layer


def _npu_dequant(weight_e, scale_e, offset_e, k_dim):
    x = torch.randn(M, k_dim, dtype=DTYPE).npu()
    out = torch_npu.npu_weight_quant_batchmatmul(
        x, _clean_npu(weight_e),
        antiquant_scale=_clean_npu(scale_e),
        antiquant_offset=_clean_npu(offset_e),
        antiquant_group_size=GS,
    )
    return x, out


# ---------------------------------------------------------------------------
# GPTQ MoE W4A16
# ---------------------------------------------------------------------------


class TestGPTQMoEW4A16:
    """T4 (4-bit): standard-order unpack -> int4pack; offset = -(zp-8)."""

    def _setup(self, use_v2=True):
        from vllm_ascend.quantization.gptq_config import GPTQConfig
        from vllm_ascend.quantization.methods.gptq import (
            AscendW4A16GPTQFusedMoEMethod,
        )
        cfg = GPTQConfig(weight_bits=4, group_size=GS, desc_act=False,
                         checkpoint_format="gptq_v2" if use_v2 else "")
        scheme = AscendW4A16GPTQFusedMoEMethod(cfg)
        return scheme, scheme.get_weight(E, IN, H, DTYPE), \
            scheme.get_dynamic_quant_param(E, IN, H, DTYPE), use_v2

    def _fill(self, scheme, w, pq, use_v2, which):
        K, N = (H, 2 * IN) if which == "w13" else (IN, H)
        qw, sc, qz = f"{which}_qweight", f"{which}_scales", f"{which}_qzeros"
        q_uint = torch.randint(0, 16, (E, K, N))
        zp_uint = torch.randint(0, 16, (E, K // GS, N))
        scale = torch.rand(E, K // GS, N) * 0.1 + 0.01
        layer = _build_layer(
            scheme, w, pq,
            {"qw": qw, "packed": pack_gptq_weight(q_uint, 4)},
            {"qz": qz, "packed": pack_gptq_qzeros(zp_uint, 4)},
            {"sc": sc, "val": scale}, device="npu")  # int4pack needs NPU
        zp_eff = zp_uint + (0 if use_v2 else 1)
        return layer, q_uint, zp_eff, scale, (K, N)

    def test_w13_correctness(self):
        layer, q_uint, zp_eff, scale, (K, N) = self._fill(*self._setup(), "w13")
        for e in range(E):
            x, out = _npu_dequant(
                layer.w13_qweight.data[e], layer.w13_scales.data[e],
                layer.w13_qzeros.data[e], K)
            _assert_rel(out, reference_output(x.cpu(), q_uint, zp_eff, scale, GS)[e],
                        "GPTQ W4 w13", e)

    def test_w2_correctness(self):
        layer, q_uint, zp_eff, scale, (K, N) = self._fill(*self._setup(), "w2")
        for e in range(E):
            x, out = _npu_dequant(
                layer.w2_qweight.data[e], layer.w2_scales.data[e],
                layer.w2_qzeros.data[e], K)
            _assert_rel(out, reference_output(x.cpu(), q_uint, zp_eff, scale, GS)[e],
                        "GPTQ W4 w2", e)

    def test_layout_is_input_first(self):
        # Must match the shipping W4A16 target layout consumed by fused_experts.
        layer, *_ = self._fill(*self._setup(), "w13")
        assert layer.w13_qweight.shape == (E, H, 2 * IN // 8)
        assert layer.w13_scales.shape == (E, H // GS, 2 * IN)
        assert layer.w13_qzeros.shape == (E, H // GS, 2 * IN)


# ---------------------------------------------------------------------------
# GPTQ MoE W8A16
# ---------------------------------------------------------------------------


class TestGPTQMoEW8A16:
    """T4 (8-bit): standard-order unpack -> int8 (no int4pack); int32 storage
    view for grouped_matmul; offset = -(zp-128)."""

    def _setup(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig
        from vllm_ascend.quantization.methods.gptq import (
            AscendW8A16GPTQFusedMoEMethod,
        )
        cfg = GPTQConfig(weight_bits=8, group_size=GS, desc_act=False,
                         checkpoint_format="gptq_v2")
        scheme = AscendW8A16GPTQFusedMoEMethod(cfg)
        return scheme, scheme.get_weight(E, IN, H, DTYPE), \
            scheme.get_dynamic_quant_param(E, IN, H, DTYPE)

    def _fill(self, scheme, w, pq, which):
        K, N = (H, 2 * IN) if which == "w13" else (IN, H)
        qw, sc, qz = f"{which}_qweight", f"{which}_scales", f"{which}_qzeros"
        q_uint = torch.randint(0, 256, (E, K, N))
        zp_uint = torch.randint(0, 256, (E, K // GS, N))
        scale = torch.rand(E, K // GS, N) * 0.01 + 0.001
        layer = _build_layer(
            scheme, w, pq,
            {"qw": qw, "packed": pack_gptq_weight(q_uint, 8)},
            {"qz": qz, "packed": pack_gptq_qzeros(zp_uint, 8)},
            {"sc": sc, "val": scale}, device="cpu")
        return layer, q_uint, zp_uint, scale, (K, N)

    def test_dequant_math_w13(self):
        # npu_weight_quant_batchmatmul rejects the int32 storage view (it
        # misreads it as int4), so feed the int8 weight directly to verify the
        # dequant math; the int32 view is verified separately below.
        from vllm_ascend.quantization.methods.gptq import (
            _unpack_qweight_from_int32,
        )
        scheme, w, pq = self._setup()
        layer, q_uint, zp_uint, scale, (K, N) = self._fill(scheme, w, pq, "w13")
        sint8 = _unpack_qweight_from_int32(
            pack_gptq_weight(q_uint, 8).flatten(0, 1), 8).view(E, K, N)
        for e in range(E):
            offset = -(zp_uint.to(torch.float32) - 128)[e]
            x = torch.randn(M, K, dtype=DTYPE).npu()
            out = torch_npu.npu_weight_quant_batchmatmul(
                x, _clean_npu(sint8[e]),  # int8 weight, like the linear 8-bit path
                antiquant_scale=_clean_npu(scale.to(DTYPE)[e]),
                antiquant_offset=_clean_npu(offset.to(DTYPE)),
                antiquant_group_size=GS)
            _assert_rel(out, reference_output(x.cpu(), q_uint, zp_uint, scale, GS)[e],
                        "GPTQ W8 w13", e)

    def test_int32_view_roundtrip(self):
        # Production 8-bit MoE views int8 (E,K,N) as int32 (E,K,N//4) for
        # grouped_matmul storage. Verify the view preserves data exactly.
        from vllm_ascend.quantization.methods.gptq import (
            _unpack_qweight_from_int32,
        )
        scheme, w, pq = self._setup()
        layer, q_uint, *_ = self._fill(scheme, w, pq, "w13")
        assert layer.w13_qweight.shape == (E, H, 2 * IN // 4)
        expected = _unpack_qweight_from_int32(
            pack_gptq_weight(q_uint, 8).flatten(0, 1), 8).view(E, H, 2 * IN)
        got = layer.w13_qweight.data.view(torch.int8).reshape(E, H, 2 * IN)
        assert torch.equal(got, expected.to(torch.int8)), (
            "8-bit int32 storage view corrupted weight data")

    def test_layout_is_input_first(self):
        scheme, w, pq = self._setup()
        layer, *_ = self._fill(scheme, w, pq, "w13")
        assert layer.w13_qweight.shape == (E, H, 2 * IN // 4)
        assert layer.w13_scales.shape == (E, H // GS, 2 * IN)


# ---------------------------------------------------------------------------
# AWQ MoE W4A16
# ---------------------------------------------------------------------------


class TestAWQMoEW4A16:
    """T3: interleaved-order unpack + XOR; offset = -(zp-8)."""

    def _setup(self):
        from vllm_ascend.quantization.awq_config import AWQConfig
        from vllm_ascend.quantization.methods.w4a16_awq import (
            AscendW4A16AWQFusedMoEMethod,
        )
        scheme = AscendW4A16AWQFusedMoEMethod(
            AWQConfig(weight_bits=4, group_size=GS, zero_point=True))
        return scheme, scheme.get_weight(E, IN, H, DTYPE), \
            scheme.get_dynamic_quant_param(E, IN, H, DTYPE)

    def _fill(self, scheme, w, pq, which):
        K, N = (H, 2 * IN) if which == "w13" else (IN, H)
        qw, sc, qz = f"{which}_qweight", f"{which}_scales", f"{which}_qzeros"
        q_uint = torch.randint(0, 16, (E, K, N))
        zp_uint = torch.randint(0, 16, (E, K // GS, N))
        scale = torch.rand(E, K // GS, N) * 0.1 + 0.01
        layer = _build_layer(
            scheme, w, pq,
            {"qw": qw, "packed": pack_awq(q_uint, 4)},
            {"qz": qz, "packed": pack_awq(zp_uint, 4)},
            {"sc": sc, "val": scale}, device="cpu")
        return layer, q_uint, zp_uint, scale, (K, N)

    def test_w13_correctness(self):
        layer, q_uint, zp_uint, scale, (K, N) = self._fill(*self._setup(), "w13")
        for e in range(E):
            x, out = _npu_dequant(
                layer.w13_qweight.data[e], layer.w13_scales.data[e],
                layer.w13_qzeros.data[e], K)
            _assert_rel(out, reference_output(x.cpu(), q_uint, zp_uint, scale, GS)[e],
                        "AWQ w13", e)

    def test_w2_correctness(self):
        layer, q_uint, zp_uint, scale, (K, N) = self._fill(*self._setup(), "w2")
        for e in range(E):
            x, out = _npu_dequant(
                layer.w2_qweight.data[e], layer.w2_scales.data[e],
                layer.w2_qzeros.data[e], K)
            _assert_rel(out, reference_output(x.cpu(), q_uint, zp_uint, scale, GS)[e],
                        "AWQ w2", e)

    def test_layout_is_input_first(self):
        layer, *_ = self._fill(*self._setup(), "w13")
        assert layer.w13_qweight.shape == (E, H, 2 * IN // 8)
        assert layer.w13_scales.shape == (E, H // GS, 2 * IN)


# ---------------------------------------------------------------------------
# MoE desc_act rejection (commit 6aacf5b0)
# ---------------------------------------------------------------------------


class TestMoEDescActRejection:
    """GPTQ MoE with desc_act=True must raise NotImplementedError."""

    def test_w4_rejects_desc_act(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig
        from vllm_ascend.quantization.methods.gptq import (
            AscendW4A16GPTQFusedMoEMethod,
        )
        scheme = AscendW4A16GPTQFusedMoEMethod(
            GPTQConfig(weight_bits=4, group_size=GS, desc_act=True))
        with pytest.raises(NotImplementedError, match="desc_act"):
            scheme.process_weights_after_loading(torch.nn.Module())

    def test_w8_rejects_desc_act(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig
        from vllm_ascend.quantization.methods.gptq import (
            AscendW8A16GPTQFusedMoEMethod,
        )
        scheme = AscendW8A16GPTQFusedMoEMethod(
            GPTQConfig(weight_bits=8, group_size=GS, desc_act=True))
        with pytest.raises(NotImplementedError, match="desc_act"):
            scheme.process_weights_after_loading(torch.nn.Module())


class TestMoEValueErrorGuards:
    """T18/2.10: the MoE get_weight pack_factor divisibility checks and the
    apply() activation check must raise ValueError (not assert) so they
    survive ``python -O`` in production."""

    def test_awq_moe_pack_factor_guard(self):
        from vllm_ascend.quantization.awq_config import AWQConfig
        from vllm_ascend.quantization.methods.w4a16_awq import (
            AscendW4A16AWQFusedMoEMethod,
        )
        scheme = AscendW4A16AWQFusedMoEMethod(
            AWQConfig(weight_bits=4, group_size=GS, zero_point=True))
        # hidden_sizes=257 is not divisible by pack_factor=8
        with pytest.raises(ValueError, match="pack_factor"):
            scheme.get_weight(num_experts=2, intermediate_size_per_partition=128,
                              hidden_sizes=257, params_dtype=DTYPE)

    def test_gptq_w4_moe_pack_factor_guard(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig
        from vllm_ascend.quantization.methods.gptq import (
            AscendW4A16GPTQFusedMoEMethod,
        )
        scheme = AscendW4A16GPTQFusedMoEMethod(
            GPTQConfig(weight_bits=4, group_size=GS, desc_act=False))
        with pytest.raises(ValueError, match="pack_factor"):
            scheme.get_weight(num_experts=2, intermediate_size_per_partition=128,
                              hidden_sizes=257, params_dtype=DTYPE)

    def test_gptq_w8_moe_pack_factor_guard(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig
        from vllm_ascend.quantization.methods.gptq import (
            AscendW8A16GPTQFusedMoEMethod,
        )
        scheme = AscendW8A16GPTQFusedMoEMethod(
            GPTQConfig(weight_bits=8, group_size=GS, desc_act=False))
        with pytest.raises(ValueError, match="pack_factor"):
            scheme.get_weight(num_experts=2, intermediate_size_per_partition=127,
                              hidden_sizes=256, params_dtype=DTYPE)

    def test_awq_moe_group_size_guard(self):
        from vllm_ascend.quantization.awq_config import AWQConfig
        from vllm_ascend.quantization.methods.w4a16_awq import (
            AscendW4A16AWQFusedMoEMethod,
        )
        scheme = AscendW4A16AWQFusedMoEMethod(
            AWQConfig(weight_bits=4, group_size=GS, zero_point=True))
        # intermediate=100 not divisible by group_size=64
        with pytest.raises(ValueError, match="divisible"):
            scheme.get_dynamic_quant_param(num_experts=2, intermediate_size_per_partition=100,
                                           hidden_sizes=256, params_dtype=DTYPE)

    def test_gptq_moe_group_size_guard(self):
        from vllm_ascend.quantization.gptq_config import GPTQConfig
        from vllm_ascend.quantization.methods.gptq import (
            AscendW4A16GPTQFusedMoEMethod,
        )
        scheme = AscendW4A16GPTQFusedMoEMethod(
            GPTQConfig(weight_bits=4, group_size=GS, desc_act=False))
        with pytest.raises(ValueError, match="divisible"):
            scheme.get_dynamic_quant_param(num_experts=2, intermediate_size_per_partition=100,
                                           hidden_sizes=256, params_dtype=DTYPE)

    def test_activation_guard_is_value_error(self):
        # The activation guard must raise ValueError (not assert), so it is not
        # stripped by python -O. We check the apply method raises before
        # touching hardware by passing an unsupported activation.
        from vllm_ascend.quantization.awq_config import AWQConfig
        from vllm_ascend.quantization.methods.w4a16_awq import (
            AscendW4A16AWQFusedMoEMethod,
        )
        scheme = AscendW4A16AWQFusedMoEMethod(
            AWQConfig(weight_bits=4, group_size=GS, zero_point=True))
        x = torch.zeros(1, H, dtype=DTYPE)
        router = torch.zeros(1, E, dtype=DTYPE)
        with pytest.raises(ValueError, match="SiLU"):
            scheme.apply(torch.nn.Module(), x, router, top_k=1, renormalize=True,
                         num_experts=E, activation="gelu")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
