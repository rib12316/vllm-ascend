"""Offline (CPU, no NPU) tests for AscendW4A16LinearScheme (AG10 Linear gap).

Run with ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` (NPU not started). Coverage:

* registry: ``get_scheme_class("W4A16", "linear")`` returns the new scheme
  (previously ``None`` → ``NotImplementedError`` for dense compressed-tensors
  int4 models).
* ``get_weight`` / ``get_pergroup_param`` return the compressed-tensors
  shapes (``weight_packed`` packed along the input dim, ``weight_scale``
  per-group ``[out, in//gs]``).
* ``process_weights_after_loading`` glue: the packed weight is unpacked,
  transposed to ``(K, N) = [in, out]`` and fed to the int4pack op; the scale
  is transposed to ``[in//gs, out]``; a zero offset of matching shape is built.
* ``apply`` reshapes output to the saved ``w4a16_output_size``.

The ``npu_*`` ops are mocked, so no NPU hardware is required. The real
``npu_convert_weight_to_int4pack`` + ``npu_weight_quant_batchmatmul`` math is
validated on NPU in Phase B (``run_phase_b_npu_validation``).
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch_npu  # noqa: F401  package import only; all npu_* ops are mocked below

from vllm_ascend.quantization.methods.registry import get_scheme_class
from vllm_ascend.quantization.methods.w4a16 import AscendW4A16LinearScheme, unpack_from_int32


def _make_scheme(group_size: int = 32) -> AscendW4A16LinearScheme:
    """Construct a scheme, mocking get_current_vllm_config() used in __init__."""
    fake_vllm_config = MagicMock()
    fake_vllm_config.quant_config.quant_description.get = (
        lambda key, default=None: group_size if key == "group_size" else default
    )
    with patch(
        "vllm_ascend.quantization.methods.w4a16.get_current_vllm_config",
        return_value=fake_vllm_config,
    ):
        return AscendW4A16LinearScheme()


def _pack_int4_to_int32(signed_int4: torch.Tensor) -> torch.Tensor:
    """Inverse of ``unpack_from_int32`` (packed_dim=1, LSB-first, +8 centering).

    Maps signed int4 values in [-8, 7] to packed int32 [out, in // 8], matching
    the compressed-tensors / LLM-Compressor weight_packed layout consumed by
    the scheme.
    """
    assert signed_int4.dtype == torch.int8
    out_features, in_features = signed_int4.shape
    assert in_features % 8 == 0
    uint4 = (signed_int4.to(torch.int32) + 8) & 0xF  # [-8..7] -> [0..15]
    packed = torch.zeros(out_features, in_features // 8, dtype=torch.int32)
    for i in range(8):
        packed |= (uint4[:, i::8] & 0xF) << (4 * i)
    return packed


# ---------------------------------------------------------------------------
# Registration (proves the AG10 Linear gap is closed at the routing layer)
# ---------------------------------------------------------------------------


def test_registry_returns_w4a16_linear_scheme():
    """get_scheme_class("W4A16", "linear") used to return None; now our scheme."""
    assert get_scheme_class("W4A16", "linear") is AscendW4A16LinearScheme


# ---------------------------------------------------------------------------
# Parameter shapes
# ---------------------------------------------------------------------------


def test_get_weight_shapes_and_packing_metadata():
    scheme = _make_scheme(group_size=32)
    spec = scheme.get_weight(input_size=64, output_size=16, params_dtype=torch.bfloat16)
    # weight_packed: [out, in // pack_factor] int32, packed along the input dim
    assert spec["weight_packed"].shape == torch.Size([16, 8])
    assert spec["weight_packed"].dtype == torch.int32
    # weight_shape: [2] metadata
    assert spec["weight_shape"].shape == torch.Size([2])
    assert spec["weight_shape"].dtype == torch.int32
    # packing metadata consumed by AscendLinearMethod.create_weights
    assert spec["_packed_dim"] == 1
    assert spec["_packed_factor"] == 8
    assert spec["_param_dims"]["weight_packed"] == {"input_dim": 1, "output_dim": 0}
    assert "weight_shape" in spec["_unpacked_params"]


def test_get_pergroup_param_shapes():
    scheme = _make_scheme(group_size=32)
    spec = scheme.get_pergroup_param(input_size=64, output_size=16, params_dtype=torch.bfloat16)
    # weight_scale: [out, in // group_size]
    assert spec["weight_scale"].shape == torch.Size([16, 2])
    assert spec["weight_scale"].dtype == torch.bfloat16
    assert spec["_param_dims"]["weight_scale"] == {"input_dim": 1, "output_dim": 0}


def test_get_weight_rejects_unaligned_input():
    scheme = _make_scheme(group_size=32)
    with pytest.raises(AssertionError):
        scheme.get_weight(input_size=60, output_size=16, params_dtype=torch.bfloat16)  # 60 % 8 != 0


# ---------------------------------------------------------------------------
# process_weights_after_loading glue (npu int4pack op mocked)
# ---------------------------------------------------------------------------


def test_process_weights_after_loading_transposes_to_kn_and_builds_zero_offset():
    scheme = _make_scheme(group_size=16)
    out_features, in_features = 8, 32  # in % 8 == 0, in % 16 == 0
    gs = 16

    torch.manual_seed(0)
    signed = torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8)
    weight_packed = _pack_int4_to_int32(signed)  # [out, in//8] int32
    weight_scale = torch.rand(out_features, in_features // gs, dtype=torch.bfloat16)
    weight_shape = torch.tensor([in_features, out_features], dtype=torch.int32)

    layer = torch.nn.Module()
    layer.weight_packed = torch.nn.Parameter(weight_packed, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    layer.weight_shape = torch.nn.Parameter(weight_shape, requires_grad=False)

    captured = {}

    def fake_int4pack(t):
        captured["input"] = t.clone()
        return t.clone()  # semantics irrelevant; the packed result is not asserted here

    # create=True: under TORCH_DEVICE_BACKEND_AUTOLOAD=0 some npu ops are not
    # pre-attached as module attributes, so patch must create the attribute.
    with patch("torch_npu.npu_convert_weight_to_int4pack", side_effect=fake_int4pack, create=True):
        scheme.process_weights_after_loading(layer)

    # The int4pack op must receive the weight transposed to (K, N) = [in, out],
    # int32, with values equal to the centered signed int4 weights we packed.
    expected_kn = signed.transpose(0, 1).contiguous().to(torch.int32)
    assert captured["input"].shape == torch.Size([in_features, out_features])
    assert torch.equal(captured["input"], expected_kn)
    # Scale transposed to [in // gs, out]
    assert layer.weight_scale.data.shape == torch.Size([in_features // gs, out_features])
    assert torch.equal(layer.weight_scale.data, weight_scale.transpose(0, 1).contiguous())
    # Symmetric quantization => zero antiquant offset, same shape as the scale
    assert layer.weight_offset.data.shape == layer.weight_scale.data.shape
    assert torch.equal(layer.weight_offset.data, torch.zeros_like(layer.weight_scale.data))
    # Saved output size == pre-repack output dim
    assert layer.w4a16_output_size == out_features


def test_process_weights_roundtrips_through_unpack():
    """Sanity: our _pack_int4_to_int32 helper is the exact inverse of the
    scheme's unpack_from_int32, so the unpacked weight equals the original
    signed int4 weight (validates the packing convention used by the tests)."""
    out_features, in_features = 4, 16
    torch.manual_seed(1)
    signed = torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8)
    packed = _pack_int4_to_int32(signed)
    unpacked = unpack_from_int32(packed, torch.Size([out_features, in_features]), num_bits=4)
    assert unpacked.shape == signed.shape
    assert torch.equal(unpacked, signed)


# ---------------------------------------------------------------------------
# apply output shaping (npu batchmatmul op mocked)
# ---------------------------------------------------------------------------


def test_apply_reshapes_to_saved_output_size():
    scheme = _make_scheme(group_size=16)
    out_features, in_features = 8, 32
    layer = torch.nn.Module()
    layer.weight_packed = torch.nn.Parameter(
        torch.zeros(out_features, in_features // 8, dtype=torch.int32), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.zeros(in_features // 16, out_features, dtype=torch.bfloat16), requires_grad=False
    )
    layer.weight_offset = torch.nn.Parameter(
        torch.zeros(in_features // 16, out_features, dtype=torch.bfloat16), requires_grad=False
    )
    layer.w4a16_output_size = out_features

    x = torch.randn(2, 5, in_features)

    def fake_bmm(reshaped_x, weight, **kwargs):
        # Echo an output with the true N dim so the reshape logic is exercised.
        return torch.zeros(reshaped_x.shape[0], out_features, dtype=torch.float32)

    with patch("torch_npu.npu_weight_quant_batchmatmul", side_effect=fake_bmm, create=True):
        out = scheme.apply(layer, x)

    assert out.shape == torch.Size([2, 5, out_features])


# ---------------------------------------------------------------------------
# Real NPU math (no mocks) — validates the actual int4pack + batchmatmul path.
# Skipped without NPU hardware.
# ---------------------------------------------------------------------------


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available()) and torch.npu.device_count() > 0
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _npu_available(), reason="NPU not available")
def test_apply_matches_reference_dequant_on_npu():
    """End-to-end on real NPU: process_weights (real int4pack) + apply (real
    batchmatmul) must match a pure-torch reference dequant + matmul.

    This is the validation that the AG10 Linear scheme's math is correct — the
    offline tests above only cover the glue with mocked ops.
    """
    device = "npu:0"
    gs = 32
    in_features, out_features = 64, 16  # both divisible by pack_factor=8 and gs
    scheme = _make_scheme(group_size=gs)

    torch.manual_seed(0)
    signed = torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8)
    scale = torch.rand(out_features, in_features // gs, dtype=torch.float16) * 0.05 + 0.01

    weight_packed = _pack_int4_to_int32(signed).to(device)
    weight_scale = scale.to(device)
    weight_shape = torch.tensor([in_features, out_features], dtype=torch.int32).to(device)

    layer = torch.nn.Module().to(device)
    layer.weight_packed = torch.nn.Parameter(weight_packed, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    layer.weight_shape = torch.nn.Parameter(weight_shape, requires_grad=False)

    scheme.process_weights_after_loading(layer)  # real npu_convert_weight_to_int4pack

    x = torch.randn(3, 7, in_features, dtype=torch.float16, device=device)
    out = scheme.apply(layer, x)  # real npu_weight_quant_batchmatmul

    # Reference: dequantize per-group symmetric int4, then Linear (x @ W.T).
    scale_per_elem = scale.float().repeat_interleave(gs, dim=1).to(device)  # [out, in]
    weight_dequant = signed.float().to(device) * scale_per_elem  # [out, in]
    ref = x.reshape(-1, in_features).float() @ weight_dequant.t()  # [tokens, out]
    ref = ref.reshape(3, 7, out_features).to(torch.float16)

    assert out.shape == torch.Size([3, 7, out_features])
    # int4 dequant + fused op should match the reference to fp16 precision.
    max_abs_err = (out.float() - ref.float()).abs().max().item()
    mean_abs = ref.float().abs().mean().item()
    assert max_abs_err / max(mean_abs, 1e-6) < 0.05, (
        f"AG10 Linear NPU math mismatch: max_abs_err={max_abs_err:.4f}, "
        f"mean_abs_ref={mean_abs:.4f}, rel={max_abs_err / max(mean_abs, 1e-6):.4f}"
    )
