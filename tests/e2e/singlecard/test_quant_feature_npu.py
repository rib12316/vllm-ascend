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

"""NPU correctness tests for two AWQ/GPTQ features that need real-hardware verify:

1. per-channel quantization (group_size=-1 → antiquant_group_size=0) for both
   4-bit (int4pack) and 8-bit (int8) Linear — verifies the NPU fused op accepts
   per-channel scales and matches a dense oracle.
2. GPTQ MoE negative-scale absorption — verifies that where antiquant_scale < 0,
   the weight+offset sign flip + abs(scale) keeps the NPU op output equal to a
   dense oracle computed with the ORIGINAL (negative) scales.

Requires Ascend NPU hardware; skipped otherwise.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.quantization.methods.gptq import (
    _repack_gptq_moe_qweight,
    _unpack_qzeros_from_int32,
)


def _npu_available():
    return hasattr(torch, "npu") and hasattr(torch.Tensor, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(not _npu_available(), reason="Ascend NPU not available")


@pytest.fixture(autouse=True)
def _set_device():
    torch.npu.set_device(0)
    yield


def _oracle(sint_w, scale, x):
    w_ref = sint_w.float() * scale.cpu().float()
    return x.cpu().float() @ w_ref


# ---------------------------------------------------------------------------
# per-channel (group_size=-1)
# ---------------------------------------------------------------------------


def test_int4_per_channel_matches_dense():
    """4-bit int4pack + antiquant_group_size=0 (per-channel) matches dense."""
    torch.manual_seed(0)
    M, K, N = 8, 128, 64
    sint4 = torch.randint(-8, 7, (K, N))
    packed = torch_npu.npu_convert_weight_to_int4pack(sint4.to(torch.int32).npu())
    scale = (torch.rand(N, dtype=torch.float16) * 0.1 + 0.01).npu()
    offset = torch.zeros(N, dtype=torch.float16).npu()
    x = torch.randn(M, K, dtype=torch.float16).npu()
    out = torch_npu.npu_weight_quant_batchmatmul(
        x, packed, antiquant_scale=scale, antiquant_offset=offset, antiquant_group_size=0
    )
    ref = _oracle(sint4, scale, x)
    rel = (out.cpu().float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
    assert not torch.isnan(out).any()
    assert rel < 0.15, f"4-bit per-channel rel_err {rel:.4f} > 0.15"


def test_int8_per_channel_matches_dense():
    """8-bit int8 + antiquant_group_size=0 (per-channel) matches dense."""
    torch.manual_seed(0)
    M, K, N = 8, 128, 64
    sint8 = torch.randint(-128, 127, (K, N)).to(torch.int8).npu()
    scale = (torch.rand(N, dtype=torch.float16) * 0.1 + 0.01).npu()
    offset = torch.zeros(N, dtype=torch.float16).npu()
    x = torch.randn(M, K, dtype=torch.float16).npu()
    out = torch_npu.npu_weight_quant_batchmatmul(
        x, sint8, antiquant_scale=scale, antiquant_offset=offset, antiquant_group_size=0
    )
    ref = _oracle(sint8.cpu().to(torch.int32), scale, x)
    rel = (out.cpu().float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
    assert not torch.isnan(out).any()
    assert rel < 0.15, f"8-bit per-channel rel_err {rel:.4f} > 0.15"


# ---------------------------------------------------------------------------
# GPTQ MoE negative-scale absorption
# ---------------------------------------------------------------------------


def test_gptq_moe_negative_scale_absorbed(tmp_path):
    """Negative MoE scales absorbed into weight+offset; NPU output matches dense."""
    torch.manual_seed(42)
    E, K, N, GS, M, PF = 2, 128, 64, 64, 8, 8

    q_uint = torch.randint(0, 16, (E, K, N))
    qweight = torch.zeros(E, K // PF, N, dtype=torch.int32)
    for i in range(PF):
        qweight |= q_uint[:, i::PF, :].to(torch.int32) << (4 * i)
    scales = torch.randn(E, K // GS, N) * 0.05  # contains negatives
    assert (scales < 0).any()
    zeros_uint = torch.randint(0, 16, (E, K // GS, N))
    qzeros = torch.zeros(E, K // GS, N // PF, dtype=torch.int32)
    for i in range(PF):
        qzeros |= zeros_uint[..., i::PF].to(torch.int32) << (4 * i)

    # dense oracle with ORIGINAL (negative) scales
    x = torch.randn(M, K, dtype=torch.float16).npu()
    zp_exp = zeros_uint.repeat_interleave(GS, dim=1)
    sc_exp = scales.repeat_interleave(GS, dim=1)
    w_ref = (q_uint.float() - zp_exp.float()) * sc_exp.float()
    oracle = torch.stack([x.cpu().float() @ w_ref[e] for e in range(E)])

    # production path: offset → negative-scale absorption → repack
    zeros_unpacked = _unpack_qzeros_from_int32(qzeros, 4, use_v2_format=True).float()
    offset = -(zeros_unpacked - 8)
    repacked, new_offset, new_scales = _repack_gptq_moe_qweight(
        qweight.npu(), 4, PF, scales=scales.npu(), offset=offset.npu(), group_size=GS
    )
    assert (new_scales >= 0).all()

    def clean(t):
        return t.detach().cpu().contiguous().npu()

    outs = []
    for e in range(E):
        out = torch_npu.npu_weight_quant_batchmatmul(
            x,
            clean(repacked[e]),
            antiquant_scale=clean(new_scales[e].to(torch.float16)),
            antiquant_offset=clean(new_offset[e].to(torch.float16)),
            antiquant_group_size=GS,
        )
        outs.append(out.cpu().float())
    npu_out = torch.stack(outs)

    assert not torch.isnan(npu_out).any()
    rel = (npu_out - oracle).abs().max().item() / max(oracle.abs().max().item(), 1e-6)
    assert rel < 0.15, f"negative-scale MoE rel_err {rel:.4f} > 0.15"
