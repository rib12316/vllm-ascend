"""Synthetic correctness test for GGUF MoE dense fallback (row 4).

Uses GGUF F16 type: qweight stores the dense weight verbatim as bytes, so
``dequantize`` is a bit-exact identity. This isolates the MoE *compute* (routing
+ matmul + swiglu + weighted combine) from quantization noise. Dequant
correctness for real quant types (Q8_0/Q4_K/...) is covered by the existing
``test_gguf_dequant.py`` bit-exact suite.

Weight orientation (R7, SETTLED at real e2e vs SzymonOzog/test-gguf-moe-sample):
BOTH w13 [E,2*inter,H] and w2 [E,H,inter] are [out,in] → both use F.linear.
"""

import types

import torch
import torch.nn.functional as F
from gguf import GGMLQuantizationType as WT

from vllm_ascend.quantization.methods.gguf_moe import AscendGGUFMoEMethod


def _make_layer(w13: torch.Tensor, w2: torch.Tensor) -> types.SimpleNamespace:
    """Build a minimal layer with the sideload params process_weights reads."""
    return types.SimpleNamespace(
        w13_qweight=w13.view(torch.uint8),  # [E, 2*inter, H*2] (fp16 → 2 bytes)
        w2_qweight=w2.view(torch.uint8),  # [E, H, inter*2]
        w13_qweight_type=types.SimpleNamespace(weight_type=int(WT.F16)),
        w2_qweight_type=types.SimpleNamespace(weight_type=int(WT.F16)),
    )


def _reference_moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Pure-torch reference MoE (swiglu); both weights [out,in] → F.linear."""
    inter = w13.shape[1] // 2
    ref = torch.empty_like(x)
    for tok in range(x.shape[0]):
        cur = None
        for w, i in zip(topk_weights[tok], topk_ids[tok]):
            gu = F.linear(x[tok], w13[int(i)])  # [2*inter]
            g, u = gu[:inter], gu[inter:]
            a = F.silu(g) * u  # [inter]
            d = F.linear(a, w2[int(i)]) * float(w)  # [H]
            cur = d if cur is None else cur + d
        ref[tok] = cur if cur is not None else torch.zeros_like(x[tok])
    return ref


def test_dense_fallback_matches_reference():
    torch.manual_seed(0)
    E, H, inter = 3, 16, 24
    num_tok, topk = 5, 2
    dtype = torch.float16
    w13 = torch.randn(E, 2 * inter, H, dtype=dtype)  # [out, in]
    w2 = torch.randn(E, H, inter, dtype=dtype)  # [out, in]

    layer = _make_layer(w13, w2)
    method = AscendGGUFMoEMethod(quant_config=None, moe=None)
    method.process_weights_after_loading(layer)

    # process_weights must reproduce the original dense weights bit-exact (F16).
    assert torch.equal(layer.w13_weight, w13)
    assert torch.equal(layer.w2_weight, w2)

    x = torch.randn(num_tok, H, dtype=dtype)
    topk_weights = torch.randn(num_tok, topk, dtype=dtype).softmax(dim=-1)
    topk_ids = torch.randint(0, E, (num_tok, topk))

    out = method.apply(layer, x, topk_weights, topk_ids)
    ref = _reference_moe(x, w13, w2, topk_weights, topk_ids)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_dense_fallback_single_expert_selection():
    # topk_ids all → expert 0: output == expert-0 swiglu MLP of x.
    torch.manual_seed(1)
    E, H, inter = 2, 8, 12
    dtype = torch.float16
    w13 = torch.randn(E, 2 * inter, H, dtype=dtype)  # [out, in]
    w2 = torch.randn(E, H, inter, dtype=dtype)  # [out, in]
    layer = _make_layer(w13, w2)
    method = AscendGGUFMoEMethod(quant_config=None, moe=None)
    method.process_weights_after_loading(layer)

    num_tok = 3
    x = torch.randn(num_tok, H, dtype=dtype)
    topk_weights = torch.ones(num_tok, 1, dtype=dtype)
    topk_ids = torch.zeros(num_tok, 1, dtype=torch.long)
    out = method.apply(layer, x, topk_weights, topk_ids)

    gu = F.linear(x, w13[0])  # [num_tok, 2*inter]
    g, u = gu[:, :inter], gu[:, inter:]
    ref = F.linear(F.silu(g) * u, w2[0])  # [num_tok, H]
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
