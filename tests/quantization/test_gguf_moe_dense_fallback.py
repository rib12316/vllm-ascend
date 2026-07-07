"""Synthetic correctness test for GGUF MoE dense compute (row 4).

Validates the numerically-verified dense swiglu MoE reference
(``dense_moe_reference``) — the correctness oracle for the dense fallback. The
production dense path delegates to ``AscendUnquantizedFusedMoEMethod`` for
correct Ascend-pipeline integration (validated end-to-end on a real model, not
here), but the underlying MoE math is this helper.

Weight orientation (R7, SETTLED at real e2e vs SzymonOzog/test-gguf-moe-sample):
BOTH w13 [E,2*inter,H] and w2 [E,H,inter] are [out,in] → both use F.linear.
"""

import torch
import torch.nn.functional as F

from vllm_ascend.quantization.methods.gguf_moe import dense_moe_reference


def _independent_reference(x, w13, w2, topk_weights, topk_ids):
    """Second, independently-written reference (batched matmul) to cross-check."""
    inter = w13.shape[1] // 2
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        acc = torch.zeros(w2.shape[1], dtype=x.dtype)  # [H]
        for w, i in zip(topk_weights[t], topk_ids[t]):
            gu = torch.matmul(w13[int(i)], x[t])  # [2*inter] = [out,in]@[in]
            g, u = gu[:inter], gu[inter:]
            acc = acc + torch.matmul(w2[int(i)], F.silu(g) * u) * float(w)
        out[t] = acc
    return out


def test_dense_reference_matches_independent():
    torch.manual_seed(0)
    E, H, inter = 3, 16, 24
    num_tok, topk = 5, 2
    dtype = torch.float16
    w13 = torch.randn(E, 2 * inter, H, dtype=dtype)  # [out, in]
    w2 = torch.randn(E, H, inter, dtype=dtype)  # [out, in]
    x = torch.randn(num_tok, H, dtype=dtype)
    topk_weights = torch.randn(num_tok, topk, dtype=dtype).softmax(dim=-1)
    topk_ids = torch.randint(0, E, (num_tok, topk))

    out = dense_moe_reference(x, w13, w2, topk_weights, topk_ids)
    ref = _independent_reference(x, w13, w2, topk_weights, topk_ids)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_dense_reference_single_expert():
    # topk_ids all → expert 0: output == expert-0 swiglu MLP of x.
    torch.manual_seed(1)
    E, H, inter = 2, 8, 12
    dtype = torch.float16
    w13 = torch.randn(E, 2 * inter, H, dtype=dtype)
    w2 = torch.randn(E, H, inter, dtype=dtype)
    x = torch.randn(3, H, dtype=dtype)
    topk_weights = torch.ones(3, 1, dtype=dtype)
    topk_ids = torch.zeros(3, 1, dtype=torch.long)

    out = dense_moe_reference(x, w13, w2, topk_weights, topk_ids)
    gu = F.linear(x, w13[0])
    g, u = gu[:, :inter], gu[:, inter:]
    ref = F.linear(F.silu(g) * u, w2[0])
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
