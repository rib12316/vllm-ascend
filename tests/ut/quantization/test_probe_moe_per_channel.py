"""AG6 NPU probe — does npu_grouped_matmul support per-channel antiquant?

**NPU-only.** Skipped automatically when NPU is unavailable.

Answers the AG6 question that is NOT decidable from the op doc:
``npu_grouped_matmul`` has **no** ``antiquant_group_size`` parameter (unlike
``npu_weight_quant_batchmatmul``), so per-channel (``group_size = -1``)
granularity must be expressed implicitly via the ``antiquant_scale`` shape.
Whether the op accepts a per-channel-shaped scale (one value per output
channel) and produces correct output can only be confirmed on hardware.

Uses the ``split_item=0, group_type=-1`` ("多多多") calling convention: x,
weight, antiquant_scale, antiquant_offset are **lists of one tensor per
expert** and ``group_list=None``. We use the fp16 pseudo-quant antiquant path
(``y = x · (weight + offset) · scale``) to isolate the per-channel granularity
question from int4 packing.

Run in Phase B::

    pytest tests/ut/quantization/test_probe_moe_per_channel.py -s
"""

from __future__ import annotations

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover
    torch_npu = None  # type: ignore[assignment]


def _npu_available() -> bool:
    if torch_npu is None:
        return False
    try:
        return bool(torch.npu.is_available()) and torch.npu.device_count() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _npu_available(), reason="NPU not available (run with NPU started)")
def test_grouped_matmul_per_channel_probe(capsys):
    """Diagnostic: try per-channel antiquant_scale shapes; print verdict.

    Not asserted pass/fail — the answer is genuinely unknown until run on hw.
    """
    E, N, K = 2, 32, 64  # experts, out, in
    device = "npu:0"
    torch.manual_seed(0)
    # One x tensor per expert (T_e tokens each); int8 weight [N, K] per expert.
    # Antiquant path REQUIRES int weight (fp16 weight ⇒ "nonquant" ⇒ error).
    xs = [torch.randn(4, K, dtype=torch.float16, device=device) for _ in range(E)]
    ws = [torch.randint(-16, 16, (N, K), dtype=torch.int8, device=device) for _ in range(E)]

    def ref(scale_per_expert, offset_per_expert):
        """y_e = x_e @ ((w_e + off_e) * scale_e).T, scale broadcast over K."""
        outs = []
        for x, w, s, o in zip(xs, ws, scale_per_expert, offset_per_expert, strict=True):
            deq = (w.to(torch.float32) + o.to(torch.float32)) * s.to(torch.float32)
            outs.append((x.to(torch.float32) @ deq.t()).to(torch.float16))
        return outs

    def run(scale_list, offset_list, label):
        try:
            out = torch_npu.npu_grouped_matmul(
                x=xs,
                weight=ws,
                antiquant_scale=scale_list,
                antiquant_offset=offset_list,
                group_list=None,
                split_item=0,
                group_type=-1,
            )
            return out
        except Exception as exc:  # noqa: BLE001
            return f"REJECTED ({type(exc).__name__}: {exc})"

    lines = ["=== AG6 npu_grouped_matmul per-channel probe (fp16 pseudo-quant) ==="]

    def report(label, out, scale_list, offset_list):
        if isinstance(out, str):
            lines.append(f"{label}: {out}")
            return
        ref_out = ref(scale_list, offset_list)
        max_err = max((out[e] - ref_out[e]).abs().max().item() for e in range(E))
        verdict = "MATCH" if max_err < 0.05 else "MISMATCH"
        lines.append(f"{label}: ACCEPTED {tuple(out[0].shape)}, err={max_err:.4f} ({verdict})")

    # Per-element scale [N,K] — known-good baseline (one scale per weight elem).
    s_elem = [torch.ones(N, K, dtype=torch.float16, device=device) for _ in range(E)]
    o_elem = [torch.zeros(N, K, dtype=torch.float16, device=device) for _ in range(E)]
    report("per-element [N,K]", run(s_elem, o_elem, ""), s_elem, o_elem)

    # Per-channel hypothesis A: scale [N, 1]
    s_n1 = [torch.ones(N, 1, dtype=torch.float16, device=device) for _ in range(E)]
    o_n1 = [torch.zeros(N, 1, dtype=torch.float16, device=device) for _ in range(E)]
    report("per-channel [N,1]", run(s_n1, o_n1, ""), s_n1, o_n1)

    # Per-channel hypothesis B: scale [N] (1-D)
    s_n = [torch.ones(N, dtype=torch.float16, device=device) for _ in range(E)]
    o_n = [torch.zeros(N, dtype=torch.float16, device=device) for _ in range(E)]
    report("per-channel [N] ", run(s_n, o_n, ""), s_n, o_n)

    lines.append("=== verdict (2026-07-05 NPU run) ===")
    lines.append("Standalone npu_grouped_matmul antiquant: int weight required (fp16=>nonquant")
    lines.append("error); antiquant_scale must be 1-D; per-channel shapes [N,1]/[N] rejected with")
    lines.append("conflicting shape messages. The working MoE per-group antiquant goes through")
    lines.append("fused_experts (different code path/convention). Definitive AG6 answer requires")
    lines.append("removing the MoE rejection + running test_quant_moe_synthetic with group_size=-1")
    lines.append("(code change + NPU run) — deferred: AG6 is P2/niche (group_size=-1 MoE is rare);")
    lines.append("the rejection is NOT obviously over-conservative.")
    msg = "\n".join(lines)
    with capsys.disabled():
        print("\n" + msg)
    assert "per-element" in msg
