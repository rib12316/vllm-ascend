"""Row 3 validation: 3D per-expert repack == stacked 2D Linear repack (real bytes).

The 2D ``gguf_repack.repack_to_npu`` is already verified (in production for gguf
Linear). ``repack_moe_to_npu`` only extends it to MoE by looping over experts and
stacking, so per-expert numerical correctness is INHERITED. This test validates
exactly the new part: that expert e's slice of the 3D result equals the
standalone 2D repack of expert e (packed weight + per-group scale + offset),
across a few experts, on real quantized gguf bytes. No int4pack layout
reverse-engineering needed.

Run on NPU (int4pack needs the runtime).
"""
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from gguf import GGUFReader

sys.argv = ["x"]
S = "/hub/models--SzymonOzog--test-gguf-moe-sample/snapshots/2b77eb27ae2ea4b3f68cf3042961509a9f5847b8"

from vllm_ascend.quantization.methods.gguf_moe import repack_moe_to_npu  # noqa: E402
from vllm_ascend.quantization.methods.gguf_repack import repack_to_npu  # noqa: E402


def run(qname: str, proj_idx: int) -> bool:
    r = GGUFReader(Path(S) / f"Quant_{qname}_512.gguf")
    t = sorted(r.tensors, key=lambda x: -x.data.shape[-1])[proj_idx]
    qt = int(t.tensor_type)
    w_q = torch.tensor(t.data).npu()  # [E, N, K_bytes]
    E = w_q.shape[0]

    moe = repack_moe_to_npu(w_q, qt, torch.float16)
    if moe is None:
        print(f"[{qname} proj{proj_idx}] not repackable — skip")
        return True
    m_packed, m_scale, m_offset, m_gs = moe

    ok = True
    for e in [0, 1, E // 2, E - 1]:  # spot-check experts
        qw, scale, offset, gs = repack_to_npu(w_q[e], qt, torch.float16)
        same = (
            torch.equal(m_packed[e], qw)
            and torch.equal(m_scale[e], scale)
            and torch.equal(m_offset[e], offset)
            and m_gs == gs
        )
        ok = ok and same
    print(f"[{qname} proj{proj_idx}] E={E} packed{tuple(m_packed.shape)} "
          f"scale{tuple(m_scale.shape)} gs={m_gs} {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = {}
    for q in ["Q4_K", "Q4_0", "Q8_0", "Q5_K"]:
        results[f"{q}_w13"] = run(q, 0)
        results[f"{q}_w2"] = run(q, 1)
    print("\nSUMMARY:", results)
    sys.exit(0 if all(results.values()) else 1)
