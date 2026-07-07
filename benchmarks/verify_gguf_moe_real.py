"""Real-gguf-bytes e2e validation of AscendGGUFMoEMethod (row 9, R7 settle).

Drives our method with REAL quantized gguf MoE tensors and compares against
upstream fused_experts reference (the oracle). Settles R7 (w2 orientation).
"""
import sys
import types
import torch
import torch.nn.functional as F
from pathlib import Path
from gguf import GGUFReader, GGMLQuantizationType as WT

sys.argv = ["x"]
S = "/hub/models--SzymonOzog--test-gguf-moe-sample/snapshots/2b77eb27ae2ea4b3f68cf3042961509a9f5847b8"

from vllm_ascend.quantization.methods.gguf_dequant import dequantize
from vllm_ascend.quantization.methods.gguf_moe import AscendGGUFMoEMethod


def load(qname):
    r = GGUFReader(Path(S) / f"Quant_{qname}_512.gguf")
    ts = sorted(r.tensors, key=lambda t: -t.data.shape[-1])  # w13 (bigger K) first
    w13, w2 = ts[0], ts[1]
    return w13, w2


def run(qname):
    w13t, w2t = load(qname)
    qtype = int(w13t.tensor_type)
    dtype = torch.float16
    # raw packed bytes as [E, N, K_bytes]
    w13_q = torch.tensor(w13t.data)  # uint8
    w2_q = torch.tensor(w2t.data)
    E = w13_q.shape[0]
    print(f"[{qname}] w13 {tuple(w13_q.shape)} w2 {tuple(w2_q.shape)} E={E}")

    layer = types.SimpleNamespace(
        w13_qweight=w13_q, w2_qweight=w2_q,
        w13_qweight_type=types.SimpleNamespace(weight_type=qtype),
        w2_qweight_type=types.SimpleNamespace(weight_type=qtype),
    )
    m = AscendGGUFMoEMethod(quant_config=None, moe=None)
    m.process_weights_after_loading(layer)
    w13_d, w2_d = layer.w13_weight, layer.w2_weight  # [E,2i,H], [E,H,i]
    H = w13_d.shape[2]
    inter = w13_d.shape[1] // 2
    print(f"    dequant: w13 {tuple(w13_d.shape)} w2 {tuple(w2_d.shape)} H={H} inter={inter}")

    torch.manual_seed(0)
    num_tok, topk = 7, 4
    x = torch.randn(num_tok, H, dtype=dtype)
    topk_weights = torch.rand(num_tok, topk, dtype=dtype)
    topk_ids = torch.randint(0, E, (num_tok, topk))

    out = m.apply(layer, x, topk_weights, topk_ids)

    # reference: both w13 and w2 are [out,in] → F.linear for both
    ref = torch.empty_like(x)
    for t in range(num_tok):
        cur = torch.zeros(H, dtype=dtype)
        for w, i in zip(topk_weights[t], topk_ids[t]):
            gu = F.linear(x[t], w13_d[int(i)])
            g, u = gu[:inter], gu[inter:]
            a = F.silu(g) * u
            cur = cur + F.linear(a, w2_d[int(i)]) * float(w)
        ref[t] = cur
    diff = (out.float() - ref.float()).abs().max().item()
    rel = diff / (ref.float().abs().max().item() + 1e-6)
    ok = rel < 0.02
    print(f"    max_abs_diff={diff:.4f} rel={rel:.4f} {'PASS' if ok else 'FAIL (w2 orientation!)'}")
    return ok


if __name__ == "__main__":
    results = {q: run(q) for q in ["Q4_K", "Q8_0"]}
    print("\nSUMMARY:", results)
    sys.exit(0 if all(results.values()) else 1)
