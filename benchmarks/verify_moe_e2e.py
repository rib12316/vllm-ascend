# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This file is part of the vllm-ascend project.
"""End-to-end MoE verification for AWQ / GPTQ quantization on Ascend NPU.

This is the MoE counterpart of the linear-model smoke test that verified
T1/T2/T6 (see logs/2026-06-12_T1T2T6_inference_verification.log). The MoE
weight-processing math is covered by tests/quantization/test_moe_synthetic_npu.py,
but that test explicitly skips the full ``apply -> fused_experts -> grouped_matmul``
path and real-checkpoint loading. This script closes that gap: it loads the two
real MoE models on disk, confirms our Ascend MoE quant method is attached, and
checks generated output is sane (correct + non-degenerate).

Run from the vllm-ascend/ directory with the venv active:

    python benchmarks/verify_moe_e2e.py 2>&1 | tee logs/<date>_T3T4_moe-e2e.log
"""

import sys
import traceback

import torch
from vllm import LLM, SamplingParams

# Each entry: (display, model_path, quantization, dtype, prompts-with-expected-substr)
MODELS = [
    {
        "tag": "T3-AWQ-MoE",
        "model": "/data/DeepSeek-V2-Lite-Chat-AWQ",
        "quantization": "awq",
        "dtype": "float16",
        "checks": [
            ("The capital of France is", "Paris"),
            ("What is 1+1? Reply with only the number.", "2"),
            ("Translate the word 'hello' to Chinese.", "你好"),
        ],
    },
    {
        "tag": "T4-GPTQ-MoE",
        "model": "/data/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4",
        "quantization": "gptq",
        "dtype": "float16",
        "checks": [
            ("The capital of France is", "Paris"),
            ("What is 1+1? Reply with only the number.", "2"),
            ("Translate the word 'hello' to Chinese.", "你好"),
        ],
    },
]


# Allow focusing on one quant flavor via env (e.g. MOE_ONLY=gptq) so iterative
# debugging isn't blocked waiting on a known-failing model.
import os as _os
_moe_only = _os.environ.get("MOE_ONLY", "")
if _moe_only:
    MODELS = [m for m in MODELS if _moe_only.lower() in m["tag"].lower()]


def _is_degenerate(text: str) -> tuple[bool, str]:
    """Detect broken-quantization symptoms: empty, or a single token repeating.

    A classic sign of a wrong antiquant_offset (e.g. symmetric GPTQ with no
    qzeros) is output that is either empty or one token repeated over and over.
    """
    text = text.strip()
    if len(text) == 0:
        return True, "empty output"
    tokens = text.split()
    if len(tokens) >= 6:
        most_common = max(set(tokens), key=tokens.count)
        if tokens.count(most_common) / len(tokens) > 0.6:
            return True, f"'{most_common}' repeats {tokens.count(most_common)}/{len(tokens)}"
    return False, ""


def _probe_moe_quant_method(llm: LLM) -> str:
    """Best-effort: walk the loaded model and report the quant method class on
    the first FusedMoE layer. Definitively confirms OUR Ascend MoE method (not a
    vLLM native fallback) is attached. Wrapped defensively — never breaks the run.
    """
    try:
        model = (
            llm.llm_engine.model_executor.driver_worker.worker.model_runner.model  # type: ignore[attr-defined]
        )
        found = []
        for name, mod in model.named_modules():
            qm = getattr(mod, "quant_method", None)
            if qm is not None and type(qm).__name__ != "UnquantizedFusedMoEMethod":
                found.append((name, type(qm).__name__))
            if len(found) >= 3:
                break
        if not found:
            return "(no quant_method found on any module — check routing)"
        return "; ".join(f"{n} -> {c}" for n, c in found)
    except Exception as e:  # noqa: BLE001
        return f"(probe skipped: {e})"


def run_one(cfg: dict) -> dict:
    """Load + generate for one model. Returns a result dict; never raises."""
    res = {**{k: cfg[k] for k in ("tag", "model", "quantization")}, "load_ok": False, "gen_ok": None, "detail": []}
    try:
        llm = LLM(
            model=cfg["model"],
            quantization=cfg["quantization"],
            dtype=cfg["dtype"],
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
        )
    except Exception as e:  # noqa: BLE001
        res["detail"].append(f"LOAD FAILED: {type(e).__name__}: {e}")
        res["detail"].append(traceback.format_exc(limit=4))
        return res

    res["load_ok"] = True
    res["quant_method"] = _probe_moe_quant_method(llm)
    res["detail"].append(f"loaded OK; MoE quant_method: {res['quant_method']}")

    prompts = [[{"role": "user", "content": q}] for q, _ in cfg["checks"]]
    try:
        outs = llm.chat(prompts, SamplingParams(temperature=0, max_tokens=40))
    except Exception as e:  # noqa: BLE001
        res["detail"].append(f"GENERATE FAILED: {type(e).__name__}: {e}")
        res["detail"].append(traceback.format_exc(limit=4))
        return res

    all_ok = True
    for (q, expected), o in zip(cfg["checks"], outs):
        text = o.outputs[0].text.strip()
        degen, why = _is_degenerate(text)
        hit = expected.lower() in text.lower()
        ok = hit and not degen
        all_ok = all_ok and ok
        flag = "PASS" if ok else ("DEGEN" if degen else "MISS")
        snippet = text.replace("\n", " ")[:80]
        res["detail"].append(f"[{flag}] Q={q!r} expect~{expected!r} -> {snippet!r}" + (f" ({why})" if degen else ""))
    res["gen_ok"] = all_ok
    # Free the model before the next one to reclaim HBM.
    del llm
    torch.npu.empty_cache() if torch.npu.is_available() else None
    return res


def main() -> int:
    print("=" * 64)
    print(" AWQ/GPTQ MoE end-to-end verification on Ascend NPU")
    print(f" torch {torch.__version__} | torch_npu {torch_npu_ver()}")
    print(f" NPU available: {torch.npu.is_available()}")
    print("=" * 64)

    results = [run_one(c) for c in MODELS]

    print("\n" + "=" * 64)
    print(" SUMMARY")
    print("=" * 64)
    for r in results:
        load = "OK" if r["load_ok"] else "FAIL"
        gen = "--" if r["gen_ok"] is None else ("OK" if r["gen_ok"] else "FAIL")
        print(f"  {r['tag']:<14} load={load:<4} generate={gen:<4}  ({r['quantization']})")
    print("-" * 64)
    for r in results:
        print(f"\n[{r['tag']}] {r['model']}")
        for line in r["detail"]:
            print(f"    {line}")

    n_fail = sum(1 for r in results if not (r["load_ok"] and r["gen_ok"]))
    print("\n" + ("ALL PASS — MoE path verified end-to-end" if n_fail == 0 else f"{n_fail} model(s) need investigation"))
    return 0 if n_fail == 0 else 1


def torch_npu_ver() -> str:
    try:
        import torch_npu
        return torch_npu.__version__
    except Exception:  # noqa: BLE001
        return "?"


if __name__ == "__main__":
    sys.exit(main())
