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
T1/T2/T6 (see logs/e2e/2026-06-12_awq-gptq-inference.log). The MoE
weight-processing math is covered by tests/quantization/test_moe_synthetic_npu.py,
but that test explicitly skips the full ``apply -> fused_experts -> grouped_matmul``
path and real-checkpoint loading. This script closes that gap: it loads the two
real MoE models on disk, confirms our Ascend MoE quant method is attached, and
checks generated output is sane (correct + non-degenerate).

Each model runs in its OWN SUBPROCESS. vLLM retains its KV-cache pool in global
state, so ``del llm; gc.collect(); torch.npu.empty_cache()`` in-process does NOT
reclaim the ~50 GB HBM allocation — the next model then fails to load. A fresh
process per model is the only reliable release (process death frees everything),
mirroring the subprocess-per-model design in ``benchmark_awq_gptq.py``.

Run from the vllm-ascend/ directory with the venv active:

    python benchmarks/verify_moe_e2e.py 2>&1 | tee logs/e2e/<date>_moe-e2e.log
"""

import json
import os
import subprocess
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
_moe_only = os.environ.get("MOE_ONLY", "")
if _moe_only:
    MODELS = [m for m in MODELS if _moe_only.lower() in m["tag"].lower()]

# Child-process result marker: the child prints ``<SENTINEL><json>`` so the
# parent can recover the result dict from among vLLM's log noise.
_RESULT_SENTINEL = "__MOE_E2E_RESULT__"
_CHILD_FLAG = "VLLM_MOE_E2E_CHILD"


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
    """Load + generate for one model. Returns a result dict; never raises.

    Runs in a child process (see ``main``), so there is no in-process HBM
    cleanup — process exit releases all NPU memory for the next model.
    """
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
    return res


def _run_child() -> int:
    """Child mode: MOE_ONLY filters MODELS to exactly one; run it and print the
    result as ``<SENTINEL><json>`` for the parent to collect."""
    cfg = MODELS[0]
    r = run_one(cfg)
    print(_RESULT_SENTINEL + json.dumps(r, ensure_ascii=False))
    return 0


def _run_model_subprocess(cfg: dict) -> dict:
    """Spawn a fresh Python process for one model so vLLM's global KV-cache pool
    is fully released on exit (in-process gc/empty_cache is insufficient)."""
    env = dict(os.environ)
    env[_CHILD_FLAG] = "1"
    env["MOE_ONLY"] = cfg["tag"]
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__)],
        env=env,
        cwd="/data/ascend/vllm-ascend",
        capture_output=True,
        text=True,
        timeout=900,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_SENTINEL):
            return json.loads(line[len(_RESULT_SENTINEL) :])
    # No result line — child crashed before emitting one.
    return {
        **{k: cfg[k] for k in ("tag", "model", "quantization")},
        "load_ok": False,
        "gen_ok": None,
        "detail": [f"CHILD FAILED (exit {proc.returncode})", proc.stderr[-3000:]],
    }


def main() -> int:
    # Child mode: run exactly one (MOE_ONLY-filtered) model and emit its result.
    if os.environ.get(_CHILD_FLAG) == "1":
        return _run_child()

    print("=" * 64)
    print(" AWQ/GPTQ MoE end-to-end verification on Ascend NPU")
    print(f" torch {torch.__version__} | torch_npu {torch_npu_ver()}")
    print(f" NPU available: {torch.npu.is_available()}")
    print("=" * 64)

    # Parent mode: one fresh subprocess per model (full HBM release between).
    results = [_run_model_subprocess(c) for c in MODELS]

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
    msg = "ALL PASS — MoE path verified end-to-end" if n_fail == 0 else f"{n_fail} model(s) need investigation"
    print("\n" + msg)
    return 0 if n_fail == 0 else 1


def torch_npu_ver() -> str:
    try:
        import torch_npu

        return torch_npu.__version__
    except Exception:  # noqa: BLE001
        return "?"


if __name__ == "__main__":
    sys.exit(main())
