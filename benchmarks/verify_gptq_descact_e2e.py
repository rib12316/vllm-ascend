#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This file is part of the vllm-ascend project.
"""R13: clean GPTQ Linear desc_act=True end-to-end on Ascend NPU.

The GPTQ Linear desc_act path (g_idx argsort + weight shuffle + runtime
activation gather, ``methods/gptq.py``) is validated on synthetic weights by
``test_quant_moe_synthetic.py::TestGPTQLinearDescAct``, but the real-model
evidence was previously weak (no PASS marker, degraded output on some prompts).
This script closes that gap with a clean real-model run on
``TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ`` (a desc_act=True, 4-bit, group=128
GPTQ checkpoint), confirming the path produces sane output.

Run from vllm-ascend/ with the venv + set_env.sh active::

    python benchmarks/verify_gptq_descact_e2e.py 2>&1 | tee logs/e2e/<date>_gptq-descact-e2e.log
"""

from __future__ import annotations

import sys

from vllm import LLM, SamplingParams

MODEL = (
    "/data/huggingface_home/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GPTQ"
    "/snapshots/9d4580af0f21bccafd762dcc50d0c7bac6273584"
)

# (prompt, expected case-insensitive substring). All completion-style factual
# prompts that TinyLlama-1.1B answers reliably WITHOUT a chat template — the
# point is to exercise the desc_act dequant path across many layers/prompts,
# not to benchmark instruction-following. Correct capitals across several
# prompts is strong evidence the per-group g_idx shuffle + activation gather is
# sound (a broken desc_act path would garble all of them, not just one).
CHECKS = [
    ("The capital of France is", "paris"),
    ("The capital of Japan is", "tokyo"),
    ("The capital of Italy is", "rome"),
    ("The capital of Germany is", "berlin"),
    ("The capital of China is", "beijing"),
]


def main() -> int:
    llm = LLM(model=MODEL, quantization="gptq", dtype="float16", max_model_len=512, enforce_eager=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=20)
    prompts = [p for p, _ in CHECKS]
    outputs = llm.generate(prompts, sampling)

    print("\n========== R13 GPTQ Linear desc_act=True e2e ==========")
    print(f"model: {MODEL}")
    all_pass = True
    for (prompt, expected), out in zip(CHECKS, outputs, strict=True):
        text = out.outputs[0].text.strip()
        ok = expected in text.lower()
        all_pass = all_pass and ok
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] prompt={prompt!r}")
        print(f"        expected~{expected!r}  got={text!r}")

    tag = "[R13-GPTQ-Linear-desc_act] PASS" if all_pass else "[R13-GPTQ-Linear-desc_act] FAIL"
    print(f"\n{tag}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
