# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This file is part of the vllm-ascend project.
"""Downstream-accuracy eval for real AWQ/GPTQ MoE models on Ascend NPU.

Complements ``benchmark_moe_throughput.py`` (throughput) with downstream-task
accuracy via lm_eval (``--model vllm`` offline). Uses the same 6 accuracy tasks
as the Linear 0.5B eval (§5.3): ARC-C / ARC-E / HellaSwag / LAMBADA / PIQA /
Winogrande — all cached locally (no network). Each model runs in its own
subprocess so vLLM's global KV-cache pool is fully released between models.

⚠️ UNPAIRED: no dense MoE baseline locally (disk-blocked), so these show MoE
accuracy is sane, NOT the quantized-vs-dense delta.

Usage (from vllm-ascend/ with venv + set_env + custom-op env):
    python benchmarks/eval_moe_accuracy.py --limit 500 2>&1 \\
        | tee logs/<date>_R1_moe-accuracy.log
"""

import argparse
import json
import os
import subprocess
import sys
import time

# Quantized MoE models on disk.
MODELS = [
    ("/data/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4", "gptq", "moe-gptq-int4"),
    ("/data/DeepSeek-V2-Lite-Chat-AWQ", "awq", "moe-awq-int4"),
]

TASKS = ["arc_challenge", "arc_easy", "hellaswag", "lambada_openai", "piqa", "winogrande"]
_RESULT_SENTINEL = "__MOE_ACC_RESULT__"
_CHILD_FLAG = "VLLM_MOE_ACC_CHILD"


def _child_script(model, quant, label, tasks, limit, max_model_len, batch):
    limit_arg = f"limit={limit}," if limit else ""
    return f"""
import json
import lm_eval

res = lm_eval.simple_evaluate(
    model="vllm",
    model_args="pretrained={model},quantization={quant},dtype=float16,"
               "max_model_len={max_model_len},gpu_memory_utilization=0.85,enforce_eager=True,"
               "trust_remote_code=True",
    tasks={tasks!r},
    {limit_arg}batch_size={batch},
)
out = {{}}
for t, m in res["results"].items():
    out[t] = {{k: round(float(v), 4) for k, v in m.items() if isinstance(v, (int, float))}}
print("{_RESULT_SENTINEL}" + json.dumps(out, ensure_ascii=False))
"""


def _run_model(model, quant, label, tasks, limit, max_model_len, batch):
    """Spawn a fresh process per model (full vLLM HBM release between)."""
    env = dict(os.environ)
    env[_CHILD_FLAG] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _child_script(model, quant, label, tasks, limit, max_model_len, batch)],
        env=env,
        cwd="/data/ascend/vllm-ascend",
        capture_output=True,
        text=True,
        timeout=2400,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_SENTINEL):
            return json.loads(line[len(_RESULT_SENTINEL) :])
    return {"error": f"child failed (exit {proc.returncode})", "stderr": proc.stderr[-3000:]}


def main():
    parser = argparse.ArgumentParser(description="MoE downstream-accuracy eval (Ascend NPU)")
    parser.add_argument("--limit", type=int, default=None, help="Per-task example cap (None = full dataset).")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="vLLM max_model_len (lower may avoid MLA attention tiling failures).",
    )
    parser.add_argument(
        "--tasks", default=",".join(TASKS), help="Comma-separated lm_eval tasks (default: the 6 §5.3 tasks)."
    )
    parser.add_argument(
        "--batch", default="auto", help='lm_eval batch_size ("auto" or an int; 1 avoids MLA padding-tiling).'
    )
    args = parser.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    # "auto" → '"auto"' (string); int → bare int, for the f-string in _child_script.
    batch = '"auto"' if args.batch.lower() == "auto" else int(args.batch)

    print(
        f"MoE Downstream-Accuracy Eval — lm_eval --model vllm, tasks={tasks}, "
        f"limit={args.limit}, max_model_len={args.max_model_len}, batch={batch}, UNPAIRED\n"
    )

    results = []
    for model, quant, label in MODELS:
        print(f"\n=== {label}: {model} ({quant}) — start {time.strftime('%H:%M:%S')} ===")
        r = _run_model(model, quant, label, tasks, args.limit, args.max_model_len, batch)
        r = {"label": label, "model": model, "quantization": quant, **r}
        results.append(r)
        print(f"=== {label} done: {r} ===")

    # Save
    os.makedirs("benchmarks/results", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = f"benchmarks/results/python_eval_moe_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "paired": False,
                "limit": args.limit,
                "tasks": tasks,
                "note": "quantized-only; no dense MoE baseline on disk",
                "results": results,
            },
            f,
            indent=2,
        )

    # Table — lm_eval metric keys are "acc,none" / "acc_norm,none".
    def _metric(r, t):
        m = r.get(t, {}) or {}
        return m.get("acc_norm,none") if m.get("acc_norm,none") is not None else m.get("acc,none")

    print(f"\n{'=' * 80}")
    print(f"{'Label':<16}" + "".join(f"{t.replace('_openai', '')[:10]:>11}" for t in tasks))
    print("-" * 80)
    for r in results:
        row = "".join(f"{(_metric(r, t) if _metric(r, t) is not None else '-'):>11}" for t in tasks)
        print(f"{r['label']:<16}{row}")
    print("=" * 80)
    print("(acc_norm where available else acc; UNPAIRED — no dense MoE baseline)")
    print(f"Results JSON: {out_file}")


if __name__ == "__main__":
    main()
