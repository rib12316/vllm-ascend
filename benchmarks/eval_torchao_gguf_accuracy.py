#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""lm-eval accuracy harness for torchao + gguf on Ascend NPU.

Mirrors the AWQ/GPTQ 6-task accuracy evaluation so torchao/gguf numbers are
directly comparable to the dense/AWQ/GPTQ baselines. For each config it:

  1. starts a vLLM OpenAI server (``--quantization torchao|gguf``, plus
     ``--load_format gguf`` for GGUF),
  2. runs ``lm_eval --model local-completions`` over 6 accuracy tasks
     (arc_challenge, arc_easy, hellaswag, lambada_openai, piqa, winogrande),
  3. tears the server down and parses the per-task metrics.

Run from the torchao_gguf worktree root with the venv active::

    cd /data/ascend/torchao_gguf/vllm-ascend
    source /data/ascend/.venv/bin/activate
    python benchmarks/eval_torchao_gguf_accuracy.py --limit 1000 \\
        2>&1 | tee /data/ascend/logs/accuracy/2026-07-01_torchao-gguf-accuracy.log

Config selection: ``--configs dense,torchao-int8wo,torchao-int4wo,torchao-fp8wo,
gguf-q4_0,gguf-q8_0,gguf-q5_0,gguf-q4_k_m`` (default: all eight).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Tasks mirror the AWQ/GPTQ 6-task harness (logs/accuracy/2026-06-12_*).
ACCURACY_TASKS = [
    "arc_challenge",
    "arc_easy",
    "hellaswag",
    "lambada_openai",
    "piqa",
    "winogrande",
]

GGUF_DIR = "/data/ascend/torchao_gguf/e2e_models/qwen05b-gguf"
TORCHAO_DIR = "/data/ascend/torchao_gguf/e2e_models"
DENSE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Each config: label -> (model, extra server args)
CONFIGS = {
    "dense-fp16": (DENSE_MODEL, ["--dtype", "float16"]),
    "torchao-int8wo": (
        f"{TORCHAO_DIR}/qwen05b-torchao-int8wo",
        ["--quantization", "torchao", "--dtype", "float16"],
    ),
    "torchao-int4wo": (
        f"{TORCHAO_DIR}/qwen05b-torchao-int4wo",
        ["--quantization", "torchao", "--dtype", "float16"],
    ),
    "torchao-fp8wo": (
        f"{TORCHAO_DIR}/qwen05b-torchao-fp8wo",
        ["--quantization", "torchao", "--dtype", "float16"],
    ),
    "gguf-q4_0": (
        f"{GGUF_DIR}/qwen2.5-0.5b-instruct-q4_0.gguf",
        ["--quantization", "gguf", "--load_format", "gguf", "--dtype", "float16"],
    ),
    "gguf-q8_0": (
        f"{GGUF_DIR}/qwen2.5-0.5b-instruct-q8_0.gguf",
        ["--quantization", "gguf", "--load_format", "gguf", "--dtype", "float16"],
    ),
    "gguf-q5_0": (
        f"{GGUF_DIR}/qwen2.5-0.5b-instruct-q5_0.gguf",
        ["--quantization", "gguf", "--load_format", "gguf", "--dtype", "float16"],
    ),
    "gguf-q4_k_m": (
        f"{GGUF_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        ["--quantization", "gguf", "--load_format", "gguf", "--dtype", "float16"],
    ),
    "gguf-q5_k_m": (
        f"{GGUF_DIR}/qwen2.5-0.5b-instruct-q5_k_m.gguf",
        ["--quantization", "gguf", "--load_format", "gguf", "--dtype", "float16"],
    ),
}

SERVED_NAME = "test"
PORT = 18890
PYTHON = "/data/ascend/.venv/bin/python"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def server_cmd(model: str, extra: list[str]) -> list[str]:
    return [
        PYTHON,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--served-model-name",
        SERVED_NAME,
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.8",
        "--enforce-eager",
        "--port",
        str(PORT),
    ] + extra


def wait_for_server(proc: subprocess.Popen, timeout: int = 240) -> bool:
    """Poll /v1/models until ready or the server dies."""
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False  # server exited
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(3)
    return False


def run_lm_eval(label: str, limit: int | None, out_dir: Path, eval_timeout: int) -> dict | None:
    """Run lm_eval over the 6 accuracy tasks via local-completions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [
        PYTHON,
        "-m",
        "lm_eval",
        "--model",
        "local-completions",
        "--model_args",
        f"model={SERVED_NAME},base_url=http://localhost:{PORT}/v1/completions,"
        f"tokenizer={DENSE_MODEL},num_concurrent=32,max_retries=0",
        "--tasks",
        ",".join(ACCURACY_TASKS),
        "--output_path",
        str(out_dir),
        "--log_samples",
    ]
    if limit is not None:
        args += ["--limit", str(limit)]
    log(f"  lm_eval accuracy: --tasks {','.join(ACCURACY_TASKS)}{' --limit ' + str(limit) if limit else ' (full)'}")
    try:
        cp = subprocess.run(args, capture_output=True, text=True, timeout=eval_timeout, cwd=os.getcwd())
    except subprocess.TimeoutExpired:
        log(f"  lm_eval TIMED OUT after {eval_timeout}s")
        return None
    if cp.returncode != 0:
        log(f"  lm_eval exit={cp.returncode}")
        log(cp.stderr[-2000:])
        return None
    return parse_results(out_dir)


def parse_results(out_dir: Path) -> dict | None:
    """Find the result JSON and pull the primary metric per task."""
    jsons = sorted(out_dir.rglob("results_*.json"))
    if not jsons:
        log("  no results_*.json found")
        return None
    data = json.loads(jsons[-1].read_text())
    results = data.get("results", {})
    out = {}
    for task in ACCURACY_TASKS:
        if task not in results:
            out[task] = None
            continue
        tr = results[task]
        # primary metric: acc_norm for arc/hellaswag/piqa, acc for winogrande,
        # perplexity+acc for lambada. Report both acc and acc_norm where present.
        out[task] = {
            k: round(v, 4) for k, v in tr.items() if isinstance(v, float) and ("acc" in k or "perplexity" in k)
        }
    return out


def run_config(
    label: str, limit: int | None, results_root: Path, server_timeout: int, eval_timeout: int
) -> dict | None:
    model, extra = CONFIGS[label]
    quant = " ".join(a for i, a in enumerate(extra) if i and extra[i - 1] == "--quantization") or "none"
    load_fmt = " ".join(a for i, a in enumerate(extra) if i and extra[i - 1] == "--load_format") or "auto"
    log(f"[{label}] model={model} quant={quant} load_format={load_fmt}")

    cmd = server_cmd(model, extra)
    server_log = results_root / f"{label}_server.log"
    slf = open(server_log, "w")  # noqa: SIM115
    proc = subprocess.Popen(cmd, stdout=slf, stderr=subprocess.STDOUT)
    try:
        if not wait_for_server(proc, server_timeout):
            log(f"  server failed to start within {server_timeout}s (see {server_log})")
            return None
        log(f"  server ready (pid={proc.pid})")
        return run_lm_eval(label, limit, results_root / label, eval_timeout)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        slf.close()
        log(f"  server stopped (log: {server_log})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS), help="comma-separated labels (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="samples per task (default: full)")
    ap.add_argument("--server-timeout", type=int, default=240)
    ap.add_argument("--eval-timeout", type=int, default=1500)
    ap.add_argument("--out-root", default="benchmarks/results/torchao_gguf_acc")
    args = ap.parse_args()

    labels = [c.strip() for c in args.configs.split(",") if c.strip()]
    results_root = Path(args.out_root)
    results_root.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict | None] = {}

    log(f"=== torchao/gguf accuracy eval — {len(labels)} config(s) ===")
    log(f"tasks: {', '.join(ACCURACY_TASKS)} | limit={args.limit or 'full'}")
    for label in labels:
        if label not in CONFIGS:
            log(f"unknown config '{label}', skipping")
            continue
        try:
            all_results[label] = run_config(
                label,
                args.limit,
                results_root,
                args.server_timeout,
                args.eval_timeout,
            )
        except Exception as e:  # noqa: BLE001
            log(f"[{label}] EXCEPTION: {e!r}")
            all_results[label] = None
        log("")

    # ---- summary table ----
    log("=== SUMMARY (acc_norm for arc/hellaswag/piqa, acc for winogrande, both for lambada) ===")
    header = f"{'config':<18}" + "".join(f"{t[:10]:>12}" for t in ACCURACY_TASKS)
    log(header)
    for label in labels:
        r = all_results.get(label)
        if not r:
            log(f"{label:<18}" + "".join(f"{'--':>12}" for _ in ACCURACY_TASKS))
            continue
        cells = []
        for t in ACCURACY_TASKS:
            m = r.get(t) or {}
            if t == "winogrande" or t == "lambada_openai":
                v = m.get("acc,none")
            else:
                v = m.get("acc_norm,none", m.get("acc,none"))
            cells.append(f"{v:.4f}" if v is not None else "--")
        log(f"{label:<18}" + "".join(f"{c:>12}" for c in cells))

    # dump machine-readable summary
    summary_path = results_root / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    log(f"summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
