"""
Phase 5 — Shared runner utilities.

Every experiment script imports from here to keep behaviour consistent:
  - feature_vector(task)        — same deterministic encoder as the main run
  - run_task(task, workflow)    — executes one task through one workflow
  - load_run_log(path)          — reads a JSONL episode log into a list
  - ensure_dirs()                — creates results/ and figures/ if missing

This file contains NO policy logic — it just wires up workflows + quality.
"""
from __future__ import annotations
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Make `phase5` AND `phase3` imports work regardless of cwd.
#
# Phase 3 uses unqualified imports like `from data.loaders.gsm8k_loader import ...`
# which requires `phase3/` itself to be on sys.path (not just the project root).
# We add both so everything works whether imports are written as `phase3.data.X`
# or bare `data.X`.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))                    # for `from phase5.X import ...`
sys.path.insert(0, str(ROOT / "phase3"))         # for `from data.X import ...`

from phase5.workflows.w1_basic import W1BasicWorkflow
from phase5.workflows.w2_reasoned import W2ReasonedWorkflow
from phase5.workflows.w3_heavy import W3HeavyWorkflow
from phase5.quality import score as quality_score

FEATURE_DIM = 16

WORKFLOWS = {
    "W1": W1BasicWorkflow(),
    "W2": W2ReasonedWorkflow(),
    "W3": W3HeavyWorkflow(),
}

PHASE5_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PHASE5_DIR / "results"
FIGURES_DIR = PHASE5_DIR / "figures"
RUNS_DIR = PHASE5_DIR / "data" / "runs"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def feature_vector(task_text: str, task_id: str) -> np.ndarray:
    """
    Deterministic task encoder.

    Must exactly match the function used in phase5/run_phase5.py so that
    policies trained in run_phase5 can be replayed here.
    """
    seed = int(hashlib.md5(task_id.encode()).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(seed)
    length_norm = min(len(task_text) / 500.0, 1.0)
    word_count_norm = min(len(task_text.split()) / 100.0, 1.0)
    base = [length_norm, word_count_norm] + [rng.random() for _ in range(FEATURE_DIM - 2)]
    return np.array([round(v, 6) for v in base])


def run_task(task: dict, workflow_id: str) -> dict:
    """
    Run a single task through a single workflow.

    Returns a dict with:
        output_text, cost_tokens, latency_ms, quality_score, error
    """
    wf = WORKFLOWS[workflow_id]
    t0 = time.time()
    try:
        output_text, cost_tokens = wf.run(task["task_text"])
        err = None
    except Exception as e:
        output_text = f"[ERROR: {e}]"
        cost_tokens = 0
        err = str(e)

    latency_ms = int((time.time() - t0) * 1000)

    if err is None:
        try:
            q = quality_score(
                task["task_text"],
                output_text,
                task.get("ground_truth"),
                task["task_class"],
                test_code=task.get("test_code"),
            )
        except Exception as qe:
            q = 0.0
            err = f"quality_error: {qe}"
    else:
        q = 0.0

    return {
        "output_text": output_text,
        "cost_tokens": int(cost_tokens),
        "latency_ms": latency_ms,
        "quality_score": float(q),
        "error": err,
    }


def load_run_log(path: str | Path) -> list[dict]:
    """Load a JSONL episode log into a list of dicts."""
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalise_cost(cost: float, max_cost: float) -> float:
    if max_cost <= 0:
        return 0.0
    return min(cost / max_cost, 1.0)


# Paper colour palette — used across all figures for consistency
COLOURS = {
    "W1": "#4CAF50",      # green
    "W2": "#FFA726",      # amber
    "W3": "#EF5350",      # coral
    "ours": "#7E57C2",    # purple
    "linucb": "#7E57C2",
    "epsilon_greedy": "#26A69A",  # teal
    "random": "#9E9E9E",  # gray
    "always_w1": "#4CAF50",
    "always_w3": "#EF5350",
    "oracle": "#FFD700",  # gold
}