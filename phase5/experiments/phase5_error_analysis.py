"""
Phase 5 — Experiment 5: Task-Level Error Analysis (qualitative, Section 4.4)

Find the 30 tasks where the system made the WORST routing decisions
(largest gap between oracle reward and actual reward).

Outputs a formatted table to stdout and saves it as CSV for the paper.

Usage:
    python -m phase5.experiments.phase5_error_analysis
"""
from __future__ import annotations
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase3"))  # Phase 3 uses bare `from data.X import ...`

from phase4.policy.bandit import LinUCBBandit
from phase5.experiments._shared import (
    FEATURE_DIM, RESULTS_DIR, RUNS_DIR,
    feature_vector, load_run_log, ensure_dirs,
)

ALL_ARMS_LOG = RUNS_DIR / "phase5_all_arms.jsonl"
OUTPUT_CSV = RESULTS_DIR / "error_analysis.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()

    if not ALL_ARMS_LOG.exists():
        print(f"[error_analysis] ERROR: {ALL_ARMS_LOG} missing.")
        sys.exit(1)

    all_arms = load_run_log(ALL_ARMS_LOG)

    task_order: list[str] = []
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for rec in all_arms:
        tid = rec["task_id"]
        if tid not in by_task:
            task_order.append(tid)
        by_task[tid][rec["workflow_id"]] = rec

    # Replay LinUCB to get its decisions
    bandit = LinUCBBandit(feature_dim=FEATURE_DIM, alpha=1.0, seed=args.seed)

    errors: list[dict] = []
    for tid in task_order:
        task_recs = by_task[tid]
        if len(task_recs) < 3:
            continue  # need all three arms for oracle
        any_rec = next(iter(task_recs.values()))
        fv = feature_vector(any_rec["task_text"], tid)

        chosen = bandit.select_workflow(fv)
        if chosen not in task_recs:
            continue

        chosen_rec = task_recs[chosen]
        # Oracle = best workflow for THIS task
        oracle_wf = max(task_recs, key=lambda w: task_recs[w]["reward"])
        oracle_rec = task_recs[oracle_wf]

        bandit.update(fv, chosen, chosen_rec["reward"])

        gap = oracle_rec["reward"] - chosen_rec["reward"]
        errors.append({
            "task_id": tid,
            "task_class": any_rec["task_class"],
            "task_snippet": (any_rec["task_text"][:100] + "…") if len(any_rec["task_text"]) > 100 else any_rec["task_text"],
            "chosen_workflow": chosen,
            "oracle_workflow": oracle_wf,
            "chosen_quality": round(chosen_rec["quality_score"], 3),
            "oracle_quality": round(oracle_rec["quality_score"], 3),
            "chosen_cost": chosen_rec["cost_tokens"],
            "oracle_cost": oracle_rec["cost_tokens"],
            "reward_gap": round(gap, 4),
            "feature_norm": round(float(np.linalg.norm(fv)), 3),
        })

    errors.sort(key=lambda e: e["reward_gap"], reverse=True)
    top = errors[:args.top_k]

    fieldnames = list(top[0].keys()) if top else []
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top)

    print(f"[error_analysis] Top {len(top)} failures → {OUTPUT_CSV}\n")
    print(f"{'class':<12} {'chose':<5} {'oracle':<6} {'gap':>6}  snippet")
    print("-" * 90)
    for e in top[:15]:  # show first 15 in stdout
        print(f"{e['task_class']:<12} {e['chosen_workflow']:<5} {e['oracle_workflow']:<6} "
              f"{e['reward_gap']:>6.3f}  {e['task_snippet'][:60]}")


if __name__ == "__main__":
    main()