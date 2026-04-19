"""
Phase 5 — Experiment 4: Ablation Study (Table 2)

Ablations:
    full         — LinUCB + real features + full reward signal
    no_encoder   — Features replaced with random noise
    no_nli       — Reward uses only ground-truth match (sets quality to 0.5
                   for tasks where ground truth is None)
    no_bandit    — Uniform random workflow choice
    no_w3        — Bandit can only choose W1 or W2

Outputs:
    results/ablation_results.csv

Prerequisite:
    data/runs/phase5_all_arms.jsonl

Usage:
    python -m phase5.experiments.phase5_ablations
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
from phase5.experiments.baselines import RandomPolicy

ALL_ARMS_LOG = RUNS_DIR / "phase5_all_arms.jsonl"
OUTPUT_CSV = RESULTS_DIR / "ablation_results.csv"


class TwoArmLinUCB(LinUCBBandit):
    """LinUCB restricted to W1 and W2 only — for the no_w3 ablation."""
    def select_workflow(self, feature_vector: np.ndarray) -> str:
        choice = super().select_workflow(feature_vector)
        return choice if choice in ("W1", "W2") else "W2"


def evaluate_ablation(name: str, all_arms: list[dict], seed: int) -> dict:
    # Group records
    task_order: list[str] = []
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for rec in all_arms:
        tid = rec["task_id"]
        if tid not in by_task:
            task_order.append(tid)
        by_task[tid][rec["workflow_id"]] = rec

    rng_np = np.random.RandomState(seed)

    # Build policy per ablation
    if name == "full":
        policy = LinUCBBandit(feature_dim=FEATURE_DIM, alpha=1.0, seed=seed)
        mutate_features = lambda fv, tid, text: fv
        mutate_reward = lambda rec: rec["reward"]
        restrict_arms = None
    elif name == "no_encoder":
        policy = LinUCBBandit(feature_dim=FEATURE_DIM, alpha=1.0, seed=seed)
        mutate_features = lambda fv, tid, text: rng_np.rand(FEATURE_DIM)
        mutate_reward = lambda rec: rec["reward"]
        restrict_arms = None
    elif name == "no_nli":
        # Approximated: reward uses quality only if ground_truth exists,
        # else a fixed 0.5. This simulates losing the NLI signal.
        policy = LinUCBBandit(feature_dim=FEATURE_DIM, alpha=1.0, seed=seed)
        mutate_features = lambda fv, tid, text: fv

        def mutate_reward(rec):
            if rec.get("ground_truth") is None:
                # Fake a degraded reward without NLI: quality locked at 0.5
                from phase4.policy.reward import compute_reward
                return compute_reward(0.5, rec["cost_tokens"], rec["lambda_value"])
            return rec["reward"]
        restrict_arms = None
    elif name == "no_bandit":
        policy = RandomPolicy(seed=seed)
        mutate_features = lambda fv, tid, text: fv
        mutate_reward = lambda rec: rec["reward"]
        restrict_arms = None
    elif name == "no_w3":
        policy = TwoArmLinUCB(feature_dim=FEATURE_DIM, alpha=1.0, seed=seed)
        mutate_features = lambda fv, tid, text: fv
        mutate_reward = lambda rec: rec["reward"]
        restrict_arms = {"W1", "W2"}
    else:
        raise ValueError(f"Unknown ablation: {name}")

    # Simulate
    picks: list[dict] = []
    for tid in task_order:
        task_recs = by_task[tid]
        any_rec = next(iter(task_recs.values()))
        fv_raw = feature_vector(any_rec["task_text"], tid)
        fv = mutate_features(fv_raw, tid, any_rec["task_text"])

        chosen = policy.select_workflow(fv)
        if restrict_arms is not None and chosen not in restrict_arms:
            chosen = "W2"
        if chosen not in task_recs:
            continue

        rec = task_recs[chosen]
        r = mutate_reward(rec)
        policy.update(fv, chosen, r)
        picks.append(rec)

    # Report on second half
    half = len(picks) // 2
    tail = picks[half:] if half > 0 else picks

    return {
        "ablation": name,
        "n": len(tail),
        "avg_quality": round(float(np.mean([p["quality_score"] for p in tail])), 4),
        "avg_cost":    round(float(np.mean([p["cost_tokens"]   for p in tail])), 1),
        "avg_reward":  round(float(np.mean([p["reward"]        for p in tail])), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()

    if not ALL_ARMS_LOG.exists():
        print(f"[ablations] ERROR: {ALL_ARMS_LOG} missing.")
        print("[ablations] Run phase5_main_results.py first.")
        sys.exit(1)

    all_arms = load_run_log(ALL_ARMS_LOG)
    print(f"[ablations] Loaded {len(all_arms)} records\n")

    ablations = ["full", "no_encoder", "no_nli", "no_bandit", "no_w3"]
    results = []
    for name in ablations:
        print(f"  → {name}")
        results.append(evaluate_ablation(name, all_arms, args.seed))

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ablation", "n", "avg_quality",
                                                "avg_cost", "avg_reward"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[ablations] CSV → {OUTPUT_CSV}\n")
    print(f"{'ablation':<12} {'n':>4} {'quality':>8} {'cost':>8} {'reward':>8}")
    print("-" * 52)
    for r in results:
        print(f"{r['ablation']:<12} {r['n']:>4} {r['avg_quality']:>8.3f} "
              f"{r['avg_cost']:>8.0f} {r['avg_reward']:>8.3f}")


if __name__ == "__main__":
    main()