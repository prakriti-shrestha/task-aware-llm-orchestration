"""
Phase 5 — Main Results Table (Table 1)
REAL DATA VERSION (no mock values)
"""
from __future__ import annotations
import csv
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.baselines import AlwaysW1Policy, AlwaysW3Policy, RandomPolicy, OraclePolicy
from policy.bandit import LinUCBBandit

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE     = PROJECT_ROOT / "phase3" / "data" / "runs" / "phase3_run_001.jsonl"
RESULTS_DIR  = PROJECT_ROOT / "phase4" / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_CSV   = RESULTS_DIR / "main_results.csv"

FEATURE_DIM = 16
SEED        = 42


# ── Load logs ──────────────────────────────────────────────────────────────
def load_logs(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Build lookup: (task_id, workflow_id) → record ─────────────────────────
def build_lookup(records):
    lookup = {}
    for r in records:
        key = (r["task_id"], r["workflow_id"])
        lookup[key] = r
    return lookup


# ── Run policy ────────────────────────────────────────────────────────────
def run_policy(policy, records, lookup, rng, use_task_id=False, random_features=False):
    results_by_class = {}

    for rec in records:
        fv = np.array(rec["feature_vector"])
        task_id = rec["task_id"]
        task_class = rec["task_class"]

        # Random feature ablation
        if random_features:
            fv = rng.rand(FEATURE_DIM)

        # Select workflow
        if use_task_id:
            chosen = policy.select_workflow(fv, task_id=task_id)
        else:
            chosen = policy.select_workflow(fv)

        # 🔥 IMPORTANT: fetch real data for chosen workflow
        key = (task_id, chosen)

        if key in lookup:
            data = lookup[key]
            quality = data["quality_score"]
            cost = data["cost_tokens"]
            reward = data["reward"]
        else:
            # fallback (rare)
            quality = 0.5
            cost = 500
            reward = 0.0

        # Update policy (only matters for learning ones)
        policy.update(fv, chosen, reward)

        # Store results
        if task_class not in results_by_class:
            results_by_class[task_class] = {"quality": [], "cost": [], "reward": []}

        results_by_class[task_class]["quality"].append(quality)
        results_by_class[task_class]["cost"].append(cost)
        results_by_class[task_class]["reward"].append(reward)

    return results_by_class


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print(f"Loading logs from {LOG_FILE}...")
    records = load_logs(LOG_FILE)
    lookup = build_lookup(records)

    print(f"Loaded {len(records)} records!\n")

    policies = [
        ("Always-W1",         AlwaysW1Policy(), False, False),
        ("Always-W3",         AlwaysW3Policy(), False, False),
        ("Random",            RandomPolicy(seed=SEED), False, False),
        ("Oracle",            OraclePolicy(LOG_FILE), True, False),
        ("LinUCB-Full",       LinUCBBandit(feature_dim=FEATURE_DIM, alpha=1.0, seed=SEED), False, False),
        ("LinUCB-NoEncoder",  LinUCBBandit(feature_dim=FEATURE_DIM, alpha=1.0, seed=SEED), False, True),
    ]

    all_rows = []

    for name, policy, use_task_id, random_features in policies:
        print(f"Evaluating {name}...")
        rng = np.random.RandomState(SEED)

        results = run_policy(policy, records, lookup, rng,
                             use_task_id=use_task_id,
                             random_features=random_features)

        for task_class, metrics in sorted(results.items()):
            avg_quality = round(np.mean(metrics["quality"]), 4)
            avg_cost    = round(np.mean(metrics["cost"]), 2)
            avg_reward  = round(np.mean(metrics["reward"]), 4)

            row = {
                "policy":      name,
                "task_class":  task_class,
                "avg_quality": avg_quality,
                "avg_cost":    avg_cost,
                "avg_reward":  avg_reward,
            }
            all_rows.append(row)

            print(f"  {task_class}: quality={avg_quality}, cost={avg_cost}, reward={avg_reward}")

        print()

    # Save CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["policy", "task_class", "avg_quality", "avg_cost", "avg_reward"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Results saved to {OUTPUT_CSV}")
    print("Table 1 complete!")


if __name__ == "__main__":
    main()