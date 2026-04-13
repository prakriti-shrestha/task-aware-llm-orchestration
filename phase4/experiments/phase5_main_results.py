"""
Phase 5 — Main Results Table (Table 1)
Runs all 6 policies and produces results/main_results.csv
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
from policy.reward import compute_reward

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE     = PROJECT_ROOT / "phase3" / "data" / "runs" / "phase3_run_001.jsonl"
RESULTS_DIR  = PROJECT_ROOT / "phase4" / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_CSV   = RESULTS_DIR / "main_results.csv"

LAMBDA      = 0.5
FEATURE_DIM = 16
SEED        = 42

# Mock quality per workflow per task class
# Based on what W1/W2/W3 realistically achieve
MOCK_QUALITY = {
    ("W1", "qa"):        0.82,
    ("W1", "reasoning"): 0.52,
    ("W1", "code"):      0.48,
    ("W2", "qa"):        0.75,
    ("W2", "reasoning"): 0.73,
    ("W2", "code"):      0.70,
    ("W3", "qa"):        0.76,
    ("W3", "reasoning"): 0.91,
    ("W3", "code"):      0.88,
}
COST_TOKENS = {"W1": 150, "W2": 350, "W3": 700}

# ── Load logs ──────────────────────────────────────────────────────────────
def load_logs(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

# ── Get quality for a workflow + task class ────────────────────────────────
def get_quality(workflow_id, task_class, rng):
    base = MOCK_QUALITY.get((workflow_id, task_class), 0.70)
    return float(np.clip(base + rng.normal(0, 0.04), 0, 1))

# ── Run a policy over all records ─────────────────────────────────────────
def run_policy(policy, records, rng, use_task_id=False, random_features=False):
    results_by_class = {}

    for rec in records:
        fv = np.array(rec["feature_vector"])
        task_id = rec["task_id"]
        task_class = rec["task_class"]

        # Ablation: replace features with random noise
        if random_features:
            fv = rng.rand(FEATURE_DIM)

        # Select workflow
        if use_task_id:
            chosen = policy.select_workflow(fv, task_id=task_id)
        else:
            chosen = policy.select_workflow(fv)

        # Get quality and cost
        quality = get_quality(chosen, task_class, rng)
        cost = COST_TOKENS[chosen] + rng.randint(-20, 20)
        reward = compute_reward(quality, cost, LAMBDA)

        # Update policy
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
    print(f"Loaded {len(records)} records!\n")

    policies = [
        ("Always-W1",         AlwaysW1Policy(),                                False, False),
        ("Always-W3",         AlwaysW3Policy(),                                False, False),
        ("Random",            RandomPolicy(seed=SEED),                         False, False),
        ("Oracle",            OraclePolicy(LOG_FILE),                          True,  False),
        ("LinUCB-Full",       LinUCBBandit(feature_dim=FEATURE_DIM, alpha=1.0, seed=SEED), False, False),
        ("LinUCB-NoEncoder",  LinUCBBandit(feature_dim=FEATURE_DIM, alpha=1.0, seed=SEED), False, True),
    ]

    all_rows = []

    for name, policy, use_task_id, random_features in policies:
        print(f"Evaluating {name}...")
        rng = np.random.RandomState(SEED)
        results = run_policy(policy, records, rng,
                           use_task_id=use_task_id,
                           random_features=random_features)

        for task_class, metrics in sorted(results.items()):
            avg_quality = round(sum(metrics["quality"]) / len(metrics["quality"]), 4)
            avg_cost    = round(sum(metrics["cost"])    / len(metrics["cost"]),    2)
            avg_reward  = round(sum(metrics["reward"])  / len(metrics["reward"]),  4)

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