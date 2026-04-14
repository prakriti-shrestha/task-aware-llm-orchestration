"""
Phase 5 — Main Results Table (Table 1)
Runs all 6 policies and prints a clean summary table + saves CSV.
"""
from __future__ import annotations
import csv
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "phase4"))

from experiments.baselines import AlwaysW1Policy, AlwaysW3Policy, RandomPolicy, OraclePolicy
from policy.bandit import LinUCBBandit

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE     = PROJECT_ROOT / "phase3" / "data" / "runs" / "phase3_run_001.jsonl"
RESULTS_DIR  = PROJECT_ROOT / "phase5" / "results"
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


# ── Run one policy through all records ────────────────────────────────────
def run_policy(policy, records, lookup, rng, use_task_id=False, random_features=False):
    results_by_class = {}

    for rec in records:
        fv         = np.array(rec["feature_vector"])
        task_id    = rec["task_id"]
        task_class = rec["task_class"]

        if random_features:
            fv = rng.rand(FEATURE_DIM)

        chosen = policy.select_workflow(fv, task_id=task_id) if use_task_id \
                 else policy.select_workflow(fv)

        key = (task_id, chosen)
        if key in lookup:
            data   = lookup[key]
            quality = data["quality_score"]
            cost    = data["cost_tokens"]
            reward  = data["reward"]
        else:
            quality, cost, reward = 0.5, 500, 0.0

        policy.update(fv, chosen, reward)

        if task_class not in results_by_class:
            results_by_class[task_class] = {"quality": [], "cost": [], "reward": []}

        results_by_class[task_class]["quality"].append(quality)
        results_by_class[task_class]["cost"].append(cost)
        results_by_class[task_class]["reward"].append(reward)

    return results_by_class


# ── Print a clean summary table in terminal ────────────────────────────────
def print_summary(all_rows):
    # Pivot: policy → task_class → metrics
    from collections import defaultdict
    data = defaultdict(dict)
    for row in all_rows:
        data[row["policy"]][row["task_class"]] = row

    classes = sorted({r["task_class"] for r in all_rows})

    # Build header
    col_w = 20
    num_w = 10
    header = f"{'Policy':<{col_w}}"
    for c in classes:
        header += f"{'Q-'+c:>{num_w}} {'Cost':>{num_w}}"
    header += f"{'Avg Reward':>{num_w}}"

    divider = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("TABLE 1 — MAIN RESULTS")
    print(f"{'='*len(header)}")
    print(header)
    print(divider)

    policy_order = [
        "Always-W1", "Always-W3", "Random", "Oracle",
        "LinUCB-Full", "LinUCB-NoEncoder"
    ]

    for i, policy in enumerate(policy_order):
        if i == 4:
            print(divider)  # separator before our system
        if policy not in data:
            continue
        row_str = f"{policy:<{col_w}}"
        rewards = []
        for c in classes:
            if c in data[policy]:
                q = data[policy][c]["avg_quality"]
                cost = data[policy][c]["avg_cost"]
                r = data[policy][c]["avg_reward"]
                rewards.append(r)
                row_str += f"{q:>{num_w}.4f} {int(cost):>{num_w}}"
            else:
                row_str += f"{'—':>{num_w}} {'—':>{num_w}}"
        avg_r = np.mean(rewards) if rewards else 0
        row_str += f"{avg_r:>{num_w}.4f}"
        print(row_str)

    print(divider)
    print(f"\nColumns: Q-<class> = avg quality score (0-1), Cost = avg tokens\n")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print(f"Loading logs...")
    records = load_logs(LOG_FILE)
    lookup  = build_lookup(records)
    print(f"Loaded {len(records)} records\n")

    policies = [
        ("Always-W1",        AlwaysW1Policy(),                                              False, False),
        ("Always-W3",        AlwaysW3Policy(),                                              False, False),
        ("Random",           RandomPolicy(seed=SEED),                                       False, False),
        ("Oracle",           OraclePolicy(LOG_FILE),                                        True,  False),
        ("LinUCB-Full",      LinUCBBandit(feature_dim=FEATURE_DIM, alpha=1.0, seed=SEED),  False, False),
        ("LinUCB-NoEncoder", LinUCBBandit(feature_dim=FEATURE_DIM, alpha=1.0, seed=SEED),  False, True),
    ]

    all_rows = []

    for name, policy, use_task_id, random_features in policies:
        print(f"  Running {name}...")
        rng     = np.random.RandomState(SEED)
        results = run_policy(policy, records, lookup, rng,
                             use_task_id=use_task_id,
                             random_features=random_features)

        for task_class, metrics in sorted(results.items()):
            all_rows.append({
                "policy":      name,
                "task_class":  task_class,
                "avg_quality": round(np.mean(metrics["quality"]), 4),
                "avg_cost":    round(np.mean(metrics["cost"]), 2),
                "avg_reward":  round(np.mean(metrics["reward"]), 4),
            })

    # Print clean table in terminal
    print_summary(all_rows)

    # Save CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["policy", "task_class", "avg_quality", "avg_cost", "avg_reward"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved → {OUTPUT_CSV}")
    print("Table 1 complete!")


if __name__ == "__main__":
    main()