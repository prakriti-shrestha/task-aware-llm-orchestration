"""
Phase 5 — Main Results Table
Loads the phase3 log file and computes average quality and cost
for each baseline policy. Produces results/main_results.csv (Table 1).
"""

import json
import random
import csv
import os
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "phase3" / "data" / "runs" / "phase3_run_001.jsonl"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = RESULTS_DIR / "main_results.csv"

# ── load log file ──────────────────────────────────────────────────────────
def load_logs(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records

# ── baseline policies ──────────────────────────────────────────────────────
class AlwaysW1Policy:
    name = "Always-W1"
    def select_workflow(self, features): return "W1"

class AlwaysW3Policy:
    name = "Always-W3"
    def select_workflow(self, features): return "W3"

class RandomPolicy:
    name = "Random"
    def select_workflow(self, features):
        return random.choice(["W1", "W2", "W3"])

# ── evaluate a policy against the logs ────────────────────────────────────
def evaluate_policy(policy, records):
    """
    For each task, simulate what would happen if this policy
    picked a workflow. Look up the quality and cost from the logs.
    """
    # Build a lookup: task_id + workflow -> record
    lookup = {}
    for r in records:
        lookup[(r["task_id"], r["workflow_id"])] = r

    results_by_class = {}

    # Get unique tasks
    task_ids = list({r["task_id"] for r in records})
    task_class_map = {r["task_id"]: r["task_class"] for r in records}

    for task_id in task_ids:
        task_class = task_class_map[task_id]

        # Get a dummy feature vector from the first record of this task
        first_record = next(r for r in records if r["task_id"] == task_id)
        features = first_record["feature_vector"]

        # Policy picks a workflow
        chosen_workflow = policy.select_workflow(features)

        # Look up result for that workflow
        key = (task_id, chosen_workflow)
        if key not in lookup:
            continue

        record = lookup[key]
        quality = record["quality_score"]
        cost = record["cost_tokens"]

        if task_class not in results_by_class:
            results_by_class[task_class] = {"quality": [], "cost": []}

        results_by_class[task_class]["quality"].append(quality)
        results_by_class[task_class]["cost"].append(cost)

    return results_by_class

# ── main ───────────────────────────────────────────────────────────────────
def main():
    print(f"Loading logs from {LOG_FILE}...")
    records = load_logs(LOG_FILE)
    print(f"Loaded {len(records)} records!")

    policies = [AlwaysW1Policy(), AlwaysW3Policy(), RandomPolicy()]
    all_rows = []

    for policy in policies:
        print(f"\nEvaluating {policy.name}...")
        results = evaluate_policy(policy, records)

        for task_class, metrics in results.items():
            avg_quality = sum(metrics["quality"]) / len(metrics["quality"])
            avg_cost = sum(metrics["cost"]) / len(metrics["cost"])

            row = {
                "policy": policy.name,
                "task_class": task_class,
                "avg_quality": round(avg_quality, 4),
                "avg_cost": round(avg_cost, 2),
            }
            all_rows.append(row)
            print(f"  {task_class}: quality={avg_quality:.4f}, cost={avg_cost:.2f}")

    # Save to CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["policy", "task_class", "avg_quality", "avg_cost"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults saved to {OUTPUT_CSV}")
    print("Table 1 data is ready!")

if __name__ == "__main__":
    main()