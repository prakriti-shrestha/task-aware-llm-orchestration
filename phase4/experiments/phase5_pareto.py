"""
Phase 5 — Pareto Frontier (Figure 2)
Sweeps lambda values and plots cost vs quality curve.
"""

import json
import random
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "phase3" / "data" / "runs" / "phase3_run_001.jsonl"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── load logs ──────────────────────────────────────────────────────────────
def load_logs(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records

# ── simulate system with a given lambda ───────────────────────────────────
def simulate_with_lambda(records, lambda_value):
    """
    For each task, pick the workflow with highest reward
    using this lambda value. Record quality and cost.
    """
    # Build lookup: task_id + workflow -> record
    lookup = {}
    for r in records:
        lookup[(r["task_id"], r["workflow_id"])] = r

    task_ids = list({r["task_id"] for r in records})

    qualities = []
    costs = []

    for task_id in task_ids:
        best_workflow = None
        best_reward = float("-inf")

        for workflow in ["W1", "W2", "W3"]:
            key = (task_id, workflow)
            if key not in lookup:
                continue
            r = lookup[key]
            # Recompute reward with this lambda
            cost_norm = min(r["cost_tokens"] / 1000.0, 1.0)
            reward = r["quality_score"] - lambda_value * cost_norm
            if reward > best_reward:
                best_reward = reward
                best_workflow = workflow

        if best_workflow:
            record = lookup[(task_id, best_workflow)]
            qualities.append(record["quality_score"])
            costs.append(record["cost_tokens"])

    avg_quality = sum(qualities) / len(qualities)
    avg_cost = sum(costs) / len(costs)
    return round(avg_quality, 4), round(avg_cost, 2)

# ── main ───────────────────────────────────────────────────────────────────
def main():
    print("Loading logs...")
    records = load_logs(LOG_FILE)
    print(f"Loaded {len(records)} records!")

    # Lambda values to sweep
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    pareto_data = []
    print("\nSweeping lambda values...")

    for lam in lambdas:
        avg_quality, avg_cost = simulate_with_lambda(records, lam)
        pareto_data.append({
            "lambda": lam,
            "avg_quality": avg_quality,
            "avg_cost": avg_cost
        })
        print(f"  lambda={lam}: quality={avg_quality}, cost={avg_cost}")

    # Fixed baselines
    always_w1_quality = 0.6002
    always_w1_cost = 443.30
    always_w3_quality = 0.8974
    always_w3_cost = 449.16

    # Save CSV
    csv_path = RESULTS_DIR / "pareto_data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "avg_quality", "avg_cost"])
        writer.writeheader()
        writer.writerows(pareto_data)
    print(f"\nPareto data saved to {csv_path}")

    # ── Plot ───────────────────────────────────────────────────────────────
    qualities = [d["avg_quality"] for d in pareto_data]
    costs = [d["avg_cost"] for d in pareto_data]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Our system curve
    ax.plot(costs, qualities, color="purple", linewidth=2,
            marker="o", markersize=8, label="Our System (varying λ)", zorder=3)

    # Label each lambda point
    offsets = {
        0.0:  (10, 8),
        0.1:  (10, -15),
        0.2:  (-55, -15),
        0.3:  (-55, 8),
        0.5:  (10, 8),
        0.7:  (10, 8),
        1.0:  (10, 8),
    }
    for d in pareto_data:
        offset = offsets.get(d["lambda"], (5, 5))
        ax.annotate(f"λ={d['lambda']}",
                    xy=(d["avg_cost"], d["avg_quality"]),
                    xytext=offset, textcoords="offset points", fontsize=9)

    # Baseline points
    ax.scatter(always_w1_cost, always_w1_quality, color="green",
               s=150, zorder=5, label="Always-W1 (cheap)", marker="*")
    ax.annotate("Always-W1", xy=(always_w1_cost, always_w1_quality),
                xytext=(10, -15), textcoords="offset points", fontsize=9, color="green")

    ax.scatter(always_w3_cost, always_w3_quality, color="coral",
               s=150, zorder=5, label="Always-W3 (expensive)", marker="*")
    ax.annotate("Always-W3", xy=(always_w3_cost, always_w3_quality),
                xytext=(10, 8), textcoords="offset points", fontsize=9, color="coral")

    ax.set_xlabel("Average Cost (tokens)", fontsize=13)
    ax.set_ylabel("Average Quality Score", fontsize=13)
    ax.set_title("Cost-Quality Pareto Frontier", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save both formats
    png_path = FIGURES_DIR / "pareto_frontier.png"
    pdf_path = FIGURES_DIR / "pareto_frontier.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, dpi=300)
    print(f"Figure saved to {png_path}")
    print(f"Figure saved to {pdf_path}")

    plt.show()
    print("\nPareto frontier done! This is Figure 2 of your paper!")

if __name__ == "__main__":
    main()