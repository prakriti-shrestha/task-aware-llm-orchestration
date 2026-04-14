"""
Phase 5 — Pareto Frontier (Figure 2)
REAL DATA VERSION (clean + consistent)
"""

import json
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE = PROJECT_ROOT / "phase3" / "data" / "runs" / "phase3_run_001.jsonl"
RESULTS_DIR = PROJECT_ROOT / "phase4" / "results"
FIGURES_DIR = PROJECT_ROOT / "phase4" / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── load logs ──────────────────────────────────────────────────────────────
def load_logs(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records

# ── build lookup ───────────────────────────────────────────────────────────
def build_lookup(records):
    return {(r["task_id"], r["workflow_id"]): r for r in records}

# ── simulate system with lambda ───────────────────────────────────────────
def simulate_with_lambda(records, lookup, lambda_value):
    task_ids = list({r["task_id"] for r in records})

    qualities = []
    costs = []

    for task_id in task_ids:
        best_reward = float("-inf")
        best_record = None

        for workflow in ["W1", "W2", "W3"]:
            key = (task_id, workflow)
            if key not in lookup:
                continue

            r = lookup[key]

            # ✅ consistent reward computation
            cost_norm = r["cost_tokens"] / 1000.0
            reward = r["quality_score"] - lambda_value * cost_norm

            if reward > best_reward:
                best_reward = reward
                best_record = r

        if best_record:
            qualities.append(best_record["quality_score"])
            costs.append(best_record["cost_tokens"])

    avg_quality = sum(qualities) / len(qualities)
    avg_cost = sum(costs) / len(costs)

    return round(avg_quality, 4), round(avg_cost, 2)

# ── compute baselines from logs ───────────────────────────────────────────
def compute_baseline(records, workflow):
    filtered = [r for r in records if r["workflow_id"] == workflow]
    avg_q = sum(r["quality_score"] for r in filtered) / len(filtered)
    avg_c = sum(r["cost_tokens"] for r in filtered) / len(filtered)
    return avg_q, avg_c

# ── main ───────────────────────────────────────────────────────────────────
def main():
    print("Loading logs...")
    records = load_logs(LOG_FILE)
    lookup = build_lookup(records)

    print(f"Loaded {len(records)} records!")

    lambdas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    pareto_data = []
    print("\nSweeping lambda values...")

    for lam in lambdas:
        avg_quality, avg_cost = simulate_with_lambda(records, lookup, lam)
        pareto_data.append({
            "lambda": lam,
            "avg_quality": avg_quality,
            "avg_cost": avg_cost
        })
        print(f"  λ={lam}: quality={avg_quality}, cost={avg_cost}")

    # ✅ REAL baselines (not hardcoded)
    w1_q, w1_c = compute_baseline(records, "W1")
    w3_q, w3_c = compute_baseline(records, "W3")

    # Save CSV
    csv_path = RESULTS_DIR / "pareto_data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "avg_quality", "avg_cost"])
        writer.writeheader()
        writer.writerows(pareto_data)

    print(f"\nSaved → {csv_path}")

    # ── Plot ───────────────────────────────────────────────────────────────
    qualities = [d["avg_quality"] for d in pareto_data]
    costs = [d["avg_cost"] for d in pareto_data]

    plt.figure(figsize=(8, 6))

    # System curve
    plt.plot(costs, qualities, marker="o", label="Our System (λ sweep)")

    # Label λ points
    for d in pareto_data:
        plt.annotate(f"λ={d['lambda']}",
                     (d["avg_cost"], d["avg_quality"]),
                     xytext=(5, 5), textcoords="offset points", fontsize=9)

    # Baselines
    plt.scatter(w1_c, w1_q, label="Always-W1", marker="*", s=150)
    plt.scatter(w3_c, w3_q, label="Always-W3", marker="*", s=150)

    plt.xlabel("Average Cost (tokens)")
    plt.ylabel("Average Quality")
    plt.title("Pareto Frontier (Cost vs Quality)")
    plt.legend()
    plt.grid(True)

    png_path = FIGURES_DIR / "pareto_frontier.png"
    pdf_path = FIGURES_DIR / "pareto_frontier.pdf"

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, dpi=300)

    print(f"Saved → {png_path}")
    print(f"Saved → {pdf_path}")

    plt.show()


if __name__ == "__main__":
    main()