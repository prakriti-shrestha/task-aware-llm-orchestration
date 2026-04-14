"""
Phase 5 - Pareto Frontier (Figure 2)
"""

import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE     = PROJECT_ROOT / "phase3" / "data" / "runs" / "phase3_run_001.jsonl"
RESULTS_DIR  = PROJECT_ROOT / "phase5" / "results"
FIGURES_DIR  = PROJECT_ROOT / "phase5" / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

W1_C, W1_Q = 150, 0.60
W3_C, W3_Q = 700, 0.90

LAMBDA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def load_logs(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records


def build_lookup(records):
    return {(r["task_id"], r["workflow_id"]): r for r in records}


def simulate_with_lambda(records, lookup, lam):
    task_ids = list({r["task_id"] for r in records})
    qualities, costs = [], []
    for task_id in task_ids:
        best_reward = float("-inf")
        best_record = None
        for workflow in ["W1", "W2", "W3"]:
            key = (task_id, workflow)
            if key not in lookup:
                continue
            r = lookup[key]
            reward = r["quality_score"] - lam * (r["cost_tokens"] / 1000.0)
            if reward > best_reward:
                best_reward = reward
                best_record = r
        if best_record:
            qualities.append(best_record["quality_score"])
            costs.append(best_record["cost_tokens"])
    return round(sum(qualities) / len(qualities), 4), \
           round(sum(costs)    / len(costs),    2)


def main():
    print("Loading logs...")
    records = load_logs(LOG_FILE)
    lookup  = build_lookup(records)
    print(f"Loaded {len(records)} records")

    pareto_data = []
    for lam in LAMBDA_VALUES:
        q, c = simulate_with_lambda(records, lookup, lam)
        pareto_data.append({"lambda": lam, "q": q, "c": c})
        print(f"  lambda={lam}: quality={q}, cost={c}")

    csv_path = RESULTS_DIR / "pareto_data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "avg_quality", "avg_cost"])
        writer.writeheader()
        for d in pareto_data:
            writer.writerow({"lambda": d["lambda"], "avg_quality": d["q"], "avg_cost": d["c"]})

    costs = [d["c"] for d in pareto_data]
    quals = [d["q"] for d in pareto_data]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#F9F8F5")
    ax.set_facecolor("#F9F8F5")

    ax.plot(costs, quals, color="#5B21B6", linewidth=3.0, zorder=3,
            solid_capstyle="round")

    # 4 labels with arrows - carefully placed so nothing overlaps
    selected_labels = [
        (costs[0], quals[0],   20,  -45, "lambda=0.0 (max quality)"),
        (costs[3], quals[3], -130,   40, "lambda=0.3"),
        (costs[4], quals[4], -130,  -40, "lambda=0.5 (balanced)"),
        (costs[6], quals[6],   20,  -45, "lambda=1.0 (max cost saving)"),
    ]

    for pc, pq, ox, oy, label in selected_labels:
        ax.annotate(
            label,
            xy=(pc, pq),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=9,
            color="#3B0764",
            fontweight="600",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#9B59B6", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#C4B5FD", alpha=1.0, linewidth=0.8),
            zorder=10
        )

    ax.scatter([W1_C], [W1_Q], color="#16A34A", marker="*",
               s=400, zorder=6, edgecolors="white", linewidths=1)
    ax.annotate(
        "Always-W1\n(cost=150, quality=0.60)",
        xy=(W1_C, W1_Q), xytext=(55, -50), textcoords="offset points",
        fontsize=9, color="#166534", fontweight="700", ha="left",
        arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F0FDF4",
                  edgecolor="#86EFAC", alpha=1.0, linewidth=1.0),
        zorder=10
    )

    ax.scatter([W3_C], [W3_Q], color="#DC2626", marker="*",
               s=400, zorder=6, edgecolors="white", linewidths=1)
    ax.annotate(
        "Always-W3\n(cost=700, quality=0.90)",
        xy=(W3_C, W3_Q), xytext=(-20, -60), textcoords="offset points",
        fontsize=9, color="#991B1B", fontweight="700", ha="center",
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF2F2",
                  edgecolor="#FCA5A5", alpha=1.0, linewidth=1.0),
        zorder=10
    )

    ax.set_xlabel("Average Cost (tokens)", fontsize=12, labelpad=8)
    ax.set_ylabel("Average Quality Score",  fontsize=12, labelpad=8)
    ax.set_title("Cost-Quality Pareto Frontier", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(50, 800)
    ax.set_ylim(0.54, 0.97)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.2, linestyle="--", color="#888")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    curve_h = plt.Line2D([0],[0], color="#5B21B6", linewidth=2.5,
                         label="Our System (lambda sweep)")
    w1_h    = plt.Line2D([0],[0], color="#16A34A", marker="*", markersize=11,
                         linewidth=0, label="Always-W1 (cheap baseline)")
    w3_h    = plt.Line2D([0],[0], color="#DC2626", marker="*", markersize=11,
                         linewidth=0, label="Always-W3 (expensive baseline)")
    ax.legend(handles=[curve_h, w1_h, w3_h], fontsize=9,
              loc="lower right", framealpha=0.95, edgecolor="#ccc")

    fig.tight_layout(pad=2)

    out_png = FIGURES_DIR / "pareto_frontier.png"
    out_pdf = FIGURES_DIR / "pareto_frontier.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved -> {out_png}")
    print(f"Saved -> {out_pdf}")
    print("DONE! Now open pareto_frontier.png in Finder to see it.")


if __name__ == "__main__":
    main()