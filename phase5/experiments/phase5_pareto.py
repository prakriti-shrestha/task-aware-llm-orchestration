"""
Phase 5 — Experiment 2: Cost-Quality Pareto Frontier (Figure 2)

Sweep lambda across [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0].  For each lambda,
train LinUCB on the all_arms log and record (avg_cost, avg_quality).

Plot as a curve, overlay Always-W1 and Always-W3 as fixed reference points.

Outputs:
    results/pareto_data.csv
    figures/pareto_frontier.png
    figures/pareto_frontier.pdf

Prerequisite:
    Run phase5_main_results.py first — it produces data/runs/phase5_all_arms.jsonl.

Usage:
    python -m phase5.experiments.phase5_pareto
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
from phase4.policy.reward import compute_reward
from phase5.experiments._shared import (
    FEATURE_DIM, RESULTS_DIR, FIGURES_DIR, RUNS_DIR, COLOURS,
    feature_vector, load_run_log, ensure_dirs,
)

ALL_ARMS_LOG = RUNS_DIR / "phase5_all_arms.jsonl"
OUTPUT_CSV = RESULTS_DIR / "pareto_data.csv"
FIG_PNG = FIGURES_DIR / "pareto_frontier.png"
FIG_PDF = FIGURES_DIR / "pareto_frontier.pdf"

LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def _by_task(all_arms: list[dict]) -> tuple[list[str], dict[str, dict[str, dict]]]:
    """Group records by task_id, preserving order."""
    task_order: list[str] = []
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for rec in all_arms:
        tid = rec["task_id"]
        if tid not in by_task:
            task_order.append(tid)
        by_task[tid][rec["workflow_id"]] = rec
    return task_order, by_task


def run_lambda(all_arms: list[dict], lam: float, seed: int = 42) -> dict:
    """
    Train LinUCB with reward = quality - lam * normalised_cost, then
    return {avg_cost, avg_quality} on the same tasks.
    """
    task_order, by_task = _by_task(all_arms)

    # Compute max cost for normalisation (use W3 as reference)
    w3_costs = [rec["cost_tokens"] for rec in all_arms if rec["workflow_id"] == "W3"]
    max_cost = float(np.mean(w3_costs)) if w3_costs else 1000.0

    # Re-derive rewards at this lambda
    def reward_at(rec):
        return compute_reward(rec["quality_score"], rec["cost_tokens"], lam)

    bandit = LinUCBBandit(feature_dim=FEATURE_DIM, alpha=1.0, seed=seed)
    picks: list[dict] = []

    for tid in task_order:
        task_recs = by_task[tid]
        any_rec = next(iter(task_recs.values()))
        fv = feature_vector(any_rec["task_text"], tid)

        chosen = bandit.select_workflow(fv)
        if chosen not in task_recs:
            continue
        rec = task_recs[chosen]
        r = reward_at(rec)
        bandit.update(fv, chosen, r)
        picks.append(rec)

    # Report metrics on the SECOND HALF of the run (after learning has stabilised)
    half = len(picks) // 2
    tail = picks[half:] if half > 0 else picks

    avg_q = float(np.mean([p["quality_score"] for p in tail]))
    avg_c = float(np.mean([p["cost_tokens"] for p in tail]))
    return {"avg_quality": avg_q, "avg_cost": avg_c, "lambda": lam, "n": len(tail)}


def baseline_point(all_arms: list[dict], workflow_id: str) -> tuple[float, float]:
    recs = [r for r in all_arms if r["workflow_id"] == workflow_id]
    return (float(np.mean([r["cost_tokens"] for r in recs])),
            float(np.mean([r["quality_score"] for r in recs])))


def plot(points: list[dict], w1_point: tuple[float, float],
         w3_point: tuple[float, float]) -> None:
    import matplotlib.pyplot as plt

    # Sort by cost for a monotone-ish curve
    pts = sorted(points, key=lambda p: p["avg_cost"])
    xs = [p["avg_cost"] for p in pts]
    ys = [p["avg_quality"] for p in pts]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(xs, ys, "-o", color=COLOURS["ours"], linewidth=2.2,
            markersize=8, label="Ours (LinUCB, λ sweep)", zorder=3)
    for p in pts:
        ax.annotate(f"λ={p['lambda']:.1f}", (p["avg_cost"], p["avg_quality"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.scatter([w1_point[0]], [w1_point[1]], color=COLOURS["W1"], s=140,
               marker="s", label="Always-W1", zorder=4, edgecolor="black")
    ax.scatter([w3_point[0]], [w3_point[1]], color=COLOURS["W3"], s=140,
               marker="^", label="Always-W3", zorder=4, edgecolor="black")

    ax.set_xlabel("Average cost (tokens)", fontsize=12)
    ax.set_ylabel("Average quality score", fontsize=12)
    ax.set_title("Cost–Quality Pareto Frontier", fontsize=13)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    plt.savefig(FIG_PNG, dpi=300)
    plt.savefig(FIG_PDF)
    plt.close()
    print(f"[pareto] Saved → {FIG_PNG}")
    print(f"[pareto] Saved → {FIG_PDF}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()

    if not ALL_ARMS_LOG.exists():
        print(f"[pareto] ERROR: {ALL_ARMS_LOG} missing.")
        print("[pareto] Run phase5_main_results.py first to build it.")
        sys.exit(1)

    print(f"[pareto] Loading {ALL_ARMS_LOG}")
    all_arms = load_run_log(ALL_ARMS_LOG)
    print(f"[pareto] {len(all_arms)} records")

    print(f"[pareto] Sweeping {len(LAMBDAS)} lambda values…")
    points = []
    for lam in LAMBDAS:
        pt = run_lambda(all_arms, lam, seed=args.seed)
        points.append(pt)
        print(f"  λ={lam:.1f}  quality={pt['avg_quality']:.3f}  cost={pt['avg_cost']:.0f}")

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "avg_cost", "avg_quality", "n"])
        writer.writeheader()
        writer.writerows(points)
    print(f"\n[pareto] CSV → {OUTPUT_CSV}")

    w1 = baseline_point(all_arms, "W1")
    w3 = baseline_point(all_arms, "W3")
    print(f"[pareto] Always-W1: cost={w1[0]:.0f} quality={w1[1]:.3f}")
    print(f"[pareto] Always-W3: cost={w3[0]:.0f} quality={w3[1]:.3f}")

    plot(points, w1, w3)


if __name__ == "__main__":
    main()