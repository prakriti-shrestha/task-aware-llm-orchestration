"""
Phase 5 — Figure 1: Workflow Routing Distribution

Shows what fraction of each task class the trained policy routes to W1/W2/W3.
This is the "proof" that the system has actually learned to differentiate.

Outputs:
    figures/workflow_distribution.png
    figures/workflow_distribution.pdf

Prerequisite:
    data/runs/phase5_all_arms.jsonl

Usage:
    python -m phase5.experiments.phase5_workflow_distribution
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase3"))  # Phase 3 uses bare `from data.X import ...`

from phase4.policy.bandit import LinUCBBandit
from phase5.experiments._shared import (
    FEATURE_DIM, FIGURES_DIR, RUNS_DIR, COLOURS,
    feature_vector, load_run_log, ensure_dirs,
)

ALL_ARMS_LOG = RUNS_DIR / "phase5_all_arms.jsonl"
FIG_PNG = FIGURES_DIR / "workflow_distribution.png"
FIG_PDF = FIGURES_DIR / "workflow_distribution.pdf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()

    if not ALL_ARMS_LOG.exists():
        print(f"[dist] ERROR: {ALL_ARMS_LOG} missing.")
        sys.exit(1)

    all_arms = load_run_log(ALL_ARMS_LOG)

    task_order: list[str] = []
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for rec in all_arms:
        tid = rec["task_id"]
        if tid not in by_task:
            task_order.append(tid)
        by_task[tid][rec["workflow_id"]] = rec

    # Train LinUCB once, then replay picks WITHOUT updating to get its final
    # learned policy.  (First pass learns; second pass just reads out.)
    bandit = LinUCBBandit(feature_dim=FEATURE_DIM, alpha=1.0, seed=args.seed)
    for tid in task_order:
        task_recs = by_task[tid]
        any_rec = next(iter(task_recs.values()))
        fv = feature_vector(any_rec["task_text"], tid)
        chosen = bandit.select_workflow(fv)
        if chosen in task_recs:
            bandit.update(fv, chosen, task_recs[chosen]["reward"])

    # Second pass — freeze exploration by copying with alpha=0 would be ideal;
    # as a proxy we just read the same bandit's greedy pick.  LinUCBBandit may
    # use its alpha term, so we accept small exploration noise here.
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"W1": 0, "W2": 0, "W3": 0})
    for tid in task_order:
        task_recs = by_task[tid]
        any_rec = next(iter(task_recs.values()))
        tc = any_rec["task_class"]
        fv = feature_vector(any_rec["task_text"], tid)
        chosen = bandit.select_workflow(fv)
        counts[tc][chosen] += 1

    # Normalise to percentages
    classes = sorted(counts.keys())
    w1 = [counts[c]["W1"] / max(sum(counts[c].values()), 1) for c in classes]
    w2 = [counts[c]["W2"] / max(sum(counts[c].values()), 1) for c in classes]
    w3 = [counts[c]["W3"] / max(sum(counts[c].values()), 1) for c in classes]

    print(f"{'class':<12} {'W1':>6} {'W2':>6} {'W3':>6}")
    print("-" * 32)
    for i, c in enumerate(classes):
        print(f"{c:<12} {w1[i]*100:>5.0f}% {w2[i]*100:>5.0f}% {w3[i]*100:>5.0f}%")

    # Plot stacked bar
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(classes))
    w1_arr, w2_arr, w3_arr = np.array(w1), np.array(w2), np.array(w3)

    ax.bar(x, w1_arr, color=COLOURS["W1"], label="W1 (cheap)", edgecolor="white")
    ax.bar(x, w2_arr, bottom=w1_arr, color=COLOURS["W2"],
           label="W2 (balanced)", edgecolor="white")
    ax.bar(x, w3_arr, bottom=w1_arr + w2_arr, color=COLOURS["W3"],
           label="W3 (heavy)", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylabel("Routing share", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Learned Workflow Routing by Task Class", fontsize=13)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.0, -0.1), ncol=3)

    plt.tight_layout()
    plt.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_PDF, bbox_inches="tight")
    plt.close()

    print(f"\n[dist] Saved → {FIG_PNG}")
    print(f"[dist] Saved → {FIG_PDF}")


if __name__ == "__main__":
    main()