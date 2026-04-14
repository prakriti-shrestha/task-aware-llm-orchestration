"""
Phase 5 — Workflow Distribution Plot (Figure 1)
Shows how workflows were distributed across the bandit training run.
"""

# FIX: Must read from phase3_bandit_train.jsonl, NOT phase3_run_001.jsonl.
# phase3_run_001.jsonl is the LABELING run where every workflow ran on every
# task exactly once — so by construction each workflow appears exactly 500 times.
# That tells you nothing about routing behavior.
# phase3_bandit_train.jsonl is the actual bandit training run with random routing,
# which gives a realistic ~equal-thirds distribution as the random baseline.

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ✅ FIXED: use bandit training log, not the labeling run
LOG_FILE = PROJECT_ROOT / "phase3" / "data" / "runs" / "phase3_bandit_train.jsonl"

FIG_DIR = PROJECT_ROOT / "phase5" / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_logs(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except Exception:
                    continue
    return data


def main():
    records = load_logs(LOG_FILE)

    workflows = []
    for r in records:
        wf = r.get("workflow_id") or r.get("workflow") or r.get("action")
        if wf:
            workflows.append(wf)

    if not workflows:
        print("No workflow data found.")
        return

    counts = Counter(workflows)
    print("Workflow counts from bandit training run:", dict(counts))

    # Colors per handbook: W1=green, W2=amber, W3=coral
    colors = {"W1": "green", "W2": "#F59E0B", "W3": "coral"}
    bar_colors = [colors.get(k, "gray") for k in counts.keys()]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.keys(), counts.values(), color=bar_colors)

    # Label counts on top of each bar
    for bar, val in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                str(val), ha="center", fontsize=11)

    ax.set_xlabel("Workflow", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Workflow Distribution (Bandit Training Run — Random Policy Baseline)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    fig.savefig(FIG_DIR / "workflow_distribution.png", dpi=300)
    fig.savefig(FIG_DIR / "workflow_distribution.pdf", dpi=300)
    plt.close(fig)

    print("Saved workflow_distribution.png and .pdf")


if __name__ == "__main__":
    main()