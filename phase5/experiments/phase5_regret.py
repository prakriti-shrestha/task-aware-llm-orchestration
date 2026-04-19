"""
Phase 5 — Experiment 3: Regret Curves (Figure 3)

Compare three policies' cumulative regret over episodes:
    - LinUCB     (our system)
    - ε-greedy   (from phase4/policy/bandit.py — kept as baseline)
    - Random

Regret at episode t = oracle_reward(task_t) - policy_reward(task_t), accumulated.
Oracle reward = best reward achievable for that task across all three workflows.

Repeat each policy with 3 seeds, plot mean ± 95% CI shaded band.

Outputs:
    results/regret_data.csv
    figures/regret_curves.png
    figures/regret_curves.pdf

Prerequisite:
    data/runs/phase5_all_arms.jsonl (built by phase5_main_results.py)

Usage:
    python -m phase5.experiments.phase5_regret
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
from phase5.experiments._shared import (
    FEATURE_DIM, RESULTS_DIR, FIGURES_DIR, RUNS_DIR, COLOURS,
    feature_vector, load_run_log, ensure_dirs,
)
from phase5.experiments.baselines import RandomPolicy

ALL_ARMS_LOG = RUNS_DIR / "phase5_all_arms.jsonl"
OUTPUT_CSV = RESULTS_DIR / "regret_data.csv"
FIG_PNG = FIGURES_DIR / "regret_curves.png"
FIG_PDF = FIGURES_DIR / "regret_curves.pdf"

SEEDS = [42, 123, 2026]


class EpsilonGreedyBandit:
    """
    Simple ε-greedy bandit over workflow arms (no feature conditioning).
    Kept lightweight — this is just a baseline for the regret plot.
    """
    name = "epsilon_greedy"

    def __init__(self, epsilon: float = 0.1, seed: int = 42):
        self.epsilon = epsilon
        self._rng = np.random.RandomState(seed)
        self.counts = {"W1": 0, "W2": 0, "W3": 0}
        self.values = {"W1": 0.0, "W2": 0.0, "W3": 0.0}

    def select_workflow(self, feature_vector: np.ndarray) -> str:
        if self._rng.rand() < self.epsilon:
            return self._rng.choice(["W1", "W2", "W3"])
        return max(self.values, key=self.values.get)

    def update(self, feature_vector: np.ndarray, workflow_id: str, reward: float) -> None:
        self.counts[workflow_id] += 1
        n = self.counts[workflow_id]
        self.values[workflow_id] += (reward - self.values[workflow_id]) / n


def simulate(policy, tasks_by_id: dict[str, dict[str, dict]],
             task_order: list[str]) -> list[float]:
    """
    Simulate a policy over the given task order.
    Returns the cumulative regret series (one value per episode).
    """
    cum_regret = 0.0
    series: list[float] = []

    for tid in task_order:
        task_recs = tasks_by_id[tid]
        any_rec = next(iter(task_recs.values()))
        fv = feature_vector(any_rec["task_text"], tid)

        oracle_r = max(r["reward"] for r in task_recs.values())

        chosen = policy.select_workflow(fv)
        if chosen not in task_recs:
            actual_r = min(r["reward"] for r in task_recs.values())
        else:
            actual_r = task_recs[chosen]["reward"]
            policy.update(fv, chosen, actual_r)

        cum_regret += (oracle_r - actual_r)
        series.append(cum_regret)

    return series


def plot(curves: dict[str, list[list[float]]]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    styles = {
        "linucb":         {"color": COLOURS["linucb"],         "linestyle": "-",  "label": "LinUCB (ours)"},
        "epsilon_greedy": {"color": COLOURS["epsilon_greedy"], "linestyle": "--", "label": "ε-greedy"},
        "random":         {"color": COLOURS["random"],         "linestyle": ":",  "label": "Random"},
    }

    for policy_name, runs in curves.items():
        arr = np.array(runs)  # shape: (n_seeds, n_episodes)
        mean = arr.mean(axis=0)
        # 95% CI approximated as ±1.96 * SE across seeds
        se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean)
        lo, hi = mean - 1.96 * se, mean + 1.96 * se

        episodes = np.arange(1, len(mean) + 1)
        style = styles[policy_name]
        ax.plot(episodes, mean, color=style["color"],
                linestyle=style["linestyle"], linewidth=2.2, label=style["label"])
        ax.fill_between(episodes, lo, hi, color=style["color"], alpha=0.18)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Cumulative regret", fontsize=12)
    ax.set_title("Cumulative Regret vs Episode (95% CI)", fontsize=13)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(FIG_PNG, dpi=300)
    plt.savefig(FIG_PDF)
    plt.close()
    print(f"[regret] Saved → {FIG_PNG}")
    print(f"[regret] Saved → {FIG_PDF}")


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    ensure_dirs()

    if not ALL_ARMS_LOG.exists():
        print(f"[regret] ERROR: {ALL_ARMS_LOG} missing.")
        print("[regret] Run phase5_main_results.py first.")
        sys.exit(1)

    print(f"[regret] Loading {ALL_ARMS_LOG}")
    all_arms = load_run_log(ALL_ARMS_LOG)

    task_order: list[str] = []
    tasks_by_id: dict[str, dict[str, dict]] = defaultdict(dict)
    for rec in all_arms:
        tid = rec["task_id"]
        if tid not in tasks_by_id:
            task_order.append(tid)
        tasks_by_id[tid][rec["workflow_id"]] = rec

    # Only keep tasks where we have ALL three arms (oracle needs them all)
    complete = [tid for tid in task_order if len(tasks_by_id[tid]) == 3]
    print(f"[regret] {len(complete)} tasks have all three arms (out of {len(task_order)})")
    if len(complete) < 10:
        print("[regret] Too few complete tasks — increase --n-per-class in main_results")
        sys.exit(1)
    task_order = complete

    curves: dict[str, list[list[float]]] = defaultdict(list)
    csv_rows: list[dict] = []

    for seed in SEEDS:
        # Shuffle task order per seed so each seed sees a different stream
        rng = np.random.RandomState(seed)
        order = list(task_order)
        rng.shuffle(order)

        for policy_name, policy in [
            ("linucb",         LinUCBBandit(feature_dim=FEATURE_DIM, alpha=1.0, seed=seed)),
            ("epsilon_greedy", EpsilonGreedyBandit(epsilon=0.1, seed=seed)),
            ("random",         RandomPolicy(seed=seed)),
        ]:
            series = simulate(policy, tasks_by_id, order)
            curves[policy_name].append(series)
            for ep, val in enumerate(series, start=1):
                csv_rows.append({"policy": policy_name, "seed": seed,
                                 "episode": ep, "cumulative_regret": round(val, 6)})
            print(f"  seed={seed} {policy_name:<16} final regret={series[-1]:.3f}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["policy", "seed", "episode", "cumulative_regret"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[regret] CSV → {OUTPUT_CSV}")

    plot(curves)


if __name__ == "__main__":
    main()