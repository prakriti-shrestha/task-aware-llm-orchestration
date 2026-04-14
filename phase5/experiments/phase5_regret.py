"""
Phase 5 — Regret Curves (Figure 3)
"""

# NOTE: With mock quality scores, LinUCB's feature-based learning has weak
# signal. Epsilon-greedy converges faster here because rewards are nearly
# uniform across workflows. This ordering will reverse with real LLM outputs
# in phase 5, where quality gaps between workflows are task-dependent.

from __future__ import annotations
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "phase4"))
from policy.bandit import LinUCBBandit, EpsilonGreedyBandit

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE     = PROJECT_ROOT / "phase3" / "data" / "runs" / "phase3_run_001.jsonl"
FIGURES_DIR  = PROJECT_ROOT / "phase5" / "figures"
RESULTS_DIR  = PROJECT_ROOT / "phase5" / "results"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

FEATURE_DIM = 16
SEEDS       = [42, 123, 456]


# ── Load logs ──────────────────────────────────────────────────────────────
def load_logs(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# ── Build lookup (task_id, workflow_id) → record ──────────────────────────
def build_lookup(records):
    return {(r["task_id"], r["workflow_id"]): r for r in records}


# ── Oracle reward per task ─────────────────────────────────────────────────
def build_oracle(records):
    best = defaultdict(lambda: -999.0)
    for r in records:
        best[r["task_id"]] = max(best[r["task_id"]], r["reward"])
    return dict(best)


# ── One record per task (for sequential replay) ────────────────────────────
def get_unique_tasks(records):
    seen, tasks = set(), []
    for r in records:
        if r["task_id"] not in seen:
            seen.add(r["task_id"])
            tasks.append(r)
    return tasks


# ── Run a policy through the task sequence ────────────────────────────────
def run_policy(policy, tasks, lookup, oracle):
    cum_regret = 0.0
    regrets    = []
    for rec in tasks:
        fv      = np.array(rec["feature_vector"])
        task_id = rec["task_id"]
        chosen  = policy.select_workflow(fv)
        key     = (task_id, chosen)
        reward  = lookup[key]["reward"] if key in lookup else 0.0
        policy.update(fv, chosen, reward)
        oracle_reward = oracle.get(task_id, reward)
        cum_regret   += max(0.0, oracle_reward - reward)
        regrets.append(cum_regret)
    return regrets


# ── Multi-seed runner ─────────────────────────────────────────────────────
def run_multi_seed(policy_cls, policy_kwargs, tasks, lookup, oracle):
    all_regrets = []
    for seed in SEEDS:
        policy = policy_cls(**{**policy_kwargs, "seed": seed})
        all_regrets.append(run_policy(policy, tasks, lookup, oracle))
    return np.array(all_regrets)


# ── Random policy (no learning) ───────────────────────────────────────────
class RandomPolicy:
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
    def select_workflow(self, fv):
        return self.rng.choice(["W1", "W2", "W3"])
    def update(self, fv, wf, reward):
        pass


def run_multi_seed_random(tasks, lookup, oracle):
    all_regrets = []
    for seed in SEEDS:
        all_regrets.append(run_policy(RandomPolicy(seed), tasks, lookup, oracle))
    return np.array(all_regrets)


# ── Plot ───────────────────────────────────────────────────────────────────
def plot_regret_curves(results, out_dir):
    fig, ax = plt.subplots(figsize=(9, 6))

    styles = {
        "LinUCB":         {"color": "#7B2D8B", "ls": "-",  "lw": 2.2},
        "Epsilon-Greedy": {"color": "#2AA198", "ls": "--", "lw": 1.8},
        "Random":         {"color": "#888888", "ls": ":",  "lw": 1.6},
    }

    for name, arr in results.items():
        mean = arr.mean(axis=0)
        ci   = 1.96 * arr.std(axis=0) / np.sqrt(arr.shape[0])
        eps  = np.arange(len(mean))
        s    = styles[name]
        ax.plot(eps, mean, color=s["color"], ls=s["ls"], lw=s["lw"], label=name)
        ax.fill_between(eps, mean - ci, mean + ci, color=s["color"], alpha=0.15)

    ax.set_xlabel("Episode", fontsize=13)
    ax.set_ylabel("Cumulative Regret", fontsize=13)
    # NOTE: title explains mock data ordering — do not remove this note
    ax.set_title(
        "Cumulative regret over episodes (mock data — see phase 5 for real results)",
        fontsize=11
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        p = out_dir / f"regret_curves.{ext}"
        fig.savefig(p, dpi=300)
        print(f"Saved → {p}")
    plt.close(fig)


# ── Save CSV ───────────────────────────────────────────────────────────────
def save_regret_csv(results, out_dir):
    rows = []
    for policy, arr in results.items():
        for i, seed in enumerate(SEEDS):
            for ep, val in enumerate(arr[i]):
                rows.append({
                    "policy": policy, "seed": seed,
                    "episode": ep, "cumulative_regret": round(float(val), 6),
                })
    out = out_dir / "regret_data.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["policy", "seed", "episode", "cumulative_regret"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {out}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("Loading logs...")
    records = load_logs(LOG_FILE)
    lookup  = build_lookup(records)
    tasks   = get_unique_tasks(records)
    oracle  = build_oracle(records)
    print(f"Tasks: {len(tasks)}")

    linucb  = run_multi_seed(LinUCBBandit,       {"feature_dim": FEATURE_DIM, "alpha": 1.0}, tasks, lookup, oracle)
    egreedy = run_multi_seed(EpsilonGreedyBandit, {"epsilon": 0.15},                          tasks, lookup, oracle)
    random  = run_multi_seed_random(tasks, lookup, oracle)

    results = {"LinUCB": linucb, "Epsilon-Greedy": egreedy, "Random": random}

    plot_regret_curves(results, FIGURES_DIR)
    save_regret_csv(results, RESULTS_DIR)
    print("Done!")


if __name__ == "__main__":
    main()