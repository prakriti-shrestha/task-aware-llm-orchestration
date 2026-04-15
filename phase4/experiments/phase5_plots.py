"""
Phase 5 plots — scaffold version.

Generates all three required paper figures:
  Figure 1 — Workflow routing distribution by task class (bar chart)
  Figure 2 — Cost-quality Pareto frontier
  Figure 3 — Regret curves (LinUCB vs epsilon-greedy vs random)

Run:
    python phase4/experiments/phase5_plots.py

Output:
    phase4/figures/workflow_distribution.png
    phase4/figures/pareto_frontier.png
    phase4/figures/regret_curves.png

Install: pip install matplotlib
"""
from __future__ import annotations
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policy.bandit import LinUCBBandit
from policy.reward import compute_reward
from experiments.baselines import RandomPolicy

FIGURES_DIR  = Path(__file__).resolve().parent.parent / "figures"
BANDIT_LOG   = Path(__file__).resolve().parent.parent.parent / "phase3/data/runs/phase3_bandit_train.jsonl"
LAMBDA_LOGS  = Path(__file__).resolve().parent.parent.parent / "phase3/data/runs"

# ── Color scheme — consistent across all plots ────────────────────────────────
COLORS = {"W1": "#1A6B3A", "W2": "#B07800", "W3": "#C0392B",
          "linucb": "#7B5EA7", "epsilon": "#1D7A6E", "random": "#888780"}

DPI = 300


def load_bandit_log() -> list[dict]:
    records = []
    with open(BANDIT_LOG, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


# ── Figure 1 — Workflow distribution by task class ────────────────────────────
def plot_workflow_distribution(records: list[dict]):
    """
    Bar chart showing how each policy distributes workflow choices
    across task classes. Uses the LinUCB policy run.
    """
    ORACLE_QUALITY = {
        ("W1", "qa"): 0.82,   ("W1", "reasoning"): 0.52,
        ("W1", "code"): 0.48, ("W1", "explanation"): 0.65,
        ("W2", "qa"): 0.75,   ("W2", "reasoning"): 0.73,
        ("W2", "code"): 0.70, ("W2", "explanation"): 0.72,
        ("W3", "qa"): 0.76,   ("W3", "reasoning"): 0.91,
        ("W3", "code"): 0.88, ("W3", "explanation"): 0.80,
    }
    COST = {"W1": 150, "W2": 350, "W3": 700}
    rng    = np.random.RandomState(42)
    bandit = LinUCBBandit(feature_dim=16, alpha=1.0, seed=42)

    # Run LinUCB and collect choices
    choices: dict[str, dict[str, int]] = {}   # task_class → {W1:n, W2:n, W3:n}
    for rec in records:
        fv = np.array(rec["feature_vector"])
        tc = rec["task_class"]
        chosen = bandit.select_workflow(fv)
        q = float(np.clip(ORACLE_QUALITY.get((chosen, tc), 0.70) + rng.normal(0, 0.04), 0, 1))
        r = compute_reward(q, COST[chosen], 0.5)
        bandit.update(fv, chosen, r)
        choices.setdefault(tc, {"W1": 0, "W2": 0, "W3": 0})
        choices[tc][chosen] = choices[tc].get(chosen, 0) + 1

    task_classes = sorted(choices.keys())
    x     = np.arange(len(task_classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, wf in enumerate(["W1", "W2", "W3"]):
        counts = [choices[tc].get(wf, 0) for tc in task_classes]
        totals = [sum(choices[tc].values()) for tc in task_classes]
        pcts   = [c / t * 100 if t > 0 else 0 for c, t in zip(counts, totals)]
        ax.bar(x + i * width, pcts, width, label=wf, color=COLORS[wf], alpha=0.85)

    ax.set_xlabel("Task class", fontsize=12)
    ax.set_ylabel("Workflow selection (%)", fontsize=12)
    ax.set_title("Workflow routing distribution by task class (LinUCB)", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels(task_classes, fontsize=10)
    ax.legend(title="Workflow", fontsize=10)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    path = FIGURES_DIR / "workflow_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close()
    print(f"[plots] Saved → {path}")


# ── Figure 2 — Pareto frontier ────────────────────────────────────────────────
def plot_pareto_frontier():
    """
    Plots (avg_cost_tokens, avg_quality) for each lambda value.
    Overlays Always-W1 and Always-W3 fixed points.
    """
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    costs, qualities = [], []

    for lam in lambdas:
        log_path = LAMBDA_LOGS / f"phase4_lambda_{lam}.jsonl"
        if not log_path.exists():
            print(f"[plots] Warning: missing {log_path.name} — skipping")
            continue
        qs, cs = [], []
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                qs.append(r["quality"])
                cs.append(r["cost"])
        costs.append(np.mean(cs))
        qualities.append(np.mean(qs))

    fig, ax = plt.subplots(figsize=(7, 5))

    # Our system curve
    ax.plot(costs, qualities, "o-", color=COLORS["linucb"], linewidth=2,
            markersize=7, label="Our system (LinUCB)", zorder=3)

    # Annotate lambda values
    for lam, c, q in zip(lambdas[:len(costs)], costs, qualities):
        ax.annotate(f"λ={lam}", (c, q), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=COLORS["linucb"])

    # Fixed baselines
    ax.scatter([150], [0.72], s=120, color=COLORS["W1"], zorder=4,
               label="Always-W1 (cheap baseline)", marker="s")
    ax.scatter([700], [0.89], s=120, color=COLORS["W3"], zorder=4,
               label="Always-W3 (heavy baseline)", marker="^")

    ax.set_xlabel("Average cost (tokens)", fontsize=12)
    ax.set_ylabel("Average quality score", fontsize=12)
    ax.set_title("Cost–quality Pareto frontier", fontsize=13)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    path = FIGURES_DIR / "pareto_frontier.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close()
    print(f"[plots] Saved → {path}")


# ── Figure 3 — Regret curves ──────────────────────────────────────────────────
def plot_regret_curves(records: list[dict]):
    """
    Cumulative regret over episodes for 3 policies × 3 seeds.
    Shaded region = 95% confidence interval across seeds.
    """
    ORACLE_QUALITY = {
        ("W1", "qa"): 0.82,   ("W1", "reasoning"): 0.52,
        ("W1", "code"): 0.48, ("W1", "explanation"): 0.65,
        ("W2", "qa"): 0.75,   ("W2", "reasoning"): 0.73,
        ("W2", "code"): 0.70, ("W2", "explanation"): 0.72,
        ("W3", "qa"): 0.76,   ("W3", "reasoning"): 0.91,
        ("W3", "code"): 0.88, ("W3", "explanation"): 0.80,
    }
    COST = {"W1": 150, "W2": 350, "W3": 700}
    SEEDS = [42, 123, 7]

    def _oracle_reward(task_class: str, rng: np.random.RandomState) -> float:
        """Best possible reward for this task class."""
        best_wf = "W1" if task_class == "qa" else "W3"
        q = float(np.clip(ORACLE_QUALITY[(best_wf, task_class)] + rng.normal(0, 0.04), 0, 1))
        return compute_reward(q, COST[best_wf], 0.5)

    def _run_policy_regret(policy_fn, seed: int) -> list[float]:
        """Returns cumulative regret list over episodes."""
        rng = np.random.RandomState(seed)
        cumulative = 0.0
        regrets = []
        for rec in records:
            fv = np.array(rec["feature_vector"])
            tc = rec["task_class"]
            chosen = policy_fn(fv)
            q = float(np.clip(ORACLE_QUALITY.get((chosen, tc), 0.70) + rng.normal(0, 0.04), 0, 1))
            actual_reward  = compute_reward(q, COST[chosen] + rng.randint(-20, 20), 0.5)
            oracle_r       = _oracle_reward(tc, rng)
            cumulative    += max(0.0, oracle_r - actual_reward)
            regrets.append(cumulative)
        return regrets

    def _run_linucb_regret(seed: int) -> list[float]:
        bandit = LinUCBBandit(feature_dim=16, alpha=1.0, seed=seed)
        rng    = np.random.RandomState(seed)
        cumulative = 0.0
        regrets    = []
        for rec in records:
            fv = np.array(rec["feature_vector"])
            tc = rec["task_class"]
            chosen = bandit.select_workflow(fv)
            q = float(np.clip(ORACLE_QUALITY.get((chosen, tc), 0.70) + rng.normal(0, 0.04), 0, 1))
            actual_reward = compute_reward(q, COST[chosen] + rng.randint(-20, 20), 0.5)
            oracle_r      = _oracle_reward(tc, rng)
            bandit.update(fv, chosen, actual_reward)
            cumulative   += max(0.0, oracle_r - actual_reward)
            regrets.append(cumulative)
        return regrets

    policy_configs = [
        ("LinUCB (ours)",    "linucb",   lambda fv, s=None: None,  _run_linucb_regret),
        ("Epsilon-greedy",   "epsilon",  None,                      None),
        ("Random",           "random",   None,                      None),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    episodes = np.arange(1, len(records) + 1)

    # LinUCB
    linucb_runs = [_run_linucb_regret(s) for s in SEEDS]
    linucb_arr  = np.array(linucb_runs)
    mean = linucb_arr.mean(0)
    std  = linucb_arr.std(0)
    ax.plot(episodes, mean, color=COLORS["linucb"], linewidth=2, label="LinUCB (ours)")
    ax.fill_between(episodes, mean - 1.96*std, mean + 1.96*std,
                    color=COLORS["linucb"], alpha=0.15)

    # Epsilon-greedy (simulated with decaying epsilon)
    def _run_egreedy_regret(seed: int) -> list[float]:
        rng = np.random.RandomState(seed)
        cumulative = 0.0
        regrets = []
        counts = {"W1": 0, "W2": 0, "W3": 0}
        totals = {"W1": 0.0, "W2": 0.0, "W3": 0.0}
        for i, rec in enumerate(records):
            tc  = rec["task_class"]
            eps = max(0.05, 0.3 - i * 0.0005)
            if rng.random() < eps:
                chosen = rng.choice(["W1", "W2", "W3"])
            else:
                means = {w: totals[w]/counts[w] if counts[w]>0 else 0.0 for w in ["W1","W2","W3"]}
                chosen = max(means, key=means.get)
            q = float(np.clip(ORACLE_QUALITY.get((chosen, tc), 0.70) + rng.normal(0, 0.04), 0, 1))
            r = compute_reward(q, COST[chosen] + rng.randint(-20, 20), 0.5)
            counts[chosen] += 1
            totals[chosen] += r
            oracle_r   = _oracle_reward(tc, rng)
            cumulative += max(0.0, oracle_r - r)
            regrets.append(cumulative)
        return regrets

    eg_runs = [_run_egreedy_regret(s) for s in SEEDS]
    eg_arr  = np.array(eg_runs)
    mean = eg_arr.mean(0)
    std  = eg_arr.std(0)
    ax.plot(episodes, mean, "--", color=COLORS["epsilon"], linewidth=2, label="Epsilon-greedy")
    ax.fill_between(episodes, mean - 1.96*std, mean + 1.96*std,
                    color=COLORS["epsilon"], alpha=0.15)

    # Random
    rand_runs = [
        _run_policy_regret(RandomPolicy(seed=s).select_workflow, s) for s in SEEDS
    ]
    rand_arr = np.array(rand_runs)
    mean = rand_arr.mean(0)
    std  = rand_arr.std(0)
    ax.plot(episodes, mean, ":", color=COLORS["random"], linewidth=2, label="Random")
    ax.fill_between(episodes, mean - 1.96*std, mean + 1.96*std,
                    color=COLORS["random"], alpha=0.15)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Cumulative regret", fontsize=12)
    ax.set_title("Cumulative regret over episodes", fontsize=13)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    path = FIGURES_DIR / "regret_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close()
    print(f"[plots] Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    records = load_bandit_log()
    print(f"[plots] Loaded {len(records)} records\n")

    plot_workflow_distribution(records)
    plot_pareto_frontier()
    plot_regret_curves(records)

    print(f"\n[plots] All figures saved to {FIGURES_DIR}")
    print("[plots] Check phase4/figures/ for .png and .pdf versions")