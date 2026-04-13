"""
Phase 4 — Lambda reward sweep.
Runs the bandit 200 episodes at each lambda value and saves separate logs.
Member 2 uses these to generate the Pareto frontier plot.

Run:
    python phase4/experiments/phase4_reward_sweep.py

Output:
    phase3/data/runs/phase4_lambda_0.1.jsonl
    phase3/data/runs/phase4_lambda_0.5.jsonl
    phase3/data/runs/phase4_lambda_1.0.jsonl
"""
from __future__ import annotations
import json
import sys
import hashlib
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policy.bandit import LinUCBBandit
from policy.reward import compute_reward

BANDIT_LOG = Path(__file__).resolve().parent.parent.parent / "phase3/data/runs/phase3_bandit_train.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "phase3/data/runs"
LAMBDAS    = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
N_EPISODES = 200
SEED       = 42

ORACLE_QUALITY = {
    ("W1", "qa"):          0.82, ("W1", "reasoning"): 0.52,
    ("W1", "code"):        0.48, ("W1", "explanation"): 0.65,
    ("W2", "qa"):          0.75, ("W2", "reasoning"):  0.73,
    ("W2", "code"):        0.70, ("W2", "explanation"): 0.72,
    ("W3", "qa"):          0.76, ("W3", "reasoning"):  0.91,
    ("W3", "code"):        0.88, ("W3", "explanation"): 0.80,
}
COST = {"W1": 150, "W2": 350, "W3": 700}


def run_sweep():
    records = []
    with open(BANDIT_LOG) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    records = records[:N_EPISODES]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for lam in LAMBDAS:
        rng    = np.random.RandomState(SEED)
        bandit = LinUCBBandit(feature_dim=16, alpha=1.0)
        results = []

        for rec in records:
            x          = np.array(rec["feature_vector"])
            task_class = rec["task_class"]
            chosen     = bandit.select_workflow(x)

            base_q  = ORACLE_QUALITY.get((chosen, task_class), 0.70)
            quality = float(np.clip(base_q + rng.normal(0, 0.04), 0, 1))
            cost_t  = COST[chosen] + rng.randint(-20, 20)
            reward  = compute_reward(quality, cost_t, lam)

            bandit.update(x, chosen, reward)
            results.append({
                "lambda":    lam,
                "workflow":  chosen,
                "quality":   round(quality, 4),
                "cost":      int(cost_t),
                "reward":    round(reward, 6),
                "task_class": task_class,
            })

        avg_q    = sum(r["quality"] for r in results) / len(results)
        avg_cost = sum(r["cost"]    for r in results) / len(results)
        out_path = OUTPUT_DIR / f"phase4_lambda_{lam}.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        summary.append((lam, avg_q, avg_cost))
        print(f"  λ={lam:.1f} → avg_quality={avg_q:.4f}  avg_cost={avg_cost:.1f}  → {out_path.name}")

    print("\nPareto data (copy this to Member 2):")
    print(f"  {'lambda':>6}  {'avg_quality':>11}  {'avg_cost':>9}")
    for lam, q, c in summary:
        print(f"  {lam:>6.1f}  {q:>11.4f}  {c:>9.1f}")


if __name__ == "__main__":
    print(f"[reward_sweep] Running {len(LAMBDAS)} lambda values × {N_EPISODES} episodes each")
    run_sweep()