"""
Phase 5 main results experiment — scaffold version.

Runs all 6 policies on the phase 3 bandit training log and produces
the numbers for Table 1 of the paper.

Policies evaluated:
  1. AlwaysW1
  2. AlwaysW3
  3. Random
  4. Oracle
  5. LinUCB (our system — full)
  6. LinUCB — no encoder (random feature vector ablation)

In phase 5, replace _get_quality() with a real LLM call.
Everything else stays the same.

Run:
    python phase4/experiments/phase5_main_results.py

Output:
    phase4/results/main_results.csv
    phase4/results/main_results_per_class.csv
"""
from __future__ import annotations
import csv
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.baselines import AlwaysW1Policy, AlwaysW3Policy, RandomPolicy, OraclePolicy
from policy.bandit import LinUCBBandit
from policy.reward import compute_reward
from experiments.metrics import aggregate

# ── Paths ─────────────────────────────────────────────────────────────────────
BANDIT_LOG  = Path(__file__).resolve().parent.parent.parent / "phase3/data/runs/phase3_bandit_train.jsonl"
RUN_LOG     = Path(__file__).resolve().parent.parent.parent / "phase3/data/runs/phase3_run_001.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

LAMBDA      = 0.5
FEATURE_DIM = 16
SEED        = 42

# Oracle quality lookup (mock — replace with real LLM in phase 5)
ORACLE_QUALITY = {
    ("W1", "qa"):          0.82, ("W1", "reasoning"): 0.52,
    ("W1", "code"):        0.48, ("W1", "explanation"): 0.65,
    ("W2", "qa"):          0.75, ("W2", "reasoning"): 0.73,
    ("W2", "code"):        0.70, ("W2", "explanation"): 0.72,
    ("W3", "qa"):          0.76, ("W3", "reasoning"): 0.91,
    ("W3", "code"):        0.88, ("W3", "explanation"): 0.80,
}
COST_TOKENS = {"W1": 150, "W2": 350, "W3": 700}


def _get_quality(workflow_id: str, task_class: str, rng: np.random.RandomState) -> float:
    """
    Mock quality lookup.
    ── PHASE 5 REPLACEMENT POINT ──
    Replace this entire function with a real LLM call:
        output = run_workflow(workflow_id, task_text)
        return evaluate(task_text, output, ground_truth, task_class)
    """
    base = ORACLE_QUALITY.get((workflow_id, task_class), 0.70)
    return float(np.clip(base + rng.normal(0, 0.04), 0, 1))


def _run_policy(policy, records: list[dict], rng: np.random.RandomState) -> list[dict]:
    """Run a single policy over all records and collect episode results."""
    results = []
    is_oracle = isinstance(policy, OraclePolicy)

    for rec in records:
        fv        = np.array(rec["feature_vector"])
        task_id   = rec["task_id"]
        task_class = rec["task_class"]

        chosen = policy.select_workflow(fv, task_id=task_id) if is_oracle \
                 else policy.select_workflow(fv)

        quality    = _get_quality(chosen, task_class, rng)
        cost_t     = COST_TOKENS[chosen] + rng.randint(-20, 20)
        reward     = compute_reward(quality, cost_t, LAMBDA)

        policy.update(fv, chosen, reward)

        results.append({
            "task_id":      task_id,
            "task_class":   task_class,
            "workflow_id":  chosen,
            "quality_score": round(quality, 4),
            "cost_tokens":  int(cost_t),
            "reward":       round(reward, 6),
        })

    return results


def run():
    # Load records
    records = []
    with open(BANDIT_LOG, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    print(f"[main_results] {len(records)} tasks loaded")
    print(f"[main_results] Running 6 policies...\n")

    policies = [
        ("always_w1",         AlwaysW1Policy()),
        ("always_w3",         AlwaysW3Policy()),
        ("random",            RandomPolicy(seed=SEED)),
        ("oracle",            OraclePolicy(RUN_LOG)),
        ("linucb_full",       LinUCBBandit(feature_dim=FEATURE_DIM, alpha=1.0, seed=SEED)),
        ("linucb_no_encoder", LinUCBBandit(feature_dim=FEATURE_DIM, alpha=1.0, seed=SEED)),
    ]

    all_summaries = []
    per_class_rows = []

    for name, policy in policies:
        rng = np.random.RandomState(SEED)

        # For ablation: replace feature vector with random noise
        run_records = records
        if name == "linucb_no_encoder":
            ablation_rng = np.random.RandomState(SEED + 1)
            run_records = [
                {**r, "feature_vector": ablation_rng.rand(FEATURE_DIM).tolist()}
                for r in records
            ]

        results = _run_policy(policy, run_records, rng)
        summary = aggregate(results)

        print(f"  {name:<22} quality={summary['avg_quality']:.4f}  "
              f"cost={summary['avg_cost_tokens']:.0f} tokens  "
              f"reward={summary['avg_reward']:.4f}")
        print(f"  {'':22} workflows: {summary['workflow_distribution']}")

        all_summaries.append({"policy": name, **{k: v for k, v in summary.items()
                                                  if k != "per_class" and k != "workflow_distribution"}})

        for tc, stats in summary["per_class"].items():
            per_class_rows.append({"policy": name, "task_class": tc, **stats})

    # Save CSVs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    main_csv = RESULTS_DIR / "main_results.csv"
    with open(main_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"\n[main_results] Saved → {main_csv}")

    per_class_csv = RESULTS_DIR / "main_results_per_class.csv"
    with open(per_class_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_class_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_class_rows)
    print(f"[main_results] Saved → {per_class_csv}")


if __name__ == "__main__":
    run()