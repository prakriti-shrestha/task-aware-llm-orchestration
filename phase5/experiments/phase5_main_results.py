"""
Phase 5 — Experiment 1: Main Results Table (Table 1 of paper)

Runs 6 policies on the same set of tasks (200 per class by default, but
you can reduce for free-tier testing).  Produces:
    results/main_results.csv       — rows: policy × task_class
    data/runs/phase5_all_arms.jsonl — side-effect: every task × every workflow
                                      cached here.  Needed by OraclePolicy.

Because the quality scorer uses the LLM cache, running the same task through
the same workflow only costs API calls the FIRST time.  That's how the 6-way
comparison stays affordable.

Usage:
    # Tiny smoke test (9 tasks total, ~30 API calls):
    python -m phase5.experiments.phase5_main_results --n-per-class 3 --dry-run

    # Real run (free tier friendly, ~20 tasks per class, ~150 unique calls):
    python -m phase5.experiments.phase5_main_results --n-per-class 20

    # Full paper run (requires paid tier or multi-day scheduling):
    python -m phase5.experiments.phase5_main_results --n-per-class 200
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase3"))  # Phase 3 uses bare `from data.X import ...`

from phase3.data.pipeline import task_sampler
from phase4.policy.bandit import LinUCBBandit
from phase4.policy.reward import compute_reward
from phase5.experiments._shared import (
    FEATURE_DIM, RESULTS_DIR, RUNS_DIR,
    feature_vector, run_task, ensure_dirs, write_jsonl,
)
from phase5.experiments.baselines import (
    AlwaysW1Policy, AlwaysW3Policy, RandomPolicy,
    OraclePolicy, NoEncoderWrapper,
)

ALL_ARMS_LOG = RUNS_DIR / "phase5_all_arms.jsonl"
OUTPUT_CSV = RESULTS_DIR / "main_results.csv"

TASK_CLASSES = ["qa", "reasoning", "code", "explanation"]


def sample_tasks_by_class(n_per_class: int, seed: int) -> dict[str, list[dict]]:
    """Draw n_per_class tasks for each of the four task classes."""
    by_class: dict[str, list[dict]] = defaultdict(list)
    # Oversample, then filter — pipeline gives mixed classes
    total_needed = n_per_class * len(TASK_CLASSES) * 4
    seen_ids: set = set()

    for batch in task_sampler(n=total_needed, batch_size=50, seed=seed):
        for task in batch:
            tc = task.get("task_class", "qa")
            if tc in TASK_CLASSES and task["task_id"] not in seen_ids:
                if len(by_class[tc]) < n_per_class:
                    by_class[tc].append(task)
                    seen_ids.add(task["task_id"])
        if all(len(by_class[tc]) >= n_per_class for tc in TASK_CLASSES):
            break

    # Report what we actually got
    for tc in TASK_CLASSES:
        print(f"  {tc:12s} : {len(by_class[tc])} tasks")
    return dict(by_class)


def build_all_arms_log(tasks_by_class: dict[str, list[dict]],
                       lambda_value: float,
                       dry_run: bool) -> None:
    """
    Run every task under every workflow and log the results.

    Writes to data/runs/phase5_all_arms.jsonl (overwrites).  This file is used
    by the Oracle policy and by downstream experiments.
    """
    records: list[dict] = []
    total_tasks = sum(len(v) for v in tasks_by_class.values())
    call_count = 0
    total_calls = total_tasks * 3

    print(f"\n[all_arms] Running {total_tasks} tasks × 3 workflows = {total_calls} executions")
    if dry_run:
        print("[all_arms] DRY RUN — mock results only")

    for task_class, tasks in tasks_by_class.items():
        for task in tasks:
            for wf_id in ["W1", "W2", "W3"]:
                call_count += 1
                if dry_run:
                    # Mock: fake quality, realistic costs
                    mock_q = {"W1": 0.4, "W2": 0.65, "W3": 0.85}[wf_id]
                    mock_c = {"W1": 150, "W2": 450, "W3": 750}[wf_id]
                    result = {"output_text": f"[mock {wf_id}]",
                              "cost_tokens": mock_c, "latency_ms": 10,
                              "quality_score": mock_q, "error": None}
                else:
                    result = run_task(task, wf_id)

                reward = compute_reward(
                    result["quality_score"], result["cost_tokens"], lambda_value
                )

                records.append({
                    "task_id": task["task_id"],
                    "task_text": task["task_text"],
                    "task_class": task_class,
                    "ground_truth": task.get("ground_truth"),
                    "workflow_id": wf_id,
                    "output_text": result["output_text"],
                    "quality_score": result["quality_score"],
                    "cost_tokens": result["cost_tokens"],
                    "latency_ms": result["latency_ms"],
                    "reward": reward,
                    "lambda_value": lambda_value,
                    "error": result["error"],
                })

                if call_count % 10 == 0:
                    print(f"  [{call_count}/{total_calls}] last: {task_class}/{wf_id} "
                          f"q={result['quality_score']:.2f} c={result['cost_tokens']}")

    write_jsonl(ALL_ARMS_LOG, records)
    print(f"[all_arms] Wrote {len(records)} records → {ALL_ARMS_LOG}\n")


def lookup_reward(all_arms: list[dict], task_id: str, workflow_id: str) -> dict | None:
    """Find the record for a specific (task, workflow) pair."""
    for rec in all_arms:
        if rec["task_id"] == task_id and rec["workflow_id"] == workflow_id:
            return rec
    return None


def evaluate_policy(policy, policy_name: str,
                    tasks_by_class: dict[str, list[dict]],
                    all_arms: list[dict]) -> list[dict]:
    """
    Evaluate a policy by asking it which workflow to pick for each task,
    then looking up the (quality, cost, reward) from the all_arms log.
    """
    results: list[dict] = []

    for task_class, tasks in tasks_by_class.items():
        for task in tasks:
            fv = feature_vector(task["task_text"], task["task_id"])

            # Oracle needs the task_id
            if isinstance(policy, OraclePolicy):
                policy.set_task_id(task["task_id"])

            chosen_wf = policy.select_workflow(fv)
            rec = lookup_reward(all_arms, task["task_id"], chosen_wf)

            if rec is None:
                # Shouldn't happen if all_arms is complete
                continue

            # If the policy is a learner, let it update from this reward
            policy.update(fv, chosen_wf, rec["reward"])

            results.append({
                "policy": policy_name,
                "task_id": task["task_id"],
                "task_class": task_class,
                "workflow_id": chosen_wf,
                "quality_score": rec["quality_score"],
                "cost_tokens": rec["cost_tokens"],
                "reward": rec["reward"],
            })

    return results


def train_linucb(all_arms: list[dict], seed: int = 42) -> LinUCBBandit:
    """Train a LinUCB bandit online over the all_arms log."""
    bandit = LinUCBBandit(feature_dim=FEATURE_DIM, alpha=1.0, seed=seed)

    # Group records by task_id so we can simulate: present task, pick arm,
    # then only reward the picked arm (not all of them).
    from collections import defaultdict
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    task_order: list[str] = []
    for rec in all_arms:
        tid = rec["task_id"]
        if tid not in by_task:
            task_order.append(tid)
        by_task[tid][rec["workflow_id"]] = rec

    for tid in task_order:
        task_recs = by_task[tid]
        # Any record has the task_text (they all do)
        any_rec = next(iter(task_recs.values()))
        fv = feature_vector(any_rec["task_text"], tid)
        chosen = bandit.select_workflow(fv)
        if chosen in task_recs:
            bandit.update(fv, chosen, task_recs[chosen]["reward"])

    return bandit


def summarise(results: list[dict]) -> list[dict]:
    """Average quality / cost / reward per (policy, task_class)."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["policy"], r["task_class"])].append(r)

    rows = []
    for (policy, tc), items in grouped.items():
        rows.append({
            "policy": policy,
            "task_class": tc,
            "n": len(items),
            "avg_quality": round(np.mean([x["quality_score"] for x in items]), 4),
            "avg_cost": round(np.mean([x["cost_tokens"] for x in items]), 1),
            "avg_reward": round(np.mean([x["reward"] for x in items]), 4),
        })
    rows.sort(key=lambda r: (r["policy"], r["task_class"]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=20)
    parser.add_argument("--lambda-value", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-all-arms", action="store_true",
                        help="Reuse existing all_arms log instead of re-running")
    args = parser.parse_args()

    ensure_dirs()

    print(f"[main_results] n_per_class={args.n_per_class}, lambda={args.lambda_value}")
    print(f"[main_results] Sampling tasks…")
    tasks_by_class = sample_tasks_by_class(args.n_per_class, args.seed)

    if args.skip_all_arms and ALL_ARMS_LOG.exists():
        print(f"[main_results] Reusing existing {ALL_ARMS_LOG}")
        from phase5.experiments._shared import load_run_log
        all_arms = load_run_log(ALL_ARMS_LOG)
    else:
        build_all_arms_log(tasks_by_class, args.lambda_value, args.dry_run)
        from phase5.experiments._shared import load_run_log
        all_arms = load_run_log(ALL_ARMS_LOG)

    # Build and evaluate every policy
    print("[main_results] Evaluating policies…")
    all_results: list[dict] = []

    policies = [
        ("always_w1", AlwaysW1Policy()),
        ("always_w3", AlwaysW3Policy()),
        ("random",    RandomPolicy(seed=args.seed)),
        ("oracle",    OraclePolicy(ALL_ARMS_LOG)),
        ("linucb",    train_linucb(all_arms, seed=args.seed)),
    ]

    # "No-NLI" ablation: approximated by running LinUCB with noisy rewards
    # (the NLI signal is embedded in quality_score; we stub it here).
    # A stricter ablation lives in phase5_ablations.py.

    for name, policy in policies:
        print(f"  → {name}")
        all_results.extend(evaluate_policy(policy, name, tasks_by_class, all_arms))

    # Summarise and write CSV
    summary = summarise(all_results)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["policy", "task_class", "n",
                                                "avg_quality", "avg_cost", "avg_reward"])
        writer.writeheader()
        writer.writerows(summary)

    # Print pretty table
    print(f"\n[main_results] Results → {OUTPUT_CSV}\n")
    print(f"{'policy':<12} {'class':<12} {'n':>4} {'quality':>8} {'cost':>8} {'reward':>8}")
    print("-" * 60)
    for row in summary:
        print(f"{row['policy']:<12} {row['task_class']:<12} {row['n']:>4} "
              f"{row['avg_quality']:>8.3f} {row['avg_cost']:>8.0f} {row['avg_reward']:>8.3f}")


if __name__ == "__main__":
    main()