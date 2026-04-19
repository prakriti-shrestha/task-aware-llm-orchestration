"""
Phase 5 — Real LLM training run.

Replaces mock outputs with real LLM calls through W1, W2, W3 workflows.
Runs the LinUCB bandit in online mode: one routing decision per task,
real quality evaluation, real reward, real policy update.

Outputs:
    phase5/data/runs/phase5_run_001.jsonl  — full episode log
    phase5/data/runs/phase5_bandit.pt      — trained bandit checkpoint

Usage:
    python phase5/run_phase5.py

    # Dry run (5 tasks, no API calls, confirms pipeline works):
    python phase5/run_phase5.py --dry-run

    # Specific lambda:
    python phase5/run_phase5.py --lambda-value 0.3

    # Limit tasks (for testing):
    python phase5/run_phase5.py --n-tasks 20
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "phase3"))

from phase3.data.pipeline import task_sampler
from phase3.data.logger import JSONLLogger
from phase4.policy.bandit import LinUCBBandit
from phase4.policy.reward import compute_reward
from phase5.workflows.w1_basic import W1BasicWorkflow
from phase5.workflows.w2_reasoned import W2ReasonedWorkflow
from phase5.workflows.w3_heavy import W3HeavyWorkflow
from phase5.quality import score as quality_score

# ── Config ────────────────────────────────────────────────────────────────────
N_TASKS      = 500
LAMBDA_VALUE = 0.5
FEATURE_DIM  = 16
SEED         = 42
OUTPUT_DIR   = Path(__file__).resolve().parent / "data" / "runs"
OUTPUT_FILE  = OUTPUT_DIR / "phase5_run_001.jsonl"
BANDIT_CKPT  = OUTPUT_DIR / "phase5_bandit.pt"

WORKFLOWS = {
    "W1": W1BasicWorkflow(),
    "W2": W2ReasonedWorkflow(),
    "W3": W3HeavyWorkflow(),
}


# ── Feature vector (same deterministic function as Phase 3) ──────────────────
def _feature_vector(task_text: str, task_id: str) -> list[float]:
    seed = int(hashlib.md5(task_id.encode()).hexdigest(), 16) % (2 ** 32)
    rng  = random.Random(seed)
    length_norm    = min(len(task_text) / 500.0, 1.0)
    word_count_norm = min(len(task_text.split()) / 100.0, 1.0)
    base = [length_norm, word_count_norm] + [rng.random() for _ in range(FEATURE_DIM - 2)]
    return [round(v, 6) for v in base]


# ── Dry run mock ──────────────────────────────────────────────────────────────
def _mock_run(workflow_id: str, task_text: str) -> tuple[str, int]:
    """Used in --dry-run mode only. No API calls."""
    costs = {"W1": 150, "W2": 350, "W3": 700}
    return f"[DRY RUN {workflow_id}] {task_text[:50]}", costs[workflow_id]


# ── Main ──────────────────────────────────────────────────────────────────────
def run(n_tasks: int, lambda_value: float, dry_run: bool) -> None:
    print(f"[phase5] Starting Phase 5 run")
    print(f"  Tasks       : {n_tasks}")
    print(f"  Lambda      : {lambda_value}")
    print(f"  Dry run     : {dry_run}")
    print(f"  Output      : {OUTPUT_FILE}\n")

    rng    = np.random.RandomState(SEED)
    bandit = LinUCBBandit(feature_dim=FEATURE_DIM, alpha=1.0, seed=SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    episode_id = 0
    t_start    = time.time()

    with JSONLLogger(OUTPUT_FILE, mode="w") as logger:
        for batch in task_sampler(n=n_tasks, batch_size=50, seed=SEED):
            for task in batch:
                task_id    = task["task_id"]
                task_text  = task["task_text"]
                task_class = task["task_class"]
                ground_truth = task.get("ground_truth")

                # ── Encode task ───────────────────────────────────────────
                fv = np.array(_feature_vector(task_text, task_id))

                # ── Policy selects workflow ───────────────────────────────
                chosen = bandit.select_workflow(fv)

                # ── Run workflow ──────────────────────────────────────────
                t0 = time.time()
                if dry_run:
                    output_text, cost_tokens = _mock_run(chosen, task_text)
                    quality = 0.5
                else:
                    try:
                        output_text, cost_tokens = WORKFLOWS[chosen].run(task_text)
                        quality = quality_score(
                            task_text, output_text, ground_truth, task_class
                        )
                    except Exception as e:
                        print(f"  [episode {episode_id}] ERROR: {e}")
                        output_text  = f"[ERROR: {e}]"
                        cost_tokens  = 0
                        quality      = 0.0

                latency_ms = int((time.time() - t0) * 1000)

                # ── Compute reward and update bandit ──────────────────────
                reward = compute_reward(quality, cost_tokens, lambda_value)
                bandit.update(fv, chosen, reward)

                # ── Log ───────────────────────────────────────────────────
                logger.write({
                    "task_id":        task_id,
                    "task_text":      task_text,
                    "task_class":     task_class,
                    "feature_vector": fv.tolist(),
                    "workflow_id":    chosen,
                    "output_text":    output_text,
                    "quality_score":  float(quality),
                    "cost_tokens":    int(cost_tokens),
                    "latency_ms":     latency_ms,
                    "reward":         float(reward),
                    "ground_truth":   ground_truth,
                    "episode_id":     episode_id,
                    "lambda_value":   lambda_value,
                })

                episode_id += 1

                if episode_id % 50 == 0:
                    elapsed = time.time() - t_start
                    diag    = bandit.diagnostics()
                    print(f"  [{episode_id}/{n_tasks}] {elapsed:.0f}s — "
                          f"W1:{diag['W1']['count']} "
                          f"W2:{diag['W2']['count']} "
                          f"W3:{diag['W3']['count']}")

    # Save bandit checkpoint
    bandit.save(str(BANDIT_CKPT).replace(".pt", ""))

    elapsed = time.time() - t_start
    print(f"\n[phase5] Done. {episode_id} episodes in {elapsed:.1f}s")
    print(f"[phase5] Log     → {OUTPUT_FILE}")
    print(f"[phase5] Bandit  → {BANDIT_CKPT}")

    # Print final diagnostics
    print("\nFinal bandit diagnostics:")
    for wf, d in bandit.diagnostics().items():
        print(f"  {wf}: count={d['count']:3d}  mean_reward={d['mean_reward']:.4f}")

    print("\nExpected with real data:")
    print("  W1 selected most for QA tasks  (triviaqa)")
    print("  W3 selected most for reasoning (gsm8k, arc)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks",      type=int,   default=N_TASKS)
    parser.add_argument("--lambda-value", type=float, default=LAMBDA_VALUE)
    parser.add_argument("--dry-run",      action="store_true")
    args = parser.parse_args()

    run(args.n_tasks, args.lambda_value, args.dry_run)