"""
Phase 3 — Bandit training run.

Runs 500 episodes, one routing decision per task (random policy for now).
LinUCB in Phase 4 replaces the random selection.

Output: phase3/data/runs/phase3_bandit_train.jsonl
"""

import hashlib
import random
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pipeline import task_sampler
from data.logger import JSONLLogger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42
N_TASKS = 500
LAMBDA_VALUE = 0.5
FEATURE_DIM = 16
WORKFLOWS = ["W1", "W2", "W3"]
OUTPUT_FILE = Path(__file__).parent / "runs" / "phase3_bandit_train.jsonl"

# Realistic per-class quality: hard tasks benefit more from W3
MOCK_QUALITY_BASE = {
    "W1": {"qa": 0.72, "reasoning": 0.55, "code": 0.50, "explanation": 0.65},
    "W2": {"qa": 0.75, "reasoning": 0.72, "code": 0.68, "explanation": 0.72},
    "W3": {"qa": 0.76, "reasoning": 0.88, "code": 0.85, "explanation": 0.80},
}

COST_RANGES = {
    "W1": (80, 200),
    "W2": (200, 500),
    "W3": (500, 900),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feature_vector(task_text: str, task_id: str) -> list:
    """Deterministic per task_id — identical to run_pipeline.py."""
    seed = int(hashlib.md5(task_id.encode()).hexdigest(), 16) % (2 ** 32)
    task_rng = random.Random(seed)
    length_norm = min(len(task_text) / 500.0, 1.0)
    word_count_norm = min(len(task_text.split()) / 100.0, 1.0)
    base = [length_norm, word_count_norm] + [task_rng.random() for _ in range(FEATURE_DIM - 2)]
    return [round(v, 6) for v in base]


def _quality(workflow_id: str, task_class: str, rng: random.Random) -> float:
    base = MOCK_QUALITY_BASE[workflow_id].get(task_class, 0.70)
    return round(max(0.0, min(1.0, base + rng.gauss(0, 0.05))), 4)


def _cost(workflow_id: str, rng: random.Random) -> int:
    lo, hi = COST_RANGES[workflow_id]
    return rng.randint(lo, hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    rng = random.Random(SEED)
    episode_id = 0
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"[bandit_train] Starting — {N_TASKS} episodes, seed={SEED}")
    print(f"[bandit_train] Output: {OUTPUT_FILE}")
    t_start = time.time()

    with JSONLLogger(OUTPUT_FILE, mode="w") as logger:
        for batch in task_sampler(n=N_TASKS, batch_size=50, seed=SEED):
            for task in batch:
                # Random policy — LinUCB replaces this in Phase 4
                workflow_id = rng.choice(WORKFLOWS)

                feature_vector = _feature_vector(task["task_text"], task["task_id"])
                quality_score = _quality(workflow_id, task["task_class"], rng)
                cost_tokens = _cost(workflow_id, rng)
                latency_ms = rng.randint(100, 3000)
                cost_norm = min(cost_tokens / 1000.0, 1.0)
                reward = round(quality_score - LAMBDA_VALUE * cost_norm, 6)

                logger.write({
                    "task_id":        task["task_id"],
                    "task_text":      task["task_text"],
                    "task_class":     task["task_class"],
                    "feature_vector": feature_vector,
                    "workflow_id":    workflow_id,
                    "output_text":    f"[Mock {workflow_id} output for {task['task_id']}]",
                    "quality_score":  quality_score,
                    "cost_tokens":    cost_tokens,
                    "latency_ms":     latency_ms,
                    "reward":         reward,
                    "ground_truth":   task["ground_truth"],
                    "episode_id":     episode_id,
                    "lambda_value":   LAMBDA_VALUE,
                })
                episode_id += 1

    elapsed = time.time() - t_start
    print(f"[bandit_train] Done. {episode_id} episodes → {OUTPUT_FILE} in {elapsed:.2f}s")


if __name__ == "__main__":
    run()