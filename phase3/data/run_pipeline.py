"""
Run execution script — Phase 3, Run 001.

Executes 500 episodes using the unified task pipeline and the strict JSONL
logger. Because the ML routing models are separate components (not yet
implemented), this script uses deterministic mock outputs for:
  - workflow_id    (cycling W1 → W2 → W3)
  - feature_vector (fixed-length vector derived from task text length)
  - quality_score  (mock based on workflow)
  - reward         (mock based on quality_score)

Output: data/runs/phase3_run_001.jsonl
"""

import random
import hashlib 
import time
import sys
import os
from pathlib import Path

# Allow running from the phase3/ directory or the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pipeline import task_sampler
from data.logger import JSONLLogger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
N_EPISODES = 1500
BATCH_SIZE = 50
LAMBDA_VALUE = 0.5  # Fixed trade-off weight for this run
OUTPUT_FILE = Path(__file__).parent / "runs" / "phase3_run_001.jsonl"

WORKFLOWS = ["W1", "W2", "W3"]

# Mock quality scores per workflow (simulating W3 > W2 > W1 quality)
MOCK_QUALITY = {"W1": 0.60, "W2": 0.75, "W3": 0.90}

# Feature vector dimension
FEATURE_DIM = 16


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_feature_vector(task_text: str, task_id: str) -> list:
    """
    Deterministic mock feature vector seeded by task_id.
    The same task always produces the same vector regardless of call order.
    """
    seed = int(hashlib.md5(task_id.encode()).hexdigest(), 16) % (2 ** 32)
    task_rng = random.Random(seed)
    length_norm = min(len(task_text) / 500.0, 1.0)
    word_count_norm = min(len(task_text.split()) / 100.0, 1.0)
    base = [length_norm, word_count_norm] + [task_rng.random() for _ in range(FEATURE_DIM - 2)]
    return [round(v, 6) for v in base]


def _mock_workflow(episode_id: int) -> str:
    """Cycle through W1 → W2 → W3 deterministically."""
    return WORKFLOWS[episode_id % 3]


def _mock_quality_score(workflow_id: str, rng: random.Random) -> float:
    """Base quality per workflow with small Gaussian noise."""
    base = MOCK_QUALITY[workflow_id]
    noise = rng.gauss(0, 0.05)
    return round(max(0.0, min(1.0, base + noise)), 4)


def _mock_output_text(task: dict, workflow_id: str) -> str:
    return f"[Mock {workflow_id} output for task {task['task_id']}]"


def _mock_cost_tokens(rng: random.Random) -> int:
    return rng.randint(80, 800)


def _mock_latency_ms(rng: random.Random) -> int:
    return rng.randint(100, 3000)


def _compute_reward(quality_score: float, cost_tokens: int, lambda_value: float) -> float:
    """
    Simple reward: quality_score - lambda * normalised_cost.
    Cost is normalised to [0, 1] assuming max 1000 tokens.
    """
    cost_norm = min(cost_tokens / 1000.0, 1.0)
    reward = quality_score - lambda_value * cost_norm
    return round(reward, 6)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def run(n_episodes: int = N_EPISODES, output_file: Path = OUTPUT_FILE) -> None:
    print(f"[run_pipeline] Starting Phase 3 Run 001")
    print(f"  Episodes  : {n_episodes}")
    print(f"  Seed      : {SEED}")
    print(f"  Output    : {output_file}")

    rng = random.Random(SEED)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    episode_id = 0
    t_start = time.time()

    with JSONLLogger(output_file, mode="w") as logger:
        # Evaluate all 3 workflows per task
        n_tasks_needed = 500
        for batch in task_sampler(n=n_tasks_needed, batch_size=BATCH_SIZE, seed=SEED):
            for task in batch:
                for workflow_id in WORKFLOWS:
                    if episode_id >= n_episodes:
                        break

                    feature_vector = _mock_feature_vector(task["task_text"], task["task_id"])
                    quality_score = _mock_quality_score(workflow_id, rng)
                    cost_tokens = _mock_cost_tokens(rng)
                    latency_ms = _mock_latency_ms(rng)
                    reward = _compute_reward(quality_score, cost_tokens, LAMBDA_VALUE)

                    record = {
                        "task_id":       task["task_id"],
                        "task_text":     task["task_text"],
                        "task_class":    task["task_class"],
                        "feature_vector": feature_vector,
                        "workflow_id":   workflow_id,
                        "output_text":   _mock_output_text(task, workflow_id),
                        "quality_score": quality_score,
                        "cost_tokens":   cost_tokens,
                        "latency_ms":    latency_ms,
                        "reward":        reward,
                        "ground_truth":  task["ground_truth"],
                        "episode_id":    episode_id,
                        "lambda_value":  LAMBDA_VALUE,
                    }

                    logger.write(record)
                    episode_id += 1

                    if episode_id % 100 == 0:
                        elapsed = time.time() - t_start
                        print(f"  [{episode_id}/{n_episodes}] episodes written — {elapsed:.1f}s elapsed")

    elapsed = time.time() - t_start
    print(f"\n[run_pipeline] Done. {episode_id} episodes written to {output_file} in {elapsed:.2f}s")


if __name__ == "__main__":
    run()
