"""
Baseline policies for workflow selection.

Four baselines required for the paper comparison table:
  - AlwaysW1Policy   : always selects W1 (cheap baseline)
  - AlwaysW3Policy   : always selects W3 (heavy baseline)
  - RandomPolicy     : selects uniformly at random
  - OraclePolicy     : retrospective best workflow per task (upper bound)

All classes share the same interface:
    select_workflow(feature_vector: np.ndarray) -> str
    update(feature_vector, workflow_id, reward)  -> None  (no-op for static policies)
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WORKFLOWS = ["W1", "W2", "W3"]


class AlwaysW1Policy:
    name = "always_w1"

    def select_workflow(self, feature_vector: np.ndarray) -> str:
        return "W1"

    def update(self, feature_vector: np.ndarray, workflow_id: str, reward: float) -> None:
        pass


class AlwaysW3Policy:
    name = "always_w3"

    def select_workflow(self, feature_vector: np.ndarray) -> str:
        return "W3"

    def update(self, feature_vector: np.ndarray, workflow_id: str, reward: float) -> None:
        pass


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def select_workflow(self, feature_vector: np.ndarray) -> str:
        return self.rng.choice(WORKFLOWS)

    def update(self, feature_vector: np.ndarray, workflow_id: str, reward: float) -> None:
        pass


class OraclePolicy:
    """
    Retrospective oracle — always picks the workflow that produced the
    highest reward for this task_id in the training log.

    At inference time, if a task_id has not been seen before, falls back
    to the workflow with the highest average reward across all seen tasks.
    """
    name = "oracle"

    def __init__(self, log_path: str | Path):
        self._best: dict[str, str] = {}       # task_id → best workflow_id
        self._fallback: str = "W3"
        self._load(Path(log_path))

    def _load(self, log_path: Path) -> None:
        if not log_path.exists():
            raise FileNotFoundError(f"[OraclePolicy] Log not found: {log_path}")

        # {task_id: {workflow_id: [rewards]}}
        scores: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                scores[rec["task_id"]][rec["workflow_id"]].append(rec["reward"])

        # Pick best workflow per task
        for task_id, wf_scores in scores.items():
            best_wf = max(wf_scores, key=lambda w: sum(wf_scores[w]) / len(wf_scores[w]))
            self._best[task_id] = best_wf

        # Global fallback: workflow with best average reward across all tasks
        global_means: dict[str, list] = defaultdict(list)
        for wf_scores in scores.values():
            for wf, rewards in wf_scores.items():
                global_means[wf].extend(rewards)
        self._fallback = max(global_means, key=lambda w: sum(global_means[w]) / len(global_means[w]))

        print(f"[OraclePolicy] Loaded {len(self._best)} task entries from {log_path.name}")
        print(f"[OraclePolicy] Fallback workflow: {self._fallback}")

    def select_workflow(self, feature_vector: np.ndarray, task_id: str = "") -> str:
        return self._best.get(task_id, self._fallback)

    def update(self, feature_vector: np.ndarray, workflow_id: str, reward: float) -> None:
        pass


if __name__ == "__main__":
    log = Path(__file__).resolve().parent.parent.parent / "phase3/data/runs/phase3_run_001.jsonl"
    dummy_fv = np.zeros(16)

    p1 = AlwaysW1Policy()
    p3 = AlwaysW3Policy()
    rp = RandomPolicy()
    op = OraclePolicy(log)

    print(f"\nAlwaysW1  → {p1.select_workflow(dummy_fv)}")
    print(f"AlwaysW3  → {p3.select_workflow(dummy_fv)}")
    print(f"Random    → {rp.select_workflow(dummy_fv)}")
    print(f"Oracle    → {op.select_workflow(dummy_fv, task_id='gsm8k_train_00036')}")
    print(f"Oracle (unseen task) → {op.select_workflow(dummy_fv, task_id='unknown_task_xyz')}")