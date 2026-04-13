"""
Baseline policies for workflow selection.

Four baselines required for the paper comparison table:
  - AlwaysW1Policy   : always selects W1 (cheap baseline)
  - AlwaysW3Policy   : always selects W3 (heavy baseline)
  - RandomPolicy     : selects uniformly at random
  - OraclePolicy     : retrospective best workflow per task (upper bound)

All classes share the same interface:
    select_workflow(feature_vector: np.ndarray) -> str
    update(feature_vector, workflow_id, reward)  -> None (no-op for static policies)
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
    """Always picks the cheap, fast workflow — lazy baseline."""
    name = "always_w1"

    def select_workflow(self, feature_vector: np.ndarray) -> str:
        return "W1"

    def update(self, feature_vector: np.ndarray, workflow_id: str, reward: float) -> None:
        pass


class AlwaysW3Policy:
    """Always picks the expensive, heavy workflow — wasteful baseline."""
    name = "always_w3"

    def select_workflow(self, feature_vector: np.ndarray) -> str:
        return "W3"

    def update(self, feature_vector: np.ndarray, workflow_id: str, reward: float) -> None:
        pass


class RandomPolicy:
    """Picks a workflow randomly each time — clueless baseline."""
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
    Falls back to the globally best workflow for unseen tasks.
    """
    name = "oracle"

    def __init__(self, log_path: str | Path = None):
        self._best: dict[str, str] = {}
        self._fallback: str = "W3"
        if log_path:
            self._load(Path(log_path))

    def _load(self, log_path: Path) -> None:
        if not log_path.exists():
            print(f"[OraclePolicy] Log not found: {log_path}. Using fallback W3.")
            return

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
            best_wf = max(
                wf_scores,
                key=lambda w: sum(wf_scores[w]) / len(wf_scores[w])
            )
            self._best[task_id] = best_wf

        # Global fallback: workflow with best average reward across all tasks
        global_means: dict[str, list] = defaultdict(list)
        for wf_scores in scores.values():
            for wf, rewards in wf_scores.items():
                global_means[wf].extend(rewards)
        self._fallback = max(
            global_means,
            key=lambda w: sum(global_means[w]) / len(global_means[w])
        )

        print(f"[OraclePolicy] Loaded {len(self._best)} tasks from {log_path.name}")
        print(f"[OraclePolicy] Fallback workflow: {self._fallback}")

    def select_workflow(self, feature_vector: np.ndarray, task_id: str = "") -> str:
        return self._best.get(task_id, self._fallback)

    def update(self, feature_vector: np.ndarray, workflow_id: str, reward: float) -> None:
        pass


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    log = PROJECT_ROOT / "phase3" / "data" / "runs" / "phase3_run_001.jsonl"
    print(f"Project root: {PROJECT_ROOT}")
    dummy_fv = np.zeros(16)

    p1 = AlwaysW1Policy()
    p3 = AlwaysW3Policy()
    rp = RandomPolicy()
    op = OraclePolicy(log)

    print(f"AlwaysW1 → {p1.select_workflow(dummy_fv)}")
    print(f"AlwaysW3 → {p3.select_workflow(dummy_fv)}")
    print(f"Random   → {rp.select_workflow(dummy_fv)}")
    print(f"Oracle   → {op.select_workflow(dummy_fv, task_id='gsm8k_train_00036')}")
    print(f"Oracle (unseen) → {op.select_workflow(dummy_fv, task_id='unknown_xyz')}")
    print("All baselines working!")