"""
Phase 5 — Baseline policies.

Each policy has the same interface:
    select_workflow(feature_vector) -> str in {"W1", "W2", "W3"}
    update(feature_vector, workflow_id, reward) -> None  (no-op for baselines)

Policies:
    AlwaysW1Policy   — always pick W1 (cheap)
    AlwaysW3Policy   — always pick W3 (expensive)
    RandomPolicy     — uniform random
    OraclePolicy     — retroactive oracle: looks up which workflow actually
                       produced the best reward for each task_id in a log file

The OraclePolicy is special: it needs access to a log that contains every
task run under every workflow (an "all-arms" log).  To build that log, run
each task under all three workflows and record the reward — this is what
phase5_main_results.py does when --build-oracle is passed.
"""
from __future__ import annotations
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


class BasePolicy:
    name: str = "base"

    def select_workflow(self, feature_vector: np.ndarray) -> str:
        raise NotImplementedError

    def update(self, feature_vector: np.ndarray, workflow_id: str, reward: float) -> None:
        pass  # stateless by default


class AlwaysW1Policy(BasePolicy):
    name = "always_w1"

    def select_workflow(self, feature_vector: np.ndarray) -> str:
        return "W1"


class AlwaysW2Policy(BasePolicy):
    name = "always_w2"

    def select_workflow(self, feature_vector: np.ndarray) -> str:
        return "W2"


class AlwaysW3Policy(BasePolicy):
    name = "always_w3"

    def select_workflow(self, feature_vector: np.ndarray) -> str:
        return "W3"


class RandomPolicy(BasePolicy):
    name = "random"

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def select_workflow(self, feature_vector: np.ndarray) -> str:
        return self._rng.choice(["W1", "W2", "W3"])


class OraclePolicy(BasePolicy):
    """
    Retroactive oracle.

    Given a log file containing every (task_id, workflow, reward) triple,
    pick the workflow with the best reward for each task_id.

    If a task_id has never been seen under all three workflows, falls back
    to the best available option.
    """
    name = "oracle"

    def __init__(self, all_arms_log: str | Path):
        """
        Args:
            all_arms_log: JSONL file where each line has at least
                'task_id', 'workflow_id', and 'reward'.
        """
        self._best: dict[str, str] = {}
        self._build_table(all_arms_log)

    def _build_table(self, log_path: str | Path) -> None:
        # task_id -> workflow -> best reward seen
        best_rewards: dict[str, dict[str, float]] = defaultdict(dict)

        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tid = rec["task_id"]
                wf = rec["workflow_id"]
                r = float(rec["reward"])
                if wf not in best_rewards[tid] or r > best_rewards[tid][wf]:
                    best_rewards[tid][wf] = r

        for tid, rewards in best_rewards.items():
            # Argmax over whichever workflows we've seen for this task
            self._best[tid] = max(rewards, key=rewards.get)

        print(f"[OraclePolicy] Loaded {len(self._best)} oracle decisions from {log_path}")

    # The oracle needs the task_id, not just the feature vector.  We add a
    # convenience method and keep select_workflow() for interface compatibility.
    _current_task_id: str | None = None

    def set_task_id(self, task_id: str) -> None:
        self._current_task_id = task_id

    def select_workflow(self, feature_vector: np.ndarray) -> str:
        if self._current_task_id is None:
            raise RuntimeError(
                "OraclePolicy.set_task_id(task_id) must be called before select_workflow"
            )
        choice = self._best.get(self._current_task_id, "W2")  # default to W2
        self._current_task_id = None  # consume
        return choice


# ── Helper: uniform wrapper so the runner can treat all policies the same ────
class NoEncoderWrapper(BasePolicy):
    """
    Wraps any policy but replaces the real feature vector with random noise.
    Used for the 'no encoder' ablation.
    """
    name = "no_encoder"

    def __init__(self, inner: BasePolicy, feature_dim: int, seed: int = 42):
        self.inner = inner
        self.feature_dim = feature_dim
        self._rng = np.random.RandomState(seed)

    def _noise(self) -> np.ndarray:
        return self._rng.rand(self.feature_dim)

    def select_workflow(self, feature_vector: np.ndarray) -> str:
        return self.inner.select_workflow(self._noise())

    def update(self, feature_vector: np.ndarray, workflow_id: str, reward: float) -> None:
        self.inner.update(self._noise(), workflow_id, reward)