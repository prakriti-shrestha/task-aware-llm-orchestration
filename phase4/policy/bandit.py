"""
Contextual bandit policies for workflow selection.

Two implementations:
  - EpsilonGreedyBandit: original Phase 1/2 policy (kept for ablation)
  - LinUCBBandit:        Phase 4 upgrade — Linear Upper Confidence Bound

LinUCB maintains per-workflow linear models and explicit confidence intervals.
This produces regret curves (the bandit knows what it doesn't know) and
is more sample-efficient than epsilon-greedy.

Reference: Li et al. (2010) "A Contextual-Bandit Approach to Personalized
           News Article Recommendation", WWW 2010.
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional


WORKFLOW_IDS = ["W1", "W2", "W3"]


# ── Epsilon-Greedy (Phase 1/2 — keep for ablation) ───────────────────────────

class EpsilonGreedyBandit:
    """
    Simple epsilon-greedy contextual bandit.
    Maintains a running mean reward estimate per workflow.
    """

    def __init__(
        self,
        workflows: List[str] = WORKFLOW_IDS,
        epsilon: float = 0.15,
        seed: int = 42,
    ):
        self.workflows = workflows
        self.epsilon = epsilon
        self.rng = np.random.RandomState(seed)
        # Per-workflow: total reward and count
        self._totals = {w: 0.0 for w in workflows}
        self._counts = {w: 0 for w in workflows}

    def select_workflow(self, feature_vector: np.ndarray) -> str:
        """Select a workflow. feature_vector is accepted but not used in pure e-greedy."""
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.workflows)
        # Greedy: pick workflow with highest mean reward (break ties randomly)
        means = {w: (self._totals[w] / self._counts[w]) if self._counts[w] > 0 else 0.0
                 for w in self.workflows}
        best_val = max(means.values())
        best = [w for w, v in means.items() if v == best_val]
        return self.rng.choice(best)

    def update(self, feature_vector: np.ndarray, workflow_id: str, reward: float) -> None:
        self._totals[workflow_id] += reward
        self._counts[workflow_id] += 1

    @property
    def name(self) -> str:
        return "epsilon_greedy"


# ── LinUCB (Phase 4 — your main policy) ──────────────────────────────────────

class LinUCBBandit:
    """
    Linear Upper Confidence Bound contextual bandit.

    For each workflow k, maintains:
        A_k  — (d x d) feature covariance matrix, initialised to I
        b_k  — (d,)   reward accumulator, initialised to 0

    At each step:
        theta_k = A_k^{-1} b_k           (estimated reward weights)
        UCB_k   = theta_k.T @ x + alpha * sqrt(x.T @ A_k^{-1} @ x)

    The workflow with the highest UCB is selected (upper confidence bound
    encourages exploration of uncertain arms).

    After observing reward r for workflow k with context x:
        A_k += x @ x.T
        b_k += r * x

    Args:
        feature_dim: Dimensionality of the task feature vector.
        alpha:       Exploration coefficient. Higher = more exploration.
                     Start at 1.0, tune down to 0.5 if over-exploring.
        workflows:   List of workflow IDs.
        seed:        Random seed for tie-breaking.
    """

    def __init__(
        self,
        feature_dim: int = 16,
        alpha: float = 1.0,
        workflows: List[str] = WORKFLOW_IDS,
        seed: int = 42,
    ):
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.workflows = workflows
        self.rng = np.random.RandomState(seed)

        # Per-workflow matrices
        self._A: dict[str, np.ndarray] = {
            w: np.identity(feature_dim) for w in workflows
        }
        self._b: dict[str, np.ndarray] = {
            w: np.zeros(feature_dim) for w in workflows
        }
        # Track counts and cumulative reward for diagnostics
        self._counts = {w: 0 for w in workflows}
        self._total_reward = {w: 0.0 for w in workflows}

    def _ucb_score(self, workflow_id: str, x: np.ndarray) -> float:
        """Compute UCB score for a single workflow given context x."""
        A_inv = np.linalg.inv(self._A[workflow_id])
        theta = A_inv @ self._b[workflow_id]
        mean = theta @ x
        confidence = self.alpha * np.sqrt(x @ A_inv @ x)
        return float(mean + confidence)

    def select_workflow(self, feature_vector: np.ndarray) -> str:
        """
        Select the workflow with the highest UCB score.

        Args:
            feature_vector: np.ndarray of shape (feature_dim,)

        Returns:
            Workflow ID string: "W1", "W2", or "W3"
        """
        x = np.asarray(feature_vector, dtype=float)
        scores = {w: self._ucb_score(w, x) for w in self.workflows}
        best_val = max(scores.values())
        # Break ties randomly to avoid bias
        best = [w for w, s in scores.items() if s == best_val]
        return self.rng.choice(best)

    def update(self, feature_vector: np.ndarray, workflow_id: str, reward: float) -> None:
        """
        Update internal state after observing a reward.

        Args:
            feature_vector: The same feature vector used during select_workflow.
            workflow_id:    The workflow that was chosen.
            reward:         The observed reward (quality - lambda * cost).
        """
        x = np.asarray(feature_vector, dtype=float)
        self._A[workflow_id] += np.outer(x, x)
        self._b[workflow_id] += reward * x
        self._counts[workflow_id] += 1
        self._total_reward[workflow_id] += reward

    def get_theta(self, workflow_id: str) -> np.ndarray:
        """Return the current estimated reward weight vector for a workflow."""
        A_inv = np.linalg.inv(self._A[workflow_id])
        return A_inv @ self._b[workflow_id]

    def diagnostics(self) -> dict:
        """Return per-workflow selection counts and mean rewards."""
        return {
            w: {
                "count": self._counts[w],
                "mean_reward": (self._total_reward[w] / self._counts[w])
                if self._counts[w] > 0 else 0.0,
            }
            for w in self.workflows
        }

    def save(self, path: str) -> None:
        """Save policy state to a .npz file."""
        data = {"alpha": np.array([self.alpha]), "workflows": np.array(self.workflows)}
        for w in self.workflows:
            data[f"A_{w}"] = self._A[w]
            data[f"b_{w}"] = self._b[w]
            data[f"count_{w}"] = np.array([self._counts[w]])
        np.savez(path, **data)
        print(f"[bandit] Saved policy → {path}.npz")

    def load(self, path: str) -> "LinUCBBandit":
        """Load policy state from a .npz file."""
        data = np.load(path, allow_pickle=True)
        self.alpha = float(data["alpha"][0])
        for w in self.workflows:
            self._A[w] = data[f"A_{w}"]
            self._b[w] = data[f"b_{w}"]
            self._counts[w] = int(data[f"count_{w}"][0])
        print(f"[bandit] Loaded policy ← {path}")
        return self

    @property
    def name(self) -> str:
        return f"linucb_alpha{self.alpha}"


if __name__ == "__main__":
    import json
    from pathlib import Path
    import numpy as np

    log_path = Path(__file__).resolve().parent.parent.parent / "phase3/data/runs/phase3_bandit_train.jsonl"
    if not log_path.exists():
        print(f"Log not found: {log_path}")
        exit()

    records = []
    with open(log_path) as f:
        for line in f:
            records.append(json.loads(line.strip()))

    bandit = LinUCBBandit(feature_dim=16, alpha=1.0)

    # Instead of using the logged reward (which is from a random policy),
    # simulate the ORACLE reward for whatever workflow the bandit picks.
    # This is what Phase 5 will do with real LLM outputs.
    # For now: W3 gets high reward on reasoning/code, W1 gets high reward on QA.
    ORACLE_QUALITY = {
        ("W1", "qa"):        0.82,
        ("W1", "reasoning"): 0.52,
        ("W1", "code"):      0.48,
        ("W2", "qa"):        0.75,
        ("W2", "reasoning"): 0.73,
        ("W2", "code"):      0.70,
        ("W3", "qa"):        0.76,
        ("W3", "reasoning"): 0.91,
        ("W3", "code"):      0.88,
    }
    COST = {"W1": 150, "W2": 350, "W3": 700}
    rng = np.random.RandomState(42)

    print("Running LinUCB with oracle-quality simulation...")
    for i, rec in enumerate(records):
        x = np.array(rec["feature_vector"])
        task_class = rec["task_class"]
        chosen = bandit.select_workflow(x)

        base_q = ORACLE_QUALITY.get((chosen, task_class), 0.70)
        quality = float(np.clip(base_q + rng.normal(0, 0.04), 0, 1))
        cost_norm = COST[chosen] / 1000.0
        reward = round(quality - 0.5 * cost_norm, 6)

        bandit.update(x, chosen, reward)

        if (i + 1) % 100 == 0:
            diag = bandit.diagnostics()
            print(f"\nEpisode {i+1}:")
            for w, d in diag.items():
                print(f"  {w}: count={d['count']:3d}  mean_reward={d['mean_reward']:.4f}")

    print("\nFinal diagnostics:")
    for w, d in bandit.diagnostics().items():
        print(f"  {w}: count={d['count']:3d}  mean_reward={d['mean_reward']:.4f}")
    print("\nExpected: W1 count highest (QA tasks are common), W3 mean_reward highest")