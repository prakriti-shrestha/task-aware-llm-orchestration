"""
Evaluation metrics for the experiment runner.

Functions:
    exact_match(prediction, ground_truth) -> float
    compute_cost_usd(cost_tokens, workflow_id) -> float
    aggregate(results) -> dict
"""
from __future__ import annotations
import re
from typing import Optional


# Approximate cost per 1000 tokens (output tokens, using gpt-3.5-turbo pricing as proxy)
# W1 is one call, W2 is ~2 calls, W3 is ~3 calls
COST_PER_1K_TOKENS_USD = 0.002
WORKFLOW_CALL_MULTIPLIER = {"W1": 1, "W2": 2, "W3": 3}


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Normalised exact match.
    Extracts final number from GSM8K-style answers.
    Returns 1.0 if match, 0.0 otherwise.
    """
    def _norm(s: str) -> str:
        s = s.strip().lower()
        m = re.search(r"####\s*(.+)", s)
        if m:
            s = m.group(1).strip()
        s = re.sub(r"[.,;:!?]+$", "", s)
        nums = re.findall(r"-?\d+(?:\.\d+)?", s)
        return nums[-1] if nums else s

    return 1.0 if _norm(prediction) == _norm(ground_truth) else 0.0


def compute_cost_usd(cost_tokens: int, workflow_id: str) -> float:
    """Approximate USD cost for a single workflow execution."""
    multiplier = WORKFLOW_CALL_MULTIPLIER.get(workflow_id, 1)
    return round((cost_tokens / 1000) * COST_PER_1K_TOKENS_USD * multiplier, 6)


def aggregate(results: list[dict]) -> dict:
    """
    Aggregate a list of episode result dicts into summary statistics.

    Each dict must have: quality_score, cost_tokens, reward, workflow_id, task_class.

    Returns:
        {
          avg_quality, avg_cost_tokens, avg_cost_usd, avg_reward,
          workflow_distribution: {W1: %, W2: %, W3: %},
          per_class: {task_class: {avg_quality, avg_cost_tokens, count}}
        }
    """
    if not results:
        return {}

    n = len(results)
    avg_quality     = sum(r["quality_score"] for r in results) / n
    avg_cost_tokens = sum(r["cost_tokens"]   for r in results) / n
    avg_reward      = sum(r["reward"]        for r in results) / n
    avg_cost_usd    = sum(
        compute_cost_usd(r["cost_tokens"], r["workflow_id"]) for r in results
    ) / n

    # Workflow distribution
    wf_counts = {"W1": 0, "W2": 0, "W3": 0}
    for r in results:
        wf_counts[r["workflow_id"]] = wf_counts.get(r["workflow_id"], 0) + 1
    wf_dist = {w: round(c / n * 100, 1) for w, c in wf_counts.items()}

    # Per task class breakdown
    per_class: dict[str, dict] = {}
    for r in results:
        tc = r.get("task_class", "unknown")
        if tc not in per_class:
            per_class[tc] = {"quality_sum": 0.0, "cost_sum": 0, "count": 0}
        per_class[tc]["quality_sum"] += r["quality_score"]
        per_class[tc]["cost_sum"]    += r["cost_tokens"]
        per_class[tc]["count"]       += 1
    per_class_summary = {
        tc: {
            "avg_quality":     round(v["quality_sum"] / v["count"], 4),
            "avg_cost_tokens": round(v["cost_sum"]    / v["count"], 1),
            "count":           v["count"],
        }
        for tc, v in per_class.items()
    }

    return {
        "avg_quality":          round(avg_quality, 4),
        "avg_cost_tokens":      round(avg_cost_tokens, 1),
        "avg_cost_usd":         round(avg_cost_usd, 6),
        "avg_reward":           round(avg_reward, 4),
        "n_episodes":           n,
        "workflow_distribution": wf_dist,
        "per_class":            per_class_summary,
    }


if __name__ == "__main__":
    # Smoke test
    dummy = [
        {"quality_score": 0.9, "cost_tokens": 150, "reward": 0.82, "workflow_id": "W1", "task_class": "qa"},
        {"quality_score": 0.75, "cost_tokens": 350, "reward": 0.57, "workflow_id": "W2", "task_class": "reasoning"},
        {"quality_score": 0.88, "cost_tokens": 700, "reward": 0.53, "workflow_id": "W3", "task_class": "reasoning"},
        {"quality_score": 0.72, "cost_tokens": 120, "reward": 0.66, "workflow_id": "W1", "task_class": "qa"},
    ]
    result = aggregate(dummy)
    import json
    print(json.dumps(result, indent=2))