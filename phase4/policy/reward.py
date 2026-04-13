"""
Reward function for the contextual bandit.

reward = quality_score - lambda_value * normalised_cost

normalised_cost = cost_tokens / MAX_COST_TOKENS
MAX_COST_TOKENS = 1000 (approximate max for W3)

Lambda controls the cost-quality tradeoff:
  lambda=0.0 → maximize quality only (always W3)
  lambda=1.0 → maximize quality-cost equally (prefers W1 on easy tasks)
  lambda=0.5 → balanced (default)
"""
from __future__ import annotations

MAX_COST_TOKENS: int = 1000  # W3 rough ceiling — used for normalisation


def compute_reward(
    quality_score: float,
    cost_tokens: int,
    lambda_value: float = 0.5,
) -> float:
    """
    Compute reward for a single episode.

    Args:
        quality_score: float in [0, 1], higher is better.
        cost_tokens:   Integer token count consumed by the workflow.
        lambda_value:  Cost penalty weight. 0 = ignore cost, 1 = equal weight.

    Returns:
        float reward, typically in [-1, 1].
    """
    if not 0.0 <= quality_score <= 1.0:
        raise ValueError(f"quality_score must be in [0, 1], got {quality_score}")
    if cost_tokens < 0:
        raise ValueError(f"cost_tokens must be >= 0, got {cost_tokens}")
    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError(f"lambda_value must be in [0, 1], got {lambda_value}")

    cost_normalised = min(cost_tokens / MAX_COST_TOKENS, 1.0)
    reward = quality_score - lambda_value * cost_normalised
    return round(reward, 6)


def sweep_lambdas(
    quality_score: float,
    cost_tokens: int,
    lambdas: list[float] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
) -> dict[float, float]:
    """
    Compute rewards for multiple lambda values.
    Used by Member 2's phase5_pareto.py to generate the Pareto frontier.
    """
    return {lam: compute_reward(quality_score, cost_tokens, lam) for lam in lambdas}


if __name__ == "__main__":
    # Quick sanity check
    print("W1 result (cheap, ok quality):")
    print(f"  reward(q=0.60, cost=150, λ=0.5) = {compute_reward(0.60, 150, 0.5)}")
    print("W3 result (expensive, high quality):")
    print(f"  reward(q=0.90, cost=800, λ=0.5) = {compute_reward(0.90, 800, 0.5)}")
    print("W3 with λ=0.0 (cost ignored):")
    print(f"  reward(q=0.90, cost=800, λ=0.0) = {compute_reward(0.90, 800, 0.0)}")
    print("\nLambda sweep for W3:")
    for lam, r in sweep_lambdas(0.90, 800).items():
        print(f"  λ={lam}: reward={r:.4f}")