def compute_reward(quality, cost, lambda_cost=0.1):
    return quality - lambda_cost * cost