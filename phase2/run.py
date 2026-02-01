import os
import torch
import numpy as np

from tasks import TASKS
from workflows import run_workflow, WORKFLOWS
from policy import PolicyLearner
from evaluator import compute_reward

from features.encoder import TaskEncoder
from features.predictors import TaskPropertyPredictor

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(
    BASE_DIR, "features", "checkpoints", "task_feature_model.pt"
)

encoder = TaskEncoder()

predictor = TaskPropertyPredictor(embed_dim=384)
predictor.load_state_dict(torch.load(MODEL_PATH))
predictor.eval()  # IMPORTANT: inference mode

# Feature dim = 3 → [reasoning_depth, ambiguity, error_risk]
policy = PolicyLearner(
    workflows=list(WORKFLOWS.keys()),
    feature_dim=3
)

EPISODES = 200

for episode in range(EPISODES):
    task = TASKS[episode % len(TASKS)]

    # Phase 2 feature extraction (learned, not heuristic)
    with torch.no_grad():
        embedding = encoder.encode(task["text"])
        features = predictor(
            torch.tensor(embedding).float()
        ).numpy()

    # Policy selects workflow
    workflow = policy.select_workflow(features)

    # Execute workflow
    result = run_workflow(workflow, task)

    # Compute reward
    reward = compute_reward(result["quality"], result["cost"])

    # Update policy
    policy.update(workflow, features, reward)

    # Logging
    print(
        f"Episode {episode:3d} | "
        f"Task {task['id']} | "
        f"Workflow {workflow} | "
        f"Reward {reward:.3f}"
    )

print("\nPhase 2 run complete.")