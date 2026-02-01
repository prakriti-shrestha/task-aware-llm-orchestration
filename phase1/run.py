from tasks import TASKS
from workflows import run_workflow, WORKFLOWS
from features import extract_features
from policy import PolicyLearner
from evaluator import compute_reward

import json
import os

os.makedirs("D:/projects/task-aware-llm-orchestration/phase2/data", exist_ok=True)

log_path = "D:/projects/task-aware-llm-orchestration/phase2/data/phase1_logs.jsonl"
log_file = open(log_path, "a")

policy = PolicyLearner(
    workflows=list(WORKFLOWS.keys()),
    feature_dim=3
)

EPISODES = 500  # use more data for Phase 2

for episode in range(EPISODES):
    task = TASKS[episode % len(TASKS)]
    features = extract_features(task["text"])

    workflow = policy.select_workflow(features)
    result = run_workflow(workflow, task)

    reward = compute_reward(result["quality"], result["cost"])
    policy.update(workflow, features, reward)

    print(
        f"Episode {episode:3d} | "
        f"Task {task['id']} | "
        f"Workflow {workflow} | "
        f"Reward {reward:.3f}"
    )

    log_entry = {
        "task_text": task["text"],
        "task_type": task["type"],
        "workflow": workflow,
        "quality": result["quality"],
        "cost": result["cost"],
        "reward": reward
    }

    log_file.write(json.dumps(log_entry) + "\n")

# Close file AFTER loop
log_file.close()

print(f"\nPhase-1 logging complete. Data written to {log_path}")