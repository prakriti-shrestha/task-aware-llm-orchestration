from tasks import TASKS
from workflows import run_workflow, WORKFLOWS
from features import extract_features
from policy import PolicyLearner
from evaluator import compute_reward

policy = PolicyLearner(
    workflows=list(WORKFLOWS.keys()),
    feature_dim=3
)

EPISODES = 200

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