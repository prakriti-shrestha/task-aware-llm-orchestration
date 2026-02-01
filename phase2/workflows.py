# Hardcoded workflows 
# Workflows are the actions the policy chooses.

# Workflow      Strength        Cost
#  W1              weak         cheap
#  W2              medium       medium
#  W3              strong       expensive

import random

WORKFLOWS = {
    "W1": {"cost": 1},
    "W2": {"cost": 3},
    "W3": {"cost": 6},
}

def run_workflow(workflow_name, task):
    task_type = task["type"]

    # Base difficulty penalty by task type
    difficulty_penalty = {
        "qa": 0.0,
        "explanation": 0.1,
        "code": 0.2,
        "reasoning": 0.25,
        "design": 0.35
    }[task_type]

    if workflow_name == "W1":
        base = random.uniform(0.4, 0.7)
    elif workflow_name == "W2":
        base = random.uniform(0.6, 0.85)
    elif workflow_name == "W3":
        base = random.uniform(0.75, 0.95)

    quality = max(0.0, base - difficulty_penalty)

    return {
        "quality": quality,
        "cost": WORKFLOWS[workflow_name]["cost"]
    }