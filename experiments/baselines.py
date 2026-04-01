import numpy as np
import random
import json

class AlwaysW1Policy:
    """Always picks the cheap, fast workflow — lazy baseline."""
    def select_workflow(self, features):
        return "W1"

class AlwaysW3Policy:
    """Always picks the expensive, heavy workflow — wasteful baseline."""
    def select_workflow(self, features):
        return "W3"

class RandomPolicy:
    """Picks a workflow randomly each time — clueless baseline."""
    def select_workflow(self, features):
        return random.choice(["W1", "W2", "W3"])

class OraclePolicy:
    """
    Looks at past logs to always pick the best workflow per task.
    This is the theoretical ceiling — impossible to beat in practice.
    Needs Aryan's log file to work.
    """
    def __init__(self, log_path=None):
        self.best_workflow = {}
        if log_path:
            self._load_logs(log_path)

    def _load_logs(self, log_path):
        task_rewards = {}
        try:
            with open(log_path, "r") as f:
                for line in f:
                    record = json.loads(line.strip())
                    task_id = record["task_id"]
                    workflow = record["workflow_id"]
                    reward = record["reward"]

                    if task_id not in task_rewards:
                        task_rewards[task_id] = {}
                    task_rewards[task_id][workflow] = reward

            for task_id, workflows in task_rewards.items():
                self.best_workflow[task_id] = max(
                    workflows, key=workflows.get
                )
            print(f"Oracle loaded {len(self.best_workflow)} tasks from logs.")

        except FileNotFoundError:
            print(f"Log file not found. Oracle will default to W2.")

    def select_workflow(self, features, task_id=None):
        if task_id and task_id in self.best_workflow:
            return self.best_workflow[task_id]
        return "W2"  # sensible fallback