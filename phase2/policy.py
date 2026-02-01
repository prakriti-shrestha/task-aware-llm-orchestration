import numpy as np
import random

class PolicyLearner:
    def __init__(self, workflows, feature_dim, epsilon=0.2):
        self.workflows = workflows
        self.epsilon = epsilon

        # one weight vector per workflow
        self.weights = {
            wf: np.zeros(feature_dim)
            for wf in workflows
        }

    def select_workflow(self, features):
        # exploration
        if random.random() < self.epsilon:
            return random.choice(list(self.workflows))

        # exploitation
        scores = {
            wf: np.dot(self.weights[wf], features)
            for wf in self.workflows
        }
        return max(scores, key=scores.get)

    def update(self, workflow, features, reward, lr=0.01):
        prediction = np.dot(self.weights[workflow], features)
        error = reward - prediction
        self.weights[workflow] += lr * error * features