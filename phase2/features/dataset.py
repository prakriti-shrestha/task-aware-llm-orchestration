import json
import numpy as np
from collections import defaultdict

def load_phase1_data(path):
    data = defaultdict(list)

    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            data[entry["task_text"]].append(entry)

    return data

def compute_labels(task_runs):
    qualities = [r["quality"] for r in task_runs]

    reasoning_depth = np.mean(qualities)  # proxy
    error_risk = np.std(qualities)
    ambiguity = np.var(qualities)

    return np.array([reasoning_depth, ambiguity, error_risk])