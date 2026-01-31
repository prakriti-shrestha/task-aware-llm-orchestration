import numpy as np

KEYWORDS = ["design", "explain", "solve", "write"]

def extract_features(task_text):
    text = task_text.lower()

    length = len(text)
    word_count = len(text.split())
    keyword_count = sum(1 for k in KEYWORDS if k in text)

    return np.array([
        length,
        word_count,
        keyword_count
    ], dtype=float)