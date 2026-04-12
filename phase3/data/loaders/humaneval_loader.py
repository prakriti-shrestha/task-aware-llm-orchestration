"""
Loader for the OpenAI HumanEval dataset.
task_class = "code"
ground_truth is set to None as the canonical check is via unit tests,
not a plain answer string.
"""

from datasets import load_dataset


def load_humaneval(split: str = "test"):
    """
    Load openai_humaneval and yield task dicts.

    Args:
        split: HuggingFace dataset split (only 'test' exists for HumanEval).

    Yields:
        dict with keys: task_id, task_text, task_class, ground_truth
    """
    dataset = load_dataset("openai_humaneval", split=split, trust_remote_code=False, streaming=True)

    for idx, example in enumerate(dataset):
        yield {
            "task_id": f"humaneval_{split}_{idx:05d}",
            "task_text": example["prompt"],
            "task_class": "code",
            "ground_truth": None,  # Evaluation via unit tests; no plain string answer
        }


if __name__ == "__main__":
    for i, task in enumerate(load_humaneval()):
        print(task)
        if i >= 4:
            break
