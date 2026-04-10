"""
Loader for the GSM8K dataset (subset: main).
task_class = "reasoning"
"""

from datasets import load_dataset


def load_gsm8k(split: str = "train"):
    """
    Load GSM8K (main subset) and yield task dicts.

    Args:
        split: HuggingFace dataset split ('train' or 'test').

    Yields:
        dict with keys: task_id, task_text, task_class, ground_truth
    """
    dataset = load_dataset("gsm8k", "main", split=split, trust_remote_code=False, streaming=True)

    for idx, example in enumerate(dataset):
        yield {
            "task_id": f"gsm8k_{split}_{idx:05d}",
            "task_text": example["question"],
            "task_class": "reasoning",
            "ground_truth": example["answer"],
        }


if __name__ == "__main__":
    for i, task in enumerate(load_gsm8k()):
        print(task)
        if i >= 4:
            break
