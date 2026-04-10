"""
Loader for the AI2 ARC dataset (subset: ARC-Challenge).
task_class = "reasoning"
ground_truth is the text of the correct answer choice.
"""

from datasets import load_dataset


def load_arc(split: str = "train"):
    """
    Load ai2_arc (ARC-Challenge subset) and yield task dicts.

    Args:
        split: HuggingFace dataset split ('train', 'validation', or 'test').

    Yields:
        dict with keys: task_id, task_text, task_class, ground_truth
    """
    dataset = load_dataset("ai2_arc", "ARC-Challenge", split=split, trust_remote_code=False, streaming=True)

    for idx, example in enumerate(dataset):
        # answerKey is a letter like 'A', 'B', 'C', 'D'
        answer_key = example.get("answerKey", "")

        # choices = {'label': ['A','B','C','D'], 'text': ['...','...','...','...']}
        choices = example.get("choices", {})
        labels = choices.get("label", [])
        texts = choices.get("text", [])

        ground_truth = None
        if answer_key in labels:
            ground_truth = texts[labels.index(answer_key)]

        yield {
            "task_id": f"arc_{split}_{idx:05d}",
            "task_text": example["question"],
            "task_class": "reasoning",
            "ground_truth": ground_truth,
        }


if __name__ == "__main__":
    for i, task in enumerate(load_arc()):
        print(task)
        if i >= 4:
            break
