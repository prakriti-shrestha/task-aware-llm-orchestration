"""
Loader for the TriviaQA dataset (subset: rc).
task_class = "qa"
ground_truth is the first accepted alias from the answer field.
"""

from datasets import load_dataset


def load_triviaqa(split: str = "train"):
    """
    Load trivia_qa (rc subset) and yield task dicts.

    Args:
        split: HuggingFace dataset split ('train', 'validation', or 'test').

    Yields:
        dict with keys: task_id, task_text, task_class, ground_truth
    """
    dataset = load_dataset("trivia_qa", "rc", split=split, trust_remote_code=False, streaming=True)

    for idx, example in enumerate(dataset):
        # 'answer' field contains {'value': str, 'aliases': [str], ...}
        answer_field = example.get("answer", {})
        ground_truth = answer_field.get("value", None)

        yield {
            "task_id": f"triviaqa_{split}_{idx:05d}",
            "task_text": example["question"],
            "task_class": "qa",
            "ground_truth": ground_truth,
        }


if __name__ == "__main__":
    for i, task in enumerate(load_triviaqa()):
        print(task)
        if i >= 4:
            break
