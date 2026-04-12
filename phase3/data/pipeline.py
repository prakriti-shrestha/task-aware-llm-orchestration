"""
Unified task pipeline.

Aggregates tasks from all four dataset loaders and yields deterministic
batches using a fixed random seed (seed=42).  The same seed always
produces the same task ordering, guaranteeing reproducible runs.
"""

import os
import random
from typing import Iterator, List, Dict, Any

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from data.loaders.gsm8k_loader import load_gsm8k
from data.loaders.humaneval_loader import load_humaneval
from data.loaders.triviaqa_loader import load_triviaqa
from data.loaders.arc_loader import load_arc

SEED = 42


def _collect_all_tasks(
    gsm8k_split: str = "train",
    humaneval_split: str = "test",
    triviaqa_split: str = "train",
    arc_split: str = "train",
    max_per_source: int = 200,
) -> List[Dict[str, Any]]:
    """
    Collect up to `max_per_source` tasks from each loader into a single list.
    Capped to avoid loading entire multi-million-row datasets into memory.
    """
    all_tasks: List[Dict[str, Any]] = []

    print(f"[pipeline] Loading GSM8K ({gsm8k_split})...")
    for i, task in enumerate(load_gsm8k(split=gsm8k_split)):
        all_tasks.append(task)
        if i + 1 >= max_per_source:
            break

    print(f"[pipeline] Loading HumanEval ({humaneval_split})...")
    for i, task in enumerate(load_humaneval(split=humaneval_split)):
        all_tasks.append(task)
        if i + 1 >= max_per_source:
            break

    print(f"[pipeline] Loading TriviaQA RC ({triviaqa_split})...")
    for i, task in enumerate(load_triviaqa(split=triviaqa_split)):
        all_tasks.append(task)
        if i + 1 >= max_per_source:
            break

    print(f"[pipeline] Loading ARC-Challenge ({arc_split})...")
    for i, task in enumerate(load_arc(split=arc_split)):
        all_tasks.append(task)
        if i + 1 >= max_per_source:
            break

    return all_tasks


def task_sampler(
    n: int = 500,
    batch_size: int = 50,
    gsm8k_split: str = "train",
    humaneval_split: str = "test",
    triviaqa_split: str = "train",
    arc_split: str = "train",
    max_per_source: int = 200,
    seed: int = SEED,
) -> Iterator[List[Dict[str, Any]]]:
    """
    Aggregate tasks from all four loaders, shuffle deterministically with
    `seed`, then yield batches of `batch_size` until `n` tasks are produced.

    Args:
        n:              Total number of tasks to sample across all batches.
        batch_size:     Tasks per yielded batch.
        *_split:        Dataset split name for each source.
        max_per_source: Maximum tasks loaded from each individual source.
        seed:           Random seed for deterministic shuffling (default 42).

    Yields:
        List[dict] of length <= batch_size, each item matching the schema:
        {task_id, task_text, task_class, ground_truth}
    """
    all_tasks = _collect_all_tasks(
        gsm8k_split=gsm8k_split,
        humaneval_split=humaneval_split,
        triviaqa_split=triviaqa_split,
        arc_split=arc_split,
        max_per_source=max_per_source,
    )

    rng = random.Random(seed)
    rng.shuffle(all_tasks)

    # Sample exactly n tasks (with wrap-around if pool is smaller than n)
    sampled: List[Dict[str, Any]] = []
    while len(sampled) < n:
        needed = n - len(sampled)
        sampled.extend(all_tasks[:needed])

    # Yield in batches
    for start in range(0, n, batch_size):
        yield sampled[start : start + batch_size]


if __name__ == "__main__":
    total = 0
    for batch in task_sampler(n=10, batch_size=5):
        for task in batch:
            print(task["task_id"], task["task_class"])
            total += 1
    print(f"\nTotal tasks sampled: {total}")
