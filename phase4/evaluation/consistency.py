"""
Self-consistency scorer for W2 workflow.
Generates N outputs for the same task and measures agreement on the final answer.
Returns float 0.0-1.0 where 1.0 = all outputs agree.

This module works with any callable LLM function.
For Phase 4 it uses the real W2 workflow callable.
"""
from __future__ import annotations
import re
from collections import Counter
from typing import Callable, List
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _extract_final_answer(text: str) -> str:
    """
    Extract the final answer from an LLM output.
    Handles: "#### 42", "The answer is 42", "= 42", bare numbers.
    Returns lowercased, stripped string for comparison.
    """
    # GSM8K style: #### <answer>
    m = re.search(r"####\s*(.+)", text)
    if m:
        return m.group(1).strip().lower()

    # "the answer is X" / "answer: X"
    m = re.search(r"(?:the\s+)?answer\s+is\s*[:\-]?\s*(.+?)(?:\.|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()

    # Last number in the text (fallback for math)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]

    # Fallback: last non-empty line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1].lower() if lines else text.strip().lower()


def score(
    task_text: str,
    llm_fn: Callable[[str], str],
    n_samples: int = 3,
) -> float:
    """
    Generate n_samples outputs and measure answer agreement.

    Args:
        task_text: The input task text.
        llm_fn:    Callable that takes task_text and returns an output string.
                   This should be your W2 workflow function.
        n_samples: Number of independent samples to generate (default 3).

    Returns:
        float in [0.0, 1.0]
        - 1.0: all samples agree on the same answer
        - 0.0: all samples give different answers
    """
    outputs = [llm_fn(task_text) for _ in range(n_samples)]
    answers = [_extract_final_answer(out) for out in outputs]

    # Fraction of samples that agree with the majority answer
    most_common, count = Counter(answers).most_common(1)[0]
    consistency = count / n_samples
    return round(consistency, 4)


def score_from_outputs(outputs: List[str]) -> float:
    """
    Compute consistency from pre-generated outputs (avoids re-calling the LLM).
    Use this when you already have N outputs and just need the score.

    Args:
        outputs: List of LLM output strings.

    Returns:
        float in [0.0, 1.0]
    """
    if not outputs:
        return 0.0
    answers = [_extract_final_answer(out) for out in outputs]
    most_common, count = Counter(answers).most_common(1)[0]
    return round(count / len(outputs), 4)


if __name__ == "__main__":
    # Smoke test with a fake LLM
    import random
    rng = random.Random(42)

    def fake_llm_consistent(text):
        return "The answer is #### 42"

    def fake_llm_noisy(text):
        return f"The answer is #### {rng.randint(1, 10)}"

    print(f"All agree   (expect 1.0): {score('test', fake_llm_consistent, n_samples=5)}")
    print(f"All random  (expect low): {score('test', fake_llm_noisy,      n_samples=5)}")

    outputs_mixed = ["#### 42", "#### 42", "#### 7"]
    print(f"2/3 agree   (expect 0.67): {score_from_outputs(outputs_mixed)}")