"""
Real quality scorer for Phase 5.
Replaces the mock _get_quality() function from Phase 4 experiments.

Handles all four task classes:
  qa         — exact match against ground truth
  reasoning  — exact match on final numeric answer (GSM8K format)
  code       — execution-based pass rate against HumanEval test cases
  explanation — NLI-based score (no ground truth available)
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase4.evaluation.evaluator import evaluate as nli_evaluate


def _normalise(s: str) -> str:
    s = s.strip().lower()
    m = re.search(r"final answer:\s*(.+)", s, re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    m = re.search(r"####\s*(.+)", s)
    if m:
        s = m.group(1).strip()
    s = re.sub(r"[.,;:!?]+$", "", s)
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else s


def _exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if _normalise(prediction) == _normalise(ground_truth) else 0.0


def _code_pass_rate(output_text: str, test_code: Optional[str]) -> float:
    """
    Execute the generated code against HumanEval test cases.
    test_code is the 'test' field from HumanEval — a string of assert statements.
    """
    if not test_code:
        return 0.5  # no tests available

    # Extract code block if wrapped in markdown
    code_match = re.search(r"```python\n(.*?)```", output_text, re.DOTALL)
    code = code_match.group(1) if code_match else output_text

    try:
        namespace: dict = {}
        exec(code, namespace)        # define the function
        exec(test_code, namespace)   # run the assert statements
        return 1.0                   # all assertions passed
    except AssertionError:
        return 0.0
    except Exception:
        return 0.0


def score(
    task_text: str,
    output_text: str,
    ground_truth: Optional[str],
    task_class: str,
    test_code: Optional[str] = None,
) -> float:
    """
    Compute real quality score for a (task, output) pair.

    Args:
        task_text:    Original task prompt.
        output_text:  LLM-generated answer.
        ground_truth: Reference answer string, or None.
        task_class:   One of 'qa', 'reasoning', 'code', 'explanation'.
        test_code:    HumanEval test string for code tasks, or None.

    Returns:
        float in [0.0, 1.0]
    """
    if task_class == "code":
        return _code_pass_rate(output_text, test_code)

    if ground_truth is not None and task_class in ("qa", "reasoning"):
        external = _exact_match(output_text, ground_truth)
        internal = nli_evaluate(task_text, output_text, use_nli=True)
        return round(0.6 * external + 0.4 * internal, 4)

    # explanation or no ground truth — NLI only
    return nli_evaluate(task_text, output_text, use_nli=True)