"""
Unified quality evaluator.
Blends external quality (exact match, LLM judge) with internal quality
(NLI contradiction score + self-consistency).

External quality is used when ground_truth is available.
Internal quality acts as fallback when ground_truth is None.

Blending formula:
    if ground_truth available:
        quality = 0.6 * external + 0.4 * internal_nli
    else:
        quality = internal_nli  (NLI only; consistency requires re-calling LLM)
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.contradiction import score as nli_score


# ── External quality helpers ─────────────────────────────────────────────────

def _exact_match(prediction: str, ground_truth: str) -> float:
    def _normalise(s: str) -> str:
        s = s.strip().lower()
        # Extract answer after #### (GSM8K format)
        m = re.search(r"####\s*(.+)", s)
        if m:
            s = m.group(1).strip()
        # Remove trailing punctuation
        s = re.sub(r"[.,;:!?]+$", "", s)
        # Extract last number if present (handles "2 + 2 = 4" → "4")
        numbers = re.findall(r"-?\d+(?:\.\d+)?", s)
        if numbers:
            s = numbers[-1]
        return s

    return 1.0 if _normalise(prediction) == _normalise(ground_truth) else 0.0


def _code_pass_rate(output_text: str, test_cases: Optional[list] = None) -> float:
    """
    Placeholder for code evaluation.
    In Phase 4 with mock outputs this returns 0.5.
    Replace with real execution in Phase 5.
    """
    if test_cases is None:
        # Cannot evaluate without test cases — return neutral score
        return 0.5
    passed = 0
    for test in test_cases:
        try:
            namespace = {}
            exec(output_text, namespace)
            result = eval(test["call"], namespace)
            if result == test["expected"]:
                passed += 1
        except Exception:
            pass
    return passed / len(test_cases) if test_cases else 0.5


# ── Main evaluate function ────────────────────────────────────────────────────

def evaluate(
    task_text: str,
    output_text: str,
    ground_truth: Optional[str] = None,
    task_class: str = "qa",
    use_nli: bool = True,
) -> float:
    """
    Compute a quality score for a single (task, output) pair.

    Args:
        task_text:    The original input task.
        output_text:  The LLM-generated output to evaluate.
        ground_truth: Reference answer string, or None if unavailable.
        task_class:   One of 'qa', 'reasoning', 'code', 'explanation'.
        use_nli:      Whether to compute NLI score (set False for speed in tests).

    Returns:
        float in [0.0, 1.0]
    """
    # Skip NLI for mock outputs — they contain no real content
    is_mock = output_text.startswith("[Mock")

    # ── Internal quality (NLI) ────────────────────────────────────────────
    if use_nli and not is_mock:
        internal = nli_score(task_text, output_text)
    else:
        internal = 0.5  # neutral placeholder for mock outputs

    # ── External quality ─────────────────────────────────────────────────
    if ground_truth is not None and not is_mock:
        if task_class in ("qa", "reasoning"):
            external = _exact_match(output_text, ground_truth)
        elif task_class == "code":
            external = _code_pass_rate(output_text)
        else:
            # explanation: fall back to NLI only
            external = internal
        # Blend
        quality = 0.6 * external + 0.4 * internal
    else:
        # No ground truth or mock output — use internal signal only
        quality = internal

    return round(quality, 4)


if __name__ == "__main__":
    # Smoke test
    q1 = evaluate("What is 2 + 2?", "2 + 2 = 4", ground_truth="4", task_class="reasoning", use_nli=False)
    q2 = evaluate("What is 2 + 2?", "2 + 2 = 7", ground_truth="4", task_class="reasoning", use_nli=False)
    q3 = evaluate("What is 2 + 2?", "[Mock W1 output for task gsm8k_00001]", ground_truth="4", task_class="reasoning")
    print(f"Correct answer  (expect ~0.6): {q1}")
    print(f"Wrong answer    (expect ~0.0): {q2}")
    print(f"Mock output     (expect 0.5 ): {q3}")