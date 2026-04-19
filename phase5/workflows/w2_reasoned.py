"""
W2 — Chain-of-thought + self-consistency (balanced).
Generates N=3 CoT answers and picks the majority answer.
"""
from __future__ import annotations
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase5.workflows.base import BaseWorkflow
from phase5.llm_clients import call_llm

N_SAMPLES = 3

COT_PROMPT = """{task}

Think step by step. Show your reasoning, then give the final answer on the 
last line in the format:
FINAL ANSWER: <your answer>"""


def _extract_final_answer(text: str) -> str:
    """Extract the answer after 'FINAL ANSWER:'."""
    m = re.search(r"FINAL ANSWER:\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: last non-empty line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else text.strip()


class W2ReasonedWorkflow(BaseWorkflow):
    workflow_id = "W2"

    def run(self, task_text: str) -> tuple[str, int]:
        prompt      = COT_PROMPT.format(task=task_text)
        total_tokens = 0
        outputs      = []

        for _ in range(N_SAMPLES):
            text, tokens = call_llm(prompt, temperature=0.7)
            outputs.append(text)
            total_tokens += tokens

        # Majority vote on extracted answers
        answers      = [_extract_final_answer(o) for o in outputs]
        best_answer, _ = Counter(answers).most_common(1)[0]

        # Return the full output that produced the majority answer
        for out in outputs:
            if _extract_final_answer(out) == best_answer:
                return out, total_tokens

        return outputs[0], total_tokens


if __name__ == "__main__":
    wf       = W2ReasonedWorkflow()
    out, tok = wf.run("If a train travels at 60 mph for 2.5 hours, how far does it travel?")
    print(f"Output : {out}")
    print(f"Tokens : {tok}")