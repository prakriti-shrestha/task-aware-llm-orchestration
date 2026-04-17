"""
W3 — Critique-and-revise (heavy, highest quality).
Step 1: Generate initial answer.
Step 2: Critique the answer for errors.
Step 3: Revise based on the critique.
Three sequential LLM calls.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase5.workflows.base import BaseWorkflow
from phase5.llm_clients import call_llm

STEP1_PROMPT = """{task}

Think step by step and provide a detailed answer."""

STEP2_PROMPT = """You are a critical reviewer.

Original task: {task}

Proposed answer:
{answer}

Identify any errors, logical flaws, or missing steps in the answer above.
Be specific. If the answer is correct, say "No errors found."
Critique:"""

STEP3_PROMPT = """You are revising an answer based on a critique.

Original task: {task}

Original answer:
{answer}

Critique of the answer:
{critique}

Now write a corrected, improved final answer. On the last line write:
FINAL ANSWER: <your answer>"""


class W3HeavyWorkflow(BaseWorkflow):
    workflow_id = "W3"

    def run(self, task_text: str) -> tuple[str, int]:
        total_tokens = 0

        # Step 1 — Generate
        answer, t1 = call_llm(STEP1_PROMPT.format(task=task_text))
        total_tokens += t1

        # Step 2 — Critique
        critique, t2 = call_llm(
            STEP2_PROMPT.format(task=task_text, answer=answer)
        )
        total_tokens += t2

        # Step 3 — Revise
        final, t3 = call_llm(
            STEP3_PROMPT.format(task=task_text, answer=answer, critique=critique)
        )
        total_tokens += t3

        return final, total_tokens


if __name__ == "__main__":
    wf       = W3HeavyWorkflow()
    out, tok = wf.run("Prove that the sum of two odd numbers is always even.")
    print(f"Output : {out}")
    print(f"Tokens : {tok}")