"""
W1 — Single-pass workflow (cheap).
One direct LLM call. No chain-of-thought, no self-check.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase5.workflows.base import BaseWorkflow
from phase5.llm_clients import call_llm

PROMPT_TEMPLATE = """{task}

Answer directly and concisely. Give only the final answer."""


class W1BasicWorkflow(BaseWorkflow):
    workflow_id = "W1"

    def run(self, task_text: str) -> tuple[str, int]:
        prompt       = PROMPT_TEMPLATE.format(task=task_text)
        output, tokens = call_llm(prompt)
        return output, tokens


if __name__ == "__main__":
    wf  = W1BasicWorkflow()
    out, tok = wf.run("What is 12 × 15?")
    print(f"Output : {out}")
    print(f"Tokens : {tok}")