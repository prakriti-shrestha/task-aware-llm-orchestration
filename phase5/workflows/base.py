"""
Base class for all workflows.
Every workflow must implement run(task_text) -> (output_text, token_count).
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class BaseWorkflow(ABC):
    workflow_id: str = ""

    @abstractmethod
    def run(self, task_text: str) -> tuple[str, int]:
        """
        Run the workflow on a task.

        Returns:
            output_text: str — the LLM's final answer
            token_count: int — total tokens consumed across all LLM calls
        """
        ...