"""
Strict schema-validated JSONL logger.

Writes one JSON object per line (UTF-8, ensure_ascii=False).
Raises ValueError immediately if any required field is missing or has
the wrong type.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, IO, Optional, Union

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: Dict[str, type] = {
    "task_id":       str,
    "task_text":     str,
    "task_class":    str,
    "feature_vector": list,
    "workflow_id":   str,
    "output_text":   str,
    "quality_score": float,
    "cost_tokens":   int,
    "latency_ms":    int,
    "reward":        float,
    "ground_truth":  (str, type(None)),   # str or null
    "episode_id":    int,
    "lambda_value":  float,
}

VALID_TASK_CLASSES = {"qa", "reasoning", "code", "explanation"}
VALID_WORKFLOW_IDS = {"W1", "W2", "W3"}


def _validate_record(record: Dict[str, Any]) -> None:
    """
    Validate a single record against the required schema.

    Raises:
        ValueError: if any required field is missing or has an invalid value.
    """
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in record:
            raise ValueError(
                f"[logger] Missing required field: '{field}'. "
                f"Record: {list(record.keys())}"
            )
        value = record[field]
        if not isinstance(value, expected_type):
            raise ValueError(
                f"[logger] Field '{field}' has wrong type. "
                f"Expected {expected_type}, got {type(value).__name__}. "
                f"Value: {value!r}"
            )

    # Enum validation
    if record["task_class"] not in VALID_TASK_CLASSES:
        raise ValueError(
            f"[logger] 'task_class' must be one of {VALID_TASK_CLASSES}, "
            f"got {record['task_class']!r}"
        )
    if record["workflow_id"] not in VALID_WORKFLOW_IDS:
        raise ValueError(
            f"[logger] 'workflow_id' must be one of {VALID_WORKFLOW_IDS}, "
            f"got {record['workflow_id']!r}"
        )

    # feature_vector must contain only floats/ints (JSON numbers)
    fv = record["feature_vector"]
    if not all(isinstance(x, (int, float)) for x in fv):
        raise ValueError(
            f"[logger] 'feature_vector' must be a list of numbers, "
            f"got: {fv!r}"
        )


class JSONLLogger:
    """
    Context-manager style JSONL logger with schema validation.

    Usage:
        with JSONLLogger("path/to/output.jsonl") as logger:
            logger.write(record_dict)
    """

    def __init__(self, filepath: Union[str, Path], mode: str = "a"):
        """
        Args:
            filepath: Path to the output .jsonl file.
            mode:     File open mode. 'a' appends; 'w' overwrites.
        """
        self.filepath = Path(filepath)
        self.mode = mode
        self._file: Optional[IO] = None
        self._records_written: int = 0

    def open(self) -> "JSONLLogger":
        """Open the file for writing (creates parent directories if needed)."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, self.mode, encoding="utf-8")
        return self

    def close(self) -> None:
        """Flush and close the underlying file."""
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None

    def write(self, record: Dict[str, Any]) -> None:
        """
        Validate and write a single record as a JSONL line.

        Args:
            record: Dictionary matching the required schema.

        Raises:
            ValueError:  Schema validation failure.
            RuntimeError: Logger not open.
        """
        if self._file is None:
            raise RuntimeError(
                "[logger] Logger is not open. Call open() or use as context manager."
            )
        _validate_record(record)
        line = json.dumps(record, ensure_ascii=False)
        self._file.write(line + "\n")
        self._records_written += 1

    @property
    def records_written(self) -> int:
        return self._records_written

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "JSONLLogger":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def write_record(record: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Validate and append a single record to a JSONL file.
    Convenience wrapper; prefer JSONLLogger for batch writes.
    """
    with JSONLLogger(filepath, mode="a") as logger:
        logger.write(record)


if __name__ == "__main__":
    import tempfile

    sample = {
        "task_id": "gsm8k_train_00001",
        "task_text": "What is 2 + 2?",
        "task_class": "reasoning",
        "feature_vector": [0.1, 0.2, 0.3],
        "workflow_id": "W1",
        "output_text": "4",
        "quality_score": 0.95,
        "cost_tokens": 120,
        "latency_ms": 340,
        "reward": 0.9,
        "ground_truth": "4",
        "episode_id": 1,
        "lambda_value": 0.5,
    }

    with tempfile.NamedTemporaryFile(
        suffix=".jsonl", mode="w", delete=False
    ) as tmp:
        tmppath = tmp.name

    with JSONLLogger(tmppath, mode="w") as logger:
        logger.write(sample)
        print(f"Written: {logger.records_written} record(s) to {tmppath}")

    with open(tmppath, encoding="utf-8") as f:
        print("Content:", f.read().strip())
