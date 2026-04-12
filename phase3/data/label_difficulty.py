"""
Retroactive Difficulty Labeler.

Reads phase3_run_001.jsonl and computes a difficulty label for every
unique task_id based on the quality_score gap between W1 and W3 runs:

  gap = mean_quality(W3) - mean_quality(W1)

  gap > 0.3  → "high difficulty"   (W3 much better; task is hard)
  gap < 0.1  → "easy"              (workflows perform similarly)
  otherwise  → "medium"

Output: data/task_difficulty_labels.jsonl
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# Allow running from phase3/ or project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_FILE = Path(__file__).parent / "runs" / "phase3_run_001.jsonl"
OUTPUT_FILE = Path(__file__).parent / "task_difficulty_labels.jsonl"

# ---------------------------------------------------------------------------
# Thresholds (as specified in the prompt)
# ---------------------------------------------------------------------------

HIGH_DIFFICULTY_THRESHOLD = 0.3
EASY_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_run(filepath: Path) -> List[dict]:
    """Read all records from a JSONL file."""
    records = []
    with open(filepath, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[labeler] Warning: skipping malformed line {line_no}: {e}")
    return records


def compute_difficulty_labels(records: List[dict]) -> List[dict]:
    """
    Group quality scores by (task_id, workflow_id) and compute labels.

    Returns a list of dicts:
      {task_id, task_class, mean_quality_W1, mean_quality_W3, gap, difficulty}
    """
    # {task_id: {workflow_id: [quality_score, ...]}}
    scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    task_classes: Dict[str, str] = {}

    for rec in records:
        tid = rec["task_id"]
        wid = rec["workflow_id"]
        scores[tid][wid].append(rec["quality_score"])
        task_classes[tid] = rec.get("task_class", "unknown")

    results = []
    for task_id, wf_scores in sorted(scores.items()):
        w1_scores = wf_scores.get("W1", [])
        w3_scores = wf_scores.get("W3", [])

        mean_w1 = sum(w1_scores) / len(w1_scores) if w1_scores else None
        mean_w3 = sum(w3_scores) / len(w3_scores) if w3_scores else None

        if mean_w1 is None or mean_w3 is None:
            # Cannot compute gap — skip or label as "undetermined"
            difficulty = "undetermined"
            gap = None
        else:
            gap = round(mean_w3 - mean_w1, 6)
            if gap > HIGH_DIFFICULTY_THRESHOLD:
                difficulty = "high difficulty"
            elif gap < EASY_THRESHOLD:
                difficulty = "easy"
            else:
                difficulty = "medium"

        results.append(
            {
                "task_id": task_id,
                "task_class": task_classes[task_id],
                "mean_quality_W1": round(mean_w1, 6) if mean_w1 is not None else None,
                "mean_quality_W3": round(mean_w3, 6) if mean_w3 is not None else None,
                "gap": gap,
                "difficulty": difficulty,
            }
        )

    return results


def save_labels(labels: List[dict], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(input_file: Path = INPUT_FILE, output_file: Path = OUTPUT_FILE) -> None:
    if not input_file.exists():
        print(f"[labeler] ERROR: Input file not found: {input_file}")
        print("         Run `python data/run_pipeline.py` first.")
        sys.exit(1)

    print(f"[labeler] Reading: {input_file}")
    records = load_run(input_file)
    print(f"[labeler] Loaded {len(records)} episode records.")

    labels = compute_difficulty_labels(records)

    counts = {"high difficulty": 0, "easy": 0, "medium": 0, "undetermined": 0}
    for lbl in labels:
        counts[lbl["difficulty"]] += 1

    print(f"[labeler] Difficulty distribution:")
    for k, v in counts.items():
        print(f"    {k:<16}: {v}")

    save_labels(labels, output_file)
    print(f"\n[labeler] Saved {len(labels)} task labels → {output_file}")


if __name__ == "__main__":
    run()
