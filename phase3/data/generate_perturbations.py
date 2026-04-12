"""
Synthetic Perturbations Generator.

Samples 50 tasks from GSM8K and 50 from ARC-Challenge (100 total),
then applies three programmatic perturbation types to create harder
variants:

  (a) irrelevant_info  — Prepend an irrelevant distractor sentence.
  (b) negation         — Negate one clause in the question.
  (c) ambiguous_pronoun — Introduce an ambiguous pronoun reference.

Output: data/perturbations.jsonl
One JSONL record per (task, perturbation_type) triple.
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

# Allow running from phase3/ or project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loaders.gsm8k_loader import load_gsm8k
from data.loaders.arc_loader import load_arc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
SAMPLES_PER_SOURCE = 50
OUTPUT_FILE = Path(__file__).parent / "perturbations.jsonl"

# ---------------------------------------------------------------------------
# Perturbation templates
# ---------------------------------------------------------------------------

IRRELEVANT_SENTENCES = [
    "Note that the weather outside is sunny today.",
    "Scientists recently discovered a new species of deep-sea fish.",
    "The capital of France is Paris.",
    "A recent study found that coffee improves morning focus.",
    "The Eiffel Tower was completed in 1889.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The Great Wall of China stretches over 13,000 miles.",
    "Mount Everest is the tallest mountain above sea level on Earth.",
    "The speed of light in a vacuum is approximately 3×10⁸ m/s.",
    "Honeybees communicate through a waggle dance.",
]

NEGATION_PHRASES = [
    ("is", "is not"),
    ("are", "are not"),
    ("can", "cannot"),
    ("will", "will not"),
    ("does", "does not"),
    ("has", "has not"),
    ("have", "have not"),
    ("was", "was not"),
    ("were", "were not"),
]

AMBIGUOUS_PRONOUNS = [
    "They were involved in solving this.",
    "It was also a factor in the calculation.",
    "She had previously worked on a similar problem.",
    "He mentioned this to someone earlier.",
    "This had been noted by them before.",
]


# ---------------------------------------------------------------------------
# Perturbation functions
# ---------------------------------------------------------------------------

def perturb_irrelevant_info(text: str, rng: random.Random) -> str:
    """Prepend an irrelevant distractor sentence to the question."""
    distractor = rng.choice(IRRELEVANT_SENTENCES)
    return f"{distractor} {text}"


def perturb_negation(text: str, rng: random.Random) -> str:
    """
    Negate the first matching verb phrase in the question.
    Falls back to appending '(Assume this is not the case.)' if no match.
    """
    rng.shuffle(NEGATION_PHRASES)  # randomise which phrase is tried first
    for original, negated in NEGATION_PHRASES:
        if f" {original} " in text:
            # Only negate the first occurrence
            return text.replace(f" {original} ", f" {negated} ", 1)
    # Fallback: append a generic negation note
    return text + " (Assume the opposite of what is stated is true.)"


def perturb_ambiguous_pronoun(text: str, rng: random.Random) -> str:
    """Append a sentence with an ambiguous pronoun reference."""
    ambiguous = rng.choice(AMBIGUOUS_PRONOUNS)
    return text + " " + ambiguous


PERTURBATION_FUNCTIONS = {
    "irrelevant_info": perturb_irrelevant_info,
    "negation": perturb_negation,
    "ambiguous_pronoun": perturb_ambiguous_pronoun,
}


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_tasks(loader_fn, split: str, n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Load all tasks from a loader then sample n deterministically."""
    all_tasks = list(loader_fn(split=split))
    rng.shuffle(all_tasks)
    return all_tasks[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(output_file: Path = OUTPUT_FILE) -> None:
    rng = random.Random(SEED)

    print(f"[perturbations] Sampling {SAMPLES_PER_SOURCE} tasks from GSM8K (train)...")
    gsm8k_tasks = sample_tasks(load_gsm8k, "train", SAMPLES_PER_SOURCE, rng)

    print(f"[perturbations] Sampling {SAMPLES_PER_SOURCE} tasks from ARC-Challenge (train)...")
    arc_tasks = sample_tasks(load_arc, "train", SAMPLES_PER_SOURCE, rng)

    source_tasks = gsm8k_tasks + arc_tasks  # 100 total

    records = []
    for task in source_tasks:
        for ptype, pfunc in PERTURBATION_FUNCTIONS.items():
            perturbed_text = pfunc(task["task_text"], rng)
            record = {
                "original_task_id":    task["task_id"],
                "perturbation_type":   ptype,
                "original_text":       task["task_text"],
                "perturbed_text":      perturbed_text,
                "task_class":          task["task_class"],
                "ground_truth":        task["ground_truth"],
            }
            records.append(record)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[perturbations] Written {len(records)} records → {output_file}")
    print(f"  Sources      : {len(source_tasks)} tasks (50 GSM8K + 50 ARC)")
    print(f"  Perturbations: {len(PERTURBATION_FUNCTIONS)} per task")
    print(f"  Total records: {len(records)}")


if __name__ == "__main__":
    run()
