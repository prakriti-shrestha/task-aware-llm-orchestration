"""
NLI-based contradiction scorer.
Uses DeBERTa-v3-small (cross-encoder/nli-deberta-v3-small) to score whether
the output text entails, is neutral to, or contradicts the input task.
Returns a float 0.0-1.0 where 1.0 = fully entailing (no contradiction).

Install: pip install transformers torch
"""
from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        print(f"[contradiction] Loading NLI model: {_MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        _model.eval()
        print("[contradiction] Model loaded.")


def score(task_text: str, output_text: str) -> float:
    """
    Score whether output_text entails (does not contradict) task_text.

    Args:
        task_text:   The original input task / question.
        output_text: The LLM-generated answer to score.

    Returns:
        float in [0.0, 1.0]
        - 1.0: output fully entails the task (consistent, no contradiction)
        - 0.0: output directly contradicts the task
        - ~0.5: neutral / unrelated
    """
    _load_model()

    # NLI convention: premise = task, hypothesis = output
    inputs = _tokenizer(
        task_text,
        output_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        logits = _model(**inputs).logits  # shape: (1, 3)

    # DeBERTa NLI label order: contradiction=0, neutral=1, entailment=2
    probs = torch.softmax(logits, dim=-1).squeeze()
    entailment_prob = probs[2].item()
    contradiction_prob = probs[0].item()

    # Score = entailment - contradiction, shifted to [0, 1]
    raw = entailment_prob - contradiction_prob
    return round((raw + 1.0) / 2.0, 4)


if __name__ == "__main__":
    # Quick smoke test
    s1 = score("What is 2 + 2?", "2 + 2 equals 4.")
    s2 = score("What is 2 + 2?", "2 + 2 equals 7.")
    s3 = score("What is the capital of France?", "The capital of France is Berlin.")
    print(f"Consistent answer (expect ~1.0): {s1}")
    print(f"Wrong answer     (expect ~0.0): {s2}")
    print(f"Wrong capital    (expect ~0.0): {s3}")