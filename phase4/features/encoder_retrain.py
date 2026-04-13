"""
Phase 4 — Encoder retraining on real difficulty labels.

Loads task_difficulty_labels.jsonl from Phase 3, trains the three
prediction heads (difficulty, ambiguity, error_risk) on real signal
instead of Phase 2 synthetic data.

Run:
    python phase4/features/encoder_retrain.py

Output:
    phase4/features/checkpoints/encoder_retrained.pt
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
LABELS_FILE   = Path(__file__).resolve().parent.parent.parent / "phase3/data/task_difficulty_labels.jsonl"
RUN_FILE      = Path(__file__).resolve().parent.parent.parent / "phase3/data/runs/phase3_run_001.jsonl"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "encoder_retrained.pt"

EMBED_MODEL   = "all-MiniLM-L6-v2"
EMBED_DIM     = 384
HIDDEN_DIM    = 128
EPOCHS        = 100
LR            = 1e-3
BATCH_SIZE    = 32
SEED          = 42

DIFFICULTY_MAP = {"easy": 0.0, "medium": 0.5, "high difficulty": 1.0, "undetermined": 0.5}

# ── Dataset ───────────────────────────────────────────────────────────────────
class TaskLabelDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels,     dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# ── Model ─────────────────────────────────────────────────────────────────────
class TaskFeatureHead(nn.Module):
    """
    Three-headed MLP on top of sentence embeddings.
    Outputs: [difficulty, ambiguity, error_risk] each in [0, 1].
    """
    def __init__(self, input_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        out = hidden_dim // 2
        self.difficulty_head  = nn.Sequential(nn.Linear(out, 1), nn.Sigmoid())
        self.ambiguity_head   = nn.Sequential(nn.Linear(out, 1), nn.Sigmoid())
        self.error_risk_head  = nn.Sequential(nn.Linear(out, 1), nn.Sigmoid())

    def forward(self, x):
        h = self.shared(x)
        return torch.cat([
            self.difficulty_head(h),
            self.ambiguity_head(h),
            self.error_risk_head(h),
        ], dim=1)  # (batch, 3)

# ── Label building ────────────────────────────────────────────────────────────
def build_labels(labels_path: Path, run_path: Path):
    """
    Build (task_text, [difficulty, ambiguity, error_risk]) pairs.

    difficulty  — from task_difficulty_labels.jsonl (real signal)
    ambiguity   — proxy: std of quality scores across workflows for this task
    error_risk  — proxy: 1 - mean_quality_W1 (how badly the cheap workflow fails)
    """
    # Load difficulty labels
    diff_map = {}
    with open(labels_path) as f:
        for line in f:
            rec = json.loads(line)
            diff_map[rec["task_id"]] = {
                "difficulty":    DIFFICULTY_MAP.get(rec["difficulty"], 0.5),
                "mean_quality_W1": rec["mean_quality_W1"] or 0.6,
                "mean_quality_W3": rec["mean_quality_W3"] or 0.9,
                "gap":           rec["gap"] or 0.3,
            }

    # Load run to compute per-task quality std (ambiguity proxy)
    task_qualities = {}
    task_texts     = {}
    with open(run_path) as f:
        for line in f:
            rec = json.loads(line)
            tid = rec["task_id"]
            task_qualities.setdefault(tid, []).append(rec["quality_score"])
            task_texts[tid] = rec["task_text"]

    texts, label_vecs = [], []
    for tid, info in diff_map.items():
        if tid not in task_texts:
            continue
        qs    = task_qualities.get(tid, [0.6, 0.75, 0.9])
        mean_q = sum(qs) / len(qs)
        std_q  = float(np.std(qs)) if len(qs) > 1 else 0.15

        difficulty  = info["difficulty"]
        ambiguity   = min(std_q * 4.0, 1.0)   # scale std to [0,1]
        error_risk  = max(0.0, 1.0 - info["mean_quality_W1"])

        texts.append(task_texts[tid])
        label_vecs.append([difficulty, ambiguity, error_risk])

    return texts, np.array(label_vecs, dtype=np.float32)

# ── Training ──────────────────────────────────────────────────────────────────
def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("[encoder_retrain] Building labels...")
    texts, labels = build_labels(LABELS_FILE, RUN_FILE)
    print(f"  Tasks: {len(texts)}")
    print(f"  Label stats — difficulty: {labels[:,0].mean():.3f}  "
          f"ambiguity: {labels[:,1].mean():.3f}  "
          f"error_risk: {labels[:,2].mean():.3f}")

    print(f"\n[encoder_retrain] Embedding with {EMBED_MODEL}...")
    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True,
                                  convert_to_numpy=True)
    print(f"  Embedding shape: {embeddings.shape}")

    # Train / val split (80/20)
    n = len(embeddings)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    tr_idx, val_idx = idx[:split], idx[split:]

    tr_ds  = TaskLabelDataset(embeddings[tr_idx], labels[tr_idx])
    val_ds = TaskLabelDataset(embeddings[val_idx], labels[val_idx])
    tr_dl  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model     = TaskFeatureHead()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    print(f"\n[encoder_retrain] Training for {EPOCHS} epochs...")

    patience = 10
    epochs_no_improve = 0
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        tr_loss = 0.0
        for X, y in tr_dl:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(X)
        tr_loss /= len(tr_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_dl:
                pred = model(X)
                val_loss += criterion(pred, y).item() * len(X)
        val_loss /= len(val_ds)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} — train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\n[encoder_retrain] Best val_loss: {best_val_loss:.4f}")
    print(f"[encoder_retrain] Checkpoint saved → {CHECKPOINT_PATH}")
    return model


# ── Inference helper (used by Phase 4+ pipeline) ──────────────────────────────
def load_encoder() -> tuple:
    """
    Returns (embedder, model) ready for inference.
    Call encode(task_text) using the returned objects.
    """
    embedder = SentenceTransformer(EMBED_MODEL)
    model    = TaskFeatureHead()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    return embedder, model


def encode_task(task_text: str, embedder, model) -> np.ndarray:
    """
    Encode a single task text into a 3-dim feature vector.
    Returns np.ndarray([difficulty, ambiguity, error_risk]).
    """
    emb  = embedder.encode([task_text], convert_to_numpy=True)
    with torch.no_grad():
        vec = model(torch.tensor(emb, dtype=torch.float32))
    return vec.numpy().squeeze()  # shape (3,)


if __name__ == "__main__":
    train()
    print("\n[encoder_retrain] Smoke-testing inference...")
    embedder, model = load_encoder()

    test_tasks = [
        ("What year was the Eiffel Tower built?",           "qa — should be easy, low difficulty"),
        ("If a train travels 60mph for 2.5 hours...",       "reasoning — medium difficulty"),
        ("Prove that sqrt(2) is irrational using contradiction.", "hard reasoning — high difficulty"),
        ("Write a function to reverse a linked list.",      "code — high error risk"),
    ]
    print("\nTask encoding results:")
    print(f"  {'Task snippet':<52} {'diff':>5} {'ambig':>5} {'risk':>5}")
    print(f"  {'-'*52} {'-'*5} {'-'*5} {'-'*5}")
    for text, label in test_tasks:
        vec = encode_task(text, embedder, model)
        print(f"  {text[:51]:<52} {vec[0]:>5.3f} {vec[1]:>5.3f} {vec[2]:>5.3f}  ({label})")