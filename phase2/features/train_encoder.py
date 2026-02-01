import torch
import numpy as np
import os
from encoder import TaskEncoder
from predictors import TaskPropertyPredictor
from dataset import load_phase1_data, compute_labels

encoder = TaskEncoder()
model = TaskPropertyPredictor(embed_dim=384)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

data = load_phase1_data("../data/phase1_logs.jsonl")

for epoch in range(20):
    total_loss = 0
    for task_text, runs in data.items():
        emb = torch.tensor(encoder.encode(task_text)).float()
        target = torch.tensor(compute_labels(runs)).float()

        pred = model(emb)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} | Loss {total_loss:.4f}")

os.makedirs("checkpoints", exist_ok=True)
torch.save(
    model.state_dict(),
    "checkpoints/task_feature_model.pt"
)
print("Saved task feature model.")