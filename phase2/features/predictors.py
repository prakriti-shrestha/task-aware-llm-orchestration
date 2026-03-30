import torch
import torch.nn as nn

class TaskPropertyPredictor(nn.Module):

    # Small NN to convert vectors to meaningful task signals
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()  # Keeps value between 0-1
        )

    def forward(self, x):
        return self.net(x)