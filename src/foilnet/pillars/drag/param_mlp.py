"""A small MLP baseline predicting total drag from hull parameters (L, B, T)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DragMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3] -> [B, 1]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
