"""A PointNet-style encoder + hull-scale MLP predicting total drag from a point cloud."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetShapeEncoder(nn.Module):
    """Shared per-point MLP + global max-pool, matching geometry_repr/pointnet_ae.py."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, 3] -> [B, 3, N] for the per-point shared MLP (1D convs)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return torch.max(x, dim=2).values  # global max-pool over points -> [B, 512]


class PointCloudDrag(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape_encoder = PointNetShapeEncoder()

        self.scale_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(512 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, points: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # points: [B, N, 3], scale: [B, 3] -> [B, 1]
        shape_feat = self.shape_encoder(points)  # [B, 512]
        scale_feat = self.scale_mlp(scale)  # [B, 32]
        combined = torch.cat([shape_feat, scale_feat], dim=1)  # [B, 544]
        return self.head(combined)
