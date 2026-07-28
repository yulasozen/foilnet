"""A PointNet-style point-cloud autoencoder for hull shape representation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, 3] -> [B, 3, N] for the per-point shared MLP (1D convs)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = torch.max(x, dim=2).values  # global max-pool over points -> [B, 512]

        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PointNetDecoder(nn.Module):
    def __init__(self, latent_dim: int = 16, n_points: int = 2048):
        super().__init__()
        self.n_points = n_points
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, n_points * 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.n_points, 3)


class PointCloudAE(nn.Module):
    def __init__(self, latent_dim: int = 16, n_points: int = 2048):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_points = n_points
        self.encoder = PointNetEncoder(latent_dim)
        self.decoder = PointNetDecoder(latent_dim, n_points)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
