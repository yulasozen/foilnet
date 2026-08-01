"""A variational autoencoder for hull point clouds, built on the geometry_repr PointNet backbone."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hullnet.pillars.geometry_repr.pointnet_ae import PointNetDecoder


class PointNetVAEEncoder(nn.Module):
    """Same PointNet backbone as geometry_repr, with two heads for mu and logvar."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, N, 3] -> [B, 3, N] for the per-point shared MLP (1D convs)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = torch.max(x, dim=2).values  # global max-pool over points -> [B, 512]

        x = F.relu(self.fc1(x))
        return self.fc_mu(x), self.fc_logvar(x)


class PointCloudVAE(nn.Module):
    def __init__(self, latent_dim: int = 16, n_points: int = 2048):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_points = n_points
        self.encoder = PointNetVAEEncoder(latent_dim)
        self.decoder = PointNetDecoder(latent_dim, n_points)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
