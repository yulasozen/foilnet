"""Chamfer distance between two point clouds, pure torch."""

import torch


def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    """Mean nearest-neighbor distance in both directions between pc1, pc2: [B, N, 3]."""
    dists = torch.cdist(pc1, pc2)  # [B, N1, N2]
    nearest_1_to_2 = dists.min(dim=2).values  # [B, N1]
    nearest_2_to_1 = dists.min(dim=1).values  # [B, N2]
    return (nearest_1_to_2.mean() + nearest_2_to_1.mean()) / 2
