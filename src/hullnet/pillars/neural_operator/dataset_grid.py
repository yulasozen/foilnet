"""Load regular-grid FNO training data from data/processed/grids/wigley_*.npz files."""

import glob
import os
import random

import numpy as np
import torch

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
GRIDS_DIR = os.path.join(REPO_ROOT, "data", "processed", "grids")

TEST_FRACTION = 0.2
SPLIT_SEED = 42


def _coordinate_grids(bounds: tuple[float, ...], grid_shape: tuple[int, int, int]) -> torch.Tensor:
    """Normalized (0-1) X, Y, Z coordinate grids of shape [3, Nx, Ny, Nz], from domain bounds."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    nx, ny, nz = grid_shape
    x = (torch.linspace(xmin, xmax, nx) - xmin) / (xmax - xmin)
    y = (torch.linspace(ymin, ymax, ny) - ymin) / (ymax - ymin)
    z = (torch.linspace(zmin, zmax, nz) - zmin) / (zmax - zmin)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    return torch.stack([X, Y, Z], dim=0).float()


def load_hull(path: str, coord_grids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one wigley_NN.npz into (input, target) tensors, each [4, Nx, Ny, Nz]."""
    data = np.load(path)
    fields = torch.from_numpy(data["fields"]).float()
    mask = torch.from_numpy(data["mask"]).float().unsqueeze(0)
    grid_shape = tuple(int(v) for v in data["grid_shape"])
    bounds = tuple(float(v) for v in data["bounds"])

    if coord_grids is None:
        coord_grids = _coordinate_grids(bounds, grid_shape)

    input_tensor = torch.cat([mask, coord_grids], dim=0)
    return input_tensor, fields


def load_all_hulls() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Load every hull, sharing one set of coordinate grids (the domain box is fixed)."""
    paths = sorted(glob.glob(os.path.join(GRIDS_DIR, "wigley_*.npz")))
    if not paths:
        raise FileNotFoundError(f"no wigley_*.npz files found in {GRIDS_DIR}")

    first = np.load(paths[0])
    coord_grids = _coordinate_grids(
        tuple(float(v) for v in first["bounds"]),
        tuple(int(v) for v in first["grid_shape"]),
    )

    return {
        os.path.splitext(os.path.basename(path))[0]: load_hull(path, coord_grids=coord_grids)
        for path in paths
    }


def split_hulls(hull_ids: list[str]) -> tuple[list[str], list[str]]:
    ids = sorted(hull_ids)
    rng = random.Random(SPLIT_SEED)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_test = max(1, round(len(ids) * TEST_FRACTION))
    test_ids = sorted(shuffled[:n_test])
    train_ids = sorted(shuffled[n_test:])
    return train_ids, test_ids


def load_train_test_split() -> tuple[
    list[tuple[torch.Tensor, torch.Tensor]],
    list[tuple[torch.Tensor, torch.Tensor]],
    list[str],
    list[str],
]:
    """Assemble the 80/20 (seed 42) train/test split as lists of (input, target) tensors."""
    hulls = load_all_hulls()
    train_ids, test_ids = split_hulls(list(hulls.keys()))
    train_data = [hulls[h] for h in train_ids]
    test_data = [hulls[h] for h in test_ids]
    return train_data, test_data, train_ids, test_ids
