"""Load hull point clouds from data/processed/pointclouds/wigley_*.npy files."""

import glob
import os
import random

import numpy as np
import torch

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
POINTCLOUDS_DIR = os.path.join(REPO_ROOT, "data", "processed", "pointclouds")

TEST_FRACTION = 0.2
SPLIT_SEED = 42


def load_all_hulls() -> dict[str, torch.Tensor]:
    """Load every hull's point cloud into a {hull_id: [N, 3]} dict."""
    paths = sorted(glob.glob(os.path.join(POINTCLOUDS_DIR, "wigley_*.npy")))
    if not paths:
        raise FileNotFoundError(f"no wigley_*.npy files found in {POINTCLOUDS_DIR}")

    return {
        os.path.splitext(os.path.basename(path))[0]: torch.from_numpy(np.load(path)).float()
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


def load_train_test_split() -> tuple[list[torch.Tensor], list[torch.Tensor], list[str], list[str]]:
    """Assemble the 80/20 (seed 42) train/test split as lists of point-cloud tensors."""
    hulls = load_all_hulls()
    train_ids, test_ids = split_hulls(list(hulls.keys()))
    train_data = [hulls[h] for h in train_ids]
    test_data = [hulls[h] for h in test_ids]
    return train_data, test_data, train_ids, test_ids
