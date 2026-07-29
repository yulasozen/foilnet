"""Load hull parameters + total drag from data/processed/drag.csv."""

import csv
import os
import random

import torch

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
DRAG_CSV_PATH = os.path.join(REPO_ROOT, "data", "processed", "drag.csv")

TEST_FRACTION = 0.2
SPLIT_SEED = 42


def load_all_hulls() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Load every hull's (L, B, T) features and total_drag target into a {hull_id: (x, y)} dict."""
    if not os.path.isfile(DRAG_CSV_PATH):
        raise FileNotFoundError(f"drag CSV not found at {DRAG_CSV_PATH}")

    hulls = {}
    with open(DRAG_CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            features = torch.tensor(
                [float(row["L"]), float(row["B"]), float(row["T"])], dtype=torch.float32
            )
            target = torch.tensor([float(row["total_drag"])], dtype=torch.float32)
            hulls[row["hull_id"]] = (features, target)

    if not hulls:
        raise ValueError(f"no rows found in {DRAG_CSV_PATH}")

    return hulls


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
    list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor],
    list[str], list[str],
]:
    """Assemble the 80/20 (seed 42) train/test split as lists of feature/target tensors."""
    hulls = load_all_hulls()
    train_ids, test_ids = split_hulls(list(hulls.keys()))
    train_x = [hulls[h][0] for h in train_ids]
    train_y = [hulls[h][1] for h in train_ids]
    test_x = [hulls[h][0] for h in test_ids]
    test_y = [hulls[h][1] for h in test_ids]
    return train_x, train_y, test_x, test_y, train_ids, test_ids
