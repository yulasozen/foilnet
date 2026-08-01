"""Train HullGNN on the full hull dataset with a held-out test split."""

import glob
import os
import random
import time

import torch
from torch_geometric.data import Data

from hullnet.pillars.mesh_gnn.model import HullGNN

EPOCHS = 3000
LOG_EVERY = 200
LR = 1e-3
TEST_FRACTION = 0.2
SPLIT_SEED = 42

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
MODEL_OUT_PATH = os.path.join(PROCESSED_DIR, "hullgnn_trained.pt")


def fit_scaler(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = t.mean(dim=0, keepdim=True)
    std = t.std(dim=0, keepdim=True).clamp_min(1e-8)
    return mean, std


def apply_scaler(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (t - mean) / std


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def load_hulls() -> dict[str, Data]:
    paths = sorted(glob.glob(os.path.join(PROCESSED_DIR, "wigley_*.pt")))
    return {
        os.path.splitext(os.path.basename(p))[0]: torch.load(p, weights_only=False)
        for p in paths
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


def main() -> None:
    device = torch.device("cpu")
    hulls = load_hulls()

    train_ids, test_ids = split_hulls(list(hulls.keys()))
    print(f"train hulls ({len(train_ids)}): {train_ids}")
    print(f"test hulls  ({len(test_ids)}): {test_ids}")

    train_data = [hulls[h].to(device) for h in train_ids]
    test_data = {h: hulls[h].to(device) for h in test_ids}

    # Fit scalers on pooled TRAINING data only -- test hulls never touch this fit.
    x_pool = torch.cat([d.x for d in train_data], dim=0)
    y_pool = torch.cat([d.y for d in train_data], dim=0)
    x_mean, x_std = fit_scaler(x_pool)
    y_mean, y_std = fit_scaler(y_pool)

    def scale(d: Data) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_scaler(d.x, x_mean, x_std), apply_scaler(d.y, y_mean, y_std)

    train_xy = [scale(d) for d in train_data]

    model = HullGNN(in_channels=x_pool.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        losses = [
            loss_fn(model(x, d.edge_index), y)
            for d, (x, y) in zip(train_data, train_xy)
        ]
        mean_loss = torch.stack(losses).mean()
        mean_loss.backward()
        optimizer.step()

        if epoch % LOG_EVERY == 0:
            print(f"epoch {epoch:5d} | mean train MSE (standardized) {mean_loss.item():.6f}")
    train_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        train_preds, train_targets = [], []
        for d, (x, y) in zip(train_data, train_xy):
            pred = model(x, d.edge_index)
            train_preds.append(pred * y_std + y_mean)
            train_targets.append(d.y)
        train_r2 = r2_score(torch.cat(train_preds, dim=0), torch.cat(train_targets, dim=0))

        per_hull_test_r2 = {}
        test_preds, test_targets = [], []
        for hull_id, d in test_data.items():
            x, y = scale(d)
            pred = model(x, d.edge_index)
            pred_unscaled = pred * y_std + y_mean
            per_hull_test_r2[hull_id] = r2_score(pred_unscaled, d.y)
            test_preds.append(pred_unscaled)
            test_targets.append(d.y)
        test_r2 = r2_score(torch.cat(test_preds, dim=0), torch.cat(test_targets, dim=0))

    print(f"\ntrain R^2 (original units, pooled over {len(train_ids)} hulls): {train_r2:.6f}")
    print(f"test R^2  (original units, pooled over {len(test_ids)} hulls): {test_r2:.6f}")
    print("\nper-hull test R^2:")
    for hull_id in test_ids:
        print(f"  {hull_id}: {per_hull_test_r2[hull_id]:.6f}")

    print(f"\ntotal training time: {train_time:.1f}s ({EPOCHS} epochs)")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": x_pool.shape[1],
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "train_hulls": train_ids,
            "test_hulls": test_ids,
        },
        MODEL_OUT_PATH,
    )
    print(f"saved trained model + scalers to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
