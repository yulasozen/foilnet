"""Train PointCloudDrag to predict total drag from hull point clouds + scale (L, B, T)."""

import csv
import glob
import os
import time

import numpy as np
import torch

from foilnet.pillars.drag.dataset_drag import split_hulls
from foilnet.pillars.drag.pointcloud_drag import PointCloudDrag

EPOCHS = 1000
LOG_EVERY = 100
LR = 1e-3
BATCH_SIZE = 8
SEED = 42

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
POINTCLOUDS_DIR = os.path.join(PROCESSED_DIR, "pointclouds")
DRAG_CSV_PATH = os.path.join(PROCESSED_DIR, "drag.csv")
MODEL_OUT_PATH = os.path.join(PROCESSED_DIR, "pc_drag_trained.pt")


def load_drag_rows() -> dict[str, dict[str, float]]:
    with open(DRAG_CSV_PATH, newline="") as f:
        return {
            row["hull_id"]: {
                "L": float(row["L"]),
                "B": float(row["B"]),
                "T": float(row["T"]),
                "total_drag": float(row["total_drag"]),
            }
            for row in csv.DictReader(f)
        }


def load_hulls() -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load {hull_id: (points [N,3], scale [3], total_drag [1])}."""
    drag_rows = load_drag_rows()
    paths = sorted(glob.glob(os.path.join(POINTCLOUDS_DIR, "wigley_*.npy")))
    if not paths:
        raise FileNotFoundError(f"no wigley_*.npy point clouds found in {POINTCLOUDS_DIR}")

    hulls = {}
    for path in paths:
        hull_id = os.path.splitext(os.path.basename(path))[0]
        if hull_id not in drag_rows:
            print(f"WARNING: skipping {hull_id}: no drag row in {DRAG_CSV_PATH}")
            continue
        row = drag_rows[hull_id]
        points = torch.from_numpy(np.load(path)).float()
        scale = torch.tensor([row["L"], row["B"], row["T"]], dtype=torch.float32)
        target = torch.tensor([row["total_drag"]], dtype=torch.float32)
        hulls[hull_id] = (points, scale, target)

    return hulls


def r2_score(pred: torch.Tensor, true: torch.Tensor) -> float:
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    hulls = load_hulls()
    train_ids, test_ids = split_hulls(list(hulls.keys()))
    print(f"train hulls ({len(train_ids)}): {train_ids}")
    print(f"test hulls  ({len(test_ids)}): {test_ids}")

    train_points = torch.stack([hulls[h][0] for h in train_ids]).to(device)  # [80, N, 3]
    train_scale = torch.stack([hulls[h][1] for h in train_ids]).to(device)  # [80, 3]
    train_y = torch.stack([hulls[h][2] for h in train_ids]).to(device)  # [80, 1]

    test_points = torch.stack([hulls[h][0] for h in test_ids]).to(device)
    test_scale = torch.stack([hulls[h][1] for h in test_ids]).to(device)
    test_y = torch.stack([hulls[h][2] for h in test_ids]).to(device)

    # Standardize scale inputs and target using TRAIN stats only. Point clouds stay as-is.
    scale_mean = train_scale.mean(dim=0, keepdim=True)
    scale_std = train_scale.std(dim=0, keepdim=True)
    y_mean = train_y.mean(dim=0, keepdim=True)
    y_std = train_y.std(dim=0, keepdim=True)

    train_scale_s = (train_scale - scale_mean) / scale_std
    test_scale_s = (test_scale - scale_mean) / scale_std
    train_y_s = (train_y - y_mean) / y_std

    n_train = train_points.shape[0]

    model = PointCloudDrag().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        batch_losses = []
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]

            optimizer.zero_grad()
            pred = model(train_points[idx], train_scale_s[idx])
            loss = loss_fn(pred, train_y_s[idx])
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        mean_loss = sum(batch_losses) / len(batch_losses)
        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"epoch {epoch:4d} | train MSE (standardized) {mean_loss:.6f}")
    train_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        train_pred_s = model(train_points, train_scale_s)
        test_pred_s = model(test_points, test_scale_s)

        train_pred = (train_pred_s * y_std + y_mean).cpu()
        test_pred = (test_pred_s * y_std + y_mean).cpu()
        train_true = train_y.cpu()
        test_true = test_y.cpu()

        train_r2 = r2_score(train_pred, train_true)
        test_r2 = r2_score(test_pred, test_true)

        train_mae = (train_pred - train_true).abs().mean().item()
        test_mae = (test_pred - test_true).abs().mean().item()

        mean_drag = train_true.mean().item()
        train_mae_pct = 100 * train_mae / mean_drag
        test_mae_pct = 100 * test_mae / mean_drag

    print(f"\ntotal training time: {train_time:.1f}s ({EPOCHS} epochs)")

    print("\nresults (original units, Newtons):")
    print(f"  train R²  = {train_r2:.4f} | train MAE = {train_mae:.4f} N ({train_mae_pct:.2f}% of mean drag)")
    print(f"  test  R²  = {test_r2:.4f} | test  MAE = {test_mae:.4f} N ({test_mae_pct:.2f}% of mean drag)")

    print("\nheld-out hulls: predicted vs true total drag (N)")
    print(f"  {'hull_id':<12}{'predicted':>12}{'true':>12}{'abs err':>12}")
    for i, hull_id in enumerate(test_ids):
        pred_val = test_pred[i, 0].item()
        true_val = test_true[i, 0].item()
        print(f"  {hull_id:<12}{pred_val:>12.4f}{true_val:>12.4f}{abs(pred_val - true_val):>12.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scale_mean": scale_mean.cpu(),
            "scale_std": scale_std.cpu(),
            "y_mean": y_mean.cpu(),
            "y_std": y_std.cpu(),
            "train_hulls": train_ids,
            "test_hulls": test_ids,
        },
        MODEL_OUT_PATH,
    )
    print(f"\nsaved trained model + scalers to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
