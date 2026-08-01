"""Train DragMLP on the 100-hull parametric drag dataset with a held-out test split."""

import os
import time

import numpy as np
import torch

from hullnet.pillars.drag.dataset_drag import REPO_ROOT, load_train_test_split
from hullnet.pillars.drag.param_mlp import DragMLP

EPOCHS = 2000
LOG_EVERY = 200
LR = 1e-3
SEED = 42

PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
MODEL_OUT_PATH = os.path.join(PROCESSED_DIR, "drag_mlp_trained.pt")


def standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def unstandardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


def r2_score(pred: torch.Tensor, true: torch.Tensor) -> float:
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cpu")
    train_x, train_y, test_x, test_y, train_ids, test_ids = load_train_test_split()
    print(f"train hulls ({len(train_ids)}): {train_ids}")
    print(f"test hulls  ({len(test_ids)}): {test_ids}")

    train_x = torch.stack(train_x).to(device)  # [80, 3]
    train_y = torch.stack(train_y).to(device)  # [80, 1]
    test_x = torch.stack(test_x).to(device)  # [20, 3]
    test_y = torch.stack(test_y).to(device)  # [20, 1]

    # Standardize inputs and target using TRAIN statistics only.
    x_mean = train_x.mean(dim=0, keepdim=True)
    x_std = train_x.std(dim=0, keepdim=True)
    y_mean = train_y.mean(dim=0, keepdim=True)
    y_std = train_y.std(dim=0, keepdim=True)

    train_x_s = standardize(train_x, x_mean, x_std)
    train_y_s = standardize(train_y, y_mean, y_std)
    test_x_s = standardize(test_x, x_mean, x_std)
    test_y_s = standardize(test_y, y_mean, y_std)

    model = DragMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(train_x_s)
        loss = loss_fn(pred, train_y_s)
        loss.backward()
        optimizer.step()

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"epoch {epoch:4d} | train MSE (standardized) {loss.item():.6f}")
    train_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        train_pred_s = model(train_x_s)
        test_pred_s = model(test_x_s)

        train_pred = unstandardize(train_pred_s, y_mean, y_std)
        test_pred = unstandardize(test_pred_s, y_mean, y_std)

        train_r2 = r2_score(train_pred, train_y)
        test_r2 = r2_score(test_pred, test_y)

        train_mae = (train_pred - train_y).abs().mean().item()
        test_mae = (test_pred - test_y).abs().mean().item()

        mean_drag = train_y.mean().item()
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
        true_val = test_y[i, 0].item()
        print(f"  {hull_id:<12}{pred_val:>12.4f}{true_val:>12.4f}{abs(pred_val - true_val):>12.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "train_hulls": train_ids,
            "test_hulls": test_ids,
        },
        MODEL_OUT_PATH,
    )
    print(f"\nsaved trained model + scalers to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
