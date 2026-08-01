"""Train FNO3d on the 40-hull regular-grid CFD dataset with a held-out test split."""

import os
import time

import torch

from hullnet.pillars.neural_operator.dataset_grid import REPO_ROOT, load_train_test_split
from hullnet.pillars.neural_operator.fno import FNO3d

EPOCHS = 500
LOG_EVERY = 50
LR = 1e-3
MODES = (8, 8, 8)
WIDTH = 20
CHANNEL_NAMES = ["Ux", "Uy", "Uz", "p"]

PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
MODEL_OUT_PATH = os.path.join(PROCESSED_DIR, "fno_trained.pt")


def fit_target_scaler(
    targets: list[torch.Tensor], masks: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std over fluid cells only, pooled across the given hulls."""
    n_channels = targets[0].shape[0]
    means = torch.empty(n_channels)
    stds = torch.empty(n_channels)
    for c in range(n_channels):
        pooled = torch.cat([target[c][mask[0].bool()] for target, mask in zip(targets, masks)])
        means[c] = pooled.mean()
        stds[c] = pooled.std().clamp_min(1e-8)
    return means.view(-1, 1, 1, 1), stds.view(-1, 1, 1, 1)


def apply_scaler(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (t - mean) / std


def unscale(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return t * std + mean


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over fluid cells only; mask is [B, 1, Nx, Ny, Nz], broadcast over channels."""
    mask = mask.expand_as(pred)
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp_min(1)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def main() -> None:
    device = torch.device("cpu")
    train_data, test_data, train_ids, test_ids = load_train_test_split()
    print(f"train hulls ({len(train_ids)}): {train_ids}")
    print(f"test hulls  ({len(test_ids)}): {test_ids}")

    train_inputs = torch.stack([x for x, _ in train_data]).to(device)
    train_targets = torch.stack([y for _, y in train_data]).to(device)
    train_masks = train_inputs[:, 0:1]  # mask is input channel 0, shape [n_train, 1, Nx, Ny, Nz]

    test_inputs = torch.stack([x for x, _ in test_data]).to(device)
    test_targets = torch.stack([y for _, y in test_data]).to(device)
    test_masks = test_inputs[:, 0:1]

    y_mean, y_std = fit_target_scaler(list(train_targets), list(train_masks))
    train_targets_scaled = apply_scaler(train_targets, y_mean, y_std)

    model = FNO3d(modes=MODES, width=WIDTH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(train_inputs)
        loss = masked_mse(pred, train_targets_scaled, train_masks)
        loss.backward()
        optimizer.step()

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"epoch {epoch:4d} | mean train MSE (standardized, fluid cells) {loss.item():.6f}")
    train_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        test_pred = unscale(model(test_inputs), y_mean, y_std)

        preds_all, trues_all = [], []
        preds_by_channel = [[] for _ in CHANNEL_NAMES]
        trues_by_channel = [[] for _ in CHANNEL_NAMES]
        per_hull_r2 = {}

        for i, hull_id in enumerate(test_ids):
            fluid = test_masks[i, 0].bool()
            pred_i = test_pred[i][:, fluid]  # [4, n_valid]
            true_i = test_targets[i][:, fluid]

            per_hull_r2[hull_id] = r2_score(pred_i, true_i)
            preds_all.append(pred_i.reshape(-1))
            trues_all.append(true_i.reshape(-1))
            for c in range(len(CHANNEL_NAMES)):
                preds_by_channel[c].append(pred_i[c])
                trues_by_channel[c].append(true_i[c])

        test_r2 = r2_score(torch.cat(preds_all), torch.cat(trues_all))
        per_channel_r2 = {
            name: r2_score(torch.cat(preds_by_channel[c]), torch.cat(trues_by_channel[c]))
            for c, name in enumerate(CHANNEL_NAMES)
        }

    print(f"\ntest R^2 (original units, pooled over {len(test_ids)} hulls, fluid cells): {test_r2:.6f}")
    print("\nper-channel test R^2:")
    for name in CHANNEL_NAMES:
        print(f"  {name}: {per_channel_r2[name]:.6f}")
    print("\nper-hull test R^2:")
    for hull_id in test_ids:
        print(f"  {hull_id}: {per_hull_r2[hull_id]:.6f}")

    print(f"\ntotal training time: {train_time:.1f}s ({EPOCHS} epochs)")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "modes": MODES,
            "width": WIDTH,
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
