"""Train HullDragGNN to predict total drag directly from hull surface graphs.

Reuses the same surface-graph representation as pillar 1 (mesh_gnn): nodes are
hull surface faces with 7 geometry features [center xyz, normal xyz, area],
edges from face adjacency (data/processed/wigley_*.pt, generated for all 100
hulls via scripts/convert_batch_graphs.py). Each graph's per-node y (pressure,
used by pillar 1) is replaced here with a single graph-level target: the
hull's total_drag from data/processed/drag.csv.
"""

import csv
import glob
import os
import time

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from foilnet.pillars.drag.dataset_drag import split_hulls
from foilnet.pillars.drag.hull_drag_gnn import HullDragGNN

EPOCHS = 500
LOG_EVERY = 50
LR = 1e-3
BATCH_SIZE = 8
IN_CHANNELS = 7
SEED = 42

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
DRAG_CSV_PATH = os.path.join(PROCESSED_DIR, "drag.csv")
MODEL_OUT_PATH = os.path.join(PROCESSED_DIR, "drag_gnn_trained.pt")


def load_drag_map() -> dict[str, float]:
    with open(DRAG_CSV_PATH, newline="") as f:
        return {row["hull_id"]: float(row["total_drag"]) for row in csv.DictReader(f)}


def load_graphs_with_drag() -> dict[str, Data]:
    """Load every hull's surface graph and attach its total_drag as a graph-level y."""
    drag_map = load_drag_map()
    paths = sorted(glob.glob(os.path.join(PROCESSED_DIR, "wigley_*.pt")))
    if not paths:
        raise FileNotFoundError(f"no wigley_*.pt surface graphs found in {PROCESSED_DIR}")

    hulls = {}
    for path in paths:
        hull_id = os.path.splitext(os.path.basename(path))[0]
        if hull_id not in drag_map:
            print(f"WARNING: skipping {hull_id}: no drag row in {DRAG_CSV_PATH}")
            continue
        data = torch.load(path, weights_only=False)
        data.y = torch.tensor([[drag_map[hull_id]]], dtype=torch.float32)
        hulls[hull_id] = data

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

    hulls = load_graphs_with_drag()
    train_ids, test_ids = split_hulls(list(hulls.keys()))
    print(f"train hulls ({len(train_ids)}): {train_ids}")
    print(f"test hulls  ({len(test_ids)}): {test_ids}")

    train_list = [hulls[h] for h in train_ids]
    test_list = [hulls[h] for h in test_ids]

    # Standardize the target using TRAIN stats only.
    train_y = torch.cat([d.y for d in train_list], dim=0)
    y_mean = train_y.mean(dim=0, keepdim=True)
    y_std = train_y.std(dim=0, keepdim=True)

    train_loader = DataLoader(train_list, batch_size=BATCH_SIZE, shuffle=True)
    train_eval_loader = DataLoader(train_list, batch_size=len(train_list), shuffle=False)
    test_eval_loader = DataLoader(test_list, batch_size=len(test_list), shuffle=False)

    model = HullDragGNN(in_channels=IN_CHANNELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    y_mean_dev = y_mean.to(device)
    y_std_dev = y_std.to(device)

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            target = (batch.y - y_mean_dev) / y_std_dev

            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        mean_loss = sum(batch_losses) / len(batch_losses)
        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"epoch {epoch:4d} | train MSE (standardized) {mean_loss:.6f}")
    train_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        train_batch = next(iter(train_eval_loader)).to(device)
        train_pred_s = model(train_batch)
        train_pred = (train_pred_s * y_std_dev + y_mean_dev).cpu()
        train_true = train_batch.y.cpu()

        test_batch = next(iter(test_eval_loader)).to(device)
        test_pred_s = model(test_batch)
        test_pred = (test_pred_s * y_std_dev + y_mean_dev).cpu()
        test_true = test_batch.y.cpu()

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
            "in_channels": IN_CHANNELS,
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
