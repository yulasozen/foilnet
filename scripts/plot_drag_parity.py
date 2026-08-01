"""Drag parity figure -- the headline result of the drag-prediction pillar.

Loads all three trained drag models (param MLP baseline, surface GNN, PointNet
+ scale) and scores each on its own held-out test hulls (80/20 split, seed 42
-- all three checkpoints agree on the same 20 test hulls). Reuses the exact
dataset-loading and standardization code from each train_*.py script so the
numbers match the training logs exactly.
"""

import os

import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader

from hullnet.pillars.drag.dataset_drag import load_train_test_split, split_hulls
from hullnet.pillars.drag.hull_drag_gnn import HullDragGNN
from hullnet.pillars.drag.param_mlp import DragMLP
from hullnet.pillars.drag.pointcloud_drag import PointCloudDrag
from hullnet.pillars.drag.train_drag_gnn import load_graphs_with_drag
from hullnet.pillars.drag.train_pc_drag import load_hulls as load_pc_hulls

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(REPO_ROOT, "reports", "figures")

MLP_CKPT_PATH = os.path.join(PROCESSED_DIR, "drag_mlp_trained.pt")
GNN_CKPT_PATH = os.path.join(PROCESSED_DIR, "drag_gnn_trained.pt")
PC_CKPT_PATH = os.path.join(PROCESSED_DIR, "pc_drag_trained.pt")

# dataviz reference palette (see .claude skill "dataviz"): categorical slots 1/2, status-good.
COLOR_MARKER = "#2a78d6"      # slot 1, blue -- scatter points, all subplots (small multiples)
COLOR_REFLINE = "#898781"     # muted ink -- y = x reference line
COLOR_BEST_TEXT = "#006300"   # success text -- GNN-advantage annotation
COLOR_BEST_TINT = "#eafbf3"   # faint aqua tint -- winning subplot background
COLOR_BEST_EDGE = "#1baf7a"   # slot 3, aqua -- winning subplot border


def r2_score(pred: torch.Tensor, true: torch.Tensor) -> float:
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def eval_param_mlp() -> dict:
    ckpt = torch.load(MLP_CKPT_PATH, weights_only=False, map_location="cpu")
    train_x, train_y, test_x, test_y, _, test_ids = load_train_test_split()
    train_y = torch.stack(train_y)
    test_x = torch.stack(test_x)
    test_y = torch.stack(test_y)

    model = DragMLP()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        test_x_s = (test_x - ckpt["x_mean"]) / ckpt["x_std"]
        test_pred = model(test_x_s) * ckpt["y_std"] + ckpt["y_mean"]

    return _summarize("Param MLP (L, B, T)", test_ids, test_y, test_pred, train_y)


def eval_drag_gnn() -> dict:
    ckpt = torch.load(GNN_CKPT_PATH, weights_only=False, map_location="cpu")
    hulls = load_graphs_with_drag()
    train_ids, test_ids = split_hulls(list(hulls.keys()))

    train_y = torch.cat([hulls[h].y for h in train_ids], dim=0)
    test_list = [hulls[h] for h in test_ids]
    test_batch = next(iter(DataLoader(test_list, batch_size=len(test_list), shuffle=False)))

    model = HullDragGNN(in_channels=ckpt["in_channels"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        test_pred = model(test_batch) * ckpt["y_std"] + ckpt["y_mean"]

    return _summarize("Surface GNN", test_ids, test_batch.y, test_pred, train_y)


def eval_pointcloud_drag() -> dict:
    ckpt = torch.load(PC_CKPT_PATH, weights_only=False, map_location="cpu")
    hulls = load_pc_hulls()
    train_ids, test_ids = split_hulls(list(hulls.keys()))

    train_y = torch.stack([hulls[h][2] for h in train_ids])
    test_points = torch.stack([hulls[h][0] for h in test_ids])
    test_scale = torch.stack([hulls[h][1] for h in test_ids])
    test_y = torch.stack([hulls[h][2] for h in test_ids])

    model = PointCloudDrag()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        test_scale_s = (test_scale - ckpt["scale_mean"]) / ckpt["scale_std"]
        test_pred = model(test_points, test_scale_s) * ckpt["y_std"] + ckpt["y_mean"]

    return _summarize("PointNet + scale", test_ids, test_y, test_pred, train_y)


def _summarize(
    name: str,
    test_ids: list[str],
    test_true: torch.Tensor,
    test_pred: torch.Tensor,
    train_y: torch.Tensor,
) -> dict:
    test_r2 = r2_score(test_pred, test_true)
    test_mae = (test_pred - test_true).abs().mean().item()
    # MAE% denominator is the TRAIN mean drag, matching train_*.py's own convention.
    test_mae_pct = 100 * test_mae / train_y.mean().item()
    return {
        "name": name,
        "test_ids": test_ids,
        "true": test_true.squeeze(-1).numpy(),
        "pred": test_pred.squeeze(-1).numpy(),
        "r2": test_r2,
        "mae_pct": test_mae_pct,
    }


def plot_parity(results: list[dict], out_path: str) -> None:
    best = max(results, key=lambda r: r["r2"])
    baseline = results[0]

    fig, axes = plt.subplots(1, len(results), figsize=(14, 5))

    for ax, result in zip(axes, results):
        is_best = result["name"] == best["name"]

        lo = min(result["true"].min(), result["pred"].min())
        hi = max(result["true"].max(), result["pred"].max())
        pad = 0.05 * (hi - lo)
        lims = (lo - pad, hi + pad)

        if is_best:
            ax.set_facecolor(COLOR_BEST_TINT)
            for spine in ax.spines.values():
                spine.set_edgecolor(COLOR_BEST_EDGE)
                spine.set_linewidth(1.8)

        ax.plot(lims, lims, "--", color=COLOR_REFLINE, linewidth=1.2, label="y = x", zorder=1)
        ax.scatter(
            result["true"], result["pred"],
            s=34, alpha=0.8, color=COLOR_MARKER,
            edgecolors="white", linewidths=0.5, zorder=2,
        )

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("True drag (N)")
        ax.set_ylabel("Predicted drag (N)")
        ax.set_title(result["name"], fontsize=12, fontweight="bold")

        ax.text(
            0.05, 0.95,
            f"test R² = {result['r2']:.3f}\ntest MAE = {result['mae_pct']:.2f}%",
            transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#c3c2b7", alpha=0.9),
        )

        if is_best:
            delta_r2 = result["r2"] - baseline["r2"]
            ax.text(
                0.95, 0.05,
                f"▲ best — ΔR² = +{delta_r2:.3f} vs {baseline['name']}",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
                color=COLOR_BEST_TEXT, fontweight="bold",
            )

    fig.suptitle(
        "Drag prediction: predicted vs. true total drag on held-out test hulls (n=20, seed 42)",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.12, wspace=0.35)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=200)


def main() -> None:
    results = [eval_param_mlp(), eval_drag_gnn(), eval_pointcloud_drag()]

    print("test-set results (held-out hulls, n=20):")
    for r in results:
        print(f"  {r['name']:<20} R² = {r['r2']:.4f}   MAE = {r['mae_pct']:.2f}%")

    out_path = os.path.join(FIGURES_DIR, "drag_parity.png")
    plot_parity(results, out_path)
    print(f"\nfigure saved to {out_path}")


if __name__ == "__main__":
    main()
