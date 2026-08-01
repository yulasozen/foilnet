"""Visualize HullGNN pressure predictions vs CFD ground truth on a held-out hull."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from hullnet.pillars.mesh_gnn.model import HullGNN

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
CHECKPOINT_PATH = os.path.join(PROCESSED_DIR, "hullgnn_trained.pt")
FIGURES_DIR = os.path.join(REPO_ROOT, "reports", "figures")


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def set_hull_aspect(ax, pos: np.ndarray) -> None:
    # Hull is long and thin (beam/draft << length): true-proportion box_aspect
    # squeezes the y/z axes into a sliver where tick labels can't help but
    # overlap, so only the length (x) axis keeps numeric ticks.
    ranges = pos.max(axis=0) - pos.min(axis=0)
    ranges = np.clip(ranges, 1e-6, None)
    ax.set_box_aspect(tuple(ranges))
    ax.view_init(elev=22, azim=-55)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="z", length=0)
    ax.tick_params(axis="x", labelsize=7, pad=0)
    ax.set_xlabel("length [m]", fontsize=8, labelpad=6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "hull_id", nargs="?", default=None,
        help="Test hull id to visualize (default: first hull in the saved test list)",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=device)

    hull_id = args.hull_id or ckpt["test_hulls"][0]

    hull_path = os.path.join(PROCESSED_DIR, f"{hull_id}.pt")
    data = torch.load(hull_path, weights_only=False).to(device)

    model = HullGNN(in_channels=ckpt["in_channels"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_mean, x_std = ckpt["x_mean"], ckpt["x_std"]
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]

    with torch.no_grad():
        x = (data.x - x_mean) / x_std
        pred_scaled = model(x, data.edge_index)
        pred = pred_scaled * y_std + y_mean

    true_p = data.y.squeeze(-1).numpy()
    pred_p = pred.squeeze(-1).numpy()
    abs_err = np.abs(pred_p - true_p)
    pos = data.pos.numpy()

    r2 = r2_score(pred, data.y)
    in_train = hull_id in ckpt.get("train_hulls", [])
    split_label = "train" if in_train else "test"

    vmin = min(true_p.min(), pred_p.min())
    vmax = max(true_p.max(), pred_p.max())

    fig = plt.figure(figsize=(22, 5.5))

    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    sc1 = ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=true_p, cmap="viridis",
                       vmin=vmin, vmax=vmax, s=6)
    set_hull_aspect(ax1, pos)
    ax1.set_title("True pressure (CFD)")
    fig.colorbar(sc1, ax=ax1, shrink=0.6, pad=0.1, label="p")

    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    sc2 = ax2.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=pred_p, cmap="viridis",
                       vmin=vmin, vmax=vmax, s=6)
    set_hull_aspect(ax2, pos)
    ax2.set_title("Predicted pressure (GNN)")
    fig.colorbar(sc2, ax=ax2, shrink=0.6, pad=0.1, label="p")

    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    sc3 = ax3.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=abs_err, cmap="inferno", s=6)
    set_hull_aspect(ax3, pos)
    ax3.set_title("Absolute error |pred - true|")
    fig.colorbar(sc3, ax=ax3, shrink=0.6, pad=0.1, label="|error|")

    ax4 = fig.add_subplot(1, 4, 4)
    lims = [min(true_p.min(), pred_p.min()), max(true_p.max(), pred_p.max())]
    ax4.plot(lims, lims, "k--", linewidth=1, label="y = x")
    ax4.scatter(true_p, pred_p, s=6, alpha=0.5, color="tab:blue")
    ax4.set_xlabel("true pressure")
    ax4.set_ylabel("predicted pressure")
    ax4.set_title("Parity plot")
    ax4.set_xlim(lims)
    ax4.set_ylim(lims)
    ax4.set_aspect("equal", adjustable="box")
    ax4.legend(loc="upper left")

    fig.suptitle(f"{hull_id} ({split_label} hull) - R² = {r2:.4f}", fontsize=14)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.85, bottom=0.1, wspace=0.4)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, f"{hull_id}_prediction.png")
    fig.savefig(out_path, dpi=150)

    print(f"figure saved to {out_path}")
    print(f"{hull_id} R^2: {r2:.6f}")


if __name__ == "__main__":
    main()
