"""Visualize the FOILNET design loop -- the capstone: imagine a hull, get its drag instantly.

Runs the same latent interpolation as `design_loop.py --mode interp` (VAE encode
the two real endpoint hulls, linearly interpolate latent code + scale, decode,
score with the drag predictor) and reuses its model-loading code directly so
the numbers match exactly. Renders a single figure: top row is the interpolated
hull point clouds (thin -> wide), bottom is predicted drag vs. interpolation
step t, with the two real-hull endpoints marked against their true CFD drag.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from foilnet.pillars.generative.design_loop import (
    load_drag_csv_rows,
    load_drag_model,
    load_hull_pointcloud,
    load_vae,
    predict_drag,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_ROOT, "reports", "figures")

# dataviz reference palette (see .claude skill "dataviz"): categorical slot 1 + status colors.
COLOR_CLOUD = "#2a78d6"       # slot 1, blue -- interpolated point clouds
COLOR_PRED_LINE = "#2a78d6"   # slot 1, blue -- predicted-drag line
COLOR_TRUE_MARK = "#e34948"   # slot 8, red -- true CFD drag reference points
COLOR_ENDPOINT_EDGE = "#1baf7a"  # slot 3, aqua -- real-hull endpoint highlight


def set_hull_aspect(ax, points: np.ndarray) -> None:
    # Hull is long and thin (beam/draft << length): true-proportion box_aspect
    # squeezes the y/z axes into a sliver where tick labels can't help but
    # overlap, so only the length (x) axis keeps numeric ticks.
    ranges = points.max(axis=0) - points.min(axis=0)
    ranges = np.clip(ranges, 1e-6, None)
    ax.set_box_aspect(tuple(ranges))
    ax.view_init(elev=22, azim=-55)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="z", length=0)
    ax.tick_params(axis="x", labelsize=7, pad=0)


def run_interp(vae, drag_model, scalers, drag_rows, hull_a, hull_b, m):
    scale_mean, scale_std, y_mean, y_std, _ = scalers

    if hull_a not in drag_rows or hull_b not in drag_rows:
        raise ValueError(f"both endpoints need a drag.csv row; got hull_a={hull_a!r} hull_b={hull_b!r}")

    pc_a = load_hull_pointcloud(hull_a).unsqueeze(0)
    pc_b = load_hull_pointcloud(hull_b).unsqueeze(0)

    row_a, row_b = drag_rows[hull_a], drag_rows[hull_b]
    scale_a = torch.tensor([row_a["L"], row_a["B"], row_a["T"]], dtype=torch.float32)
    scale_b = torch.tensor([row_b["L"], row_b["B"], row_b["T"]], dtype=torch.float32)

    with torch.no_grad():
        mu_a, _ = vae.encode(pc_a)
        mu_b, _ = vae.encode(pc_b)

    ts = torch.linspace(0.0, 1.0, m)
    zs = torch.stack([mu_a[0] * (1 - t) + mu_b[0] * t for t in ts])
    scales = torch.stack([scale_a * (1 - t) + scale_b * t for t in ts])

    with torch.no_grad():
        points = vae.decode(zs)

    pred = predict_drag(drag_model, points, scales, scale_mean, scale_std, y_mean, y_std)

    return ts.numpy(), points.numpy(), pred.numpy(), row_a["total_drag"], row_b["total_drag"]


def plot_design_loop(hull_a, hull_b, ts, points, pred, true_a, true_b, out_path) -> None:
    m = len(ts)
    fig = plt.figure(figsize=(2.8 * m, 8))
    gs = fig.add_gridspec(2, m, height_ratios=[1.1, 1])

    for i in range(m):
        ax = fig.add_subplot(gs[0, i], projection="3d")
        ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2], s=4, alpha=0.75, color=COLOR_CLOUD)
        set_hull_aspect(ax, points[i])

        is_endpoint = i == 0 or i == m - 1
        label = f"t = {ts[i]:.2f}"
        if i == 0:
            label += f"\n({hull_a}, real)"
        elif i == m - 1:
            label += f"\n({hull_b}, real)"
        ax.set_title(label, fontsize=9, fontweight="bold" if is_endpoint else "normal")
        if is_endpoint:
            for spine in ax.spines.values():
                spine.set_edgecolor(COLOR_ENDPOINT_EDGE)

    ax_line = fig.add_subplot(gs[1, :])
    ax_line.plot(ts, pred, "-o", color=COLOR_PRED_LINE, linewidth=2, markersize=6, label="predicted drag (interpolated hull)")
    ax_line.scatter(
        [ts[0], ts[-1]], [true_a, true_b],
        marker="*", s=260, color=COLOR_TRUE_MARK, edgecolors="white", linewidths=0.8,
        zorder=5, label="true CFD drag (real endpoint hulls)",
    )
    for t, true_val, pred_val, hull_id in [
        (ts[0], true_a, pred[0], hull_a),
        (ts[-1], true_b, pred[-1], hull_b),
    ]:
        ax_line.annotate(
            f"{hull_id}\ntrue = {true_val:.2f} N\npred = {pred_val:.2f} N",
            xy=(t, true_val), xytext=(0, 18 if t == ts[0] else -46),
            textcoords="offset points", ha="center", fontsize=8.5,
        )

    ax_line.set_xlabel("interpolation step t")
    ax_line.set_ylabel("Predicted drag (N)")
    ax_line.set_title("Predicted drag along the latent interpolation", fontsize=12)
    ax_line.legend(loc="upper left", fontsize=9)
    ax_line.margins(y=0.25)

    fig.suptitle(
        f"Design loop: {hull_a} → {hull_b} -- VAE latent interpolation scored instantly by the drag predictor",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.08, hspace=0.35, wspace=0.25)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hull_a", default="wigley_05", help="First (real) hull for interpolation")
    parser.add_argument("--hull_b", default="wigley_20", help="Second (real) hull for interpolation")
    parser.add_argument("--m", type=int, default=6, help="Interpolation steps (default: 6)")
    parser.add_argument("--output", default="design_loop.png", help="Output filename, saved under reports/figures/")
    args = parser.parse_args()

    vae = load_vae()
    drag_model, scale_mean, scale_std, y_mean, y_std, train_hulls = load_drag_model()
    scalers = (scale_mean, scale_std, y_mean, y_std, train_hulls)
    drag_rows = load_drag_csv_rows()

    ts, points, pred, true_a, true_b = run_interp(
        vae, drag_model, scalers, drag_rows, args.hull_a, args.hull_b, args.m
    )

    print(f"design loop: {args.hull_a} -> {args.hull_b}, {args.m} steps")
    print(f"  {'step':<6}{'t':>6}{'predicted_drag_N':>20}{'true_drag_N':>14}")
    for i in range(args.m):
        true_str = f"{true_a:.4f}" if i == 0 else (f"{true_b:.4f}" if i == args.m - 1 else "--")
        print(f"  {i:<6}{ts[i]:>6.2f}{pred[i]:>20.4f}{true_str:>14}")

    out_path = os.path.join(FIGURES_DIR, args.output)
    plot_design_loop(args.hull_a, args.hull_b, ts, points, pred, true_a, true_b, out_path)
    print(f"\nfigure saved to {out_path}")


if __name__ == "__main__":
    main()
