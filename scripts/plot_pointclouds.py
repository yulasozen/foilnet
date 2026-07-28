"""Render a row of 3D scatter subplots for a set of point-cloud .npy files."""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_ROOT, "reports", "figures")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pattern",
        help="Glob pattern for point-cloud .npy files, e.g. 'data/processed/generated/interp_*.npy'",
    )
    parser.add_argument("output", help="Output filename, saved under reports/figures/")
    parser.add_argument("--title", default=None, help="Figure title")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"no files matched pattern '{args.pattern}'")

    n = len(paths)
    fig = plt.figure(figsize=(3.2 * n, 4))

    for i, path in enumerate(paths):
        points = np.load(path)
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=4, alpha=0.7, color="tab:blue")
        set_hull_aspect(ax, points)
        ax.set_title(os.path.splitext(os.path.basename(path))[0], fontsize=9)

    top = 0.85 if args.title else 0.92
    if args.title:
        fig.suptitle(args.title, fontsize=14)
    fig.subplots_adjust(left=0.02, right=0.98, top=top, bottom=0.05, wspace=0.3)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, args.output)
    fig.savefig(out_path, dpi=150)

    print(f"rendered {n} point clouds: {[os.path.basename(p) for p in paths]}")
    print(f"figure saved to {out_path}")


if __name__ == "__main__":
    main()
