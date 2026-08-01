"""Generate new hull point clouds from the trained PointCloudVAE — sampling or latent interpolation."""

import argparse
import os

import numpy as np
import torch

from hullnet.pillars.generative.pointcloud_vae import PointCloudVAE
from hullnet.pillars.geometry_repr.dataset_pc import POINTCLOUDS_DIR, REPO_ROOT

PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
CHECKPOINT_PATH = os.path.join(PROCESSED_DIR, "vae_trained.pt")
OUT_DIR = os.path.join(PROCESSED_DIR, "generated")


def load_model() -> PointCloudVAE:
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location="cpu")
    model = PointCloudVAE(latent_dim=ckpt["latent_dim"], n_points=ckpt["n_points"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_hull_pointcloud(hull_id: str) -> torch.Tensor:
    path = os.path.join(POINTCLOUDS_DIR, f"{hull_id}.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no point cloud found for '{hull_id}' at {path}")
    return torch.from_numpy(np.load(path)).float()


def print_bounds(label: str, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    print(
        f"  {label}: bounds x[{mins[0]:.3f},{maxs[0]:.3f}] "
        f"y[{mins[1]:.3f},{maxs[1]:.3f}] z[{mins[2]:.3f},{maxs[2]:.3f}]"
    )


def generate_samples(model: PointCloudVAE, k: int, out_dir: str) -> None:
    with torch.no_grad():
        z = torch.randn(k, model.latent_dim)
        points = model.decode(z).numpy()

    print(f"sampling mode: {k} random latent draws (z ~ N(0, I))")
    for i in range(k):
        out_path = os.path.join(out_dir, f"sample_{i:02d}.npy")
        np.save(out_path, points[i])
        print_bounds(f"sample_{i:02d} -> {out_path}", points[i])


def generate_interpolation(model: PointCloudVAE, hull_a: str, hull_b: str, m: int, out_dir: str) -> None:
    pc_a = load_hull_pointcloud(hull_a).unsqueeze(0)
    pc_b = load_hull_pointcloud(hull_b).unsqueeze(0)

    with torch.no_grad():
        mu_a, _ = model.encode(pc_a)
        mu_b, _ = model.encode(pc_b)

        ts = torch.linspace(0.0, 1.0, m)
        zs = torch.stack([mu_a[0] * (1 - t) + mu_b[0] * t for t in ts])
        points = model.decode(zs).numpy()

    print(f"interpolation mode: {hull_a} -> {hull_b}, {m} steps")
    for i in range(m):
        out_path = os.path.join(out_dir, f"interp_{i:02d}.npy")
        np.save(out_path, points[i])
        print_bounds(f"interp_{i:02d} (t={ts[i]:.2f}) -> {out_path}", points[i])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sample", "interp"], required=True, help="Generation mode")
    parser.add_argument("--k", type=int, default=5, help="Number of random samples (sample mode, default: 5)")
    parser.add_argument("--m", type=int, default=6, help="Number of interpolation steps (interp mode, default: 6)")
    parser.add_argument("--hull_a", default="wigley_05", help="First hull for interpolation (default: wigley_05)")
    parser.add_argument("--hull_b", default="wigley_20", help="Second hull for interpolation (default: wigley_20)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_model()

    if args.mode == "sample":
        generate_samples(model, args.k, OUT_DIR)
    else:
        generate_interpolation(model, args.hull_a, args.hull_b, args.m, OUT_DIR)


if __name__ == "__main__":
    main()
