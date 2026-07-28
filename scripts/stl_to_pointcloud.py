"""
Sample normalized surface point clouds from every hull STL for autoencoder training.

Runs on the Mac. For each openfoam/geometries/batch/wigley_NN.stl, uniformly samples
points from the hull surface with trimesh, centers and rescales to a unit bounding
sphere, and writes data/processed/pointclouds/wigley_NN.npy. The per-hull centroid
and scale are also written to normalization.csv so the transform is reversible.
"""
import argparse
import csv
import glob
import os

import numpy as np
import trimesh

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
STL_DIR = os.path.join(REPO_ROOT, "openfoam", "geometries", "batch")
OUT_DIR = os.path.join(REPO_ROOT, "data", "processed", "pointclouds")


def sample_normalized_point_cloud(stl_path: str, n_points: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample n_points from the hull surface, centered at origin and scaled to unit radius."""
    mesh = trimesh.load(stl_path)
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    points = points.astype(np.float32)

    centroid = points.mean(axis=0)
    points = points - centroid
    scale = float(np.linalg.norm(points, axis=1).max())
    points = points / scale

    return points, centroid, scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_points", type=int, default=2048, help="Points to sample per hull (default: 2048)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    stl_paths = sorted(glob.glob(os.path.join(STL_DIR, "wigley_*.stl")))
    if not stl_paths:
        raise SystemExit(f"no wigley_*.stl files found in {STL_DIR}")

    normalization_rows = []
    for stl_path in stl_paths:
        hull_id = os.path.splitext(os.path.basename(stl_path))[0]

        points, centroid, scale = sample_normalized_point_cloud(stl_path, args.n_points)

        out_path = os.path.join(OUT_DIR, f"{hull_id}.npy")
        np.save(out_path, points)

        max_radius = np.linalg.norm(points, axis=1).max()
        print(f"{hull_id}: {points.shape[0]} points, max radius {max_radius:.4f}")

        normalization_rows.append(
            {
                "hull_id": hull_id,
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "centroid_z": centroid[2],
                "scale": scale,
            }
        )

    normalization_path = os.path.join(OUT_DIR, "normalization.csv")
    with open(normalization_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["hull_id", "centroid_x", "centroid_y", "centroid_z", "scale"])
        writer.writeheader()
        writer.writerows(normalization_rows)

    print(f"\n{len(stl_paths)} hulls written to {OUT_DIR}")
    print(f"normalization table written to {normalization_path}")


if __name__ == "__main__":
    main()
