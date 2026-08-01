"""HullNet design loop -- the capstone: imagine a hull, get its drag instantly.

Chains pillar 4 (generative, `PointCloudVAE`) straight into the drag pillar
(`PointCloudDrag`): the VAE decodes a latent code into a new hull point cloud,
and that point cloud is fed directly to the drag predictor -- no CFD run, no
meshing step, no manual parameterization in between.

Caveat -- generated-hull scale is an ASSUMPTION, not a CFD-derived value:
`PointCloudDrag` needs a physical scale (L, B, T) alongside the point cloud,
but the VAE only ever produces points normalized to a unit sphere; it has no
notion of physical size. For freshly sampled hulls (mode=sample) there is no
ground truth to draw a scale from, so this script falls back to the MEAN
(L, B, T) of the drag model's own training hulls (data/processed/drag.csv),
unless --L/--B/--T are given explicitly. Treat sampled drag numbers as "what a
hull around this size would drag like," not a prediction for a specific real
geometry. For interpolation (mode=interp) between two REAL hulls, both
endpoints' true scale is known, so scale is linearly interpolated alongside
the latent code -- a more grounded assumption for the in-between synthetic
hulls, and the two endpoints' true CFD drag is printed for comparison.
"""

import argparse
import csv
import os

import numpy as np
import torch

from hullnet.pillars.drag.pointcloud_drag import PointCloudDrag
from hullnet.pillars.generative.pointcloud_vae import PointCloudVAE
from hullnet.pillars.geometry_repr.dataset_pc import POINTCLOUDS_DIR, REPO_ROOT

PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
VAE_CHECKPOINT_PATH = os.path.join(PROCESSED_DIR, "vae_trained.pt")
DRAG_CHECKPOINT_PATH = os.path.join(PROCESSED_DIR, "pc_drag_trained.pt")
DRAG_CSV_PATH = os.path.join(PROCESSED_DIR, "drag.csv")
OUT_DIR = os.path.join(PROCESSED_DIR, "generated")


def load_vae() -> PointCloudVAE:
    ckpt = torch.load(VAE_CHECKPOINT_PATH, weights_only=False, map_location="cpu")
    model = PointCloudVAE(latent_dim=ckpt["latent_dim"], n_points=ckpt["n_points"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_drag_model():
    ckpt = torch.load(DRAG_CHECKPOINT_PATH, weights_only=False, map_location="cpu")
    model = PointCloudDrag()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["scale_mean"], ckpt["scale_std"], ckpt["y_mean"], ckpt["y_std"], ckpt["train_hulls"]


def load_drag_csv_rows() -> dict[str, dict[str, float]]:
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


def load_hull_pointcloud(hull_id: str) -> torch.Tensor:
    path = os.path.join(POINTCLOUDS_DIR, f"{hull_id}.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no point cloud found for '{hull_id}' at {path}")
    return torch.from_numpy(np.load(path)).float()


def default_scale(train_hulls: list[str], drag_rows: dict[str, dict[str, float]]) -> torch.Tensor:
    """Mean (L, B, T) over the drag model's own training hulls -- the fallback scale
    used for hulls with no ground truth (freshly sampled from the VAE prior)."""
    arr = np.array([[drag_rows[h]["L"], drag_rows[h]["B"], drag_rows[h]["T"]] for h in train_hulls])
    return torch.tensor(arr.mean(axis=0), dtype=torch.float32)


def predict_drag(
    drag_model: PointCloudDrag,
    points: torch.Tensor,
    scale: torch.Tensor,
    scale_mean: torch.Tensor,
    scale_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        scale_s = (scale - scale_mean) / scale_std
        pred_s = drag_model(points, scale_s)
        pred = pred_s * y_std + y_mean
    return pred.squeeze(1)


def run_sample(vae, drag_model, scalers, drag_rows, k, out_dir, override_scale) -> None:
    scale_mean, scale_std, y_mean, y_std, train_hulls = scalers

    if override_scale is not None:
        base_scale = torch.tensor(override_scale, dtype=torch.float32)
        scale_note = (
            f"user-specified scale L={override_scale[0]:.3f} "
            f"B={override_scale[1]:.3f} T={override_scale[2]:.3f}"
        )
    else:
        base_scale = default_scale(train_hulls, drag_rows)
        scale_note = (
            f"ASSUMED scale = mean(L,B,T) over {len(train_hulls)} training hulls = "
            f"L={base_scale[0]:.3f} B={base_scale[1]:.3f} T={base_scale[2]:.3f} "
            "(no ground truth for freshly sampled hulls -- see module docstring)"
        )

    with torch.no_grad():
        z = torch.randn(k, vae.latent_dim)
        points = vae.decode(z)

    scale_batch = base_scale.unsqueeze(0).repeat(k, 1)
    pred = predict_drag(drag_model, points, scale_batch, scale_mean, scale_std, y_mean, y_std)

    print(f"\nsample mode: {k} random latent draws (z ~ N(0, I)), decoded and scored instantly")
    print(f"  {scale_note}")
    print(f"\n  {'sample_id':<18}{'L':>8}{'B':>8}{'T':>8}{'predicted_drag_N':>20}")
    for i in range(k):
        hull_name = f"design_sample_{i:02d}"
        out_path = os.path.join(out_dir, f"{hull_name}.npy")
        np.save(out_path, points[i].numpy())
        print(
            f"  {hull_name:<18}{base_scale[0]:>8.3f}{base_scale[1]:>8.3f}{base_scale[2]:>8.3f}"
            f"{pred[i].item():>20.4f}"
        )

    print(f"\n  saved {k} generated point clouds to {out_dir}")


def run_interp(vae, drag_model, scalers, drag_rows, hull_a, hull_b, m, out_dir) -> None:
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

    print(f"\ninterp mode: {hull_a} -> {hull_b}, {m} steps, decoded and scored at each step")
    print("  scale (L,B,T) linearly interpolated between the two REAL hulls' true values")
    print("  true_drag_N (from CFD, drag.csv) is only known at the t=0.0 and t=1.0 endpoints --")
    print("  every hull in between is synthetic and has no CFD ground truth.")
    print(f"\n  {'step':<6}{'t':>6}{'L':>8}{'B':>8}{'T':>8}{'predicted_drag_N':>18}{'true_drag_N':>14}")
    for i in range(m):
        hull_name = f"design_interp_{i:02d}"
        out_path = os.path.join(out_dir, f"{hull_name}.npy")
        np.save(out_path, points[i].numpy())

        t = ts[i].item()
        true_str = "--"
        if t == 0.0:
            true_str = f"{row_a['total_drag']:.4f}"
        elif t == 1.0:
            true_str = f"{row_b['total_drag']:.4f}"

        print(
            f"  {i:<6}{t:>6.2f}{scales[i, 0].item():>8.3f}{scales[i, 1].item():>8.3f}"
            f"{scales[i, 2].item():>8.3f}{pred[i].item():>18.4f}{true_str:>14}"
        )

    print(f"\n  saved {m} generated point clouds to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["sample", "interp"], required=True, help="Design-loop mode")
    parser.add_argument("--k", type=int, default=5, help="Random latent samples (sample mode, default: 5)")
    parser.add_argument("--m", type=int, default=6, help="Interpolation steps (interp mode, default: 6)")
    parser.add_argument("--hull_a", default="wigley_05", help="First hull for interpolation (default: wigley_05)")
    parser.add_argument("--hull_b", default="wigley_20", help="Second hull for interpolation (default: wigley_20)")
    parser.add_argument("--L", type=float, default=None, help="Override assumed hull length (sample mode)")
    parser.add_argument("--B", type=float, default=None, help="Override assumed hull beam (sample mode)")
    parser.add_argument("--T", type=float, default=None, help="Override assumed hull draft (sample mode)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    vae = load_vae()
    drag_model, scale_mean, scale_std, y_mean, y_std, train_hulls = load_drag_model()
    scalers = (scale_mean, scale_std, y_mean, y_std, train_hulls)
    drag_rows = load_drag_csv_rows()

    if args.mode == "sample":
        override_scale = None
        if args.L is not None or args.B is not None or args.T is not None:
            if args.L is None or args.B is None or args.T is None:
                raise ValueError("--L, --B, --T must all be given together to override scale")
            override_scale = [args.L, args.B, args.T]
        run_sample(vae, drag_model, scalers, drag_rows, args.k, OUT_DIR, override_scale)
    else:
        run_interp(vae, drag_model, scalers, drag_rows, args.hull_a, args.hull_b, args.m, OUT_DIR)


if __name__ == "__main__":
    main()
