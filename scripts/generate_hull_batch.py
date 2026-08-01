"""
Batch-generate Wigley hull STLs by Latin Hypercube Sampling over (L, B/L, T/L).

Reuses the closed double-body Wigley formula and STL-writing approach from
openfoam/scripts/make_wigley.py:
  y = (B/2)(1-(2x/L)^2)(1-(z/T)^2)
Double body: z spans [-T, +T] so the surface closes at top/bottom;
x spans [-L/2, +L/2] so it closes at bow/stern. -> watertight by construction.
Units: metres.
"""
import argparse
import csv
import os

import numpy as np
from scipy.stats import qmc
from stl import mesh

nx, nz = 120, 60        # surface resolution (matches make_wigley.py)

L_RANGE = (4.0, 8.0)
B_OVER_L_RANGE = (0.07, 0.13)
T_OVER_L_RANGE = (0.045, 0.08)

SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "openfoam", "geometries", "batch")


def halfbreadth(x, z, B, L, T):
    return (B / 2.0) * (1.0 - (2.0 * x / L) ** 2) * (1.0 - (z / T) ** 2)


def surface_points(sign, xs, zs, B, L, T):
    P = np.zeros((nx, nz, 3))
    for i, x in enumerate(xs):
        for k, z in enumerate(zs):
            y = sign * halfbreadth(x, z, B, L, T)
            P[i, k] = (x, y, z)
    return P


def add_quad(tris, p1, p2, p3, p4):
    tris.append([p1, p2, p3])
    tris.append([p1, p3, p4])


def build_hull(L, B, T):
    xs = np.linspace(-L / 2, L / 2, nx)
    zs = np.linspace(-T, T, nz)

    tris = []
    for sign, flip in ((+1, False), (-1, True)):
        P = surface_points(sign, xs, zs, B, L, T)
        for i in range(nx - 1):
            for k in range(nz - 1):
                a, b, c, d = P[i, k], P[i + 1, k], P[i + 1, k + 1], P[i, k + 1]
                if flip:   # keep outward normals consistent on the -y side
                    add_quad(tris, a, d, c, b)
                else:
                    add_quad(tris, a, b, c, d)

    return np.array(tris)


def sample_params(n):
    sampler = qmc.LatinHypercube(d=3, seed=SEED)
    unit = sampler.random(n=n)
    lo = np.array([L_RANGE[0], B_OVER_L_RANGE[0], T_OVER_L_RANGE[0]])
    hi = np.array([L_RANGE[1], B_OVER_L_RANGE[1], T_OVER_L_RANGE[1]])
    scaled = qmc.scale(unit, lo, hi)
    return scaled  # columns: L, B_over_L, T_over_L


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="Number of hulls to generate")
    args = parser.parse_args()
    n = args.n

    os.makedirs(OUT_DIR, exist_ok=True)

    samples = sample_params(n)
    width = max(2, len(str(n - 1)))

    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hull_id", "L", "B", "T", "B_over_L", "T_over_L"])

        for idx in range(n):
            L, b_over_l, t_over_l = samples[idx]
            B = b_over_l * L
            T = t_over_l * L

            hull_id = f"wigley_{idx:0{width}d}"

            tris = build_hull(L, B, T)
            m = mesh.Mesh(np.zeros(len(tris), dtype=mesh.Mesh.dtype))
            m.vectors = tris

            stl_path = os.path.join(OUT_DIR, f"{hull_id}.stl")
            m.save(stl_path)

            print(f"{hull_id}: L={L:.4f} B={B:.4f} T={T:.4f} "
                  f"B/L={b_over_l:.4f} T/L={t_over_l:.4f}")

            writer.writerow([hull_id, L, B, T, b_over_l, t_over_l])

    print(f"\n{n} hulls written to {OUT_DIR}")
    print(f"manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
