"""
Batch-generate Wigley hull STLs for a set of (B/L, T/L) ratios.

Reuses the closed double-body Wigley formula and STL-writing approach from
openfoam/scripts/make_wigley.py:
  y = (B/2)(1-(2x/L)^2)(1-(z/T)^2)
Double body: z spans [-T, +T] so the surface closes at top/bottom;
x spans [-L/2, +L/2] so it closes at bow/stern. -> watertight by construction.
Units: metres.
"""
import csv
import os

import numpy as np
from stl import mesh

L = 6.0                 # length (m), fixed across the batch
nx, nz = 120, 60        # surface resolution (matches make_wigley.py)

HULLS = [
    ("wigley_00", 0.08,  0.0625),
    ("wigley_01", 0.10,  0.0625),
    ("wigley_02", 0.12,  0.0625),
    ("wigley_03", 0.10,  0.05),
    ("wigley_04", 0.10,  0.075),
]

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


def build_hull(B, T):
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hull_id", "L", "B", "T", "B_over_L", "T_over_L"])

        for hull_id, b_over_l, t_over_l in HULLS:
            B = b_over_l * L
            T = t_over_l * L

            tris = build_hull(B, T)
            m = mesh.Mesh(np.zeros(len(tris), dtype=mesh.Mesh.dtype))
            m.vectors = tris

            stl_path = os.path.join(OUT_DIR, f"{hull_id}.stl")
            m.save(stl_path)

            mins = tris.reshape(-1, 3).min(axis=0)
            maxs = tris.reshape(-1, 3).max(axis=0)
            print(f"{hull_id}: B/L={b_over_l:.4f} T/L={t_over_l:.4f}  "
                  f"L={L} B={B:.4f} T={T:.4f}")
            print(f"  bbox min: {mins}")
            print(f"  bbox max: {maxs}")

            writer.writerow([hull_id, L, B, T, b_over_l, t_over_l])

    print(f"\nmanifest written to {manifest_path}")


if __name__ == "__main__":
    main()
