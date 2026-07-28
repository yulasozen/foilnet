"""Interpolate an OpenFOAM volume VTK file onto a regular 3D grid for FNO training."""

import argparse

import numpy as np
import pyvista as pv


def build_grid(bounds: tuple[float, float, float, float, float, float], nx: int, ny: int, nz: int) -> pv.ImageData:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    spacing = (
        (xmax - xmin) / (nx - 1),
        (ymax - ymin) / (ny - 1),
        (zmax - zmin) / (nz - 1),
    )
    grid = pv.ImageData(dimensions=(nx, ny, nz), origin=(xmin, ymin, zmin), spacing=spacing)
    return grid


def to_grid_shape(flat: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    """Reshape a flat point-data array (VTK's x-fastest ordering) to (nx, ny, nz, ...)."""
    trailing = flat.shape[1:]
    return flat.reshape((nz, ny, nx) + trailing).transpose((2, 1, 0) + tuple(range(3, 3 + len(trailing))))


def convert(input_path: str, output_path: str, nx: int, ny: int, nz: int) -> None:
    mesh = pv.read(input_path)
    bounds = mesh.bounds

    grid = build_grid(bounds, nx, ny, nz)
    sampled = grid.sample(mesh)

    valid = sampled.point_data["vtkValidPointMask"].astype(bool)
    U = sampled.point_data["U"].astype(np.float32).copy()
    p = sampled.point_data["p"].astype(np.float32).copy()
    U[~valid] = 0.0
    p[~valid] = 0.0

    U_grid = to_grid_shape(U, nx, ny, nz)
    p_grid = to_grid_shape(p, nx, ny, nz)
    mask_grid = to_grid_shape(valid, nx, ny, nz).astype(np.uint8)

    fields = np.stack([U_grid[..., 0], U_grid[..., 1], U_grid[..., 2], p_grid], axis=0).astype(np.float32)

    np.savez(
        output_path,
        fields=fields,
        mask=mask_grid,
        grid_shape=np.array([nx, ny, nz], dtype=np.int64),
        bounds=np.array(bounds, dtype=np.float64),
    )

    names = ["Ux", "Uy", "Uz", "p"]
    print(f"grid shape: {(nx, ny, nz)}")
    print(f"bounds: {bounds}")
    print(f"fluid-cell fraction (mask mean): {mask_grid.mean():.6g}")
    for i, name in enumerate(names):
        field = fields[i]
        print(f"{name} min/max: {field.min():.6g} / {field.max():.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input VTK file (foamToVTK internal mesh)")
    parser.add_argument("output", help="Output .npz file")
    parser.add_argument("--nx", type=int, default=64, help="Grid resolution in x (default: 64)")
    parser.add_argument("--ny", type=int, default=32, help="Grid resolution in y (default: 32)")
    parser.add_argument("--nz", type=int, default=32, help="Grid resolution in z (default: 32)")
    args = parser.parse_args()
    convert(args.input, args.output, args.nx, args.ny, args.nz)


if __name__ == "__main__":
    main()
