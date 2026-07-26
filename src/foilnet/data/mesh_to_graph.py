"""Convert an OpenFOAM hull-surface VTK file into a PyTorch Geometric graph."""

import argparse

import numpy as np
import pyvista as pv
import torch
from torch_geometric.data import Data


def build_edge_index(mesh: pv.PolyData) -> np.ndarray:
    """Connect faces that share a mesh edge, using the surface's own face connectivity."""
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for cell_id in range(mesh.n_cells):
        point_ids = mesh.get_cell(cell_id).point_ids
        n = len(point_ids)
        for i in range(n):
            a, b = point_ids[i], point_ids[(i + 1) % n]
            edge = (a, b) if a < b else (b, a)
            edge_to_faces.setdefault(edge, []).append(cell_id)

    edges = set()
    for faces in edge_to_faces.values():
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                f1, f2 = faces[i], faces[j]
                edges.add((f1, f2))
                edges.add((f2, f1))

    return np.array(sorted(edges), dtype=np.int64).T


def convert(input_path: str, output_path: str) -> None:
    mesh = pv.read(input_path)

    if "p" in mesh.point_data and "p" not in mesh.cell_data:
        mesh = mesh.point_data_to_cell_data()

    mesh = mesh.compute_normals(cell_normals=True, point_normals=False)
    mesh = mesh.compute_cell_sizes(length=False, area=True, volume=False)

    edge_index = build_edge_index(mesh)

    centers = mesh.cell_centers()
    pos = centers.points.astype(np.float32)
    normals = centers.point_data["Normals"].astype(np.float32)
    areas = centers.point_data["Area"].astype(np.float32).reshape(-1, 1)
    p = centers.point_data["p"].astype(np.float32).reshape(-1, 1)

    x = np.concatenate([pos, normals, areas], axis=1)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        y=torch.from_numpy(p),
        pos=torch.from_numpy(pos),
    )
    torch.save(data, output_path)

    print(f"nodes: {data.num_nodes}")
    print(f"edges: {data.num_edges}")
    print(f"x shape: {tuple(data.x.shape)}")
    print(f"y shape: {tuple(data.y.shape)}")
    print(f"p min/max: {p.min():.6g} / {p.max():.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input VTK file (foamToVTK hull surface)")
    parser.add_argument("output", help="Output .pt file")
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
