"""
Convert every OpenFOAM CFD run's hull-surface VTK into a PyTorch Geometric graph.

Runs on the Mac (not the container) once scripts/run_batch_cfd.sh has produced
openfoam/runs/wigley_NN/VTK/hull/hull_*.vtk for each hull. Reuses the conversion
logic in src/foilnet/data/mesh_to_graph.py, importing it directly when possible
and falling back to a subprocess call if the import fails.
"""
import glob
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
MESH_TO_GRAPH_PATH = os.path.join(SRC_DIR, "foilnet", "data", "mesh_to_graph.py")

RUNS_DIR = os.path.join(REPO_ROOT, "openfoam", "runs")
OUT_DIR = os.path.join(REPO_ROOT, "data", "processed")

sys.path.insert(0, SRC_DIR)
try:
    from foilnet.data.mesh_to_graph import convert as convert_fn
except ImportError:
    convert_fn = None


def find_hull_vtk(run_dir):
    matches = sorted(glob.glob(os.path.join(run_dir, "VTK", "hull", "hull_*.vtk")))
    return matches[-1] if matches else None


def convert(vtk_path, out_path):
    if convert_fn is not None:
        convert_fn(vtk_path, out_path)
    else:
        subprocess.run(
            [sys.executable, MESH_TO_GRAPH_PATH, vtk_path, out_path],
            check=True,
        )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    run_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, "wigley_*")))
    pass_count = 0
    fail_count = 0
    failed_hulls = []

    for run_dir in run_dirs:
        hull_id = os.path.basename(run_dir)
        print(f"\n===== {hull_id} =====")

        vtk_path = find_hull_vtk(run_dir)
        if vtk_path is None:
            print(f"  FAIL: {hull_id} (no VTK/hull/hull_*.vtk found in {run_dir})")
            fail_count += 1
            failed_hulls.append(hull_id)
            continue

        out_path = os.path.join(OUT_DIR, f"{hull_id}.pt")
        try:
            convert(vtk_path, out_path)
        except Exception as e:
            print(f"  FAIL: {hull_id} ({e})")
            fail_count += 1
            failed_hulls.append(hull_id)
            continue

        if not os.path.isfile(out_path):
            print(f"  FAIL: {hull_id} (conversion ran but {out_path} not found)")
            fail_count += 1
            failed_hulls.append(hull_id)
            continue

        print(f"  PASS: {hull_id} -> {out_path}")
        pass_count += 1

    print("\n===== SUMMARY =====")
    print(f"PASS: {pass_count}")
    print(f"FAIL: {fail_count}")
    if failed_hulls:
        print(f"Failed hulls: {' '.join(failed_hulls)}")


if __name__ == "__main__":
    main()
