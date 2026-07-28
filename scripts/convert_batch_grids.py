"""
Convert every OpenFOAM CFD run's volume VTK into a regular-grid .npz for FNO training.

Runs on the Mac (not the container) once scripts/run_batch_cfd.sh has produced
openfoam/runs/wigley_NN/VTK/wigley_NN_*.vtk for each hull. Reuses the conversion
logic in src/foilnet/pillars/neural_operator/vtk_to_grid.py, importing it directly
when possible and falling back to a subprocess call if the import fails.
"""
import glob
import os
import subprocess
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
VTK_TO_GRID_PATH = os.path.join(SRC_DIR, "foilnet", "pillars", "neural_operator", "vtk_to_grid.py")

RUNS_DIR = os.path.join(REPO_ROOT, "openfoam", "runs")
OUT_DIR = os.path.join(REPO_ROOT, "data", "processed", "grids")

NX, NY, NZ = 32, 16, 16

sys.path.insert(0, SRC_DIR)
try:
    from foilnet.pillars.neural_operator.vtk_to_grid import convert as convert_fn
except ImportError:
    convert_fn = None


def find_volume_vtk(run_dir, hull_id):
    matches = sorted(glob.glob(os.path.join(run_dir, "VTK", f"{hull_id}_*.vtk")))
    return matches[-1] if matches else None


def convert(vtk_path, out_path):
    if convert_fn is not None:
        convert_fn(vtk_path, out_path, NX, NY, NZ)
    else:
        subprocess.run(
            [
                sys.executable, VTK_TO_GRID_PATH, vtk_path, out_path,
                "--nx", str(NX), "--ny", str(NY), "--nz", str(NZ),
            ],
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

        vtk_path = find_volume_vtk(run_dir, hull_id)
        if vtk_path is None:
            print(f"  FAIL: {hull_id} (no VTK/{hull_id}_*.vtk found in {run_dir})")
            fail_count += 1
            failed_hulls.append(hull_id)
            continue

        out_path = os.path.join(OUT_DIR, f"{hull_id}.npz")
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

        data = np.load(out_path)
        grid_shape = tuple(data["grid_shape"].tolist())
        fluid_fraction = float(data["mask"].mean())

        print(f"  PASS: {hull_id} -> {out_path} (grid_shape={grid_shape}, fluid_fraction={fluid_fraction:.4f})")
        pass_count += 1

    print("\n===== SUMMARY =====")
    print(f"PASS: {pass_count}")
    print(f"FAIL: {fail_count}")
    if failed_hulls:
        print(f"Failed hulls: {' '.join(failed_hulls)}")


if __name__ == "__main__":
    main()
