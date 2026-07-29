#!/usr/bin/env python3
"""Collect total drag for each hull run and merge with manifest geometry parameters."""
import csv
import glob
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_GLOB = os.path.join(REPO_ROOT, "openfoam", "runs", "wigley_*")
MANIFEST_PATH = os.path.join(REPO_ROOT, "openfoam", "geometries", "batch", "manifest.csv")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "drag.csv")

# Matches: <time> ((Fp_x Fp_y Fp_z) (Fv_x Fv_y Fv_z)) ((...) (...))
FORCE_LINE_RE = re.compile(
    r"^\s*[\d.eE+-]+\s+\(\(([^)]+)\)\s*\(([^)]+)\)\)"
)


def parse_last_line(forces_path):
    with open(forces_path) as f:
        lines = [line for line in f if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"no data lines in {forces_path}")
    last_line = lines[-1]
    match = FORCE_LINE_RE.match(last_line)
    if not match:
        raise ValueError(f"could not parse force line in {forces_path}: {last_line!r}")
    fp = [float(x) for x in match.group(1).split()]
    fv = [float(x) for x in match.group(2).split()]
    return fp[0], fv[0]


def load_manifest(path):
    manifest = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            manifest[row["hull_id"]] = row
    return manifest


def main():
    manifest = load_manifest(MANIFEST_PATH)

    run_dirs = sorted(glob.glob(RUNS_GLOB))
    results = []
    for run_dir in run_dirs:
        hull_id = os.path.basename(run_dir)
        forces_path = os.path.join(run_dir, "postProcessing", "forces", "0", "forces.dat")
        if not os.path.isfile(forces_path):
            print(f"WARNING: skipping {hull_id}: missing {forces_path}")
            continue
        try:
            fp_x, fv_x = parse_last_line(forces_path)
        except ValueError as e:
            print(f"WARNING: skipping {hull_id}: {e}")
            continue

        pressure_drag = 2 * fp_x
        viscous_drag = 2 * fv_x
        total_drag = pressure_drag + viscous_drag

        row = {
            "hull_id": hull_id,
            "pressure_drag": pressure_drag,
            "viscous_drag": viscous_drag,
            "total_drag": total_drag,
        }

        if hull_id in manifest:
            m = manifest[hull_id]
            row.update({
                "L": m["L"],
                "B": m["B"],
                "T": m["T"],
                "B_over_L": m["B_over_L"],
                "T_over_L": m["T_over_L"],
            })
        else:
            print(f"WARNING: {hull_id} not found in manifest.csv")

        results.append(row)
        print(f"{hull_id}: total_drag = {total_drag:.6f} N")

    if not results:
        print("No hull results collected; exiting without writing output.")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fieldnames = ["hull_id", "L", "B", "T", "B_over_L", "T_over_L",
                  "pressure_drag", "viscous_drag", "total_drag"]
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    totals = [r["total_drag"] for r in results]
    print(f"\nWrote {len(results)} rows to {OUTPUT_PATH}")
    print(f"Summary across {len(totals)} hulls:")
    print(f"  min total drag  = {min(totals):.6f} N")
    print(f"  max total drag  = {max(totals):.6f} N")
    print(f"  mean total drag = {sum(totals) / len(totals):.6f} N")


if __name__ == "__main__":
    main()
