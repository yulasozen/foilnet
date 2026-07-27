#!/usr/bin/env bash
#
# Run the full OpenFOAM pipeline (surfaceFeatures -> blockMesh -> snappyHexMesh
# -> foamRun -> foamToVTK) for every hull STL in a batch.
#
# Meant to run INSIDE the OpenFOAM 11 Docker container. All paths are under /data.
#
# For each /data/openfoam/geometries/batch/wigley_NN.stl:
#   1. Fresh run dir /data/openfoam/runs/wigley_NN copied from the case template.
#   2. Hull STL copied to constant/geometry/wigley.stl (so all dicts still work).
#   3. Mesh + solve, with the symmetryPlane->symmetry boundary patch fix applied
#      after snappyHexMesh (required or foamRun crashes).
#   4. foamToVTK writes the hull surface to VTK/hull/hull_*.vtk.
#
# Graph conversion (mesh_to_graph.py) runs separately, on the Mac -- see
# scripts/convert_batch_graphs.py.
#
# Continues past a failing hull -- does not abort the whole batch.

set -u
shopt -s nullglob

DATA_ROOT="/data"
CASE_TEMPLATE="${DATA_ROOT}/openfoam/cases/wigley"
BATCH_DIR="${DATA_ROOT}/openfoam/geometries/batch"
RUNS_DIR="${DATA_ROOT}/openfoam/runs"

mkdir -p "$RUNS_DIR"

process_hull() {
    local stl="$1"
    local hull_id
    hull_id=$(basename "$stl" .stl)

    echo ""
    echo "===== ${hull_id} ====="

    local run_dir="${RUNS_DIR}/${hull_id}"
    rm -rf "$run_dir"
    cp -r "$CASE_TEMPLATE" "$run_dir"
    mkdir -p "$run_dir/constant/geometry"
    cp "$stl" "$run_dir/constant/geometry/wigley.stl"

    (
        cd "$run_dir" || exit 1
        mkdir -p log

        echo "  -> surfaceFeatures"
        surfaceFeatures > log/surfaceFeatures.log 2>&1 || {
            echo "  FAIL: ${hull_id} (surfaceFeatures, see log/surfaceFeatures.log)"
            exit 1
        }

        echo "  -> blockMesh"
        blockMesh > log/blockMesh.log 2>&1 || {
            echo "  FAIL: ${hull_id} (blockMesh, see log/blockMesh.log)"
            exit 1
        }

        echo "  -> snappyHexMesh -overwrite"
        snappyHexMesh -overwrite > log/snappyHexMesh.log 2>&1 || {
            echo "  FAIL: ${hull_id} (snappyHexMesh, see log/snappyHexMesh.log)"
            exit 1
        }

        echo "  -> fixing symmetry patch type in constant/polyMesh/boundary"
        sed -i -E 's/type([[:space:]]+)symmetryPlane;/type\1symmetry;/' constant/polyMesh/boundary
        sed -i 's/1(symmetryPlane)/1(symmetry)/' constant/polyMesh/boundary

        echo "  -> foamRun"
        foamRun > log/foamRun.log 2>&1 || {
            echo "  FAIL: ${hull_id} (foamRun, see log/foamRun.log)"
            exit 1
        }

        echo "  -> foamToVTK"
        foamToVTK -excludePatches '(inlet outlet symmetry farfield)' -latestTime -noInternal \
            > log/foamToVTK.log 2>&1 || {
            echo "  FAIL: ${hull_id} (foamToVTK, see log/foamToVTK.log)"
            exit 1
        }

        vtk_file=$(ls -1 VTK/hull/hull_*.vtk 2>/dev/null | sort -V | tail -n 1)
        if [ -z "$vtk_file" ]; then
            echo "  FAIL: ${hull_id} (no VTK/hull/hull_*.vtk produced)"
            exit 1
        fi

        echo "  PASS: ${hull_id} -> ${vtk_file}"
        exit 0
    )
    return $?
}

pass_count=0
fail_count=0
failed_hulls=()

for stl in "$BATCH_DIR"/wigley_*.stl; do
    if process_hull "$stl"; then
        pass_count=$((pass_count + 1))
    else
        fail_count=$((fail_count + 1))
        failed_hulls+=("$(basename "$stl" .stl)")
    fi
done

echo ""
echo "===== SUMMARY ====="
echo "PASS: ${pass_count}"
echo "FAIL: ${fail_count}"
if [ "${fail_count}" -gt 0 ]; then
    echo "Failed hulls: ${failed_hulls[*]}"
fi
