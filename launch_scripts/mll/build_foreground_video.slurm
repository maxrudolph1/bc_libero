#!/bin/bash
# Resumable suite conversion followed by an exhaustive pixel/data check.
# Submit from the repository with:
# sbatch --export=ALL,REPO_ROOT="$PWD" launch_scripts/mll/build_foreground_video.slurm
# Select a node that can access SRC_ROOT, DST_ROOT, and the video frames.
# Pass suites as script arguments, e.g. ...slurm libero_spatial libero_object.
#SBATCH --job-name=libero-foreground-video
#SBATCH --partition=allnodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/scratch/cluster/mrudolph/documents/BC-IB}
SRC_ROOT=${SRC_ROOT:-/datastor2/mrudolph/LIBERO/datasets}
DST_ROOT=${DST_ROOT:-/datastor2/mrudolph/LIBERO/datasets_foreground_video}
FRAMES_DIR=${FRAMES_DIR:-${REPO_ROOT}/epickitchens/P01/rgb_frames/p01_01}
cd "$REPO_ROOT"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

if [[ $# -eq 0 ]]; then
    set -- libero_goal
fi
for suite in "$@"; do
    case "$suite" in
        libero_goal|libero_spatial|libero_object|libero_10|libero_90) ;;
        *) echo "Unknown suite: $suite" >&2; exit 1 ;;
    esac
    inputs=("$SRC_ROOT/$suite/"*.hdf5)
    [[ -f "${inputs[0]}" ]] || { echo "No source files for $suite" >&2; exit 1; }
done

for suite in "$@"; do
mkdir -p "$DST_ROOT/$suite"
rm -f "$DST_ROOT/$suite/VALIDATED_COMPLETE.txt"
"$REPO_ROOT/.venv/bin/python" -u scripts/build_libero_foreground_video.py \
    --src-root "$SRC_ROOT" --dst-root "$DST_ROOT" \
    --frames-dir "$FRAMES_DIR" --suites "$suite" --keep-going

"$REPO_ROOT/.venv/bin/python" -u scripts/validate_libero_foreground_video.py \
    "$DST_ROOT/$suite/"*.hdf5

date -u > "$DST_ROOT/$suite/VALIDATED_COMPLETE.txt"
done
