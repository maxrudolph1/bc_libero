#!/bin/bash
# Resumable goal-suite conversion followed by an exhaustive pixel/data check.
# Submit from the repository with:
# sbatch --export=ALL,REPO_ROOT="$PWD" launch_scripts/mll/build_foreground_video.slurm
# Select a node that can access SRC_ROOT, DST_ROOT, and the video frames.
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

"$REPO_ROOT/.venv/bin/python" -u scripts/build_libero_foreground_video.py \
    --src-root "$SRC_ROOT" --dst-root "$DST_ROOT" \
    --frames-dir "$FRAMES_DIR" --suites libero_goal --keep-going

"$REPO_ROOT/.venv/bin/python" -u scripts/validate_libero_foreground_video.py \
    "$DST_ROOT"/libero_goal/*.hdf5

date -u > "$DST_ROOT/VALIDATED_COMPLETE.txt"
