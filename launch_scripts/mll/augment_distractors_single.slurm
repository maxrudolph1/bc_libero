#!/bin/bash
# Slurm job: augment ONE LIBERO hdf5 file with a real-world video distractor.
#
# Submit via submit_distractors.py, or manually:
#   SRC_FILE=/datastor2/mrudolph/LIBERO/datasets/libero_goal/turn_on_the_stove_demo.hdf5 \
#     sbatch launch_scripts/mll/augment_distractors_single.slurm
#
# Override paths at submit time (required if compute nodes cannot see /datastor2):
#   SRC_ROOT=/u/mrudolph/LIBERO/datasets \
#   DST_ROOT=/u/mrudolph/LIBERO/datasets_distract \
#   SRC_FILE=/u/mrudolph/LIBERO/datasets/libero_goal/foo.hdf5 \
#     sbatch launch_scripts/mll/augment_distractors_single.slurm
#
#SBATCH --job-name=libero-distract
#SBATCH --output=/u/mrudolph/documents/BC-IB/slurm_jobs/distract/job_%j.out
#SBATCH --error=/u/mrudolph/documents/BC-IB/slurm_jobs/distract/job_%j.err
#SBATCH --partition=allnodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2:00:00

set -euo pipefail

REPO_ROOT=/u/mrudolph/documents/BC-IB
SRC_ROOT=${SRC_ROOT:-/datastor2/mrudolph/LIBERO/datasets}
DST_ROOT=${DST_ROOT:-/datastor2/mrudolph/LIBERO/datasets_distract}
DISTRACTOR_DIR=${DISTRACTOR_DIR:-/u/mrudolph/documents/BC-IB/epickitchens/P01/rgb_frames/p01_01}
SEED=${SEED:-0}

source ~/.bashrc
cd "${REPO_ROOT}"
source .venv/bin/activate

export TOKENIZERS_PARALLELISM=false
export OPENCV_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

if [[ -z "${SRC_FILE:-}" ]]; then
  echo "ERROR: SRC_FILE must be set to the source .hdf5 path" >&2
  exit 1
fi

if [[ ! -f "${SRC_FILE}" ]]; then
  echo "ERROR: source file not found on this node: ${SRC_FILE}" >&2
  echo "If /datastor2 is head-node only, stage datasets to /u/mrudolph and resubmit with SRC_ROOT/DST_ROOT/SRC_FILE overrides." >&2
  exit 1
fi

if [[ ! -d "${DISTRACTOR_DIR}" ]]; then
  echo "ERROR: distractor frames dir not found: ${DISTRACTOR_DIR}" >&2
  exit 1
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "Running on node: $(hostname)"
echo "Job started at $(date)"
echo "Augmenting: ${SRC_FILE}"

python scripts/augment_libero_distractors.py \
  --files "${SRC_FILE}" \
  --src-root "${SRC_ROOT}" \
  --dst-root "${DST_ROOT}" \
  --distractor-dir "${DISTRACTOR_DIR}" \
  --seed "${SEED}" \
  --overwrite

echo "Job ended at $(date)"
