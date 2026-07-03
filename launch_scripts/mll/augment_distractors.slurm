#!/bin/bash
# Slurm ARRAY job: one array task augments ONE LIBERO hdf5 file with a
# real-world video distractor (see scripts/augment_libero_distractors.py).
#
# Each array index maps to one source .hdf5 (sorted, deterministic), so every
# augmentation runs on its own compute node instead of the head node.
#
# IMPORTANT: compute nodes do NOT mount /datastor2 (it is head-node only). Before
# submitting, stage the source datasets and choose an output dir on a filesystem
# the compute nodes can see (e.g. home /u/mrudolph, ~1.2TB free; /scratch is only
# ~40GB free). Then submit with matching overrides, for example:
#     rsync -a /datastor2/mrudolph/LIBERO/datasets/ /u/mrudolph/LIBERO/datasets/
#     SRC_ROOT=/u/mrudolph/LIBERO/datasets \
#     DST_ROOT=/u/mrudolph/LIBERO/datasets_distract \
#     sbatch --array=0-39 launch_scripts/mll/augment_distractors.slurm
# and afterwards optionally copy outputs back to /datastor2 from the head node.
#
# Submit all 40 files (libero_spatial + object + goal + 10 = indices 0-39):
#     sbatch --array=0-39 launch_scripts/mll/augment_distractors.slurm
#
# Submit a subset (e.g. just the first suite's 10 files):
#     sbatch --array=0-9 launch_scripts/mll/augment_distractors.slurm
#
# Throttle concurrency (e.g. at most 8 at once, to spare the shared disk):
#     sbatch --array=0-39%8 launch_scripts/mll/augment_distractors.slurm
#
# Override suites / paths / seed via env vars at submit time, e.g.:
#     SUITES="libero_goal" sbatch --array=0-9 launch_scripts/mll/augment_distractors.slurm
#
#SBATCH --job-name=libero-distract
#SBATCH --output=/u/mrudolph/documents/BC-IB/slurm_jobs/distract/job_%A_%a.out
#SBATCH --error=/u/mrudolph/documents/BC-IB/slurm_jobs/distract/job_%A_%a.err
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
# Space-separated suite list; order defines the array index -> file mapping.
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}

source ~/.bashrc
cd "${REPO_ROOT}"
source .venv/bin/activate

export TOKENIZERS_PARALLELISM=false
export OPENCV_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: run as an array job, e.g. 'sbatch --array=0-39 $0'" >&2
  exit 1
fi

# Build the deterministic, sorted file list across the requested suites.
FILES=()
for suite in ${SUITES}; do
  while IFS= read -r f; do
    FILES+=("${f}")
  done < <(ls "${SRC_ROOT}/${suite}"/*.hdf5 2>/dev/null | sort)
done

NUM_FILES=${#FILES[@]}
echo "Discovered ${NUM_FILES} source files across suites: ${SUITES}"

if (( SLURM_ARRAY_TASK_ID >= NUM_FILES )); then
  echo "ERROR: array index ${SLURM_ARRAY_TASK_ID} >= file count ${NUM_FILES}" >&2
  exit 1
fi

SRC_FILE=${FILES[${SLURM_ARRAY_TASK_ID}]}

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Running on node: $(hostname)"
echo "Job started at $(date)"
echo "Augmenting file [${SLURM_ARRAY_TASK_ID}/${NUM_FILES}]: ${SRC_FILE}"

python scripts/augment_libero_distractors.py \
  --files "${SRC_FILE}" \
  --src-root "${SRC_ROOT}" \
  --dst-root "${DST_ROOT}" \
  --distractor-dir "${DISTRACTOR_DIR}" \
  --seed "${SEED}" \
  --overwrite

echo "Job ended at $(date)"
