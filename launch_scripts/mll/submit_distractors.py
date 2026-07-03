#!/usr/bin/env python3
"""Submit one Slurm job per LIBERO hdf5 augmentation.

Each job runs launch_scripts/mll/augment_distractors_single.slurm with SRC_FILE
set to a single source dataset. This avoids array-index mapping and makes every
augmentation an independent job on its own compute node.

Examples:
    python launch_scripts/mll/submit_distractors.py --dry-run
    python launch_scripts/mll/submit_distractors.py
    python launch_scripts/mll/submit_distractors.py --skip-existing
    python launch_scripts/mll/submit_distractors.py --suites libero_goal
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path("/u/mrudolph/documents/BC-IB")
SLURM_SCRIPT = REPO_ROOT / "launch_scripts/mll/augment_distractors_single.slurm"
DEFAULT_SRC_ROOT = "/datastor2/mrudolph/LIBERO/datasets"
DEFAULT_DST_ROOT = "/datastor2/mrudolph/LIBERO/datasets_distract"
DEFAULT_DISTRACTOR_DIR = (
    "/u/mrudolph/documents/BC-IB/epickitchens/P01/rgb_frames/p01_01"
)
DEFAULT_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def discover_files(src_root: str, suites: List[str]) -> List[str]:
    files: List[str] = []
    for suite in suites:
        pattern = os.path.join(src_root, suite, "*.hdf5")
        files.extend(sorted(glob.glob(pattern)))
    return files


def output_path(src_file: str, src_root: str, dst_root: str) -> str:
    rel = os.path.relpath(src_file, src_root)
    return os.path.join(dst_root, rel)


def submit_one(
    src_file: str,
    *,
    src_root: str,
    dst_root: str,
    distractor_dir: str,
    seed: int,
    dry_run: bool,
) -> Optional[int]:
    exports = [
        f"SRC_FILE={src_file}",
        f"SRC_ROOT={src_root}",
        f"DST_ROOT={dst_root}",
        f"DISTRACTOR_DIR={distractor_dir}",
        f"SEED={seed}",
    ]
    cmd = ["sbatch", f"--export={','.join(exports)}", str(SLURM_SCRIPT)]

    if dry_run:
        print(" ".join(cmd))
        return None

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed for {src_file}: {result.stderr.strip()}")
    return int(result.stdout.strip().split()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", default=DEFAULT_SRC_ROOT)
    parser.add_argument("--dst-root", default=DEFAULT_DST_ROOT)
    parser.add_argument("--distractor-dir", default=DEFAULT_DISTRACTOR_DIR)
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not submit jobs whose output .hdf5 already exists.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = discover_files(args.src_root, args.suites)
    if not files:
        raise SystemExit(
            f"No .hdf5 files found under {args.src_root} for suites {args.suites}"
        )

    submitted = 0
    skipped = 0
    job_ids: List[int] = []

    for src_file in files:
        dst_file = output_path(src_file, args.src_root, args.dst_root)
        if args.skip_existing and os.path.isfile(dst_file):
            print(f"skip (exists): {dst_file}")
            skipped += 1
            continue

        rel = os.path.relpath(src_file, args.src_root)
        print(f"submit: {rel}")
        jid = submit_one(
            src_file,
            src_root=args.src_root,
            dst_root=args.dst_root,
            distractor_dir=args.distractor_dir,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        submitted += 1
        if jid is not None:
            job_ids.append(jid)

    action = "Would submit" if args.dry_run else "Submitted"
    print(f"{action} {submitted} job(s), skipped {skipped}.")
    if job_ids:
        print(f"Job IDs: {', '.join(str(j) for j in job_ids)}")


if __name__ == "__main__":
    main()
