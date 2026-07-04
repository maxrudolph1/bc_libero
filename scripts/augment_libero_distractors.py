#!/usr/bin/env python
"""Augment LIBERO demonstration datasets with real-world video distractors.

For each demo, the third-person ``obs/agentview_rgb`` image is widened from
``(T, 128, 128, 3)`` to ``(T, 128, 256, 3)`` by appending a real-world
EPIC-Kitchens frame (resized to 128x128) on the right-hand side. The distractor
advances one video frame per environment timestep (a moving clip), and each demo
draws a different random contiguous window from the source video.

Everything else in the HDF5 file (actions, states, robot_states, rewards, dones,
the wrist ``eye_in_hand_rgb`` camera, proprio, and all group/dataset attributes
such as ``env_args``/``model_file``/``init_state``/``num_samples``) is copied
verbatim from the source.

Only the project venv should run this:
    /scratch/cluster/mrudolph/documents/BC-IB/.venv/bin/python
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from libero_exp.utils.distractor_utils import (  # noqa: E402
    LAYOUT,
    TARGET_H,
    TARGET_W,
    DistractorSource,
    append_distractor,
    demo_seed,
)

DEFAULT_SRC_ROOT = "/datastor2/mrudolph/LIBERO/datasets"
DEFAULT_DST_ROOT = "/datastor2/mrudolph/LIBERO/datasets_distract"
DEFAULT_DISTRACTOR_DIR = (
    "/u/mrudolph/documents/BC-IB/epickitchens/P01/rgb_frames/p01_01"
)
DEFAULT_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
GLOBAL_SEED = 0


def augment_file(
    src_path: str,
    dst_path: str,
    suite: str,
    distractor: DistractorSource,
    global_seed: int,
    manifest_path: str,
    overwrite: bool = False,
) -> dict:
    """Create ``dst_path`` from ``src_path`` with distractor-augmented agentview."""
    task_file = os.path.basename(src_path)
    if os.path.exists(dst_path) and not overwrite:
        raise FileExistsError(f"{dst_path} exists (use --overwrite to replace)")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = dst_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    manifest = {
        "source_hdf5": src_path,
        "output_hdf5": dst_path,
        "suite": suite,
        "task_file": task_file,
        "distractor_video": distractor.frames_dir,
        "distractor_num_frames": distractor.num_frames,
        "global_seed": global_seed,
        "layout": LAYOUT,
        "out_agentview_shape": [TARGET_H, 2 * TARGET_W, 3],
        "resize_interpolation": "INTER_AREA",
        "color_conversion": "BGR2RGB",
        "temporal": "one_distractor_frame_per_timestep",
        "modified_keys": ["obs/agentview_rgb"],
        "demos": {},
    }

    with h5py.File(src_path, "r") as src, h5py.File(tmp_path, "w") as dst:
        src.copy("data", dst)
        for attr_k, attr_v in src.attrs.items():
            dst.attrs[attr_k] = attr_v
        for attr_k, attr_v in src["data"].attrs.items():
            dst["data"].attrs[attr_k] = attr_v

        demo_ids = sorted(src["data"].keys(), key=lambda d: int(d.split("_")[-1]))

        for demo_id in demo_ids:
            src_agent = src[f"data/{demo_id}/obs/agentview_rgb"]
            T = src_agent.shape[0]
            robot = src_agent[()]

            rng = np.random.RandomState(demo_seed(suite, task_file, demo_id, global_seed))
            start = distractor.sample_start(rng, T)
            distract = distractor.load_window(start, T)

            widened = np.stack(
                [append_distractor(robot[t], distract[t]) for t in range(T)], axis=0
            )

            obs_grp = dst[f"data/{demo_id}/obs"]
            del obs_grp["agentview_rgb"]
            obs_grp.create_dataset(
                "agentview_rgb",
                data=widened,
                dtype="uint8",
                chunks=(1, TARGET_H, 2 * TARGET_W, 3),
                compression="gzip",
                compression_opts=4,
            )

            manifest["demos"][demo_id] = {
                "T": int(T),
                "distractor_start_frame": int(start),
                "distractor_end_frame": int(start + T - 1),
                "wrapped": bool(T > distractor.num_frames),
            }

    os.replace(tmp_path, dst_path)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", default=DEFAULT_SRC_ROOT)
    parser.add_argument("--dst-root", default=DEFAULT_DST_ROOT)
    parser.add_argument("--distractor-dir", default=DEFAULT_DISTRACTOR_DIR)
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Optional explicit list of source .hdf5 paths (overrides suites).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many files (for dry runs).",
    )
    args = parser.parse_args()

    distractor = DistractorSource(args.distractor_dir)
    print(
        f"Distractor source: {args.distractor_dir} "
        f"({distractor.num_frames} frames)"
    )

    if args.files:
        jobs = []
        for p in args.files:
            suite = os.path.basename(os.path.dirname(p))
            jobs.append((suite, p))
    else:
        jobs = []
        for suite in args.suites:
            suite_dir = os.path.join(args.src_root, suite)
            for p in sorted(glob.glob(os.path.join(suite_dir, "*.hdf5"))):
                jobs.append((suite, p))

    if args.limit is not None:
        jobs = jobs[: args.limit]

    print(f"Processing {len(jobs)} file(s).")
    for i, (suite, src_path) in enumerate(jobs, 1):
        task_file = os.path.basename(src_path)
        dst_path = os.path.join(args.dst_root, suite, task_file)
        manifest_path = os.path.join(
            args.dst_root, suite, task_file.replace(".hdf5", ".manifest.json")
        )
        print(f"[{i}/{len(jobs)}] {suite}/{task_file} -> {dst_path}")
        m = augment_file(
            src_path=src_path,
            dst_path=dst_path,
            suite=suite,
            distractor=distractor,
            global_seed=args.seed,
            manifest_path=manifest_path,
            overwrite=args.overwrite,
        )
        print(f"    demos: {len(m['demos'])}, manifest: {manifest_path}")

    print("Done.")


if __name__ == "__main__":
    main()
