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
import hashlib
import json
import os
from typing import List

import cv2
import h5py
import numpy as np

DEFAULT_SRC_ROOT = "/datastor2/mrudolph/LIBERO/datasets"
DEFAULT_DST_ROOT = "/datastor2/mrudolph/LIBERO/datasets_distract"
DEFAULT_DISTRACTOR_DIR = (
    "/u/mrudolph/documents/BC-IB/epickitchens/P01/rgb_frames/p01_01"
)
DEFAULT_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

TARGET_H = 128  # robot agentview height (unchanged)
TARGET_W = 128  # distractor width after resize (appended on the right)
GLOBAL_SEED = 0
LAYOUT = "hconcat_right"  # [robot | distractor] concatenated along width


def _demo_seed(suite: str, task_file: str, demo_id: str, global_seed: int) -> int:
    """Deterministic per-demo seed so runs are reproducible."""
    key = f"{global_seed}|{suite}|{task_file}|{demo_id}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


class DistractorSource:
    """Lazy accessor over the sorted list of EPIC-Kitchens JPG frames."""

    def __init__(self, frames_dir: str):
        self.frames_dir = frames_dir
        self.frame_paths: List[str] = sorted(
            glob.glob(os.path.join(frames_dir, "*.jpg"))
        )
        if not self.frame_paths:
            raise FileNotFoundError(f"No .jpg frames found in {frames_dir}")
        self.num_frames = len(self.frame_paths)

    def sample_start(self, rng: np.random.RandomState, length: int) -> int:
        """Pick a random start so ``[start, start+length)`` fits in the video.

        If the demo is longer than the whole video (never happens in practice),
        start at 0 and let ``load_window`` wrap around.
        """
        if length >= self.num_frames:
            return 0
        return int(rng.randint(0, self.num_frames - length + 1))

    def load_window(self, start: int, length: int) -> np.ndarray:
        """Return ``(length, TARGET_H, TARGET_W, 3)`` uint8 RGB frames.

        Frames are read with OpenCV (BGR), converted to RGB, and resized to
        ``TARGET_H x TARGET_W`` with area interpolation (best for downscaling).
        Indices wrap modulo the video length as a safety net.
        """
        out = np.empty((length, TARGET_H, TARGET_W, 3), dtype=np.uint8)
        for t in range(length):
            idx = (start + t) % self.num_frames
            bgr = cv2.imread(self.frame_paths[idx], cv2.IMREAD_COLOR)
            if bgr is None:
                raise IOError(f"Failed to read {self.frame_paths[idx]}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(
                rgb, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA
            )
            out[t] = resized
        return out


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
        # Copy the entire "data" group verbatim (datasets + group/root attrs).
        src.copy("data", dst)
        for attr_k, attr_v in src.attrs.items():
            dst.attrs[attr_k] = attr_v
        for attr_k, attr_v in src["data"].attrs.items():
            dst["data"].attrs[attr_k] = attr_v

        demo_ids = list(src["data"].keys())
        # Sort demos numerically (demo_0, demo_1, ...) for stable manifests.
        demo_ids.sort(key=lambda d: int(d.split("_")[-1]))

        for demo_id in demo_ids:
            src_agent = src[f"data/{demo_id}/obs/agentview_rgb"]
            T = src_agent.shape[0]
            robot = src_agent[()]  # (T, 128, 128, 3) uint8, unchanged

            rng = np.random.RandomState(
                _demo_seed(suite, task_file, demo_id, global_seed)
            )
            start = distractor.sample_start(rng, T)
            distract = distractor.load_window(start, T)  # (T, 128, 128, 3)

            widened = np.concatenate([robot, distract], axis=2)  # (T,128,256,3)
            widened = np.ascontiguousarray(widened, dtype=np.uint8)

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
        "--files", nargs="+", default=None,
        help="Optional explicit list of source .hdf5 paths (overrides suites).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=None,
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
