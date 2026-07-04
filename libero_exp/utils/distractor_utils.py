"""Shared real-world video distractor utilities for LIBERO datasets and eval."""

from __future__ import annotations

import glob
import hashlib
import os
from typing import List, Sequence

import cv2
import numpy as np

TARGET_H = 128
TARGET_W = 128
LAYOUT = "hconcat_right"
DEFAULT_FRAMES_DIR = (
    "/u/mrudolph/documents/BC-IB/epickitchens/P01/rgb_frames/p01_01"
)


def demo_seed(suite: str, task_file: str, demo_id: str, global_seed: int) -> int:
    """Deterministic per-demo seed for offline dataset augmentation."""
    key = f"{global_seed}|{suite}|{task_file}|{demo_id}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def rollout_env_seed(
    global_seed: int,
    env_name: str,
    task_idx: int,
    rollout_idx: int,
    env_slot: int,
) -> int:
    """Deterministic seed for one parallel env slot in eval rollouts."""
    key = f"{global_seed}|{env_name}|{task_idx}|{rollout_idx}|{env_slot}".encode(
        "utf-8"
    )
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def append_distractor(robot_hwc: np.ndarray, distractor_hwc: np.ndarray) -> np.ndarray:
    """Horizontally concat robot view (left) and distractor (right)."""
    return np.ascontiguousarray(
        np.concatenate([robot_hwc, distractor_hwc], axis=1), dtype=np.uint8
    )


class DistractorSource:
    """Lazy accessor over sorted EPIC-Kitchens JPG frames."""

    def __init__(self, frames_dir: str):
        self.frames_dir = frames_dir
        self.frame_paths: List[str] = sorted(
            glob.glob(os.path.join(frames_dir, "*.jpg"))
        )
        if not self.frame_paths:
            raise FileNotFoundError(f"No .jpg frames found in {frames_dir}")
        self.num_frames = len(self.frame_paths)

    def sample_start(self, rng: np.random.RandomState, length: int) -> int:
        if length >= self.num_frames:
            return 0
        return int(rng.randint(0, self.num_frames - length + 1))

    def load_frame(self, index: int) -> np.ndarray:
        """Return one ``(TARGET_H, TARGET_W, 3)`` uint8 RGB frame."""
        idx = int(index) % self.num_frames
        bgr = cv2.imread(self.frame_paths[idx], cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Failed to read {self.frame_paths[idx]}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

    def load_window(self, start: int, length: int) -> np.ndarray:
        """Return ``(length, TARGET_H, TARGET_W, 3)`` uint8 RGB frames."""
        out = np.empty((length, TARGET_H, TARGET_W, 3), dtype=np.uint8)
        for t in range(length):
            out[t] = self.load_frame(start + t)
        return out


class DistractorAugmentor:
    """Runtime augmentor for eval rollouts (one moving clip per episode/env slot)."""

    def __init__(self, frames_dir: str):
        self.source = DistractorSource(frames_dir)
        self.start_indices: List[int] = []
        self.step = 0

    def reset(
        self,
        env_num: int,
        *,
        global_seed: int,
        env_name: str,
        task_idx: int,
        rollout_idx: int,
        max_horizon: int,
    ) -> None:
        self.start_indices = []
        for env_slot in range(env_num):
            seed = rollout_env_seed(
                global_seed, env_name, task_idx, rollout_idx, env_slot
            )
            rng = np.random.RandomState(seed)
            self.start_indices.append(self.source.sample_start(rng, max_horizon))
        self.step = 0

    def apply_to_obs(
        self,
        obs: Sequence[dict],
        obs_key: str = "agentview_image",
        advance: bool = True,
    ) -> Sequence[dict]:
        for env_slot, ob in enumerate(obs):
            frame_idx = self.start_indices[env_slot] + self.step
            distract = self.source.load_frame(frame_idx)
            ob[obs_key] = append_distractor(ob[obs_key], distract)
        if advance:
            self.step += 1
        return obs


def distractor_enabled(cfg) -> bool:
    distractor = getattr(cfg.data, "distractor", None)
    return bool(getattr(distractor, "enable", False))


def validate_distractor_shape_meta(cfg, shape_meta) -> None:
    if not distractor_enabled(cfg):
        return
    expected = (3, 128, 256)
    actual = tuple(shape_meta["all_shapes"]["agentview_rgb"])
    if actual != expected:
        raise ValueError(
            f"distractor enabled but agentview_rgb shape is {actual}, expected "
            f"{expected}. Set data.root_dir to datasets_distract."
        )
