#!/usr/bin/env python
"""Exercise foreground video pixels, resets, and both LIBERO vector workers."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libero.libero import benchmark
from libero_exp.utils.env_utils import make_libero_env
from libero_exp.utils.distractor_utils import DistractorAugmentor, DEFAULT_FRAMES_DIR


def validate(task, env_num, frames_dir, steps):
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    env, states = make_libero_env(suite, task, 128, env_num, 0, foreground_video=True)
    augmentor = DistractorAugmentor(frames_dir, layout="keep_foreground_video")
    reset_args = dict(env_num=env_num, global_seed=0, env_name="libero_goal",
                      task_idx=task, rollout_idx=0, max_horizon=steps + 1)
    try:
        env.reset()
        obs = env.set_init_state(states)
        augmentor.reset(**reset_args)
        starts = list(augmentor.start_indices)
        for t in range(steps + 1):
            if env_num == 1:
                # Independent renderer check: masks and observed RGB must share
                # the live state and image convention (including after reset).
                raw = env.workers[0].env.env
                from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
                import robosuite.macros as macros
                from libero_exp.utils.foreground_utils import background_geometry, foreground_mask
                convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
                rgb = raw.sim.render(camera_name="agentview", width=128, height=128)
                np.testing.assert_array_equal(obs[0]["agentview_image"], rgb[::convention])
                seg = raw.sim.render(camera_name="agentview", width=128, height=128,
                                     segmentation=True)[..., 1][::convention]
                ids, _ = background_geometry(raw.sim.model)
                np.testing.assert_array_equal(obs[0]["agentview_foreground_mask"],
                                              foreground_mask(seg, ids))
            originals = [{key: np.array(value, copy=True) for key, value in ob.items()}
                         for ob in obs]
            augmentor.apply_to_obs(obs)
            for slot, (ob, original) in enumerate(zip(obs, originals)):
                keep = original["agentview_foreground_mask"]
                assert keep.dtype == np.bool_ and 0 < keep.mean() < 1
                result = ob["agentview_image"]
                assert result.shape == (128, 128, 3) and result.dtype == np.uint8
                video = augmentor.source.load_frame(starts[slot] + t)
                if original["foreground_video_flip_vertical"]:
                    video = video[::-1]
                np.testing.assert_array_equal(result[keep], original["agentview_image"][keep])
                np.testing.assert_array_equal(result[~keep], video[~keep])
                for key in original:
                    if key != "agentview_image":
                        np.testing.assert_array_equal(ob[key], original[key])
            if t < steps:
                obs, _, _, _ = env.step(np.zeros((env_num, 7)))
        # Hard resets can randomize model properties not encoded by init_states.
        # Verify the video restarts and fresh foreground pixels remain intact.
        env.reset()
        obs = env.set_init_state(states)
        augmentor.reset(**reset_args)
        assert starts == augmentor.start_indices
        originals = [ob["agentview_image"].copy() for ob in obs]
        augmentor.apply_to_obs(obs)
        for slot, (ob, original) in enumerate(zip(obs, originals)):
            keep = ob["agentview_foreground_mask"]
            video = augmentor.source.load_frame(starts[slot])
            if ob["foreground_video_flip_vertical"]:
                video = video[::-1]
            np.testing.assert_array_equal(ob["agentview_image"][keep], original[keep])
            np.testing.assert_array_equal(ob["agentview_image"][~keep], video[~keep])
        print(f"VALIDATED task={task} env_num={env_num} frames={steps + 1} plus reset", flush=True)
        if task == 0:
            validate_rollout(env, states, env_num, frames_dir)
    finally:
        env.close()


def validate_rollout(env, states, env_num, frames_dir):
    """Exercise the real policy-observation/video rollout path without weights."""
    import torch
    from omegaconf import OmegaConf
    import robomimic.utils.obs_utils as ObsUtils
    from libero_exp.utils.results_utils import rollout

    data = OmegaConf.load(Path(__file__).resolve().parents[1] /
                          "libero_exp/configs/base/data/default.yaml")
    data.distractor.enable = True
    data.distractor.layout = "keep_foreground_video"
    data.distractor.frames_dir = frames_dir
    cfg = OmegaConf.create(dict(data=data, env=dict(env_num=env_num, env_name="libero_goal"),
                                train=dict(seed=0, device="cpu")))
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": OmegaConf.to_container(data.obs.modality)})

    class ProbePolicy:
        calls = 0

        def eval(self):
            pass

        def reset(self):
            pass

        def get_action(self, cfg, data):
            assert data["obs"]["agentview_rgb"].shape == (env_num, 3, 128, 128)
            assert data["obs"]["eye_in_hand_rgb"].shape == (env_num, 3, 128, 128)
            self.calls += 1
            return np.zeros((env_num, 7))

    policy = ProbePolicy()
    result = rollout(cfg, {"libero_goal/probe": (0, 0, env, states, torch.zeros(1, 768))},
                     policy, 1, horizon=2, return_wandb_video=False, device=torch.device("cpu"))
    assert policy.calls == 2
    assert result["rollout/vis_env0_task0"].shape == (env_num, 2, 3, 128, 128)
    print(f"VALIDATED policy rollout env_num={env_num}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--env-nums", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--frames-dir", default=DEFAULT_FRAMES_DIR)
    args = parser.parse_args()
    # Exercise subprocess creation after the parent has used EGL as well.
    for env_num in args.env_nums:
        for task in args.tasks:
            validate(task, env_num, args.frames_dir, args.steps)
