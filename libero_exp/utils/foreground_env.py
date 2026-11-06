"""Live geometry masks, computed inside each LIBERO vector-env worker."""

import numpy as np
import robosuite.macros as macros
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

from .foreground_utils import background_geometry, foreground_mask

MASK_KEY = "agentview_foreground_mask"
FLIP_KEY = "foreground_video_flip_vertical"


class ForegroundMaskEnv:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _observe(self, obs):
        rgb = obs["agentview_image"]
        if rgb.shape != (128, 128, 3):
            raise ValueError("Foreground video requires 128x128 agentview RGB")
        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        # Camera observables can be cached from an earlier physics substep.
        # Render RGB and segmentation together at the returned simulator state
        # so a moving robot cannot be paired with a displaced mask.
        obs["agentview_image"] = np.ascontiguousarray(self.env.sim.render(
            camera_name="agentview", width=128, height=128
        )[::convention])
        # Resolve IDs again after resets, which may rebuild the MuJoCo model.
        ids, _ = background_geometry(self.env.sim.model)
        segmentation = self.env.sim.render(
            camera_name="agentview", width=128, height=128, segmentation=True
        )[..., 1]
        obs[MASK_KEY] = foreground_mask(segmentation[::convention], ids)
        obs[FLIP_KEY] = convention == 1
        return obs

    def reset(self, **kwargs):
        return self._observe(self.env.reset(**kwargs))

    def set_init_state(self, state):
        return self._observe(self.env.set_init_state(state))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observe(obs), reward, done, info
