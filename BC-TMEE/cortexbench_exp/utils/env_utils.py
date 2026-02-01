#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import gc
import gym
from gym.spaces.box import Box
import numpy as np
import time as timer
from tqdm import tqdm
from typing import Union
import multiprocessing as mp
from collections import namedtuple
from mjrl.utils import tensor_utils
from mjrl.utils.gym_env import GymEnv
from mujoco_vc.supported_envs import ENV_TO_SUITE
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


def build_env(config):
    env = env_constructor(
            suite=config.env.suite,
            task_name=config.env.task_name,
            pixel_based=config.env.pixel_based,
            image_height=config.env.image_height,
            image_width=config.env.image_width,
            camera_name=config.env.camera_name,
            add_proprio=config.env.add_proprio,
    )
    return env


def env_constructor(
    suite: str,
    task_name: str,
    pixel_based: bool = True,
    image_width: int = 256,
    image_height: int = 256,
    camera_name: str = None,
    seed: int = 123,
    add_proprio: bool = False,
) -> GymEnv:
    # construct basic gym environment
    assert task_name in ENV_TO_SUITE.keys()
    suite = ENV_TO_SUITE[task_name]
    if suite == "metaworld":
        # Meta world natively misses many specs. We will explicitly add them here.
        e = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]()
        e._freeze_rand_vec = False
        e.spec = namedtuple("spec", ["id", "max_episode_steps"])
        e.spec.id = task_name
        e.spec.max_episode_steps = 500
    else:
        e = gym.make(task_name)
    # seed the environment for reproducibility
    e.seed(seed)

    # get correct camera name
    camera_name = (None if (camera_name == "None" or camera_name == "default") else camera_name)
    # Use appropriate observation wrapper
    if pixel_based:
        e = MuJoCoPixelObsWrapper(
            suite=suite,
            env=e,
            width=image_width,
            height=image_height,
            camera_name=camera_name,
            device_id=0,
            add_proprio=add_proprio,
        )
        e = GymEnv(e)
    else:
        e = GymEnv(e)

    # Output wrapped env
    e.set_seed(seed)
    return e


class MuJoCoPixelObsWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        suite,
        env,
        width,
        height,
        camera_name,
        device_id=-1,
        depth=False,
        add_proprio=False,
    ):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=0.0, high=255.0, shape=(3, width, height))
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.depth = depth
        self.device_id = device_id

        # proprioception
        if add_proprio:
            self.get_proprio = lambda: get_proprioception(self.unwrapped, suite)
            proprio = self.get_proprio()
            self.proprio_dim = 0 if proprio is None else proprio.shape[0]
        else:
            self.proprio_dim = 0
            self.get_proprio = None
        
    def get_image(self):
        if self.spec.id.startswith("dmc"):
            # dmc backend
            # dmc expects camera_id as an integer and not name
            if self.camera_name == None or self.camera_name == "None":
                self.camera_name = 0
            img = self.env.unwrapped.render(
                mode="rgb_array",
                width=self.width,
                height=self.height,
                camera_id=int(self.camera_name),
            )
        else:
            # mujoco-py backend
            img = self.sim.render(
                width=self.width,
                height=self.height,
                depth=self.depth,
                camera_name=self.camera_name,
                device_id=self.device_id,
            )
            img = img[::-1, :, :]
        return img
    
    def get_proprio(self):
        proprio = self.get_proprio()
        return proprio

    def observation(self, observation):
        # This function creates observations based on the current state of the environment.
        # Argument `observation` is ignored, but `gym.ObservationWrapper` requires it.
        # Output format is (H, W, 3)
        if self.proprio_dim > 0:    # add proprioception if necessary
            return self.get_image(), self.get_proprio()
        return self.get_image()
    
    def get_obs(self):
        return self.observation(None)


def get_proprioception(env: gym.Env, suite: str) -> Union[np.ndarray, None]:
    assert isinstance(env, gym.Env)
    if suite == "metaworld":
        return env.unwrapped._get_obs()[:4]
    elif suite == "adroit":
        # In adroit, in-hand tasks like pen lock the base of the hand
        # while other tasks like relocate allow for movement of hand base
        # as if attached to an arm
        if env.unwrapped.spec.id == "pen-v0":
            return env.unwrapped.get_obs()[:24]
        elif env.unwrapped.spec.id == "relocate-v0":
            return env.unwrapped.get_obs()[:30]
        else:
            print("Unsupported environment. Proprioception is defaulting to None.")
            return None
    elif suite == "dmc":
        # no proprioception used for dm-control
        return None
    else:
        print("Unsupported environment. Proprioception is defaulting to None.")
        return None


# Single core rollout to sample trajectories
# =======================================================
def do_rollout(
        num_traj,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        base_seed = None,
        device_id = 0,
        add_proprio = False,
        env_kwargs = None,
):
    """
    :param num_traj:    number of trajectories (int)
    :param env:         environment (env class, str with env_name, or factory function)
    :param policy:      policy to use for action selection
    :param eval_mode:   use evaluation mode for action computation (bool)
    :param horizon:     max horizon length for rollout (<= env.horizon)
    :param base_seed:   base seed for rollouts (int)
    :param env_kwargs:  dictionary with parameters, will be passed to env generator
    :return:
    """

    # get the correct env behavior
    if type(env) == str:
        env = GymEnv(env)
        if env_kwargs and 'rrl_kwargs' in env_kwargs:
            from rrl.multicam import RRL
            env = RRL(env, **env_kwargs['rrl_kwargs'], device_id=device_id)
    elif isinstance(env, GymEnv):
        env = env
    elif callable(env):
        env = env(**env_kwargs)
    else:
        print("Unsupported environment format")
        raise AttributeError

    horizon = min(horizon, env.horizon)
    paths = []
    for ep in tqdm(range(num_traj)):
        # seeding
        if base_seed is not None:
            seed = base_seed + ep
            env.set_seed(seed)
            np.random.seed(seed)

        observations=[]
        actions=[]
        rewards=[]
        # agent_infos = []
        env_infos = []

        policy.reset()
        o = env.reset()
        done = False
        t = 0

        while t < horizon and done != True:
            a = policy.get_action(o)
            # a, agent_info = policy.get_action(o)
            # if eval_mode:
            #     a = agent_info['evaluation']
            env_info_base = env.get_env_infos()
            next_o, r, done, env_info_step = env.step(a)
            # below is important to ensure correct env_infos for the timestep
            env_info = env_info_step if env_info_base == {} else env_info_base
            if add_proprio:
                observations.append(o[0])
            else:
                observations.append(o)
            actions.append(a)
            rewards.append(r)
            # agent_infos.append(agent_info)
            env_infos.append(env_info)
            o = next_o
            t += 1

        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            # agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            terminated=done
        )
        paths.append(path)

    del(env)
    gc.collect()
    return paths


def sample_paths(
        num_traj,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        base_seed = None,
        num_cpu = 1,
        max_process_time = 3000,
        max_timeouts = 4,
        suppress_print = False,
        add_proprio = False,
        env_kwargs = None,
        ):
    num_cpu = 1 if num_cpu is None else num_cpu
    num_cpu = mp.cpu_count() if num_cpu == 'max' else num_cpu
    assert type(num_cpu) == int

    if num_cpu == 1:
        input_dict = dict(num_traj=num_traj, env=env, policy=policy, add_proprio=add_proprio,
                          eval_mode=eval_mode, horizon=horizon, base_seed=base_seed,
                          env_kwargs=env_kwargs)
        # dont invoke multiprocessing if not necessary
        return do_rollout(**input_dict)

    # do multiprocessing otherwise
    paths_per_cpu = int(np.ceil(num_traj / num_cpu))
    input_dict_list= []
    for i in range(num_cpu):
        input_dict = dict(num_traj=paths_per_cpu, env=env, policy=policy,
                          eval_mode=eval_mode, horizon=horizon,
                          base_seed=base_seed + i * paths_per_cpu,
                          env_kwargs=env_kwargs, device_id=i)
        input_dict_list.append(input_dict)
    if suppress_print is False:
        start_time = timer.time()
        print("####### Gathering Samples #######")

    results = _try_multiprocess(do_rollout, input_dict_list,
                                num_cpu, max_process_time, max_timeouts)
    paths = []
    # result is a paths type and results is list of paths
    for result in results:
        for path in result:
            paths.append(path)  

    if suppress_print is False:
        print("======= Samples Gathered  ======= | >>>> Time taken = %f " %(timer.time()-start_time) )

    return paths


def sample_data_batch(
        num_samples,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        base_seed = None,
        num_cpu = 1,
        paths_per_call = 1,
        env_kwargs=None,
        ):
    num_cpu = 1 if num_cpu is None else num_cpu
    num_cpu = mp.cpu_count() if num_cpu == 'max' else num_cpu
    assert type(num_cpu) == int

    start_time = timer.time()
    print("####### Gathering Samples #######")
    sampled_so_far = 0
    paths_so_far = 0
    paths = []
    base_seed = 123 if base_seed is None else base_seed
    while sampled_so_far < num_samples:
        base_seed = base_seed + 12345
        new_paths = sample_paths(paths_per_call * num_cpu, env, policy,
                                 eval_mode, horizon, base_seed, num_cpu,
                                 suppress_print=True, env_kwargs=env_kwargs)
        for path in new_paths:
            paths.append(path)
        paths_so_far += len(new_paths)
        new_samples = np.sum([len(p['rewards']) for p in new_paths])
        sampled_so_far += new_samples
    print("======= Samples Gathered  ======= | >>>> Time taken = %f " % (timer.time() - start_time))
    print("................................. | >>>> # samples = %i # trajectories = %i " % (
    sampled_so_far, paths_so_far))
    return paths


def _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts):
    # Base case
    if max_timeouts == 0:
        return None

    pool = mp.Pool(processes=num_cpu, maxtasksperchild=None)
    parallel_runs = [pool.apply_async(func, kwds=input_dict) for input_dict in input_dict_list]
    try:
        results = [p.get(timeout=max_process_time) for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("Timeout Error raised... Trying again")
        pool.close()
        pool.terminate()
        pool.join()
        return _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts-1)

    pool.close()
    pool.terminate()
    pool.join()  
    return results
