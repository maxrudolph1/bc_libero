
#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

"""
Run MoveCubePolicy to generate cube re-posititioning demos
"""

import torch
import numpy as np
import trifinger_vc.utils.data_utils as d_utils
from trifinger_vc.trifinger_envs.action import ActionType
from trifinger_vc.trifinger_envs.gym_cube_env import MoveCubeEnv
from trifinger_vc.trifinger_envs.cube_reach import CubeReachEnv
from trifinger_simulation.trifinger_platform import ObjectType

"""
Class to execute sequence of actions. Includes instance of the environment and the policy.
The main function is execute_policy which rolls out an episode using the policy and returns a dictionary containing the trajectory.
"""


def build_env(config, traj_info):
    all_sim_dict = {
        "sim_env_demo": {
            "enable_shadows": False,
            "camera_view": "default",
            "arena_color": "default",
        },
        "sim_env_real": {
            "enable_shadows": True,
            "camera_view": "real",
            "arena_color": "real",
        },
        "sim_env_shadows": {
            "enable_shadows": True,
            "camera_view": "default",
            "arena_color": "default",
        },
        "sim_env_real_camera_view": {
            "enable_shadows": False,
            "camera_view": "real",
            "arena_color": "default",
        },
        "sim_env_real_arena_color": {
            "enable_shadows": False,
            "camera_view": "default",
            "arena_color": "real",
        },
    }
    
    sim_dict = {}
    for env_name in config.env.eval_envs:
        sim_dict[env_name] = all_sim_dict[env_name]
    
    env_dict = {}
    for sim_env_name, sim_params in sim_dict.items():
        sim = Task(
            downsample_time_step=traj_info["downsample_time_step"],
            traj_scale=traj_info["scale"],
            goal_type=config.data.goal_type,
            object_type=traj_info["object_type"],
            finger_type=traj_info["finger_type"],
            enable_shadows=sim_params["enable_shadows"],
            camera_view=sim_params["camera_view"],
            arena_color=sim_params["arena_color"],
            task=config.env.task_name,
            n_fingers_to_move=config.data.action_dim // 3,  # config.env.proprio_dim // 3,
            device=config.train.device,
        )
        env_dict[f"{sim_env_name}"] = sim

    return env_dict


class Task:
    def __init__(
        self,
        downsample_time_step=0.2,
        traj_scale=1,
        goal_type=None,
        object_type="colored_cube",
        finger_type="trifingerpro",
        goal_visualization=False,
        enable_shadows=False,
        camera_view="default",
        arena_color="default",
        task="move",
        n_fingers_to_move=3,
        device='cuda',
    ):
        if task == "reach":    
            assert (goal_type == "goal_none"), f"Need to use algo.goal_type=goal_none when running {task} task"

        self.sim_time_step = 0.004
        self.downsample_time_step = downsample_time_step
        self.traj_scale = traj_scale
        self.n_fingers_to_move = n_fingers_to_move
        self.a_dim = self.n_fingers_to_move * 3
        self.task = task
        self.goal_type = goal_type
        self.device = device

        step_size = int(self.downsample_time_step / self.sim_time_step)

        if object_type == "colored_cube":
            self.object_type = ObjectType.COLORED_CUBE
        elif object_type == "green_cube":
            self.object_type = ObjectType.GREEN_CUBE
        else:
            raise NameError

        # Set env based on task
        if self.task == "move":
            self.env = MoveCubeEnv(
                goal_pose=None,  # passing None to sample a random trajectory
                action_type=ActionType.TORQUE,
                step_size=step_size,
                visualization=False,
                goal_visualization=goal_visualization,
                no_collisions=False,
                enable_cameras=True,
                finger_type=finger_type,
                time_step=self.sim_time_step,
                camera_delay_steps=0,
                object_type=self.object_type,
                enable_shadows=enable_shadows,
                camera_view=camera_view,
                arena_color=arena_color,
                visual_observation=True,
                run_rl_policy=False,
            )

        elif self.task == "reach":
            self.env = CubeReachEnv(
                action_type=ActionType.TORQUE,
                step_size=step_size,
                visualization=False,
                enable_cameras=True,
                finger_type=finger_type,
                camera_delay_steps=0,
                time_step=self.sim_time_step,
                object_type=self.object_type,
                enable_shadows=enable_shadows,
                camera_view=camera_view,
                arena_color=arena_color,
                visual_observation=True,
                run_rl_policy=False,
            )
        else:
            raise NameError

    def close(self):
        self.env.close()

    def reset(self, expert_demo_dict):
        # Reset environment with init and goal positions, scaled from cm -> m
        obj_init_pos = expert_demo_dict["o_pos_cur"][0, :] / self.traj_scale
        obj_init_ori = expert_demo_dict["o_ori_cur"][0, :]
        # Use final object position in demo as goal
        obj_goal_pos = expert_demo_dict["o_pos_cur"][-1, :] / self.traj_scale
        obj_goal_ori = expert_demo_dict["o_ori_cur"][-1, :]
        init_pose = {"position": obj_init_pos, "orientation": obj_init_ori}
        goal_pose = {"position": obj_goal_pos, "orientation": obj_goal_ori}
        qpos_init = expert_demo_dict["robot_pos"][0, :]

        if self.task == "move":
            observation = self.env.reset(
                goal_pose_dict=goal_pose,
                init_pose_dict=init_pose,
                init_robot_position=qpos_init,
            )
        elif self.task == "reach":
            observation = self.env.reset(
                init_pose_dict=init_pose,
                init_robot_position=qpos_init,
            )
        else:
            raise NameError

        # Object goal position, scaled to cm for policy
        self.o_goal_pos = (torch.FloatTensor(obj_goal_pos).to(self.device) * self.traj_scale).unsqueeze(0)

        # Relative goal, scaled to cm for policy
        self.o_goal_pos_rel = (torch.FloatTensor(obj_goal_pos - obj_init_pos).to(self.device) * self.traj_scale).unsqueeze(0)
        return observation

    def execute_policy(self, policy, expert_demo_dict):
        # Reset env and update policy network
        observation_list = []
        observation = self.reset(expert_demo_dict)
        observation_list.append(observation)

        pred_actions = []
        episode_done = False
        while not episode_done:
            # Get bc input tensor from observation
            # Scale observation by traj_scale, for bc policy
            ft_pos_cur = observation["ft_pos_cur"] * self.traj_scale
            ft_state = torch.FloatTensor(ft_pos_cur).unsqueeze(0).to(self.device)

            img = observation["camera_observation"]["camera60"]["image"]
            img = policy.process_data(img)
            img_goal = expert_demo_dict["image_60"][-1]  
            img_goal = policy.process_data(img_goal)
            
            if policy.ft_method == 'partial_ft':    
                embedding = policy.get_representations(img.to(self.device))
                embedding_goal = policy.get_representations(img_goal.to(self.device))
                input_dict = {
                    "ft_state": ft_state,
                    "embedding": embedding,
                    "embedding_goal": embedding_goal,
                    "o_goal_pos": self.o_goal_pos,
                    "o_goal_pos_rel": self.o_goal_pos_rel,
                }
            elif policy.ft_method == 'full_ft':
                input_dict = {
                    "ft_state": ft_state,
                    "img": img,
                    "img_goal": img_goal,
                    "o_goal_pos": self.o_goal_pos,
                    "o_goal_pos_rel": self.o_goal_pos_rel,
                }
            else:
                raise ValueError
            obs_dict = {"input": input_dict}

            # Get action from policy, convert back to meters
            with torch.no_grad():
                a = policy.get_action(obs_dict)
                pred_action = a / self.traj_scale

                three_finger_action = np.zeros(9)
                three_finger_action[: self.n_fingers_to_move * 3] = pred_action * self.traj_scale
                pred_actions.append(three_finger_action)

            observation, reward, episode_done, info = self.env.step(pred_action)
            observation_list.append(observation)

        d_utils.add_actions_to_obs(observation_list)

        # Get traj_dict and downsample
        traj_dict = d_utils.get_traj_dict_from_obs_list(observation_list, scale=self.traj_scale)

        # if save_dir is not None:
        #     t_utils.save_demo_to_file(save_dir, epoch-1, observation_list, expert_demo_dict, pred_actions)

        return traj_dict
