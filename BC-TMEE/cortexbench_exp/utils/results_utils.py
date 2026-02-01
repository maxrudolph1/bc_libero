import os
import wandb
import torch
import numpy as np
import pandas as pd
from typing import List
from einops import rearrange
from mjrl.utils.gym_env import GymEnv
from mujoco_vc.rollout_utils import rollout_from_init_states

from .env_utils import sample_paths
from .video_utils import video_pad_time, render_done_to_boundary, save_numpy_to_video


@torch.no_grad()
def rollout_eval(config, env, model, highest_score=-np.inf, highest_success=-np.inf, demo_score=1000.,
                 return_wandb_video=True, return_local=False, video_save_dir=None):
    if config.debug:
        config.eval.eval_num_traj = 1
        config.env.horizon = 10

    model.eval()
    print('\nRollouting in test set...')
    paths = sample_paths(
        num_traj=config.eval.eval_num_traj,
        env=env,
        policy=model,
        eval_mode=True,
        horizon=config.env.horizon,
        base_seed=config.train.seed,
        num_cpu=config.eval.num_cpu,
        add_proprio=config.env.add_proprio,
    )
    mean_score, success_percentage, highest_score, highest_success, is_best = \
        compute_metrics_from_paths(
            env=env,
            suite=config.env.suite,
            paths=paths,
            highest_score=highest_score,
            highest_success=highest_success,
            demo_score=demo_score,
        )
    
    results = {}
    videos, num_terminated = get_video_and_info(config, paths)
    videos = videos.astype(np.uint8)
    if return_wandb_video:
        results[f"rollout_eval/vis_env_{config.env.suite}_task_{config.env.task_name}"] = \
            wandb.Video(videos, fps=30, format="mp4", caption=f"env_{config.env.suite}_task_{config.env.task_name}")
    if return_local:  
        save_numpy_to_video(videos, video_save_dir, num_terminated)

    results[f"rollout_eval/score_env_{config.env.suite}_task_{config.env.task_name}"] = mean_score
    results[f"rollout_eval/success_env_{config.env.suite}_task_{config.env.task_name}"] = success_percentage
    results[f"rollout_eval/highest_success_env_{config.env.suite}_task_{config.env.task_name}"] = highest_success
    results[f"rollout_eval/highest_score_env_{config.env.suite}_task_{config.env.task_name}"] = highest_score
    return results, is_best


@torch.no_grad()
def rollout_train(config, env, init_states, model, highest_tr_score=-np.inf, highest_tr_success=-np.inf, demo_score=1000.,
                  return_wandb_video=False, return_local=False, video_save_dir=None):
    if config.debug:
        config.eval.eval_num_traj = 1
        config.env.horizon = 500

    print('\nRollouting in training set...')
    # log statistics on training paths
    if len(init_states) > 0:
        paths = rollout_from_init_states(
            init_states[: config.eval.eval_num_traj],
            env,
            model,
            eval_mode=True,
            horizon=config.env.horizon,
        )
    else:
        # use same seed as used for collecting the training paths
        paths = sample_paths(
            num_traj=config.eval.eval_num_traj,
            env=env,
            policy=model,
            eval_mode=True,
            horizon=config.env.horizon,
            base_seed=config.train.seed,
            num_cpu=config.eval.num_cpu,
        )
    tr_score, tr_success, highest_tr_score, highest_tr_success = \
        compute_metrics_from_paths(
            env=env,
            suite=config.env.suite,
            paths=paths,
            highest_score=highest_tr_score,
            highest_success=highest_tr_success,
            demo_score=demo_score,
        )
    
    results = {}
    videos, num_terminated = get_video_and_info(config, paths)
    if return_wandb_video:
        results[f"rollout_train/vis_env_{config.env.suite}_task_{config.env.task_name}"] = \
            wandb.Video(videos, fps=30, format="mp4", caption=f"env_{config.env.suite}_task_{config.env.task_name}")
    if return_local:  
        save_numpy_to_video(videos, video_save_dir, num_terminated)
        
    results = {}
    results[f"rollout_train/score_env_{config.env.suite}_task_{config.env.task_name}"] = tr_score
    results[f"rollout_train/success_env_{config.env.suite}_task_{config.env.task_name}"] = tr_success
    results[f"rollout_train/highest_score_env_{config.env.suite}_task_{config.env.task_name}"] = highest_tr_score
    results[f"rollout_train/highest_success_env_{config.env.suite}_task_{config.env.task_name}"] = highest_tr_success
    return results


def compute_metrics_from_paths(
    env: GymEnv,
    suite: str,
    paths: list,
    highest_score: float = -1.0,
    highest_success: float = -1.0,
    demo_score: float = 1000.,
):
    mean_score = np.mean([np.sum(p["rewards"]) for p in paths])
    if suite == "dmc":
        # we evaluate dmc based on returns, not success
        success_percentage = mean_score / demo_score
    if suite == "adroit":
        success_percentage = env.env.unwrapped.evaluate_success(paths)
    if suite == "metaworld":
        sc = []
        for i, path in enumerate(paths):
            sc.append(path["env_infos"]["success"][-1])
        success_percentage = np.mean(sc) * 100

    is_best = False
    if mean_score >= highest_score:
        highest_score = mean_score
        if suite == "dmc":
            is_best = True
    if success_percentage >= highest_success:
        highest_success = success_percentage
        if suite == "adroit" or suite == "metaworld":
            is_best = True

    return round(mean_score, 4), round(success_percentage, 4), round(highest_score, 4), round(highest_success, 4), is_best


def get_video_and_info(config, paths):
    videos = []
    num_terminated = []
    for path in paths:
        video = rearrange(path['observations'], "t h w c -> t c h w")
        if config.env.suite == 'metaworld':
            if 1.0 in path["env_infos"]["success"]:
                index = np.where(path["env_infos"]["success"] == 1.0)[0][0]
                success_frame = render_done_to_boundary(video[index:])    
                videos.append(np.concatenate([video[:-index], success_frame]))
            else:
                videos.append(video)
            num_terminated.append(bool(path["env_infos"]["success"][-1]))
        elif config.env.suite == 'adroit':
            if True in path['env_infos']['goal_achieved']:
                index = np.where(path['env_infos']['goal_achieved'] == True)[0][0]
                success_frame = render_done_to_boundary(video[index:])    
                videos.append(np.concatenate([video[:-index], success_frame]))
            else:
                videos.append(video)
            if config.env.task_name == 'pen-v0':
                steps = 20
            elif config.env.task_name == 'relocate-v0':
                steps = 25
            success = np.sum(path['env_infos']['goal_achieved']) > steps
            num_terminated.append(success)
        elif config.env.suite == 'dmc':
            videos.append(video)
    videos = video_pad_time(videos)
    videos = np.stack(videos, axis=0)       # [b, t, c, h, w]
    
    return videos, num_terminated


def save_success_rate(epoch, success_metrics, summary_file_path):
    success_metrics = {
        x.split("_task")[0].replace('/', '_'): value 
        for x, value in success_metrics.items()
    }
    success_heads = list(success_metrics.keys())
    success_heads = ["epoch"] + success_heads

    success_metrics["epoch"] = str(epoch)
    df = pd.DataFrame(success_metrics, index=[0])

    if os.path.exists(summary_file_path):
        old_summary = pd.read_csv(summary_file_path)
        df = pd.concat([df, old_summary], ignore_index=True)

    df = df[success_heads]
    df.to_csv(summary_file_path)


def save_final_success_rate(summary_file_path, final_summary_file_path, k=3):
    df = pd.read_csv(summary_file_path)
    score = df.iloc[:, 2] 
    success_rate = df.iloc[:, 3]  
    avg_score = score.nlargest(k).mean()
    avg_success_rate = success_rate.nlargest(k).mean()

    final_metrics = {
        f'avg_{k}_score': [avg_score],
        f'avg_{k}_success_rate': [avg_success_rate],
    }
    final_df = pd.DataFrame(final_metrics)
    final_df.to_csv(final_summary_file_path, index=False)
    print(f"Metrics saved to {final_summary_file_path}")


def merge_results(results: List[dict]):
    merged_results = {}
    for result_dict in results:
        for k, v in result_dict.items():
            if k in merged_results:
                if isinstance(v, list):
                    merged_results[k].append(v)
                else:
                    merged_results[k] = [merged_results[k], v]
            else:
                merged_results[k] = v
        
    return merged_results
