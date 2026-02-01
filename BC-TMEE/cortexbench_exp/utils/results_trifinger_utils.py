import os
import torch
import numpy as np
from typing import List
import trifinger_vc.utils.data_utils as d_utils


@torch.no_grad()
def rollout(config, env_dict, model, traj_info, max_dict, mode='train', epoch=None, eval_dir=None):
    if config.debug:
        config.data.max_demo_per_diff = 5

    model.eval()
    print(f'\nRollouting in test set (epoch: {str(epoch)})...')

    best_dict = {}
    log_dict = {}
    for sim_env_name in env_dict.keys():
        log_dict[sim_env_name] = {f"{mode}": {}}

    for env_name, env in env_dict.items():
        traj_list = traj_info[f"{mode}_demos"]
        totals_dict = {}
        plot_count_dict = {}
        for demo_i, demo in enumerate(traj_list):
            diff = traj_info[f"{mode}_demo_stats"][demo_i]["diff"]
            traj_i = traj_info[f"{mode}_demo_stats"][demo_i]["id"]

            if diff in plot_count_dict:
                if plot_count_dict[diff] >= config.data.max_demo_per_diff:
                    continue
                else:
                    plot_count_dict[diff] += 1
            else:
                plot_count_dict[diff] = 1

            print(f"Rolling out demo (diff {diff} | id: {traj_i}) for split {mode} in sim env {env_name}")
            traj_label = f"diff-{diff}_traj-{traj_i}"
            if eval_dir == None:
                traj_sim_dir = os.path.join(config.experiment_dir, 'eval_results', env_name, mode, "epoch_"+str(epoch))
            else:
                traj_sim_dir = os.path.join(eval_dir, env_name, mode, "epoch_"+str(epoch))
            if not os.path.exists(traj_sim_dir):
                os.makedirs(traj_sim_dir)

            sim_traj_dict = env.execute_policy(model, demo)

            # Save gif of sim rollout
            d_utils.save_gif(
                sim_traj_dict["image_60"],
                os.path.join(traj_sim_dir, f"viz_{traj_label}.gif"),
            )

            # Compute final error for ftpos of each finger
            final_sim_ftpos = np.expand_dims(sim_traj_dict["ft_pos_cur"][-1], 0)
            final_demo_ftpos = np.expand_dims(demo["ft_pos_cur"][-1], 0)
            final_ftpos_dist = d_utils.get_per_finger_ftpos_err(final_demo_ftpos, final_sim_ftpos, fnum=3)
            final_ftpos_dist = np.squeeze(final_ftpos_dist)

            # Achieved object distance to goal
            sim_obj_pos_err = sim_traj_dict["position_error"][-1]

            # Compute scaled error and reward, based on task
            scaled_reward = sim_traj_dict["scaled_success"][-1]
            scaled_err = 1 - scaled_reward

            # Per traj log
            log_dict[env_name][mode][traj_label] = {
                "sim_obj_pos_err": sim_obj_pos_err,
                "scaled_err": scaled_err,
                "scaled_reward": scaled_reward,
                "final_ftpos_dist_0": final_ftpos_dist[0],
                "final_ftpos_dist_1": final_ftpos_dist[1],
                "final_ftpos_dist_2": final_ftpos_dist[2],
            }

            if diff in totals_dict:
                totals_dict[diff]["sim_obj_pos_err"] += sim_obj_pos_err
                totals_dict[diff]["scaled_err"] += scaled_err
                totals_dict[diff]["scaled_reward"] += scaled_reward
                totals_dict[diff]["final_ftpos_dist_0"] += final_ftpos_dist[0]
                totals_dict[diff]["final_ftpos_dist_1"] += final_ftpos_dist[1]
                totals_dict[diff]["final_ftpos_dist_2"] += final_ftpos_dist[2]
            else:
                totals_dict[diff] = {
                    "sim_obj_pos_err": sim_obj_pos_err,
                    "scaled_err": scaled_err,
                    "scaled_reward": scaled_reward,
                    "final_ftpos_dist_0": final_ftpos_dist[0],
                    "final_ftpos_dist_1": final_ftpos_dist[1],
                    "final_ftpos_dist_2": final_ftpos_dist[2],
                }

        # Log avg obj pos err for each diff
        for diff, per_diff_totals_dict in totals_dict.items():
            if (f"diff-{diff}_max_avg_scaled_reward_{mode}" not in max_dict[env_name][mode].keys()):
                max_dict[env_name][mode][f"diff-{diff}_max_avg_scaled_reward_{mode}"] = 0.0

            for key, total in per_diff_totals_dict.items():
                log_dict[env_name][mode][f"diff-{diff}_avg_{key}_{mode}"] = total / plot_count_dict[diff]

            curr_avg_scaled_reward = log_dict[env_name][mode][f"diff-{diff}_avg_scaled_reward_{mode}"]
            if curr_avg_scaled_reward > max_dict[env_name][mode][f"diff-{diff}_max_avg_scaled_reward_{mode}"]:
                max_dict[env_name][mode][f"diff-{diff}_max_avg_scaled_reward_{mode}"] = curr_avg_scaled_reward
                log_dict[env_name][mode][f"diff-{diff}_max_avg_scaled_reward_{mode}"] = curr_avg_scaled_reward
                best_dict[env_name] = True
            else:
                log_dict[env_name][mode][f"diff-{diff}_max_avg_scaled_reward_{mode}"] = \
                    max_dict[env_name][mode][f"diff-{diff}_max_avg_scaled_reward_{mode}"]   
                best_dict[env_name] = False
                    
    return log_dict, max_dict, best_dict


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
