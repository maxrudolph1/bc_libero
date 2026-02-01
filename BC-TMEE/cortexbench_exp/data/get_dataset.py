import os
import json
import pickle
import logging
import numpy as np

from .dataset import SequenceDataset_Full_FT, SequenceDataset_Partial_FT, TrifingerDataset
from ..utils.data_utils import get_traj_list


def get_dataset(config, policy=None, return_demo_score=False):
    # infer the demo location
    demo_paths_loc = os.path.join(config.data.data_dir, config.env.task_name + ".pickle")
    try:
        print("\nLoading data...")
        demo_paths = pickle.load(open(demo_paths_loc, "rb"))
    except:
        print("Unable to load the data. Check the data path.")
        print(demo_paths_loc)
        quit()

    demo_paths = demo_paths[: config.data.num_demos]
    demo_score = np.mean([np.sum(p["rewards"]) for p in demo_paths])

    if return_demo_score:
        return demo_score
    
    logging.info("Number of demonstrations used : %i" % len(demo_paths))
    logging.info("Demonstration score : %.2f " % demo_score)

    # store init_states for evaluation on training trajectories
    if config.env.suite == "dmc":
        init_states = [
            p["env_infos"]["internal_state"][0].astype(np.float64) for p in demo_paths
        ]
    elif config.env.suite == "adroit":
        init_states = [p["init_state_dict"] for p in demo_paths]
    elif config.env.suite == "metaworld":
        init_states = []
    else:
        print("\n\n Unsupported environment suite.")
        quit()

    if config.train.ft_method == 'full_ft':
        dataset = SequenceDataset_Full_FT(
            demo_paths,
            history_window=config.env.history_window,
            policy=policy,
            use_spatial=config.policy.use_spatial,
            proprio_key=config.env.proprio_key,
        )
    elif config.train.ft_method == 'partial_ft':
        dataset = SequenceDataset_Partial_FT(
            demo_paths,
            history_window=config.env.history_window,
            policy=policy,
            device=config.train.device,
            use_spatial=config.policy.use_spatial,
            proprio_key=config.env.proprio_key,
        )
    else:
        raise ValueError

    return dataset, init_states, demo_score


def get_dataset_trifinger(config, policy=None, return_traj_info=False):
    root_dir = config.data.split_dir
    if config.env.task_name == 'move':
        split_path = os.path.join(root_dir, "feb_demos_dtrain-110_train-0-100_dtest-110_test-100-125_scale-100_dts-0p4.json")
        fingers_to_move = 3
    elif config.env.task_name == 'reach':
        split_path = os.path.join(root_dir, "demos_dtrain-110_train-0-100_dtest-110_test-100-125_scale-100_dts-0p4.json")
        fingers_to_move = 1

    with open(split_path, "r") as f:
        traj_info = json.load(f)
    train_traj_stats = traj_info["train_demo_stats"][:config.data.num_demos]
    test_traj_stats = traj_info["test_demo_stats"][:config.eval.eval_num_traj]

    traj_info["train_demos"] = get_traj_list(config.data.data_dir, train_traj_stats, "pos")
    traj_info["test_demos"] = get_traj_list(config.data.data_dir, test_traj_stats, "pos")

    if return_traj_info:
        return traj_info

    print(f"\nLoading training dataset...")
    train_dataset = TrifingerDataset(
        train_traj_stats,
        policy=policy,
        device=config.train.device,
        fingers_to_move=fingers_to_move,
        demo_root_dir=config.data.data_dir,
    )
    
    print(f"Loading test dataset...")
    test_dataset = TrifingerDataset(
        test_traj_stats,
        policy=policy,
        device=config.train.device,
        fingers_to_move=fingers_to_move,
        demo_root_dir=config.data.data_dir,
    )
    
    return train_dataset, test_dataset, traj_info
