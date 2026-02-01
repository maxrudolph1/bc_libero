import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class SequenceDataset_Full_FT(Dataset):
    def __init__(
        self,
        paths: list,
        history_window: int,
        policy: nn.Module,
        use_spatial: bool,
        proprio_key: str = None,
    ):  
        paths = self.process_demo(paths, policy)
        self.paths = paths

        # assume equal length trajectories code will work even otherwise but may have some edge cases
        self.path_length = max([p["actions"].shape[0] for p in paths])     # 500
        self.num_paths = len(self.paths)    # 100
        self.history_window = history_window
        self.use_spatial = use_spatial
        self.proprio_key = proprio_key

    def __len__(self):
        return self.path_length * self.num_paths

    def __getitem__(self, index):
        traj_idx = int(index // self.path_length)
        timestep = int(index - traj_idx * self.path_length)
        timestep = min(timestep, self.paths[traj_idx]["actions"].shape[0] - 1)
        extra_states = None

        if timestep >= self.history_window - 1:
            images = self.paths[traj_idx]["processed_images"][timestep-self.history_window+1: timestep+1]
            if self.proprio_key not in [None, "None"]:
                extra_states = self.paths[traj_idx]["env_infos"][self.proprio_key][timestep-self.history_window+1: timestep+1]
        else:
            images = [
                self.paths[traj_idx]["processed_images"][max(timestep - k, 0)]
                for k in range(self.history_window)
            ][::-1]
            images = torch.stack(images)
            if self.proprio_key not in [None, "None"]:
                extra_states = [
                    self.paths[traj_idx]["env_infos"][self.proprio_key][max(timestep - k, 0)]
                    for k in range(self.history_window)
                ][::-1]
                extra_states = np.stack(extra_states)

        if self.use_spatial:
            actions = self.paths[traj_idx]["actions"][timestep]
        else:
            if timestep >= self.history_window - 1:
                actions = self.paths[traj_idx]["actions"][timestep-self.history_window+1: timestep+1]
            else:
                actions = [
                    self.paths[traj_idx]["actions"][max(timestep - k, 0)]
                    for k in range(self.history_window)
                ][::-1]
                actions = np.stack(actions)

        if self.proprio_key not in [None, "None"]:
            return {"images": images, "actions": actions.astype(np.float32), "extra_states": extra_states.astype(np.float32)}
        
        return {"images": images, "actions": actions.astype(np.float32)}

    def process_demo(self, paths, policy):
        print("Processing dataset...")
        for path in tqdm(paths):
            inp = path["images"] 
            path["processed_images"] = policy.process_data(inp).to(policy.device)
            del path["images"]
        return paths


class SequenceDataset_Partial_FT(Dataset):
    def __init__(
        self,
        paths: list,
        history_window: int,
        policy: nn.Module,
        device,
        use_spatial: bool,
        proprio_key: str = None,
    ):  
        paths = self.process_demo(paths, policy, device)
        self.paths = paths

        # assume equal length trajectories code will work even otherwise but may have some edge cases
        self.path_length = max([p["actions"].shape[0] for p in paths])     # 500
        self.num_paths = len(self.paths)    # 100
        self.history_window = history_window
        self.use_spatial = use_spatial
        self.proprio_key = proprio_key

    def __len__(self):
        return self.path_length * self.num_paths

    def __getitem__(self, index):
        traj_idx = int(index // self.path_length)
        timestep = int(index - traj_idx * self.path_length)
        timestep = min(timestep, self.paths[traj_idx]["actions"].shape[0] - 1)
        extra_states = None

        if timestep >= self.history_window - 1:
            embeddings = self.paths[traj_idx]["embeddings"][timestep-self.history_window+1: timestep+1]
            if self.proprio_key not in [None, "None"]:
                extra_states = self.paths[traj_idx]["env_infos"][self.proprio_key][timestep-self.history_window+1: timestep+1]
        else:
            embeddings = [
                self.paths[traj_idx]["embeddings"][max(timestep - k, 0)]
                for k in range(self.history_window)
            ][::-1]
            embeddings = torch.stack(embeddings)
            if self.proprio_key not in [None, "None"]:
                extra_states = [
                    self.paths[traj_idx]["env_infos"][self.proprio_key][max(timestep - k, 0)]
                    for k in range(self.history_window)
                ][::-1]
                extra_states = np.stack(extra_states)

        if self.use_spatial:
            actions = self.paths[traj_idx]["actions"][timestep]
        else:
            if timestep >= self.history_window - 1:
                actions = self.paths[traj_idx]["actions"][timestep-self.history_window+1: timestep+1]
            else:
                actions = [
                    self.paths[traj_idx]["actions"][max(timestep - k, 0)]
                    for k in range(self.history_window)
                ][::-1]
                actions = np.stack(actions)

        if self.proprio_key not in [None, "None"]:
            return {"embeddings": embeddings, "actions": actions.astype(np.float32), "extra_states": extra_states.astype(np.float32)}
        
        return {"embeddings": embeddings, "actions": actions.astype(np.float32)}

    def process_demo(self, paths, policy, device):
        print("Processing dataset...")
        with torch.no_grad():
            for i, path in enumerate(tqdm(paths)):
                preprocessed_inp = policy.process_data(path["images"])
                preprocessed_inp = preprocessed_inp.to(device)
                if policy.embedding_type in ['R3M']:
                    path["embeddings"] = policy.feature_extractor.get_representations(preprocessed_inp * 255.0)
                elif policy.embedding_type in ['ResNet', 'ViT', 'VC1']:
                    path["embeddings"] = policy.feature_extractor(preprocessed_inp)
                elif policy.embedding_type in ['MVP', 'Voltron']:
                    path["embeddings"] = policy.feature_extractor.get_representations(preprocessed_inp)
                elif policy.embedding_type in ['MPI']:
                    preprocessed_inp = torch.stack((preprocessed_inp, preprocessed_inp), dim=1)
                    batch_size = preprocessed_inp.shape[0] // 2
                    embedding_1 = policy.feature_extractor.get_representations(preprocessed_inp[:batch_size])
                    embedding_2 = policy.feature_extractor.get_representations(preprocessed_inp[batch_size:])
                    path["embeddings"] = torch.cat((embedding_1, embedding_2), dim=0) #.cpu()
                del path["images"]
        return paths


class TrifingerDataset(Dataset):
    def __init__(
        self,
        demo_list,
        policy=None,
        device="cpu",
        fingers_to_move=3,
        demo_root_dir="assets/data/trifinger-demos",
    ):  
        self.dataset = []
        self.policy = policy
        self.device = device
        self.n_fingers_to_move = fingers_to_move

        for demo_stats in tqdm(demo_list):
            if demo_root_dir is not None:
                demo_dir = os.path.join(demo_root_dir, demo_stats["path"])
            else:
                demo_dir = demo_stats["path"]
            self.add_new_traj(demo_dir, policy)

        # Dimensions
        self.action_dim = self.dataset[0]["output"]["action"].shape[0]

    def add_new_traj(self, demo_dir, policy):
        downsample_data_path = os.path.join(demo_dir, "downsample.pth")
        if not os.path.exists(downsample_data_path):
            raise ValueError(f"{downsample_data_path} not found")
        demo = torch.load(downsample_data_path)
        num_obs = demo["o_pos_cur"].shape[0]

        # Goal position (absolute)
        o_goal_pos = torch.FloatTensor(demo["o_pos_cur"][-1]).to(self.device)

        # Goal position (relative)
        o_init_pos = torch.FloatTensor(demo["o_pos_cur"][0]).to(self.device)
        o_goal_pos_rel = o_goal_pos - o_init_pos

        imgs = demo["image_60"]
        imgs = policy.process_data(imgs)
        img_goal = imgs[-1]
        
        if policy.ft_method == 'partial_ft':
            embeddings = policy.get_representations(imgs.to(self.device)).cpu()
            embedding_goal = embeddings[-1]

        for i in range(num_obs - 1):
            ft_pos_cur = demo["ft_pos_cur"][i]
            action = torch.FloatTensor(demo["delta_ftpos"][i])
            # Get subset of delta_ftpos that corresonds to diff (number of fingers that move)
            # For the <reach> task this will be [:3], and for other tasks [:9]
            action = action[: self.n_fingers_to_move * 3]
            
            # Observation dict (current state and action)
            if policy.ft_method == 'partial_ft':
                input_dict = {
                    "ft_state": torch.FloatTensor(ft_pos_cur),
                    "embedding": embeddings[i],
                    "embedding_goal": embedding_goal,   # not use
                    "o_goal_pos": o_goal_pos,
                    "o_goal_pos_rel": o_goal_pos_rel,
                }
            elif policy.ft_method == 'full_ft':
                input_dict = {
                    "ft_state": torch.FloatTensor(ft_pos_cur),
                    "img": imgs[i],
                    "img_goal": img_goal,   # not use
                    "o_goal_pos": o_goal_pos,
                    "o_goal_pos_rel": o_goal_pos_rel,
                }

            output_dict = {
                "action": torch.FloatTensor(action),
            }

            data_dict = {"input": input_dict, "output": output_dict}

            self.dataset.append(data_dict)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

