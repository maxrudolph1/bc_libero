import os
import torch
from trifinger_vc.utils.model_utils import MODEL_NAMES


def fuse_embeddings(embeddings, method='cat'):
    '''
    method: ['cat', 'sub_cat']
    embeddings: tesnor [B, T, emb_dim]
    '''
    b, t, emb_dim = embeddings.shape
    if method == 'cat':
        return embeddings.reshape(b, -1)
    elif method == 'sub_cat':
        result = []
        # for batch_embeddings in embeddings:     # embeddings: [B, T, emb_dim]
        #     history_window = batch_embeddings.shape[0]
        #     delta = [batch_embeddings[i + 1] - batch_embeddings[i] for i in range(history_window - 1)]
        #     delta.append(batch_embeddings[-1])
        #     result.append(torch.cat(delta, dim=-1))
        # result = torch.stack(result, dim=0)
        # return result
        batch_deltas = embeddings[:, 1:] - embeddings[:, :-1]  
        last_embeddings = embeddings[:, -1:]
        result = torch.cat([batch_deltas, last_embeddings], dim=1)
        return result.reshape(b, -1)
    else:
        raise ValueError("Unsupported embedding fusion method!")


def fuse_goal(state, obs_dict, goal_type, goal_encoder):
    if goal_type == "goal_none":
        obs_vec = state
    elif goal_type == "goal_cond":
        obs_vec = torch.cat([state, goal_encoder(obs_dict["o_goal_pos"].to(state.device))], dim=1)
    elif goal_type == "goal_o_pos":
        # Use object position goal state - relative to init position of object
        obs_vec = torch.cat([state, goal_encoder(obs_dict["o_goal_pos_rel"].to(state.device))], dim=1)
    else:
        raise NameError("Invalid goal_type")
    return obs_vec


def get_traj_list(demo_root_dir, demo_stats_list, obj_state_type):
    """Given list of demo stats demo_stats_list, load demo dicts and save in traj_list"""
    traj_list = []
    for demo_stat in demo_stats_list:
        demo_dir = demo_stat["path"]
        downsample_data_path = os.path.join(demo_root_dir, demo_dir, "downsample.pth")
        if not os.path.exists(downsample_data_path):
            raise ValueError(f"{downsample_data_path} not found")

        demo_dict = torch.load(downsample_data_path)

        if obj_state_type in MODEL_NAMES:
            # Load latent state from obj_state_type.pth file
            latent_data_path = os.path.join(demo_dir, f"{obj_state_type}.pth")
            if not os.path.exists(latent_data_path):
                raise ValueError(f"{latent_data_path} not found")

            latent_data = torch.load(latent_data_path)["data"]

            demo_dict[obj_state_type] = latent_data

        traj_list.append(demo_dict)

    return traj_list