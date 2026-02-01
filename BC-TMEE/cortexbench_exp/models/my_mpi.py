'''
@article{zeng2024learning,
  title={Learning Manipulation by Predicting Interaction},
  author={Zeng, Jia and Bu, Qingwen and Wang, Bangjun and Xia, Wenke and Chen, Li and Dong, Hao and Song, Haoming and Wang, Dong and Hu, Di and Luo, Ping and others},
  booktitle={Robotics: Science and Systems},
  year={2024}
}

Adapted from https://github.com/OpenDriveLab/MPI
'''

import os
import torch
import numpy as np
from einops import rearrange
from mpi import load_mpi

from .base_model import BaseModel
from .modules.extraction import instantiate_extractor


class MPI(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        cfg.policy.embedding_dim = 384
        if cfg.policy.embedding == 'mpi-base' or 'mpi-small':
            root_dir = os.path.join(cfg.policy.embedding_dir, 'mpi', cfg.policy.embedding)
            language_model_path = "/root/model/distilbert-base-uncased"
            self.feature_extractor = load_mpi(root_dir, language_model_path=language_model_path)
            if cfg.train.ft_method == 'partial_ft':
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
            self.vector_extractor = instantiate_extractor(self.feature_extractor)()
        else:
            raise ValueError("MPI model type is wrong! The repo only suits for [\"mpi-base\", \"mpi-small\"].")

    def forward(self, data, return_latent=False):
        cfg = self.cfg
        if cfg.train.ft_method == 'full_ft':
            b, t, l, c, h, w = data["images"].shape
            data["images"] = data["images"].to(self.device)
            images = rearrange(data["images"], "b t l c h w -> (b t) l c h w")
            embeddings = self.get_representations(images) # (b*t, n, emb_dim)
            _, n, e = embeddings.shape
            embeddings = embeddings.reshape(b, t, n, e)
        else:
            b, t, n, e = data["embeddings"].shape
            embeddings = data["embeddings"].to(self.device)

        extra = None
        if cfg.env.add_proprio:
            extra = self.extra_states_encoder(data["extra_states"].reshape(b * t, -1).to(self.device))
            extra = extra.reshape(b, t, -1)

        if cfg.policy.use_spatial:
            embeddings = self.vector_extractor(embeddings.reshape(b*t, n, e)).reshape(b, t, -1)
            if extra != None:
                x = torch.cat([embeddings, extra], dim=-1)
            else:
                x = embeddings
            z = self.fuse_encode(x)     
            dist = self.policy_head(z)
        else:
            embeddings = self.vector_extractor(embeddings.reshape(b*t, n, e))
            embeddings = self.spatial_down_sample(embeddings).reshape(b, t, -1)
            if extra != None:
                encoded = torch.stack([embeddings, extra], dim=-2)
            else:
                encoded = embeddings.unsqueeze(-2)   # (b, t, emb_dim) -> (b, t, 1, emb_dim)
            action_token = self.action_token.unsqueeze(0).expand(b, t, -1, -1)  # (b, t, 1, emb_dim)
            x = torch.cat([action_token, encoded], -2)
            z, z0 = self.fuse_encode(x)
            dist = self.policy_head(z0)

        if return_latent:
            return x, z, dist
        
        return dist
    
    def get_action(self, obeservation):
        cfg = self.cfg
        if cfg.env.add_proprio:
            image, proprio = obeservation
        else:
            image = obeservation
        transformed_image = self.process_data(image).to(self.device)  # (1, c, h, w)
        embedding = self.get_representations(transformed_image)    # (1, c, h, w) -> (1, emb_dim)
        embedding = self.vector_extractor(embedding)
        if cfg.env.add_proprio:
            proprio = torch.from_numpy(proprio.astype(np.float32)).unsqueeze(0).to(self.device)
            extra = self.extra_states_encoder(proprio)
            if cfg.policy.use_spatial:
                embedding = torch.cat([embedding, extra], dim=-1)
            else:
                self.latent_queue_proprio.append(extra)     
                if len(self.latent_queue_proprio) < self.history_window:
                    for i in range(self.history_window - len(self.latent_queue_proprio)):
                        self.latent_queue_proprio.append(extra)
                if len(self.latent_queue_proprio) > self.history_window:
                    self.latent_queue_proprio.pop(0)

        self.latent_queue.append(embedding)     
        if len(self.latent_queue) < self.history_window:
            for i in range(self.history_window - len(self.latent_queue)):
                self.latent_queue.append(embedding)
        if len(self.latent_queue) > self.history_window:
            self.latent_queue.pop(0)

        embeddings = torch.cat(self.latent_queue, dim=0).unsqueeze(0)    # (1, t, emb_dim)
        if cfg.policy.use_spatial:
            z = self.fuse_encode(embeddings)     
            dist = self.policy_head(z)
        else:
            b, t, _ = embeddings.shape
            embeddings = self.spatial_down_sample(embeddings.reshape(b*t, -1)).reshape(b, t, -1)
            if cfg.env.add_proprio:
                extras = torch.cat(self.latent_queue_proprio, dim=0).unsqueeze(0)
                encoded = torch.stack([embeddings, extras], dim=-2)
            else:
                encoded = embeddings.unsqueeze(-2)   # (b, t, emb_dim) -> (b, t, 1, emb_dim)
            action_token = self.action_token.unsqueeze(0).expand(b, t, -1, -1)  # (b, t, 1, emb_dim)
            x = torch.cat([action_token, encoded], -2)
            z, z0 = self.fuse_encode(x)
            dist = self.policy_head(z0[:, -1])
        return dist.detach().squeeze(0).cpu().numpy()
        