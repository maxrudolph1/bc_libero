'''
@inproceedings{nair2023r3m,
  title={R3M: A Universal Visual Representation for Robot Manipulation},
  author={Nair, Suraj and Rajeswaran, Aravind and Kumar, Vikash and Finn, Chelsea and Gupta, Abhinav},
  booktitle={Conference on Robot Learning},
  pages={892--909},
  year={2023},
  organization={PMLR}
}

Adapted from https://github.com/facebookresearch/r3m and https://github.com/siddk/voltron-robotics
'''

import os
import torch
import numpy as np
from einops import rearrange
from voltron import load as load_r3m

from .base_model import BaseModel
from .modules.extraction import instantiate_extractor


class R3M(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.policy.embedding in ['r3m-rn50', 'r3m-small']:
            load_path = os.path.join(cfg.policy.embedding_dir, 'r3m', cfg.policy.embedding)
            self.feature_extractor = load_r3m("r-r3m-vit", load_path=load_path, only_return_model=True)
            if cfg.train.ft_method == 'partial_ft':
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
            self.vector_extractor = instantiate_extractor(self.feature_extractor)()
        else:
            raise ValueError("R3M model type is wrong! The repo only suits for [\"r3m-rn50\" and \"r3m-small\"].")
    
    def forward(self, data, return_latent=False):
        cfg = self.cfg
        if cfg.train.ft_method == 'full_ft':
            b, t, c, h, w = data["images"].shape
            images = rearrange(data["images"], "b t c h w -> (b t) c h w")
            embeddings = self.get_representations(images.to(self.device)) 
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
            embeddings = self.spatial_down_sample(embeddings.reshape(b*t, -1)).reshape(b, t, -1)
            if extra != None:
                encoded = torch.stack([embeddings, extra], dim=-2)
            else:
                encoded = embeddings.unsqueeze(-2)   # (b, t, emb_dim) -> (b, t, 1, emb_dim)
            action_token = self.action_token.unsqueeze(0).expand(b, t, -1, -1)  # (b, t, 1, emb_dim)
            x = torch.cat([action_token, encoded], -2)
            z, z0 = self.fuse_encode(x, return_latent=return_latent)
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
        # embedding = self.spatial_down_sample(embedding)
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
        b, t, _ = embeddings.shape
        if cfg.policy.use_spatial:
            z = self.fuse_encode(embeddings)     
            dist = self.policy_head(z)
        else:
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
    