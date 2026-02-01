'''
@inproceedings{radosavovic2023real,
  title={Real-world robot learning with masked visual pre-training},
  author={Radosavovic, Ilija and Xiao, Tete and James, Stephen and Abbeel, Pieter and Malik, Jitendra and Darrell, Trevor},
  booktitle={Conference on Robot Learning},
  pages={416--426},
  year={2023},
  organization={PMLR}
}

Adapted from https://github.com/ir413/mvp and https://github.com/siddk/voltron-robotics
'''

import os
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from voltron import load as load_mvp

from .base_model import BaseModel
from .modules.mlp import DynamicMLP
from .modules.policy_head import DeterministicHead
from .modules.extraction import instantiate_extractor
from .modules.transformer_modules import SinusoidalPositionEncoding, TransformerDecoder
from ..utils.data_utils import fuse_embeddings


class MVP(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.policy.embedding == 'mvp-small':
            load_path = os.path.join(cfg.policy.embedding_dir, 'mvp', cfg.policy.embedding)
            self.feature_extractor = load_mvp("r-mvp", load_path=load_path, only_return_model=True)
            if cfg.train.ft_method == 'partial_ft':
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
            self.vector_extractor = instantiate_extractor(self.feature_extractor)()
        else:
            raise ValueError("MVP model type is wrong! The repo only suits for [\"mvp-small\"].")
        
    def forward(self, data, return_latent=False):
        cfg = self.cfg
        if cfg.train.ft_method == 'full_ft':
            preprocessed_imgs = self.process_data(data["input"]["img"])
            embeddings = self.get_representations(preprocessed_imgs.to(self.device))  # (b*t, emb_dim) -> (b, t, emb_dim)
        else:
            embeddings = data["input"]["embedding"].to(self.device)

        extra = None
        if cfg.env.add_proprio:
            extra = self.extra_states_encoder(data["input"]["ft_state"].to(self.device))

        embeddings = self.vector_extractor(embeddings)
        if cfg.policy.use_spatial:
            x, z, dist = self.spatial_encode(embeddings, extra, data)
        else:
            x, z, dist = self.temporal_encode(embeddings, extra, data)

        if return_latent:
            return x, z, dist

        return dist
        