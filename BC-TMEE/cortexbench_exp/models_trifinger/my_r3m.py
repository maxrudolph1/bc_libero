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
            raise ValueError("R3M model type is wrong! The repo only suits for [\"r3m-rn50\"].")
    
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