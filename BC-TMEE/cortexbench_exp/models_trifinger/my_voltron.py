'''
@inproceedings{karamcheti2023language,
  title={Language-driven representation learning for robotics},
  author={Karamcheti, Siddharth and Nair, Suraj and Chen, Annie S and Kollar, Thomas and Finn, Chelsea and Sadigh, Dorsa and Liang, Percy},
  booktitle={Robotics: Science and Systems},
  year={2023}
}

Adapted from  and https://github.com/siddk/voltron-robotics
'''

import os
from voltron import load as load_voltron

from .base_model import BaseModel
from .modules.extraction import instantiate_extractor


class Voltron(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.policy.embedding == 'v-cond-small' or cfg.policy.embedding == 'v-conda-base':
            load_path = os.path.join(cfg.policy.embedding_dir, 'voltron', cfg.policy.embedding)
            self.feature_extractor = load_voltron("v-cond", load_path=load_path, only_return_model=True)
            if cfg.train.ft_method == 'partial_ft':
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
            self.vector_extractor = instantiate_extractor(self.feature_extractor)()
        else:
            raise ValueError("Voltron model type is wrong! The repo only suits for [\"v-cond-small\", \"v-conda-base\"].")
        
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
        