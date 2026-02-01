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
from mpi import load_mpi

from .base_model import BaseModel
from .modules.extraction import instantiate_extractor


class MPI(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.policy.embedding == 'mpi-base' or 'mpi-small':
            root_dir = os.path.join(cfg.policy.embedding_dir, 'mpi', cfg.policy.embedding)
            language_model_path = "/baishuanghao/model/distilbert-base-uncased"
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
    