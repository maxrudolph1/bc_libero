'''
@article{majumdar2023we,
  title={Where are we in the search for an artificial visual cortex for embodied intelligence?},
  author={Majumdar, Arjun and Yadav, Karmesh and Arnaud, Sergio and Ma, Jason and Chen, Claire and Silwal, Sneha and Jain, Aryan and Berges, Vincent-Pierre and Wu, Tingfan and Vakil, Jay and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={655--677},
  year={2023}
}

Adapted from https://github.com/facebookresearch/eai-vc
'''

import os
from vc_models.models.vit.model_utils import load_model as load_vc1

from .base_model import BaseModel


class VC1(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.policy.embedding == 'vc1_vitb' or 'vc1_vitl':
            self.feature_extractor, _, _, _ = load_vc1(cfg.policy.embedding, load_path=os.path.join(cfg.policy.embedding_dir, 'vc-1'))
            if cfg.train.ft_method == 'partial_ft':
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
        else:
            raise ValueError("VC-1 model type is wrong! The repo only suits for [\"vc1_vitb\"].")
        