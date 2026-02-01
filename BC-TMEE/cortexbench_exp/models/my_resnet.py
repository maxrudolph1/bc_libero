'''
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}

Adapted from https://github.com/KaimingHe/deep-residual-networks
'''

from .base_model import BaseModel
from .modules.image_encoder import ResnetEncoder


class ResNet(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.policy.embedding == 'resnet18':
            self.feature_extractor = ResnetEncoder(output_size=cfg.policy.embedding_dim)    # 0.978752 M
            if cfg.train.ft_method == 'partial_ft':
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
        else:
            raise ValueError("ResNet model type is wrong! The repo only suits for [\"resnet18\"].")
        