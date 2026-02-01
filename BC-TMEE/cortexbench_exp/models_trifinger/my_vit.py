'''
@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

Adapted from https://github.com/google-research/vision_transformer
'''

from .base_model import BaseModel
from .modules.transformer_modules import TransformerEncoder


class ViT(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.policy.embedding == 'vit':
            self.feature_extractor = TransformerEncoder( 
                input_size=64,
                num_layers=7,
                num_heads=8,
                head_output_size=120,
                mlp_hidden_size=64,
                dropout=0.1,
            )
            if self.cfg.ft_method == 'partial_ft':
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
        else:
            raise ValueError("ViT model type is wrong! The repo only suits for [\"vit\"].")
        