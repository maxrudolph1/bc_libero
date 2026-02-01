import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np
from einops import rearrange

from .modules.mlp import DynamicMLP
from .modules.policy_head import DeterministicHead
from .modules.transformer_modules import SinusoidalPositionEncoding, TransformerDecoder
from ..utils.data_utils import fuse_goal

REGISTERED_MODELS = {}


def register_model(model_class):
    """Register a policy class with the registry."""
    model_name = model_class.__name__.lower()
    if model_name in REGISTERED_MODELS:
        raise ValueError("Cannot register duplicate policy ({})".format(model_name))

    REGISTERED_MODELS[model_name] = model_class


def get_model_class(model_name):
    """Get the policy class from the registry."""
    if model_name.lower() not in REGISTERED_MODELS:
        raise ValueError(
            "Policy class with name {} not found in registry".format(model_name)
        )
    return REGISTERED_MODELS[model_name.lower()]


def get_model_list():
    return REGISTERED_MODELS


class ModelMeta(type):
    """Metaclass for registering environments"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all models that should not be registered here.
        _unregistered_models = ["BaseModel"]

        if cls.__name__ not in _unregistered_models:
            register_model(cls)
        return cls


class BaseModel(nn.Module, metaclass=ModelMeta):
    def __init__(self, cfg):
        super().__init__()
        print('\nBuilding model...')
        if cfg.policy.embedding_type == 'ResNet':
            cfg.policy.embedding = 'resnet18'
            cfg.policy.embedding_dim = 512
        elif cfg.policy.embedding_type == 'ViT':
            cfg.policy.embedding = 'vit'
            cfg.policy.embedding_dim = 64
        elif cfg.policy.embedding_type == 'R3M':
            cfg.policy.embedding = 'r3m-small'
            cfg.policy.embedding_dim = 384
        elif cfg.policy.embedding_type == 'VC1':
            cfg.policy.embedding = 'vc1_vitb'
            cfg.policy.embedding_dim = 768
        elif cfg.policy.embedding_type == 'MVP':
            cfg.policy.embedding = 'mvp-small'
            cfg.policy.embedding_dim = 384
        elif cfg.policy.embedding_type == 'Voltron':
            cfg.policy.embedding = 'v-cond-small'
            cfg.policy.embedding_dim = 384
        elif cfg.policy.embedding_type == 'MPI':
            cfg.policy.embedding = 'mpi-small'
            cfg.policy.embedding_dim = 384
        else:
            raise ValueError

        self.device = cfg.train.device
        self.embedding_type = cfg.policy.embedding_type
        self.ft_method = cfg.train.ft_method
        self.cfg = cfg

        if cfg.env.add_proprio:
            self.extra_states_encoder = DynamicMLP([cfg.policy.extra_states_encoder.input_size, cfg.policy.extra_states_encoder.output_size])
            
        self.goal_encoder = None
        if cfg.data.goal_type in ["goal_cond", "goal_o_pos"]:
            self.goal_encoder = DynamicMLP([cfg.policy.goal_encoder.input_size, cfg.policy.goal_encoder.output_size])

        if cfg.policy.use_spatial:
            policy_head_input_size = cfg.policy.embedding_dim + cfg.policy.extra_states_encoder.input_size
            if cfg.data.goal_type in ['goal_cond', 'goal_o_pos']:
                policy_head_input_size += 3
            self.fuse_emb = DynamicMLP([int(x * policy_head_input_size) for x in cfg.policy.spatial_fuse.mlp_layer_dim])
        else:
            self.spatial_down_sample = nn.Linear(cfg.policy.embedding_dim, cfg.policy.spatial_down_sample.input_size)
            self.temporal_position_encoding = SinusoidalPositionEncoding(cfg.policy.temporal_fuse.input_size)
            self.fuse_emb = TransformerDecoder(
                input_size=cfg.policy.temporal_fuse.input_size,
                num_layers=cfg.policy.temporal_fuse.transformer_num_layers,
                num_heads=cfg.policy.temporal_fuse.transformer_num_heads,
                head_output_size=cfg.policy.temporal_fuse.transformer_head_output_size,
                mlp_hidden_size=cfg.policy.temporal_fuse.transformer_mlp_hidden_size,
                dropout=cfg.policy.temporal_fuse.transformer_dropout,
            )
            action_token = nn.Parameter(torch.randn(1, 1, cfg.policy.temporal_fuse.input_size))
            self.register_parameter("action_token", action_token)
            policy_head_input_size = cfg.policy.temporal_fuse.input_size
            
        self.policy_head = DeterministicHead(
            input_size=policy_head_input_size, 
            hidden_size=cfg.policy.policy_head.hidden_size,
            output_size=cfg.policy.policy_head.output_size,
            num_layers=cfg.policy.policy_head.num_layers,
        )
    
    def fuse_encode(self, x):
        if self.cfg.policy.use_spatial:
            x = self.fuse_emb(x)
            return x
        else:
            pos_emb = self.temporal_position_encoding(x)
            x = x + pos_emb.unsqueeze(1)  
            sh = x.shape
            self.fuse_emb.compute_mask(x.shape)

            x = rearrange(x, "b t n e -> b (t n) e")
            x = self.fuse_emb(x)           
            x = x.reshape(*sh)                          
            return x, x[:, :, 0]
        
    def spatial_encode(self, embeddings, extra, data):
        if extra != None:
            x = torch.cat([embeddings, extra], dim=-1)
        else:
            x = embeddings
        x = fuse_goal(x, data['input'], goal_type=self.cfg.data.goal_type, goal_encoder=self.goal_encoder)
        z = self.fuse_encode(x)     
        dist = self.policy_head(z)
        return x, z, dist
    
    def temporal_encode(self, embeddings, extra, data):
        embeddings = self.spatial_down_sample(embeddings).unsqueeze(1)  # (b, emb_dim) -> (b, t, emb_dim)
        if extra != None:
            extra = extra.unsqueeze(1)
            encoded = torch.stack([embeddings, extra], dim=-2)
        else:
            encoded = embeddings.unsqueeze(-2)   # (b, t, emb_dim) -> (b, t, 1, emb_dim)

        if self.cfg.data.goal_type == "goal_cond":
            goal = self.goal_encoder(data["input"]["o_goal_pos"].to(self.device)).unsqueeze(1).unsqueeze(2)
            encoded = torch.cat([encoded, goal], -2)
        elif self.cfg.data.goal_type == "goal_o_pos":
            goal = self.goal_encoder(data["input"]["o_goal_pos_rel"].to(self.device)).unsqueeze(1).unsqueeze(2)
            encoded = torch.cat([encoded, goal], -2)

        action_token = self.action_token.unsqueeze(0).expand(embeddings.shape[0], 1, -1, -1)  # (b, t, 1, emb_dim)
        x = torch.cat([action_token, encoded], -2)
        
        z, z0 = self.fuse_encode(x)
        dist = self.policy_head(z0).squeeze(1)
        return x, z, dist

    def forward(self, data, return_latent=False):
        cfg = self.cfg
        if cfg.train.ft_method == 'full_ft':
            embeddings = self.get_representations(data["input"]["img"].to(self.device))  # (b*t, emb_dim) -> (b, t, emb_dim)
        else:
            embeddings = data["input"]["embedding"].to(self.device)

        extra = None
        if cfg.env.add_proprio:
            extra = self.extra_states_encoder(data["input"]["ft_state"].to(self.device))

        if cfg.policy.use_spatial:
            x, z, dist = self.spatial_encode(embeddings, extra, data)
        else:
            x, z, dist = self.temporal_encode(embeddings, extra, data)

        if return_latent:
            return x, z, dist

        return dist

    def get_action(self, data):
        dist = self.forward(data)
        return dist.detach().squeeze(0).cpu().numpy()

    def compute_loss(self, data, reduction="mean"):
        dist = self.forward(data)
        loss = self.policy_head.loss_fn(dist, data["output"]["action"].to(self.device), reduction)
        return loss
    
    def reset(self):
        pass

    def save(self, path, epoch, optimizer, scheduler):
        state_dict = self.state_dict()
        if self.cfg.train.ft_method == "partial_ft":
            keys_to_remove = [key for key in state_dict if key.startswith('feature_extractor')]
            for key in keys_to_remove:
                del state_dict[key]

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, path)

    def load(self, path, optimizer=None, scheduler=None, map_location='cpu'):
        checkpoint = torch.load(path, map_location=map_location)
        epoch = checkpoint['epoch']
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if optimizer != None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler != None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return epoch

    def process_data(self, images):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if len(images.shape) == 3:
            images_tensor = images.unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
        else:
            images_tensor = images.permute(0, 3, 1, 2).float() / 255.0
        images_resized = F.resize(images_tensor, 256, interpolation=F.InterpolationMode.BICUBIC)
        images_cropped = F.center_crop(images_resized, 224)
        if self.embedding_type in ['R3M']:
            return images_cropped
        elif self.embedding_type in ['ResNet', 'ViT', 'VC1' , 'MVP', 'Voltron', 'MPI']:
            images_normalized = F.normalize(
                images_cropped, 
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
            return images_normalized
    
    def get_representations(self, imgs):
        if self.embedding_type in ['R3M']:
            embeddings = self.feature_extractor.get_representations(imgs * 255.0)
        elif self.embedding_type in ['ResNet', 'ViT', 'VC1']:
            embeddings = self.feature_extractor(imgs)
        elif self.embedding_type in ['MVP', 'Voltron']:
            embeddings = self.feature_extractor.get_representations(imgs)
        elif self.embedding_type in ['MPI']:
            imgs = torch.stack((imgs, imgs), dim=1)
            embeddings = self.feature_extractor.get_representations(imgs, None, with_lang_tokens=False)
        return embeddings
