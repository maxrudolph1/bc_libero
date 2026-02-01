import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np
from einops import rearrange

from .modules.mlp import DynamicMLP
from .modules.policy_head import DeterministicHead
from .modules.transformer_modules import SinusoidalPositionEncoding, TransformerDecoder
from ..utils.data_utils import fuse_embeddings

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
        elif cfg.policy.embedding_type == 'ViT_2':
            cfg.policy.embedding = 'vit'
            cfg.policy.embedding_dim = 128
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
        
        self.latent_queue = []
        if cfg.env.add_proprio:
            self.latent_queue_proprio = []
        self.history_window = cfg.env.history_window
        self.device = cfg.train.device
        self.embedding_type = cfg.policy.embedding_type
        self.cfg = cfg

        if cfg.env.add_proprio:
            self.extra_states_encoder = DynamicMLP([cfg.policy.extra_states_encoder.input_size, cfg.policy.extra_states_encoder.output_size])

        if cfg.policy.use_spatial:
            policy_head_input_size = (cfg.policy.embedding_dim + cfg.policy.extra_states_encoder.input_size) * cfg.env.history_window
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
        cfg= self.cfg
        if cfg.policy.use_spatial:
            x = fuse_embeddings(x, cfg.policy.spatial_fuse.fuse_method)   
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
        
    def forward(self, data, return_latent=False):
        cfg= self.cfg
        if cfg.train.ft_method == 'full_ft':
            b, t, c, h, w = data["images"].shape
            images = rearrange(data["images"], "b t c h w -> (b t) c h w")
            embeddings = self.get_representations(images.to(self.device)).view(b, t, -1)  # (b*t, emb_dim) -> (b, t, emb_dim)
        else:
            b, t, _ = data["embeddings"].shape
            embeddings = data["embeddings"].to(self.device)

        extra = None
        if cfg.env.add_proprio:
            extra = self.extra_states_encoder(data["extra_states"].reshape(b * t, -1).to(self.device))
            extra = extra.reshape(b, t, -1)

        if cfg.policy.use_spatial:
            if extra != None:
                x = torch.cat([embeddings, extra], dim=-1)
            else:
                x = embeddings
            z = self.fuse_encode(x)     
            dist = self.policy_head(z)
        else:
            embeddings = self.spatial_down_sample(embeddings.reshape(b*t, -1)).reshape(b, t, -1)
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

    def compute_loss(self, data, reduction="mean"):
        dist = self.forward(data)
        loss = self.policy_head.loss_fn(dist, data["actions"].to(self.device), reduction)
        return loss

    def reset(self):
        self.latent_queue = []
        if self.cfg.env.add_proprio:
            self.latent_queue_proprio = []

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
        if len(images.shape) == 3:
            images_tensor = torch.from_numpy(images.copy()).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
        else:
            images_tensor = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).float() / 255.0
        images_resized = F.resize(images_tensor, 256, interpolation=F.InterpolationMode.BICUBIC)
        images_cropped = F.center_crop(images_resized, 224)
        if self.embedding_type in ['R3M']:
            return images_cropped
        elif self.embedding_type in ['ResNet', 'ViT', 'ViT_2', 'VC1', 'MVP', 'Voltron', 'MPI']:
            images_normalized = F.normalize(
                images_cropped, 
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
            return images_normalized
        
    def get_representations(self, imgs):
        if self.embedding_type in ['R3M']:
            embeddings = self.feature_extractor.get_representations(imgs * 255.0)
        elif self.embedding_type in ['ResNet', 'ViT', 'ViT_2', 'VC1']:
            embeddings = self.feature_extractor(imgs)
        elif self.embedding_type in ['MVP', 'Voltron']:
            embeddings = self.feature_extractor.get_representations(imgs)
        elif self.embedding_type in ['MPI']:
            imgs = torch.stack((imgs, imgs), dim=1)
            embeddings = self.feature_extractor.get_representations(imgs, None, with_lang_tokens=False)
        return embeddings
    