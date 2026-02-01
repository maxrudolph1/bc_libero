import os
import torch
from torchvision import models

from r3m import load_r3m_reproduce
from voltron import load as load_voltron
from vc_models.models.vit.model_utils import load_model as load_vc1
from mpi import load_mpi


def get_downloaded_model(cfg):
    if cfg.embedding == 'resnet50':
        resnet50 = models.resnet50(pretrained=False)
        state_dict = torch.load(os.path.join(cfg.embedding_dir, 'resnet', 'resnet50-11ad3fa6.pth'))
        resnet50.load_state_dict(state_dict)
        return resnet50
    elif cfg.embedding == 'vit_b_16':
        vit_b_16 = models.vit_b_16(pretrained=False)
        state_dict = torch.load(os.path.join(cfg.embedding_dir, 'vit', 'vit_b_16-c867db91.pth'))
        vit_b_16.load_state_dict(state_dict)
        return vit_b_16
    elif 'r3m' in cfg.embedding:   # ["r3m-rn50", "r3m-small"]
        if cfg.embedding == 'r3m-rn50':
            r3m_rn50 = load_r3m_reproduce("r3m", load_path=os.path.join(cfg.embedding_dir, 'r3m', cfg.embedding))
            return r3m_rn50
        elif cfg.embedding == '"r3m-small':
            r3m_vit_small = load_voltron("r-r3m-vit", freeze=False, only_return_model=True,
                                         load_path=os.path.join(cfg.embedding_dir, 'r3m', cfg.embedding))
            return r3m_vit_small
        else:
            raise ValueError("R3M model type is wrong! The repo only suits for [\"r3m-rn50\", \"r3m-small\"].")
    elif 'mvp' in cfg.embedding:
        if cfg.embedding == 'mvp-small':
            mvp_vit_small = load_voltron("r-mvp", freeze=False, only_return_model=True,
                                         load_path=os.path.join(cfg.embedding_dir, 'r3m', cfg.embedding))
            return mvp_vit_small
        else:
            raise ValueError("MVP model type is wrong! The repo only suits for [\"mvp-small\"].")
    elif 'v-cond' in cfg.embedding:
        if cfg.embedding == 'v-cond-small':  
            v_cond_small_model = load_voltron("v-cond", freeze=False, only_return_model=True,
                                              load_path=os.path.join(cfg.embedding_dir, 'voltron', cfg.embedding))
            return v_cond_small_model   
        else:
            raise ValueError("Voltron model type is wrong! The repo only suits for [\"v-cond-small\"].")
    elif 'vc1' in cfg.embedding:
        if cfg.embedding == 'vc1_vitb' or 'vc1_vitl':
            vc1_vit_model, _, _, _ = load_vc1(cfg.embedding, load_path=os.path.join(cfg.embedding_dir, 'vc-1'))
            return vc1_vit_model
        else:
            raise ValueError("VC-1 model type is wrong! The repo only suits for [\"vc1_vitb\"].")
    elif 'mpi' in cfg.embedding:
        if cfg.embedding == 'mpi-small' or 'mpi-base':
            mpi_model = load_mpi(load_path=os.path.join(cfg.embedding_dir, 'mpi', cfg.embedding), freeze=False, 
                                 language_model_path="/baishuanghao/model/distilbert-base-uncased")
            return mpi_model
        else:
            raise ValueError("MPI model type is wrong! The repo only suits for [\"mpi-small\", \"mpi-base\"].")
    else:
        raise ValueError("Model type is wrong! \
                The repo only suits for [\"ResNet50\", \"ViT-B-16\", \"R3M\", \"MVP\", \"Voltron\", \"VC-1\", \"MPI\"]!")
    
    
def get_undownloaded_model(cfg):
    pass

