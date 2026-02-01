import torch
import torch.nn as nn

from .base_trifinger import BaseAlgo_Trifinger
from ..utils.mine_utils import Mine


class BC_IB_Policy_Trifinger(BaseAlgo_Trifinger):
    def __init__(self, cfg, inference=False):
        super().__init__(cfg, inference)
        if not inference:
            if cfg.policy.use_spatial:
                input_dim = cfg.policy.embedding_dim + cfg.policy.extra_states_encoder.input_size
                if cfg.data.goal_type in ['goal_cond', 'goal_o_pos']:
                    input_dim += 3
            else:
                mul = 1
                if cfg.env.add_proprio:
                    mul += 1
                if cfg.data.goal_type in ['goal_cond', 'goal_o_pos']:
                    mul += 1
                input_dim = cfg.policy.spatial_down_sample.input_size * (mul+1) * cfg.data.history_window
            hidden_dim = 512
            mine_model =  nn.Sequential(
                nn.Linear(input_dim*2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.mine = Mine(
                model = mine_model,
                loss_type = 'mine',  # mine_biased, fdiv
            )
            self.mine.model = self.mine.model.to(self.device)
            self.mine_optimizer = torch.optim.Adam(self.mine.model.parameters(), lr=1e-5)

    def forward_backward(self, data):
        bc_loss, mi_loss = self.compute_loss(data)
        loss = bc_loss + mi_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.train.grad_clip)
        self.optimizer.step()

        mi_loss_2 = self.compute_mine_model_loss(data)
        self.mine_optimizer.zero_grad()
        mi_loss_2.backward()
        torch.nn.utils.clip_grad_norm_(self.mine.model.parameters(), max_norm=self.cfg.train.grad_clip)
        self.mine_optimizer.step()

        ret_dict = {
            "loss": loss.item(),
            "bc_loss": bc_loss.item(),
            "mi_loss": mi_loss.item(),
            "mine": -mi_loss_2.item(),
        }

        return ret_dict
    
    def compute_loss(self, data):
        x, z, dist = self.model(data, return_latent=True)
        bc_loss = self.model.policy_head.loss_fn(dist, data["output"]["action"].to(self.device), reduction="mean")
        mi_loss = self.mine.get_mi(x, z) * self.cfg.train.mi_loss_scale
        return bc_loss, mi_loss
    
    def compute_mine_model_loss(self, data):
        with torch.no_grad():
            x, z, _ = self.model(data, return_latent=True)
        x, z = x.detach(), z.detach()
        mi_loss = self.mine(x, z) * self.cfg.train.mine_mi_loss_scale

        return mi_loss

