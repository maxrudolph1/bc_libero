import torch

from .base_trifinger import BaseAlgo_Trifinger


class BC_Policy_Trifinger(BaseAlgo_Trifinger):
    def __init__(self, cfg, inference=False):
        super().__init__(cfg, inference)

    def forward_backward(self, data):
        bc_loss = self.compute_loss(data)
        loss = bc_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.train.grad_clip)
        self.optimizer.step()

        ret_dict = {
            "loss": loss.item(),
            "bc_loss": bc_loss.item(),
        }

        return ret_dict
    
    def compute_loss(self, data):
        x, z, dist = self.model(data, return_latent=True)
        bc_loss = self.model.policy_head.loss_fn(dist, data["output"]["action"].to(self.device), reduction="mean")
        return bc_loss

