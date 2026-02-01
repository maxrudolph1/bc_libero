import torch
from tqdm import tqdm
from .base import BaseAlgo

class BC_MEE_Policy(BaseAlgo):
    def __init__(self, cfg, inference=False, device='cuda'):
        super().__init__(cfg, inference, device)

    def forward_backward(self, data):
        loss_dict = self.compute_loss(data)
        if self.training and (self.epoch >= self.cfg.train.n_epochs / 3):
            loss = loss_dict['bc_loss'] + loss_dict['mee_loss'] * self.cfg.train.mee_weight
        else:
            loss = loss_dict['bc_loss']
        self.optimizer.zero_grad()
        self.fabric.backward(loss)
        self.optimizer.step()

        ret_dict = {
            "loss": loss.detach(),
            "bc_loss": loss_dict['bc_loss'].detach(),
            "mee_loss": loss_dict['mee_loss'].detach(),
        }

        return ret_dict

    def compute_loss(self, data, augmentation=None):
        data = self.model.preprocess_input(data, augmentation=augmentation)
        x, z, dist = self.model(data, return_latent=True)

        if self.cfg.policy.policy_type == 'BCMLPPolicy':
            bc_loss = self.model.policy_head.loss_fn(dist, data["actions"][:, -1], reduction="mean")
            mee_loss = self.simple_minimum_error_entropy_loss(dist, data["actions"][:, -1], sigma=self.cfg.train.sigma)
        elif self.cfg.policy.policy_type == 'BCDPPolicy':
            repeated_diffusion_steps = self.cfg.policy.policy_head.network_kwargs.repeated_diffusion_steps
            actions = data["actions"]
            actions_repeated = actions.repeat(repeated_diffusion_steps, 1, 1)
            dist = dist.mean(dim=1, keepdim=True)
            features_repeated = dist.repeat(repeated_diffusion_steps, 1, 1)
            bc_loss, mee_loss = self.model.policy_head.loss(actions_repeated, features_repeated,
                                                            sigma=self.cfg.train.sigma)
        else:
            bc_loss = self.model.policy_head.loss_fn(dist, data["actions"], reduction="mean")
            mee_loss = self.simple_minimum_error_entropy_loss(dist, data["actions"], sigma=self.cfg.train.sigma)

        return {
            'bc_loss': bc_loss,
            'mee_loss': mee_loss,
        }

    def simple_minimum_error_entropy_loss(self, y_pred, y_true, sigma=0.5, eps=1e-8):
        # e: [batch, dim]
        e = y_pred - y_true
        e = e.view(e.size(0), -1)
        n = e.size(0)

        # pairwise differences
        diff = e.unsqueeze(1) - e.unsqueeze(0)  # [n, n, d]
        dist_sq = (diff ** 2).sum(dim=-1)  # [n, n]
        kernel = torch.exp(-dist_sq / (2 * sigma ** 2))

        # Renyi quadratic entropy estimator
        loss = -torch.log(kernel.sum() / (n ** 2) + eps)
        return loss

    @torch.no_grad()
    def evaluate(self, tag="val"):
        cfg = self.cfg
        tot_loss_dict, tot_items = {}, 0
        self.model.eval()

        print('Evaluating...')
        for data in tqdm(self.val_loader):
            loss_dict = self.compute_loss(data, augmentation=False)
            mee_loss = loss_dict['mee_loss']
            bc_loss = loss_dict['bc_loss']
            loss = bc_loss + mee_loss

            ret_dict = {
                "loss": loss.item(),
                "bc_loss": bc_loss.item(),
                "mee_loss": mee_loss.item(),
                "mmd": 0.0,
                "kl_div": 0.0,
            }

            for k, v in ret_dict.items():
                if k not in tot_loss_dict:
                    tot_loss_dict[k] = 0.0
                tot_loss_dict[k] += v
            tot_items += 1

            if cfg.train.debug:
                break

        out_dict = {}
        for k, v in tot_loss_dict.items():
            out_dict[f"{tag}/{k}"] = v / max(tot_items, 1)

        return out_dict