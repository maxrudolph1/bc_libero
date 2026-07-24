"""
@inproceedings{ma2023vip,
  title={VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training},
  author={Ma, Yecheng Jason and Sodhani, Shagun and Jayaraman, Dinesh and Bastani, Osbert and Kumar, Vikash and Zhang, Amy},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

VIP as an auxiliary representation-learning objective (analogous to the CardPol /
VAE / CURL baselines), adapted from https://github.com/facebookresearch/vip.

VIP learns a goal-conditioned value V(s; g) = -||phi(s) - phi(g)||_2 from
action-free sub-trajectories via the KL-regularized dual of goal-conditioned RL.
Here phi(.) is a projection head on top of the policy's pooled visual
representation, and the frames (o_0, g, o_t, o_{t+1}) come from the dual-task
loader running in the 'vip' mixed mode (see DualTaskBatchDataset).
"""

import torch
import torch.nn as nn

from ..utils.train_utils import setup_optimizer
from .bc_cardpol_policy import BC_CARDPOL_Policy

EPSILON = 1e-8


class RepresentationVIP(nn.Module):
    """Projection head mapping a pooled spatial representation into the VIP
    value-embedding space (mirrors VIP's ``convnet.fc`` projection)."""

    def __init__(self, rep_dim, embed_dim=1024, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rep_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class BC_VIP_Policy(BC_CARDPOL_Policy):
    """
    CARD-style dual-task training with a Value-Implicit Pre-training (VIP)
    auxiliary loss on pooled spatial input representations.

      - focused: standard BC on the configured task
      - mixed:   VIP value-implicit objective on (o_0, g, o_t, o_{t+1}) frames
                 sampled per trajectory (requires data.dual_task.mixed_mode=vip)
    """

    def build_model(self, cfg, shape_meta):
        # Skip CardPol's task-pair classifier; build the base policy directly.
        super(BC_CARDPOL_Policy, self).build_model(cfg, shape_meta)
        rep_dim = self._get_rep_dim(cfg, shape_meta)
        embed_dim = cfg.train.get("vip_embed_dim", 1024)
        hidden_dim = cfg.train.get(
            "vip_hidden_dim", cfg.train.get("rep_classifier_hidden", 256)
        )
        self.model.add_module(
            "rep_vip",
            RepresentationVIP(rep_dim, embed_dim=embed_dim, hidden_dim=hidden_dim),
        )
        self.vip_gamma = cfg.train.get("vip_gamma", 0.98)
        # VIP's original standalone pretraining uses l1/l2 embedding regularizers with
        # weight 1.0, but there they are the dominant training signal. As an auxiliary
        # loss added to BC (and controlled by rep_loss_scale) those raw norms would
        # swamp the O(1) value objective, so they default to 0 here and stay tunable.
        self.vip_l2weight = cfg.train.get("vip_l2weight", 0.0)
        self.vip_l1weight = cfg.train.get("vip_l1weight", 0.0)
        self.vip_num_negatives = cfg.train.get("vip_num_negatives", 0)
        self.optimizer = setup_optimizer(cfg.train.optimizer, self.model)

    def _encode_vip_embedding(self, data, *, obs_key, augmentation=None):
        x = self.get_input_representation(
            data, obs_key=obs_key, augmentation=augmentation
        )
        x = self._pool_representation(x)
        return self.model.rep_vip(x)

    @staticmethod
    def _value(a, b):
        """VIP value / similarity: V(a; b) = -||a - b||_2."""
        return -torch.linalg.norm(a - b, dim=-1)

    def _vip_loss_and_metrics(self, mixed_data, augmentation=None):
        if "reward" not in mixed_data:
            raise ValueError(
                "Mixed batch is missing 'reward'. BC_VIP_Policy requires "
                "data.dual_task.mixed_mode=vip so the loader emits VIP frames."
            )

        e0 = self._encode_vip_embedding(
            mixed_data, obs_key="obs_initial", augmentation=augmentation
        )  # phi(o_0)
        eg = self._encode_vip_embedding(
            mixed_data, obs_key="obs_goal", augmentation=augmentation
        )  # phi(g)
        es0 = self._encode_vip_embedding(
            mixed_data, obs_key="obs", augmentation=augmentation
        )  # phi(o_t)
        es1 = self._encode_vip_embedding(
            mixed_data, obs_key="obs_next", augmentation=augmentation
        )  # phi(o_{t+1})

        reward = mixed_data["reward"].to(device=e0.device, dtype=e0.dtype)

        V_0 = self._value(e0, eg)
        V_s = self._value(es0, eg)
        V_s_next = self._value(es1, eg)

        # KL-regularized dual VIP objective (Ma et al., 2023):
        #   (1 - gamma) * E[-V(o_0; g)] + log E[exp(-(r + gamma V(o'; g) - V(o; g)))]
        vip_loss = (1.0 - self.vip_gamma) * (-V_0.mean())
        td = reward + self.vip_gamma * V_s_next - V_s
        vip_loss = vip_loss + torch.log(EPSILON + torch.mean(torch.exp(-td)))

        if self.vip_num_negatives > 0:
            neg_td = []
            for _ in range(self.vip_num_negatives):
                perm = torch.randperm(es0.size(0), device=es0.device)
                V_s_neg = self._value(es0[perm], eg)
                V_s_next_neg = self._value(es1[perm], eg)
                r_neg = -torch.ones_like(V_s_neg)
                neg_td.append(r_neg + self.vip_gamma * V_s_next_neg - V_s_neg)
            neg_td = torch.cat(neg_td)
            vip_loss = vip_loss + torch.log(EPSILON + torch.mean(torch.exp(-neg_td)))

        # L1 / L2 regularization on the embeddings (as in VIP).
        all_emb = torch.stack([e0, eg, es0, es1], dim=0)
        l2loss = torch.linalg.norm(all_emb, ord=2, dim=-1).mean()
        l1loss = torch.linalg.norm(all_emb, ord=1, dim=-1).mean()
        loss = vip_loss + self.vip_l2weight * l2loss + self.vip_l1weight * l1loss

        metrics = {
            "rep_loss": loss.item(),
            "vip_loss": vip_loss.item(),
            "vip_l2loss": l2loss.item(),
            "vip_l1loss": l1loss.item(),
            "vip_value_init": V_0.mean().item(),
            "vip_value_cur": V_s.mean().item(),
            "vip_value_next": V_s_next.mean().item(),
        }
        return loss, metrics

    def forward_backward(self, data):
        focused, mixed = self._split_batch(data)

        bc_loss = self.compute_bc_loss(focused)
        rep_loss, rep_metrics = self._vip_loss_and_metrics(mixed)
        rep_scale = self.cfg.train.get("rep_loss_scale", 1.0)
        loss = bc_loss + rep_scale * rep_loss

        self.optimizer.zero_grad()
        self.fabric.backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.cfg.train.grad_clip
        )
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "bc_loss": bc_loss.item(),
            **rep_metrics,
        }

    @torch.no_grad()
    def compute_eval_batch_metrics(self, data):
        focused, mixed = self._split_batch(data)
        bc_loss = self.compute_bc_loss(focused, augmentation=False)
        rep_loss, rep_metrics = self._vip_loss_and_metrics(mixed, augmentation=False)
        rep_scale = self.cfg.train.get("rep_loss_scale", 1.0)
        loss = bc_loss + rep_scale * rep_loss

        return {
            "loss": loss.item(),
            "bc_loss": bc_loss.item(),
            **rep_metrics,
        }
