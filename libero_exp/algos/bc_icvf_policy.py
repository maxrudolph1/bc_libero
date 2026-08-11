"""
@inproceedings{ghosh2023icvf,
  title={Reinforcement Learning from Passive Data via Latent Intentions},
  author={Ghosh, Dibya and Bhateja, Chethan and Levine, Sergey},
  booktitle={International Conference on Machine Learning},
  year={2023}
}

ICVF as an auxiliary representation-learning objective (analogous to the CardPol /
VAE / CURL / VIP baselines), adapted from dibyaghosh/icvf_release and the AFR
PyTorch port.

ICVF learns a multi-linear intention-conditioned value
V(s, s_+, z) = phi(s)^T T(z) psi(s_+) from action-free transitions via an
IQL-style expectile TD objective. Here phi/psi/T are small MLPs on top of the
policy's pooled visual representation, and the frames (s, s', s_+, z) come from
the dual-task loader running in the 'icvf' mixed mode (see DualTaskBatchDataset).
"""

import copy

import torch
import torch.nn as nn

from ..utils.train_utils import setup_optimizer
from .bc_cardpol_policy import BC_CARDPOL_Policy


class MultilinearVF(nn.Module):
    """Low-rank multi-linear ICVF head: V(s, s_+, z) = phi(s)^T T(z) psi(s_+).

    Matches the MultilinearVF parameterization in icvf_release / AFR:
    T(z) ≈ diag(Tz) A B diag(Tz) with shared psi for outcomes and intentions.
    """

    def __init__(self, rep_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.phi_net = self._mlp(rep_dim, hidden_dim, num_layers)
        self.psi_net = self._mlp(rep_dim, hidden_dim, num_layers)
        self.T_net = self._mlp(hidden_dim, hidden_dim, num_layers)
        self.matrix_a = nn.Linear(hidden_dim, hidden_dim)
        self.matrix_b = nn.Linear(hidden_dim, hidden_dim)

    @staticmethod
    def _mlp(in_dim, hidden_dim, num_layers):
        layers = []
        dim = in_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU(inplace=True)])
            dim = hidden_dim
        return nn.Sequential(*layers)

    def forward(self, observations, outcomes, intents):
        """
        Args:
            observations: (B, D) pooled state features
            outcomes: (B, D) pooled outcome / goal features (s_+)
            intents: (B, D) pooled intention features (z)
        """
        phi = self.phi_net(observations)
        psi = self.psi_net(outcomes)
        z = self.psi_net(intents)
        Tz = self.T_net(z)
        phi_z = self.matrix_a(Tz * phi)
        psi_z = self.matrix_b(Tz * psi)
        return (phi_z * psi_z).sum(dim=-1)


class BC_ICVF_Policy(BC_CARDPOL_Policy):
    """
    CARD-style dual-task training with an Intention-Conditioned Value Function
    (ICVF) auxiliary loss on pooled spatial input representations.

      - focused: standard BC on the configured task
      - mixed:   ICVF expectile TD objective on (s, s', s_+, z)
                 (requires data.dual_task.mixed_mode=icvf)
    """

    def build_model(self, cfg, shape_meta):
        # Skip CardPol's task-pair classifier; build the base policy directly.
        super(BC_CARDPOL_Policy, self).build_model(cfg, shape_meta)
        rep_dim = self._get_rep_dim(cfg, shape_meta)
        hidden_dim = cfg.train.get(
            "icvf_hidden_dim", cfg.train.get("rep_classifier_hidden", 256)
        )
        num_layers = cfg.train.get("icvf_num_layers", 2)
        self.model.add_module(
            "rep_icvf",
            MultilinearVF(rep_dim, hidden_dim=hidden_dim, num_layers=num_layers),
        )
        self.target_icvf = copy.deepcopy(self.model.rep_icvf)
        for p in self.target_icvf.parameters():
            p.requires_grad_(False)
        # fabric.setup only moves self.model; keep the EMA target on the train device.
        self.target_icvf.to(cfg.train.device)

        self.icvf_gamma = cfg.train.get("icvf_gamma", 0.99)
        self.icvf_expectile = cfg.train.get("icvf_expectile", 0.9)
        self.icvf_target_tau = cfg.train.get("icvf_target_tau", 0.005)
        self.optimizer = setup_optimizer(cfg.train.optimizer, self.model)

    def _encode_pooled(self, data, *, obs_key, augmentation=None):
        x = self.get_input_representation(
            data, obs_key=obs_key, augmentation=augmentation
        )
        return self._pool_representation(x)

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """IQL / ICVF asymmetric L2: up-weight positive advantages."""
        weight = torch.where(adv >= 0, expectile, 1.0 - expectile)
        return weight * diff.pow(2)

    def _soft_update_target(self):
        tau = self.icvf_target_tau
        with torch.no_grad():
            for target_param, param in zip(
                self.target_icvf.parameters(), self.model.rep_icvf.parameters()
            ):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def _icvf_loss_and_metrics(self, mixed_data, augmentation=None):
        required = (
            "obs_next",
            "obs_goal",
            "obs_desired_goal",
            "reward",
            "desired_reward",
            "mask",
            "desired_mask",
        )
        missing = [k for k in required if k not in mixed_data]
        if missing:
            raise ValueError(
                "Mixed batch is missing ICVF fields "
                f"{missing}. BC_ICVF_Policy requires "
                "data.dual_task.mixed_mode=icvf so the loader emits ICVF frames."
            )

        # Online encodings (grads flow into the shared visual backbone).
        s = self._encode_pooled(
            mixed_data, obs_key="obs", augmentation=augmentation
        )
        s_next = self._encode_pooled(
            mixed_data, obs_key="obs_next", augmentation=augmentation
        )
        s_plus = self._encode_pooled(
            mixed_data, obs_key="obs_goal", augmentation=augmentation
        )
        z = self._encode_pooled(
            mixed_data, obs_key="obs_desired_goal", augmentation=augmentation
        )

        device = s.device
        dtype = s.dtype
        if next(self.target_icvf.parameters()).device != device:
            self.target_icvf.to(device)

        reward = mixed_data["reward"].to(device=device, dtype=dtype)
        desired_reward = mixed_data["desired_reward"].to(device=device, dtype=dtype)
        mask = mixed_data["mask"].to(device=device, dtype=dtype)
        desired_mask = mixed_data["desired_mask"].to(device=device, dtype=dtype)

        # TD target for outcome s_+: r(s, s_+) + gamma * mask * V_bar(s', s_+, z)
        with torch.no_grad():
            next_v_gz = self.target_icvf(s_next, s_plus, z)
            q_gz = reward + self.icvf_gamma * mask * next_v_gz

            # Advantage of s -> s' under intention z (also fully target-network).
            next_v_zz = self.target_icvf(s_next, z, z)
            v_zz = self.target_icvf(s, z, z)
            adv = desired_reward + self.icvf_gamma * desired_mask * next_v_zz - v_zz

        v_gz = self.model.rep_icvf(s, s_plus, z)
        value_loss = self.expectile_loss(adv, q_gz - v_gz, self.icvf_expectile).mean()

        metrics = {
            "rep_loss": value_loss.item(),
            "icvf_loss": value_loss.item(),
            "icvf_adv": adv.mean().item(),
            "icvf_v_gz": v_gz.mean().item(),
            "icvf_v_zz": v_zz.mean().item(),
            "icvf_reward": reward.mean().item(),
            "icvf_accept_prob": (adv >= 0).float().mean().item(),
        }
        return value_loss, metrics

    def forward_backward(self, data):
        focused, mixed = self._split_batch(data)

        bc_loss = self.compute_bc_loss(focused)
        rep_loss, rep_metrics = self._icvf_loss_and_metrics(mixed)
        rep_scale = self.cfg.train.get("rep_loss_scale", 1.0)
        loss = bc_loss + rep_scale * rep_loss

        self.optimizer.zero_grad()
        self.fabric.backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.cfg.train.grad_clip
        )
        self.optimizer.step()
        self._soft_update_target()

        return {
            "loss": loss.item(),
            "bc_loss": bc_loss.item(),
            **rep_metrics,
        }

    @torch.no_grad()
    def compute_eval_batch_metrics(self, data):
        focused, mixed = self._split_batch(data)
        bc_loss = self.compute_bc_loss(focused, augmentation=False)
        rep_loss, rep_metrics = self._icvf_loss_and_metrics(mixed, augmentation=False)
        rep_scale = self.cfg.train.get("rep_loss_scale", 1.0)
        loss = bc_loss + rep_scale * rep_loss

        return {
            "loss": loss.item(),
            "bc_loss": bc_loss.item(),
            **rep_metrics,
        }
