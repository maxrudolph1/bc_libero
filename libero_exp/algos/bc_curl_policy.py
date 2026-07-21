import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.train_utils import setup_optimizer
from .bc_cardpol_policy import BC_CARDPOL_Policy


class RepresentationContrastive(nn.Module):
    """Projection head for contrastive learning on pooled spatial representations."""

    def __init__(self, rep_dim, proj_dim=64, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(rep_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.projector = nn.Linear(hidden_dim, proj_dim)

    def forward(self, x):
        h = self.encoder(x)
        return F.normalize(self.projector(h), dim=-1)


class BC_CURL_Policy(BC_CARDPOL_Policy):
    """
    CARD-style dual-task training with a SimCLR-style contrastive auxiliary loss
    on pooled spatial input representations (VAE-style single-observation input).
    """

    def build_model(self, cfg, shape_meta):
        super(BC_CARDPOL_Policy, self).build_model(cfg, shape_meta)
        rep_dim = self._get_rep_dim(cfg, shape_meta)
        proj_dim = cfg.train.get("curl_proj_dim", 64)
        hidden_dim = cfg.train.get(
            "curl_hidden_dim", cfg.train.get("rep_classifier_hidden", 256)
        )
        self.model.add_module(
            "rep_contrastive",
            RepresentationContrastive(
                rep_dim, proj_dim=proj_dim, hidden_dim=hidden_dim
            ),
        )
        self.curl_temperature = cfg.train.get("curl_temperature", 0.1)
        self.optimizer = setup_optimizer(cfg.train.optimizer, self.model)

    def _encode_pooled_representation(self, data, *, obs_key="obs", augmentation=None):
        x = self.get_input_representation(
            data, obs_key=obs_key, augmentation=augmentation
        )
        return self._pool_representation(x)

    @staticmethod
    def _symmetric_info_nce(query, key, temperature):
        labels = torch.arange(query.size(0), device=query.device)
        logits_qk = query @ key.T / temperature
        logits_kq = key @ query.T / temperature
        loss_qk = F.cross_entropy(logits_qk, labels)
        loss_kq = F.cross_entropy(logits_kq, labels)
        acc_qk = (logits_qk.argmax(dim=-1) == labels).float().mean()
        acc_kq = (logits_kq.argmax(dim=-1) == labels).float().mean()
        loss = 0.5 * (loss_qk + loss_kq)
        acc = 0.5 * (acc_qk + acc_kq)
        return loss, acc

    def compute_representation_loss(self, mixed_data, augmentation=None):
        if augmentation is None:
            augmentation = self.model.training

        x_a = self._encode_pooled_representation(
            mixed_data, obs_key="obs", augmentation=augmentation
        )
        x_b = self._encode_pooled_representation(
            mixed_data, obs_key="obs", augmentation=augmentation
        )
        q = self.model.rep_contrastive(x_a)
        k = self.model.rep_contrastive(x_b)
        loss, _ = self._symmetric_info_nce(q, k, self.curl_temperature)
        return loss

    def compute_representation_metrics(self, mixed_data, augmentation=None):
        if augmentation is None:
            augmentation = self.model.training

        x_a = self._encode_pooled_representation(
            mixed_data, obs_key="obs", augmentation=augmentation
        )
        x_b = self._encode_pooled_representation(
            mixed_data, obs_key="obs", augmentation=augmentation
        )
        q = self.model.rep_contrastive(x_a)
        k = self.model.rep_contrastive(x_b)
        loss, acc = self._symmetric_info_nce(q, k, self.curl_temperature)
        return {
            "rep_loss": loss.item(),
            "rep_acc": acc.item(),
        }

    def forward_backward(self, data):
        focused, mixed = self._split_batch(data)

        bc_loss = self.compute_bc_loss(focused)
        rep_loss = self.compute_representation_loss(mixed)
        rep_metrics = self.compute_representation_metrics(mixed)
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
        rep_loss = self.compute_representation_loss(mixed, augmentation=True)
        rep_metrics = self.compute_representation_metrics(mixed, augmentation=True)
        rep_scale = self.cfg.train.get("rep_loss_scale", 1.0)
        loss = bc_loss + rep_scale * rep_loss

        return {
            "loss": loss.item(),
            "bc_loss": bc_loss.item(),
            **rep_metrics,
        }
