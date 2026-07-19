import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.train_utils import setup_optimizer
from .bc_cardpol_policy import BC_CARDPOL_Policy


class RepresentationVAE(nn.Module):
    """Variational autoencoder on pooled spatial input representations."""

    def __init__(self, rep_dim, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(rep_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, rep_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class BC_VAE_Policy(BC_CARDPOL_Policy):
    """
    CARD-style dual-task training with a VAE auxiliary loss on spatial input
    representations instead of task-pair classification.
    """

    def build_model(self, cfg, shape_meta):
        super(BC_CARDPOL_Policy, self).build_model(cfg, shape_meta)
        rep_dim = self._get_rep_dim(cfg, shape_meta)
        latent_dim = cfg.train.get("vae_latent_dim", 64)
        hidden_dim = cfg.train.get("vae_hidden_dim", cfg.train.get("rep_classifier_hidden", 256))
        self.model.add_module(
            "rep_vae",
            RepresentationVAE(rep_dim, latent_dim=latent_dim, hidden_dim=hidden_dim),
        )
        self.optimizer = setup_optimizer(cfg.train.optimizer, self.model)

    @staticmethod
    def _vae_loss(recon, target, mu, logvar):
        recon_loss = F.mse_loss(recon, target, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss, recon_loss, kl_loss

    def compute_representation_loss(self, x, x_future, mixed_data):
        del x_future, mixed_data
        x = self._pool_representation(x)
        recon, mu, logvar = self.model.rep_vae(x)
        loss, _, _ = self._vae_loss(recon, x, mu, logvar)
        return loss

    def compute_representation_metrics(self, x, x_future, mixed_data):
        del x_future, mixed_data
        x = self._pool_representation(x)
        recon, mu, logvar = self.model.rep_vae(x)
        loss, recon_loss, kl_loss = self._vae_loss(recon, x, mu, logvar)
        return {
            "rep_loss": loss.item(),
            "rep_recon_loss": recon_loss.item(),
            "rep_kl_loss": kl_loss.item(),
        }

    def forward_backward(self, data):
        focused, mixed = self._split_batch(data)

        bc_loss = self.compute_bc_loss(focused)
        x = self.get_input_representation(mixed, obs_key="obs")
        x_future = self.get_input_representation(mixed, obs_key="obs_future")
        rep_loss = self.compute_representation_loss(x, x_future, mixed)
        rep_metrics = self.compute_representation_metrics(x, x_future, mixed)
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
        x = self.get_input_representation(mixed, augmentation=False)
        x_future = self.get_input_representation(
            mixed, augmentation=False, obs_key="obs_future"
        )
        rep_loss = self.compute_representation_loss(x, x_future, mixed)
        rep_metrics = self.compute_representation_metrics(x, x_future, mixed)
        rep_scale = self.cfg.train.get("rep_loss_scale", 1.0)
        loss = bc_loss + rep_scale * rep_loss

        return {
            "loss": loss.item(),
            "bc_loss": bc_loss.item(),
            **rep_metrics,
        }
