import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.train_utils import setup_optimizer
from .bc_cardpol_policy import BC_CARDPOL_Policy


class TaskProjector(nn.Module):
    """Project pooled spatial input representations (optionally paired with x') to the unit sphere."""

    def __init__(self, rep_dim, proj_dim=64, hidden_dim=256, use_future=False):
        super().__init__()
        self.use_future = use_future
        in_dim = 2 * rep_dim if use_future else rep_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x, x_future=None):
        h = torch.cat([x, x_future], dim=-1) if self.use_future else x
        return F.normalize(self.net(h), dim=-1)


class BC_SUPCON_Policy(BC_CARDPOL_Policy):
    """
    Contrastive counterpart to CardPol: same dual-task batches and policy-identity
    supervision, but the task classifier is replaced by a supervised contrastive
    (multi-positive InfoNCE) loss. Mixed-batch samples from the same task are
    positives; samples from different tasks are negatives.

    By default each mixed sample is a single frame x (``train.supcon_use_future=false``);
    set it true to embed the (x, x') pair as CardPol does. Use
    ``data.dual_task.mixed_per_sample`` to enlarge the contrastive batch
    independently of the BC batch.
    """

    def build_model(self, cfg, shape_meta):
        super(BC_CARDPOL_Policy, self).build_model(cfg, shape_meta)
        rep_dim = self._get_rep_dim(cfg, shape_meta)
        self.supcon_use_future = bool(cfg.train.get("supcon_use_future", False))
        self.model.add_module(
            "rep_task_projector",
            TaskProjector(
                rep_dim,
                proj_dim=cfg.train.get("supcon_proj_dim", 64),
                hidden_dim=cfg.train.get("supcon_hidden_dim", 256),
                use_future=self.supcon_use_future,
            ),
        )
        self.supcon_temperature = cfg.train.get("supcon_temperature", 0.1)
        self.optimizer = setup_optimizer(cfg.train.optimizer, self.model)

    @staticmethod
    def _supcon_loss(z, labels, temperature):
        """
        Supervised contrastive loss (Khosla et al., 2020, L_out).

        For each anchor i, positives are the other batch samples with the same
        task id and the denominator runs over every j != i. Anchors without a
        positive in the batch are skipped. Also returns 1-NN task accuracy
        (does the most similar other sample share the anchor's task id?).
        """
        n = z.size(0)
        self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        logits = (z @ z.T / temperature).masked_fill(self_mask, float("-inf"))
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        pos_mask = (labels[:, None] == labels[None, :]) & ~self_mask
        n_pos = pos_mask.sum(dim=1)
        has_pos = n_pos > 0
        pos_log_prob = log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1)
        if has_pos.any():
            loss = -(pos_log_prob[has_pos] / n_pos[has_pos]).mean()
        else:
            # e.g. a tiny final eval batch with all-distinct task ids
            loss = z.sum() * 0.0

        nn_idx = logits.argmax(dim=1)
        acc = (labels[nn_idx] == labels).float().mean()
        return loss, acc

    def compute_supcon(self, mixed_data, augmentation=None):
        """Return (loss, 1-NN accuracy) for the mixed batch."""
        if "task_id" not in mixed_data:
            raise ValueError(
                "Mixed batch is missing 'task_id'. "
                "Use DualTaskBatchDataset with data.dual_task.enable=true."
            )
        x = self._pool_representation(
            self.get_input_representation(mixed_data, augmentation=augmentation)
        )
        x_future = None
        if self.supcon_use_future:
            x_future = self._pool_representation(
                self.get_input_representation(
                    mixed_data, augmentation=augmentation, obs_key="obs_future"
                )
            )
        z = self.model.rep_task_projector(x, x_future)
        task_ids = mixed_data["task_id"].long().to(device=z.device)
        return self._supcon_loss(z, task_ids, self.supcon_temperature)

    def forward_backward(self, data):
        focused, mixed = self._split_batch(data)

        bc_loss = self.compute_bc_loss(focused)
        rep_loss, rep_acc = self.compute_supcon(mixed)
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
            "rep_loss": rep_loss.item(),
            "rep_acc": rep_acc.item(),
        }

    @torch.no_grad()
    def compute_eval_batch_metrics(self, data):
        focused, mixed = self._split_batch(data)
        bc_loss = self.compute_bc_loss(focused, augmentation=False)
        rep_loss, rep_acc = self.compute_supcon(mixed, augmentation=False)
        rep_scale = self.cfg.train.get("rep_loss_scale", 1.0)
        loss = bc_loss + rep_scale * rep_loss

        return {
            "loss": loss.item(),
            "bc_loss": bc_loss.item(),
            "rep_loss": rep_loss.item(),
            "rep_acc": rep_acc.item(),
        }
