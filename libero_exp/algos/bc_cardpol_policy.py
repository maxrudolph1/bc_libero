import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from libero.libero.benchmark import get_benchmark
from robomimic.utils import tensor_utils as TensorUtils
from torch.utils.data import DataLoader, RandomSampler

from ..data.get_dataset import (
    DualTaskBatchDataset,
    collate_dual_task_batch,
    validate_dual_task_cfg,
)
from ..utils.train_utils import setup_optimizer
from .base import BaseAlgo


class TaskPairClassifier(nn.Module):
    """Predict task id from a pair of visual representations (z, z')."""

    def __init__(self, rep_dim, n_tasks, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * rep_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_tasks),
        )

    def forward(self, z, z_future):
        return self.net(torch.cat([z, z_future], dim=-1))


class BC_CARDPOL_Policy(BaseAlgo):
    """
    CARD-style training with dual-task batches:
      - focused: standard BC on the configured task
      - mixed: auxiliary loss on visual representations only
    """

    def __init__(self, cfg, inference=False, device="cuda"):
        if not inference:
            if not cfg.data.dual_task.enable:
                raise ValueError(
                    "BC_CARDPOL_Policy requires data.dual_task.enable=true. "
                    "Set data.dual_task.focused_task_id to the target task index."
                )
            validate_dual_task_cfg(cfg)
        super().__init__(cfg, inference, device)

    def build_dataloader(self, cfg):
        if not cfg.data.dual_task.enable:
            raise ValueError(
                "BC_CARDPOL_Policy requires data.dual_task.enable=true. "
                "Set data.dual_task.focused_task_id to the target task index."
            )
        validate_dual_task_cfg(cfg)
        return super().build_dataloader(cfg)

    def build_val_loader(self, cfg, val_datasets):
        val_dataset = DualTaskBatchDataset(
            val_datasets,
            focused_task_id=cfg.data.dual_task.focused_task_id,
            future_step_min=cfg.data.dual_task.future_step_min,
            future_step_max=cfg.data.dual_task.future_step_max,
        )
        return DataLoader(
            val_dataset,
            batch_size=cfg.eval.batch_size,
            num_workers=cfg.eval.num_workers,
            sampler=RandomSampler(val_dataset),
            collate_fn=collate_dual_task_batch,
            persistent_workers=True,
        )

    def build_model(self, cfg, shape_meta):
        super().build_model(cfg, shape_meta)
        rep_dim = self._get_rep_dim(cfg, shape_meta)
        n_tasks = get_benchmark(cfg.data.env_name)(cfg.data.task_order_index).n_tasks
        hidden_dim = cfg.train.get("rep_classifier_hidden", 256)
        self.model.add_module(
            "rep_task_classifier",
            TaskPairClassifier(rep_dim, n_tasks, hidden_dim=hidden_dim),
        )
        self.optimizer = setup_optimizer(cfg.train.optimizer, self.model)

    @staticmethod
    def _get_rep_dim(cfg, shape_meta):
        policy_type = cfg.policy.policy_type
        if policy_type in ("BCTransformerPolicy", "BCDPPolicy", "BCMLPPolicy"):
            return cfg.policy.embed_size
        if policy_type == "BCViLTPolicy":
            if cfg.policy.spatial_transformer.spatial_down_sample:
                return cfg.policy.spatial_transformer.spatial_down_sample_embed_size
            return cfg.policy.embed_size
        if policy_type == "BCRNNPolicy":
            n_img = sum(
                1
                for name in shape_meta["all_shapes"]
                if "rgb" in name or "depth" in name
            )
            return n_img * cfg.policy.image_embed_size
        raise ValueError(f"Unsupported policy_type={policy_type} for representation loss")

    @staticmethod
    def _split_batch(data):
        if "focused" not in data or "mixed" not in data:
            raise ValueError(
                "Expected a dual-task batch with keys 'focused' and 'mixed'. "
                "Enable data.dual_task.enable=true."
            )
        return data["focused"], data["mixed"]

    @staticmethod
    def _pool_representation(z):
        if z.dim() == 3:
            return z[:, -1]
        if z.dim() == 2:
            return z
        raise ValueError(f"Expected representation dim 2 or 3, got shape {tuple(z.shape)}")

    @staticmethod
    def _encode_image_tokens(model, data):
        """Per-camera image encoder outputs: (B, T, num_cameras, E)."""
        encoded = []
        for img_name in model.image_encoders.keys():
            x = data["obs"][img_name]
            b, t, c, h, w = x.shape
            img_encoded = model.image_encoders[img_name]["encoder"](
                x.reshape(b * t, c, h, w),
                langs=data["task_emb"]
                .reshape(b, 1, -1)
                .repeat(1, t, 1)
                .reshape(b * t, -1),
            ).view(b, t, 1, -1)
            encoded.append(img_encoded)
        return torch.cat(encoded, dim=-2)

    @staticmethod
    def _collapse_image_modalities(visual):
        if visual.shape[2] == 1:
            return visual
        return visual.mean(dim=2, keepdim=True)

    def _encode_visual_vilt(self, model, data):
        img_encoded = []
        for img_name in model.image_encoders.keys():
            img_encoded.append(
                rearrange(
                    TensorUtils.time_distributed(
                        data["obs"][img_name], model.image_encoders[img_name]["encoder"]
                    ),
                    "b t c h w -> b t (h w) c",
                )
            )
        img_encoded = torch.cat(img_encoded, dim=-2)
        img_encoded += model.patch_pos_embed.unsqueeze(0)
        b, t = img_encoded.shape[:2]

        patch_modality_idx = model.modality_idx[: img_encoded.shape[2]]
        img_encoded += model.modality_embed[None, :, patch_modality_idx, :]

        spatial_token = model.spatial_token.unsqueeze(0).expand(b, t, -1, -1)
        encoded = torch.cat([spatial_token, img_encoded], dim=-2)
        encoded = rearrange(encoded, "b t n e -> (b t) n e")
        out = model.spatial_transformer(encoded)[:, 0]
        if hasattr(model, "spatial_down_sample"):
            out = model.spatial_down_sample(out).view(b, t, 1, -1)
        else:
            out = out.view(b, t, 1, -1)
        return model.temporal_encode(out)

    def get_visual_representation(self, data, augmentation=None, obs_key="obs"):
        """Visual representation only (no language, proprio, or policy head)."""
        if obs_key != "obs":
            data = {**data, "obs": data[obs_key]}
        data = self.model.preprocess_input(data, augmentation=augmentation)
        policy_type = self.cfg.policy.policy_type
        model = self.model

        if policy_type == "BCMLPPolicy":
            visual = self._encode_image_tokens(model, data)
            return visual.mean(dim=2)

        if policy_type == "BCRNNPolicy":
            encoded = []
            for img_name in model.image_encoders.keys():
                x = data["obs"][img_name]
                b, t, c, h, w = x.shape
                e = model.image_encoders[img_name]["encoder"](
                    x.reshape(b * t, c, h, w),
                    langs=data["task_emb"]
                    .reshape(b, 1, -1)
                    .repeat(1, t, 1)
                    .reshape(b * t, -1),
                ).view(b, t, -1)
                encoded.append(e)
            return torch.cat(encoded, dim=-1)

        if policy_type == "BCViLTPolicy":
            return self._encode_visual_vilt(model, data)

        if policy_type in ("BCTransformerPolicy", "BCDPPolicy"):
            visual = self._collapse_image_modalities(
                self._encode_image_tokens(model, data)
            )
            return model.temporal_encode(visual)

        raise ValueError(f"Unsupported policy_type={policy_type}")

    def compute_bc_loss(self, data, augmentation=None):
        data = self.model.preprocess_input(data, augmentation=augmentation)
        _, _, dist = self.model(data, return_latent=True)

        if self.cfg.policy.policy_type == "BCMLPPolicy":
            bc_loss = self.model.policy_head.loss_fn(
                dist, data["actions"][:, -1], reduction="mean"
            )
        elif self.cfg.policy.policy_type == "BCDPPolicy":
            repeated_diffusion_steps = (
                self.cfg.policy.policy_head.network_kwargs.repeated_diffusion_steps
            )
            actions = data["actions"]
            actions_repeated = actions.repeat(repeated_diffusion_steps, 1, 1)
            dist = dist.mean(dim=1, keepdim=True)
            features_repeated = dist.repeat(repeated_diffusion_steps, 1, 1)
            bc_loss = self.model.policy_head.loss(actions_repeated, features_repeated)
        else:
            bc_loss = self.model.policy_head.loss_fn(
                dist, data["actions"], reduction="mean"
            )

        return bc_loss

    def compute_representation_loss(self, z, z_future, mixed_data):
        """
        Classify which task produced (z, z') with cross-entropy on task_id.
        """
        if "task_id" not in mixed_data:
            raise ValueError(
                "Mixed batch is missing 'task_id'. "
                "Use DualTaskBatchDataset with data.dual_task.enable=true."
            )

        z = self._pool_representation(z)
        z_future = self._pool_representation(z_future)
        logits = self.model.rep_task_classifier(z, z_future)
        task_ids = mixed_data["task_id"].long().to(device=z.device)
        return F.cross_entropy(logits, task_ids)

    def compute_representation_accuracy(self, z, z_future, mixed_data):
        z = self._pool_representation(z)
        z_future = self._pool_representation(z_future)
        logits = self.model.rep_task_classifier(z, z_future)
        task_ids = mixed_data["task_id"].long().to(device=z.device)
        preds = logits.argmax(dim=-1)
        return (preds == task_ids).float().mean()

    def forward_backward(self, data):
        focused, mixed = self._split_batch(data)

        bc_loss = self.compute_bc_loss(focused)
        z = self.get_visual_representation(mixed, obs_key="obs")
        z_future = self.get_visual_representation(mixed, obs_key="obs_future")
        rep_loss = self.compute_representation_loss(z, z_future, mixed)
        rep_acc = self.compute_representation_accuracy(z, z_future, mixed)
        rep_scale = self.cfg.train.get("rep_loss_scale", 1.0)
        loss = bc_loss + rep_scale * rep_loss

        self.optimizer.zero_grad()
        self.fabric.backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.cfg.train.grad_clip
        )
        self.optimizer.step()

        ret_dict = {
            "loss": loss.item(),
            "bc_loss": bc_loss.item(),
            "rep_loss": rep_loss.item(),
            "rep_acc": rep_acc.item(),
        }
        return ret_dict

    @torch.no_grad()
    def compute_eval_batch_metrics(self, data):
        focused, mixed = self._split_batch(data)
        bc_loss = self.compute_bc_loss(focused, augmentation=False)
        z = self.get_visual_representation(mixed, augmentation=False)
        z_future = self.get_visual_representation(
            mixed, augmentation=False, obs_key="obs_future"
        )
        rep_loss = self.compute_representation_loss(z, z_future, mixed)
        rep_acc = self.compute_representation_accuracy(z, z_future, mixed)
        rep_scale = self.cfg.train.get("rep_loss_scale", 1.0)
        loss = bc_loss + rep_scale * rep_loss

        return {
            "loss": loss.item(),
            "bc_loss": bc_loss.item(),
            "rep_loss": rep_loss.item(),
            "rep_acc": rep_acc.item(),
        }
