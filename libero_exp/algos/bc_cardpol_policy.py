import torch
import torch.nn as nn
import torch.nn.functional as F
from libero.libero.benchmark import get_benchmark
from torch.utils.data import DataLoader, RandomSampler

from ..data.get_dataset import (
    DualTaskBatchDataset,
    collate_dual_task_batch,
    validate_dual_task_cfg,
)
from ..utils.train_utils import setup_optimizer
from .base import BaseAlgo


class TaskPairClassifier(nn.Module):
    """Predict task id from a pair of spatial input representations (x, x')."""

    def __init__(self, rep_dim, n_tasks, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * rep_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_tasks),
        )

    def forward(self, x, x_future):
        return self.net(torch.cat([x, x_future], dim=-1))


class BC_CARDPOL_Policy(BaseAlgo):
    """
    CARD-style training with dual-task batches:
      - focused: standard BC on the configured task
      - mixed: auxiliary loss on spatial input representations (x)
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
            extra_dim = (
                int(cfg.data.use_joint) * 7
                + int(cfg.data.use_gripper) * 2
                + int(cfg.data.use_ee) * 3
            )
            return (
                n_img * cfg.policy.image_embed_size
                + extra_dim
                + cfg.policy.text_embed_size
            )
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
    def _pool_representation(x):
        """Pool spatial input x to a fixed-size vector for the task classifier."""
        if x.dim() == 4:
            # (B, T, num_modalities, E): last timestep, language / action token slot
            return x[:, -1, 0]
        if x.dim() == 3:
            return x[:, -1]
        if x.dim() == 2:
            return x
        raise ValueError(f"Expected representation dim 2–4, got shape {tuple(x.shape)}")

    @staticmethod
    def _encode_rnn_input(model, data, *, use_language=True):
        """RNN input features before the temporal RNN (B, T, H)."""
        encoded = []
        for img_name in model.image_encoders.keys():
            obs = data["obs"][img_name]
            b, t, c, h, w = obs.shape
            encoded.append(
                model.image_encoders[img_name]["encoder"](
                    obs.reshape(b * t, c, h, w),
                    langs=model.get_image_langs(data, b, t, use_language=use_language),
                ).view(b, t, -1)
            )
        encoded.append(model.extra_encoder(data["obs"]))
        encoded = torch.cat(encoded, dim=-1)
        if use_language:
            lang_h = model.language_encoder(data)
        else:
            lang_h = torch.zeros(
                encoded.shape[0],
                model.cfg.policy.text_embed_size,
                device=encoded.device,
                dtype=encoded.dtype,
            )
        return torch.cat(
            [encoded, lang_h.unsqueeze(1).expand(-1, encoded.shape[1], -1)],
            dim=-1,
        )

    def get_input_representation(self, data, augmentation=None, obs_key="obs"):
        """
        Spatial input representations x (pre-temporal), matching policy forward.
        Image FiLM for the rep path uses task_emb regardless of policy.use_language_conditioning.
        """
        if obs_key != "obs":
            data = {**data, "obs": data[obs_key]}
        data = self.model.preprocess_input(data, augmentation=augmentation)
        model = self.model
        policy_type = self.cfg.policy.policy_type

        if policy_type == "BCRNNPolicy":
            return self._encode_rnn_input(model, data, use_language=True)

        if hasattr(model, "spatial_encode"):
            return model.spatial_encode(data)

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

    def compute_representation_loss(self, x, x_future, mixed_data):
        """
        Classify which task produced (x, x') with cross-entropy on task_id.
        """
        if "task_id" not in mixed_data:
            raise ValueError(
                "Mixed batch is missing 'task_id'. "
                "Use DualTaskBatchDataset with data.dual_task.enable=true."
            )

        x = self._pool_representation(x)
        x_future = self._pool_representation(x_future)
        logits = self.model.rep_task_classifier(x, x_future)
        task_ids = mixed_data["task_id"].long().to(device=x.device)
        return F.cross_entropy(logits, task_ids)

    def compute_representation_accuracy(self, x, x_future, mixed_data):
        x = self._pool_representation(x)
        x_future = self._pool_representation(x_future)
        logits = self.model.rep_task_classifier(x, x_future)
        task_ids = mixed_data["task_id"].long().to(device=x.device)
        preds = logits.argmax(dim=-1)
        return (preds == task_ids).float().mean()

    def forward_backward(self, data):
        focused, mixed = self._split_batch(data)

        bc_loss = self.compute_bc_loss(focused)
        x = self.get_input_representation(mixed, obs_key="obs")
        x_future = self.get_input_representation(mixed, obs_key="obs_future")
        rep_loss = self.compute_representation_loss(x, x_future, mixed)
        rep_acc = self.compute_representation_accuracy(x, x_future, mixed)
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
        x = self.get_input_representation(mixed, augmentation=False)
        x_future = self.get_input_representation(
            mixed, augmentation=False, obs_key="obs_future"
        )
        rep_loss = self.compute_representation_loss(x, x_future, mixed)
        rep_acc = self.compute_representation_accuracy(x, x_future, mixed)
        rep_scale = self.cfg.train.get("rep_loss_scale", 1.0)
        loss = bc_loss + rep_scale * rep_loss

        return {
            "loss": loss.item(),
            "bc_loss": bc_loss.item(),
            "rep_loss": rep_loss.item(),
            "rep_acc": rep_acc.item(),
        }
