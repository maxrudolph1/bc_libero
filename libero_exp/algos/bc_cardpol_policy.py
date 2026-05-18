import torch
import torch.nn as nn
import torch.nn.functional as F
from libero.libero.benchmark import get_benchmark
from tqdm import tqdm

from ..data.get_dataset import validate_dual_task_cfg
from ..utils.train_utils import setup_optimizer
from .base import BaseAlgo


class TaskPairClassifier(nn.Module):
    """Predict task id from a pair of pre-policy representations (z, z')."""

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
      - mixed: action-free representation learning before policy_head
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

    def build_model(self, cfg, shape_meta):
        super().build_model(cfg, shape_meta)
        rep_dim = self._get_rep_dim(cfg)
        n_tasks = get_benchmark(cfg.data.env_name)(cfg.data.task_order_index).n_tasks
        hidden_dim = cfg.train.get("rep_classifier_hidden", 256)
        self.model.add_module(
            "rep_task_classifier",
            TaskPairClassifier(rep_dim, n_tasks, hidden_dim=hidden_dim),
        )
        self.optimizer = setup_optimizer(cfg.train.optimizer, self.model)

    @staticmethod
    def _get_rep_dim(cfg):
        policy_type = cfg.policy.policy_type
        if policy_type in ("BCTransformerPolicy", "BCViLTPolicy", "BCDPPolicy", "BCMLPPolicy"):
            return cfg.policy.embed_size
        if policy_type == "BCRNNPolicy":
            direction = 2 if cfg.policy.rnn_bidirectional else 1
            return direction * cfg.policy.rnn_hidden_size
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

    def get_pre_policy_representation(self, data, augmentation=None, obs_key="obs"):
        """Action-free representation immediately before policy_head."""
        if obs_key != "obs":
            data = {**data, "obs": data[obs_key]}
        data = self.model.preprocess_input(data, augmentation=augmentation)
        policy_type = self.cfg.policy.policy_type

        if policy_type == "BCMLPPolicy":
            from robomimic.utils import tensor_utils as TensorUtils

            x = self.model.spatial_encode(data)
            x = TensorUtils.join_dimensions(x, 2, 3)
            x = TensorUtils.join_dimensions(x, 1, 2)
            x = self.model.spatial_down_sample(x)
            return self.model.spatial_mlp(x)

        if policy_type == "BCRNNPolicy":
            _, output, _ = self.model(data, train_mode=True, return_latent=True)
            return output

        if policy_type in ("BCTransformerPolicy", "BCViLTPolicy", "BCDPPolicy"):
            x = self.model.spatial_encode(data)
            return self.model.temporal_encode(x)

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

    def forward_backward(self, data):
        focused, mixed = self._split_batch(data)

        bc_loss = self.compute_bc_loss(focused)
        z = self.get_pre_policy_representation(mixed, obs_key="obs")
        z_future = self.get_pre_policy_representation(mixed, obs_key="obs_future")
        rep_loss = self.compute_representation_loss(z, z_future, mixed)
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
        }
        return ret_dict

    @torch.no_grad()
    def evaluate(self, tag="val"):
        cfg = self.cfg
        tot_loss_dict, tot_items = {}, 0
        self.model.eval()

        for data in tqdm(self.val_loader):
            bc_loss = self.compute_bc_loss(data, augmentation=False)
            ret_dict = {
                "loss": bc_loss.item(),
                "bc_loss": bc_loss.item(),
            }

            for k, v in ret_dict.items():
                if k not in tot_loss_dict:
                    tot_loss_dict[k] = 0
                tot_loss_dict[k] += v
            tot_items += 1

            if cfg.train.debug:
                break

        out_dict = {}
        for k, v in tot_loss_dict.items():
            out_dict[f"{tag}/{k}"] = tot_loss_dict[f"{k}"] / tot_items

        return out_dict
