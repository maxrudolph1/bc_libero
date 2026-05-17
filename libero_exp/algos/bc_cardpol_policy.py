import torch
from tqdm import tqdm

from .base import BaseAlgo


class BC_CARDPOL_Policy(BaseAlgo):
    """
    CARD-style training with dual-task batches:
      - focused: standard BC on the configured task
      - mixed: action-free representation learning before policy_head
    """

    def __init__(self, cfg, inference=False, device="cuda"):
        super().__init__(cfg, inference, device)
        if not inference and not cfg.data.dual_task.enable:
            raise ValueError(
                "BC_CARDPOL_Policy requires data.dual_task.enable=true. "
                "Set data.dual_task.focused_task_id to the target task index."
            )

    def build_dataloader(self, cfg):
        if not cfg.data.dual_task.enable:
            raise ValueError(
                "BC_CARDPOL_Policy requires data.dual_task.enable=true. "
                "Set data.dual_task.focused_task_id to the target task index."
            )
        return super().build_dataloader(cfg)

    @staticmethod
    def _split_batch(data):
        if "focused" not in data or "mixed" not in data:
            raise ValueError(
                "Expected a dual-task batch with keys 'focused' and 'mixed'. "
                "Enable data.dual_task.enable=true."
            )
        return data["focused"], data["mixed"]

    def get_pre_policy_representation(self, data, augmentation=None):
        """Action-free representation immediately before policy_head."""
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

    def compute_representation_loss(self, z):
        """
        Action-free objective on mixed-batch representations (pre policy_head).

        Override or extend this method when the representation-learning loss is defined.
        """
        return z.new_zeros(())

    def forward_backward(self, data):
        focused, mixed = self._split_batch(data)

        bc_loss = self.compute_bc_loss(focused)
        z_mixed = self.get_pre_policy_representation(mixed)
        rep_loss = self.compute_representation_loss(z_mixed)
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
