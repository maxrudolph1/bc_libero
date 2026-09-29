# SupCon Baseline

Supervised contrastive (SupCon) auxiliary loss on **task identity**: the contrastive counterpart to CARDPol. It uses the same dual-task batches and the same task-id supervision, but replaces CARDPol's task classifier with a multi-positive InfoNCE loss (Khosla et al., 2020, the `L_out` form).

Code: `libero_exp/algos/bc_supcon_policy.py` (`BC_SUPCON_Policy`, subclasses `BC_CARDPOL_Policy`). Configs: `libero_exp/configs/bc_supcon_policy/{vilt,transformer,rnn,mlp,dp}[_distract].yaml`. Launch: `launch_scripts/mll/submit_libero.py --supcon-baseline-sweep`.

## Batches

Each step draws two batches via `DualTaskBatchDataset` (`data.dual_task.enable=true`):

- **Focused** (B = 64): samples from the one task being learned (`focused_task_id`). Used **only** by the BC loss.
- **Mixed** (B × `data.dual_task.mixed_per_sample` = 64 × 4 = 256): frames drawn uniformly from the concatenated data of all 10 tasks in the suite, each carrying its `task_id`. Used **only** by SupCon.

`mixed_per_sample` (added for this baseline, default 1 = old behavior) enlarges the contrastive batch without changing the BC batch. It is passed explicitly by the sweep because the `*_distract` configs reset `data.dual_task`.

## Loss

For each mixed frame *i*:

1. **Encode** with the policy's own encoder (`get_input_representation`, ViLT `spatial_encode`), then `_pool_representation`: at the last timestep, drop the slot-0 language/action token and mean-pool the remaining observation tokens. With image-only inputs (no joint/gripper/ee tokens) this is just the 64-d spatial summary token.
2. **Project** with `TaskProjector`: `Linear(64→256) → ReLU → Linear(256→64)`, then L2-normalize → `z_i`.
3. **Contrast** with positives `P(i)` = the other mixed frames sharing *i*'s task id, and τ = 0.1:

$$
\mathcal{L}_{\text{SupCon}} = -\frac{1}{|I'|}\sum_{i\in I'} \frac{1}{|P(i)|}\sum_{p\in P(i)} \log \frac{\exp(z_i\cdot z_p/\tau)}{\sum_{a\neq i}\exp(z_i\cdot z_a/\tau)}
$$

- Positives: all other same-task frames, from any demo and any timestep. Negatives: frames from the other 9 tasks. The denominator runs over all `a ≠ i`, so it includes the positives.
- `I'` is the set of anchors with at least one positive; anchors without one are skipped. With 256 frames over 10 tasks this is effectively every anchor, with about 25 positives each.
- `L_out`: the log-prob is averaged over positives outside the log.

Total objective: `loss = bc_loss + rep_loss_scale * supcon_loss`, with `rep_loss_scale = 0.01`. Both terms backprop into the shared encoder; the projector only receives SupCon gradients. Everything uses one AdamW optimizer.

Logged metric `rep_acc`: 1-NN task accuracy, i.e. the fraction of mixed frames whose most-similar other frame shares their task id.

## Differences from CARDPol

| | CARDPol | SupCon |
|---|---|---|
| Aux head | Task classifier (cross-entropy over task ids) | Projector + multi-positive InfoNCE |
| Input to head | (x, x′) future pair | single frame x (`train.supcon_use_future=false`; set true to embed the pair) |
| Mixed batch | 64 | 256 (`mixed_per_sample=4`) |
| `rep_loss_scale` | 0.01 | 0.01 |

With `supcon_use_future=false` the future frame is still sampled (`mixed_mode=future_pair`) but unused.

## No task-id leakage

All paper runs use `policy.use_language_conditioning=false`. In that setting:

- The ViLT patch encoder only sees pixels; there is no FiLM layer with the task embedding.
- The spatial transformer's text token and the temporal slot-0 token are zeros.
- `_pool_representation` drops slot 0 anyway.

So tasks can only be separated through the agentview image.

## Hyperparameters

| Key | Value |
|---|---|
| `train.rep_loss_scale` | 0.01 |
| `train.supcon_temperature` | 0.1 |
| `train.supcon_proj_dim` | 64 |
| `train.supcon_hidden_dim` | 256 |
| `train.supcon_use_future` | false |
| `data.dual_task.mixed_per_sample` | 4 |
| `data.dual_task.future_step_{min,max}` | 1, 10 (unused when `supcon_use_future=false`) |
| `train.batch_size` / `n_epochs` / `train_ratio` | 64 / 50 / 0.9 |

## Paper sweep (SLURM array 97564, submitted 2026-09-26)

`submit_libero.py --supcon-baseline-sweep`: 75 array tasks × 4 sequential runs = **300 runs**. ViLT, agentview image only, no proprio, no language.

- Suites: `libero_spatial`, `libero_object`, `libero_goal`
- Focused task: 0–9
- Seeds: 0–4
- Setting: clean (`vilt`) and distractors (`vilt_distract`, with rollout during and after training)

wandb project `bc-supcon-vilt`, groups `bc-supcon-baseline_img_cam-agent` and `bc-supcon-baseline_img_cam-agent_distract`. Artifacts are under `artifacts/<group>/`. Results are in the `paper_plotting` LIBERO tables (`cardpol/libero/figures/final_results/`).
