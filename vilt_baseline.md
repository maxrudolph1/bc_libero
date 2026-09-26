# ViLT Baseline

"ViLT" is not a standalone algorithm — it's the **image/language fusion backbone** (`BCViLTPolicy`) that plugs into every BC variant in this repo (`bc_policy`, `bc_ib_policy`, `bc_cardpol_policy`, `bc_vae_policy`, `bc_curl_policy`, `bc_vip_policy`, `bc_icvf_policy`). It's the default `--backbone` in the launch scripts, contrasted with the sibling `BCTransformerPolicy` (ResNet+FiLM).

Code: `libero_exp/models/bc_vilt_policy.py`. Config: `libero_exp/configs/base/policy/vilt_policy.yaml`, selected per-experiment via `libero_exp/configs/{algo}/vilt.yaml`.

## Architecture

Input: `(o_{t-H}, ..., o_t)` → Output: `a_t`

1. **Patch encoders** (per camera, e.g. `agentview_rgb`, `eye_in_hand_rgb`): `Conv2d(C→64, k=7,s=2,p=3)+BN+ReLU`, then `Conv2d(64→embed_size=128, k=8,s=8)+BN` — 128×128 image → 64×64 → 8×8=64 patches/camera → 128 patches total (2 cams).
2. **Spatial language encoder**: 1-layer MLP, BERT `[CLS]` (768-dim) → 128-dim token.
3. **Learned embeddings**: a `spatial_token` (CLS-analogue), a `patch_pos_embed` over all 128 patches, and a `modality_embed` per camera + language token.
4. **Spatial transformer** (ViT-style, self-attention over image patches + language token): 7 layers, 8 heads, head_dim=120, MLP hidden=256, dropout=0.1. Sequence = `[spatial_token, 128 image patches, language token]` (len 130), processed per-timestep (no temporal mixing). Output = spatial_token position → visual-language summary. ~3.9M params.
5. **Spatial down-sample**: `Linear(128→64)` to `temporal_embed_size=64`.
6. **Extra-modality (proprioception) encoders**: joint (7d) and gripper (2d) each via `Linear(dim→64)` (ee disabled by default) → 2 extra tokens.
7. **Temporal language encoder**: second MLP, BERT 768→64, another token per timestep.
8. **Token sequence per timestep**: `[temporal language token, spatial summary, joint token, gripper token]` → shape `(B,T,4,64)`.
9. **Sinusoidal position encoding** added across timesteps.
10. **Temporal transformer** (causal-masked): 4 layers, 6 heads, head_dim=64, MLP hidden=256, dropout=0.1, `max_seq_len=10`. ~0.53M params. Output token 0 at the last timestep = policy latent.
11. **Policy head** (`DeterministicHead` by default, or `GMMHead`): `Linear(64→1024)×2 layers → 7-dim action` (xyz, orientation, gripper). No action squashing. ~1.1M params.

`get_action` runs autoregressively with a 10-step latent queue, clamping outputs to `[-1,1]`.

## Base architecture config (`vilt_policy.yaml`)

```yaml
policy_type: BCViLTPolicy
embed_size: 128
use_language_conditioning: true
extra_state_encoder: {extra_num_layers: 0, extra_hidden_size: 128}
spatial_transformer:
  num_layers: 7, num_heads: 8, head_output_size: 120, mlp_hidden_size: 256, dropout: 0.1
  spatial_down_sample: true, spatial_down_sample_embed_size: 64
temporal_transformer:
  num_layers: 4, num_heads: 6, head_output_size: 64, mlp_hidden_size: 256, dropout: 0.1, max_seq_len: 10
```

Sub-configs: `patch_encoder` (patch_size=[8,8]), `mlp_encoder` (768→128→128, 1 layer), `sinusoidal_position_encoding` (inv_freq_factor=10), `mlp_head` (hidden=1024, 2 layers, output=[7]), color-jitter aug (0.3 brightness/contrast/saturation/hue, epsilon 0.1) + translation aug (±8px).

## Training hyperparameters (shared across all backbones — `base/train/default.yaml`)

- **Epochs**: 50, **batch size**: 64, **seed**: 0, `num_workers=8`
- **Optimizer**: AdamW, `lr=1e-4`, `betas=[0.9,0.999]`, `weight_decay=1e-4`
- **Scheduler**: CosineAnnealingLR, `T_max=50`, `eta_min=0`
- **Loss**: MSE (DeterministicHead) or GMM-NLL (GMMHead), `loss_coef=1.0`; `grad_clip=100`
- **Data**: `seq_len=10`, `frame_stack=1`, `img_size=128`, `train_ratio=0.9`, `val_demo_num=5`; language via `bert-base-cased`, `max_word_len=25`; env default `libero_goal`
- Val/save/log frequencies: every 5/10/10 epochs respectively

None of these are vilt-specific overrides — `vilt.yaml` experiment configs only swap in the `vilt_policy` architecture config and set wandb naming (`project: bc-ib-vilt`).

## Contrast with `BCTransformerPolicy` (the other backbone)

ViLT adds a **ViT-style spatial self-attention transformer** over raw image patches + language (fusing two camera views into one summary token via attention) before the temporal transformer, whereas `BCTransformerPolicy` uses a ResNet-18 CNN with FiLM-style language conditioning injected at each residual block and feeds both camera embeddings separately into the temporal transformer (`embed_size=64` there vs. 128 for ViLT).
