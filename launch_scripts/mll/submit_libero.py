#!/usr/bin/env python3
"""Submit LIBERO training runs to Slurm via sbatch.

Edit the sweep_configs block in __main__ (or import main() from another script),
then run:

    python launch_scripts/mll/submit_libero.py --dry-run
    python launch_scripts/mll/submit_libero.py
"""

from __future__ import annotations

import argparse
import copy
import itertools
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

REPO_ROOT = Path("/u/mrudolph/documents/BC-IB")
SLURM_TEMP_SCRIPT = REPO_ROOT / "launch_scripts/mll/temp_submission.slurm"
SLURM_LOG_ROOT = REPO_ROOT / "slurm_jobs/libero"
DEFAULT_SLURM_EXCLUDE = ""

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_root}/job_%j/job_%j.out
#SBATCH --error={log_root}/job_%j/job_%j.err
#SBATCH --partition={partition}
{exclude_directive}#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}

source ~/.bashrc
cd {repo_root}
source .venv/bin/activate

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false

echo "MUJOCO_GL=$MUJOCO_GL"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Job started at $(date)"
echo "Running on node: $(hostname)"

{run_commands}

echo "Job ended at $(date)"
"""


def from_sweep_config_to_config(sweep_config: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Expand list-valued keys in sweep_config to all combinations."""
    sweep_keys = [key for key in sweep_config if isinstance(sweep_config[key], list)]
    value_lists = [sweep_config[key] for key in sweep_keys]
    for combination in itertools.product(*value_lists):
        new_config = copy.deepcopy(sweep_config)
        for key, value in zip(sweep_keys, combination):
            new_config[key] = value
        yield new_config


def config_to_cli_args(config: Dict[str, Any]) -> str:
    """Hydra CLI: --config-path=... plus key=value overrides."""
    parts: List[str] = []
    for key, value in config.items():
        if key.startswith("--"):
            flag = key if key.startswith("--") else f"--{key}"
            parts.append(f"{flag}={value}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def build_train_command(
    config: Dict[str, Any],
    *,
    train_script: str = "python train_libero.py",
) -> str:
    return f"{train_script} {config_to_cli_args(config)}"


def submit_slurm_job(
    run_commands: str,
    *,
    dry_run: bool,
    job_name: str = "libero-train",
    partition: str = "allnodes",
    exclude: str = DEFAULT_SLURM_EXCLUDE,
    gpus: int = 1,
    cpus: int = 16,
    mem: str = "768GB",
    time_limit: str = "8:00:00",
) -> Optional[int]:
    exclude_directive = f"#SBATCH --exclude={exclude}\n" if exclude else ""
    slurm_script = SLURM_TEMPLATE.format(
        job_name=job_name,
        log_root=SLURM_LOG_ROOT,
        partition=partition,
        exclude_directive=exclude_directive,
        gpus=gpus,
        cpus=cpus,
        mem=mem,
        time_limit=time_limit,
        repo_root=REPO_ROOT,
        run_commands=run_commands,
    )

    if dry_run:
        print("########  DRY RUN  ##########")
        print(slurm_script)
        print("#############################")
        print()
        return None

    SLURM_TEMP_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    SLURM_TEMP_SCRIPT.write_text(slurm_script)

    result = subprocess.run(
        ["sbatch", str(SLURM_TEMP_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    print("sbatch stdout:", result.stdout.strip())
    if result.stderr:
        print("sbatch stderr:", result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed with exit code {result.returncode}")

    jid = int(result.stdout.strip().split()[-1])
    job_dir = SLURM_LOG_ROOT / f"job_{jid}"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "submission.sh").write_text(slurm_script)
    return jid


def main(
    dry_run: bool = False,
    sweep_configs: Optional[List[Dict[str, Any]]] = None,
    *,
    train_script: str = "python train_libero.py",
    num_runs_per_job: int = 1,
    job_name: str = "libero-train",
    partition: str = "allnodes",
    exclude: str = DEFAULT_SLURM_EXCLUDE,
    gpus: int = 1,
    cpus: int = 16,
    mem: str = "768GB",
    time_limit: str = "8:00:00",
) -> None:
    if sweep_configs is None:
        sweep_configs = []

    configs: List[Dict[str, Any]] = []
    for sweep_config in sweep_configs:
        configs.extend(from_sweep_config_to_config(sweep_config))

    num_jobs = (len(configs) + num_runs_per_job - 1) // num_runs_per_job
    for start in range(0, len(configs), num_runs_per_job):
        batch = configs[start : start + num_runs_per_job]
        commands = "\n".join(build_train_command(cfg, train_script=train_script) for cfg in batch)
        submit_slurm_job(
            commands,
            dry_run=dry_run,
            job_name=job_name,
            partition=partition,
            exclude=exclude,
            gpus=gpus,
            cpus=cpus,
            mem=mem,
            time_limit=time_limit,
        )

    print(f"# {'Would submit' if dry_run else 'Submitted'} {num_jobs} jobs ({len(configs)} runs)")


def policy_config_path(policy_name: str) -> str:
    return f"libero_exp/configs/{policy_name}"


VALID_MODALITIES = ("image", "proprio", "language")
_MODALITY_SHORT = {"image": "img", "proprio": "proprio", "language": "lang"}

# Camera views usable as image inputs. Each maps to a `data.obs.modality.rgb`
# key (and the corresponding `obs_key_mapping` entry used at rollout time).
VALID_CAMERAS = ("agentview", "eye_in_hand")
_CAMERA_RGB_KEY = {"agentview": "agentview_rgb", "eye_in_hand": "eye_in_hand_rgb"}
_CAMERA_SHORT = {"agentview": "agent", "eye_in_hand": "wrist"}
_CAMERA_ALIASES = {
    "agentview": "agentview",
    "agent": "agentview",
    "third": "agentview",
    "third_person": "agentview",
    "eye_in_hand": "eye_in_hand",
    "eye-in-hand": "eye_in_hand",
    "eih": "eye_in_hand",
    "wrist": "eye_in_hand",
    "hand": "eye_in_hand",
}
# Default camera set: third-person agentview only (no wrist / eye-in-hand view).
DEFAULT_CAMERAS = ["agentview"]


def parse_camera_set(spec: str) -> List[str]:
    """Parse 'agentview,wrist' into a canonical, deduped camera list."""
    raw = [c.strip().lower() for c in spec.split(",") if c.strip()]
    cams: List[str] = []
    for c in raw:
        if c not in _CAMERA_ALIASES:
            raise ValueError(
                f"Unknown camera '{c}'; valid options are {list(VALID_CAMERAS)} "
                "(aliases: agent, wrist, eih)."
            )
        cams.append(_CAMERA_ALIASES[c])
    if not cams:
        raise ValueError("At least one camera must be selected.")
    return [c for c in VALID_CAMERAS if c in cams]


def camera_tag(cameras: List[str]) -> str:
    """Short wandb-friendly tag, e.g. 'agent', 'agent+wrist'."""
    return "+".join(_CAMERA_SHORT[c] for c in cameras)


def parse_modality_set(spec: str) -> List[str]:
    """Parse 'image,proprio,language' into a canonical, deduped modality list."""
    mods = [m.strip().lower() for m in spec.split(",") if m.strip()]
    invalid = [m for m in mods if m not in VALID_MODALITIES]
    if invalid:
        raise ValueError(
            f"Unknown modalit(ies) {invalid}; valid options are {list(VALID_MODALITIES)}."
        )
    if "image" not in mods:
        raise ValueError("'image' must be included in every modality set.")
    return [m for m in VALID_MODALITIES if m in mods]


def modality_tag(modalities: List[str]) -> str:
    """Short wandb-friendly tag, e.g. 'img', 'img+proprio', 'img+proprio+lang'."""
    return "+".join(_MODALITY_SHORT[m] for m in modalities)


def wandb_run_group(
    wandb_group: str,
    env_name: str,
    task_id: int,
    rep_loss_scale: float,
) -> str:
    """Wandb group for one (env, task, rep_loss_scale) sweep; seeds share this group."""
    return f"{wandb_group}" #_{env_name}_task{task_id}_rep{rep_loss_scale:g}"


def build_cardpol_sweep_config(
    *,
    env_name: str,
    task_id: int,
    rep_loss_scale: float,
    policies: List[str],
    config_names: List[str],
    seeds: List[int],
    train_ratio: float,
    n_epochs: int,
    wandb_group: str,
    wandb_project: str = "bc-cardpol-transformer",
    modalities: List[str] = ("image", "proprio", "language"),
    cameras: List[str] = ("agentview",),
    distract: bool = False,
    enable_rollout_during_train: bool = True,
    post_train_rollout: bool = True,
) -> Dict[str, Any]:
    """One sweep entry: fixed env / task / rep scale; seeds expanded via product.

    `modalities` selects which inputs to include. 'image' is always required;
    'proprio' adds proprioceptive state, 'language' enables language conditioning.
    `cameras` selects which image views feed the policy (agentview and/or the
    eye-in-hand wrist camera); it sets ``data.obs.modality.rgb``.
    When ``distract=True``, uses ``<backbone>_distract`` configs (datasets_distract).
    """
    modalities = list(modalities)
    cameras = list(cameras)
    use_proprio = "proprio" in modalities
    use_language = "language" in modalities

    resolved_config_names = [
        name if (not distract or name.endswith("_distract")) else f"{name}_distract"
        for name in config_names
    ]

    group = wandb_run_group(wandb_group, env_name, task_id, rep_loss_scale)
    group_suffix = f"_{modality_tag(modalities)}_cam-{camera_tag(cameras)}"
    if distract:
        group_suffix += "_distract"

    config: Dict[str, Any] = {
        "--config-path": [policy_config_path(p) for p in policies],
        "--config-name": resolved_config_names,
        "data.env_name": env_name,
        "data.train_ratio": train_ratio,
        "train.n_epochs": n_epochs,
        "train.seed": seeds,
        "train.train_gpus": "[0]",
        "train.rep_loss_scale": rep_loss_scale,
        "train.rep_classifier_hidden": 256,
        "data.dual_task.enable": "true",
        "data.dual_task.focused_task_id": task_id,
        "data.dual_task.future_step_min": 1,
        "data.dual_task.future_step_max": 10,
        "policy.use_language_conditioning": str(use_language).lower(),
        "env.task_id": [task_id],
        "wandb.project": wandb_project,
        "wandb.group": f"{group}{group_suffix}",
        "wandb.policy_arch": resolved_config_names,
    }

    if distract:
        config["eval.enable_rollout"] = str(enable_rollout_during_train).lower()
        config["eval.post_train_rollout.enable"] = str(post_train_rollout).lower()

    # Select which camera views feed the policy image encoders.
    rgb_keys = [_CAMERA_RGB_KEY[c] for c in cameras]
    config["data.obs.modality.rgb"] = "[" + ",".join(rgb_keys) + "]"

    # data_modality reflects the observation inputs (image always; proprio optional).
    data_modalities = [m for m in ("image", "proprio") if m in modalities]
    config["data.data_modality"] = "[" + ",".join(data_modalities) + "]"

    if not use_proprio:
        # Drop proprioceptive state from the dataset and the policy inputs.
        config["data.obs.modality.low_dim"] = "[]"
        config["data.use_gripper"] = "false"
        config["data.use_joint"] = "false"
        config["data.use_ee"] = "false"

    return config


def _build_rep_baseline_sweep_config(
    policy_name: str,
    *,
    env_name: str,
    task_id: int,
    config_name: str,
    seeds: List[int],
    train_ratio: float,
    n_epochs: int,
    wandb_group: str,
    wandb_project: str,
    modalities: List[str] = ("image", "proprio"),
    cameras: List[str] = ("agentview",),
    distract: bool = False,
    enable_rollout_during_train: bool = True,
    post_train_rollout: bool = True,
    mixed_mode: str = "future_pair",
    rep_loss_scale: Any = None,
) -> Dict[str, Any]:
    """Hydra sweep entry for VAE/CURL/VIP representation baselines.

    `modalities` selects observation inputs: 'image' only or 'image'+'proprio'.
    `mixed_mode` selects the dual-task mixed-batch sampler ('future_pair' or 'vip').
    `rep_loss_scale` overrides train.rep_loss_scale; pass a list to sweep it.
    When ``distract=True``, uses ``<backbone>_distract`` configs (datasets_distract).
    """
    modalities = list(modalities)
    cameras = list(cameras)
    use_proprio = "proprio" in modalities

    resolved_config_name = (
        config_name
        if (not distract or config_name.endswith("_distract"))
        else f"{config_name}_distract"
    )

    group_suffix = f"_{modality_tag(modalities)}_cam-{camera_tag(cameras)}"
    if distract:
        group_suffix += "_distract"

    config: Dict[str, Any] = {
        "--config-path": policy_config_path(policy_name),
        "--config-name": resolved_config_name,
        "data.env_name": env_name,
        "data.train_ratio": train_ratio,
        "train.n_epochs": n_epochs,
        "train.seed": seeds,
        "train.train_gpus": "[0]",
        "data.dual_task.enable": "true",
        "data.dual_task.focused_task_id": task_id,
        "data.dual_task.future_step_min": 1,
        "data.dual_task.future_step_max": 10,
        "data.dual_task.mixed_mode": mixed_mode,
        "policy.use_language_conditioning": "false",
        "env.task_id": [task_id],
        "wandb.project": wandb_project,
        "wandb.group": f"{wandb_group}{group_suffix}",
        "wandb.policy_arch": resolved_config_name,
    }

    if rep_loss_scale is not None:
        config["train.rep_loss_scale"] = rep_loss_scale

    if distract:
        config["eval.enable_rollout"] = str(enable_rollout_during_train).lower()
        config["eval.post_train_rollout.enable"] = str(post_train_rollout).lower()

    rgb_keys = [_CAMERA_RGB_KEY[c] for c in cameras]
    config["data.obs.modality.rgb"] = "[" + ",".join(rgb_keys) + "]"

    data_modalities = [m for m in ("image", "proprio") if m in modalities]
    config["data.data_modality"] = "[" + ",".join(data_modalities) + "]"

    if not use_proprio:
        config["data.obs.modality.low_dim"] = "[]"
        config["data.use_gripper"] = "false"
        config["data.use_joint"] = "false"
        config["data.use_ee"] = "false"

    return config


def build_vae_sweep_config(
    *,
    env_name: str,
    task_id: int,
    config_name: str,
    seeds: List[int],
    train_ratio: float,
    n_epochs: int,
    wandb_group: str,
    wandb_project: str = "bc-vae-vilt",
    modalities: List[str] = ("image", "proprio"),
    cameras: List[str] = ("agentview",),
    distract: bool = False,
    enable_rollout_during_train: bool = True,
    post_train_rollout: bool = True,
    rep_loss_scale: Any = None,
) -> Dict[str, Any]:
    """Hydra sweep entry for bc_vae_policy (VAE representation baseline)."""
    return _build_rep_baseline_sweep_config(
        "bc_vae_policy",
        env_name=env_name,
        task_id=task_id,
        config_name=config_name,
        seeds=seeds,
        train_ratio=train_ratio,
        n_epochs=n_epochs,
        wandb_group=wandb_group,
        wandb_project=wandb_project,
        modalities=modalities,
        cameras=cameras,
        distract=distract,
        enable_rollout_during_train=enable_rollout_during_train,
        post_train_rollout=post_train_rollout,
        rep_loss_scale=rep_loss_scale,
    )


def build_curl_sweep_config(
    *,
    env_name: str,
    task_id: int,
    config_name: str,
    seeds: List[int],
    train_ratio: float,
    n_epochs: int,
    wandb_group: str,
    wandb_project: str = "bc-curl-vilt",
    modalities: List[str] = ("image", "proprio"),
    cameras: List[str] = ("agentview",),
    distract: bool = False,
    enable_rollout_during_train: bool = True,
    post_train_rollout: bool = True,
    rep_loss_scale: Any = None,
) -> Dict[str, Any]:
    """Hydra sweep entry for bc_curl_policy (CURL representation baseline)."""
    return _build_rep_baseline_sweep_config(
        "bc_curl_policy",
        env_name=env_name,
        task_id=task_id,
        config_name=config_name,
        seeds=seeds,
        train_ratio=train_ratio,
        n_epochs=n_epochs,
        wandb_group=wandb_group,
        wandb_project=wandb_project,
        modalities=modalities,
        cameras=cameras,
        distract=distract,
        enable_rollout_during_train=enable_rollout_during_train,
        post_train_rollout=post_train_rollout,
        rep_loss_scale=rep_loss_scale,
    )


def build_vip_sweep_config(
    *,
    env_name: str,
    task_id: int,
    config_name: str,
    seeds: List[int],
    train_ratio: float,
    n_epochs: int,
    wandb_group: str,
    wandb_project: str = "bc-vip-vilt",
    modalities: List[str] = ("image", "proprio"),
    cameras: List[str] = ("agentview",),
    distract: bool = False,
    enable_rollout_during_train: bool = True,
    post_train_rollout: bool = True,
    rep_loss_scale: Any = None,
) -> Dict[str, Any]:
    """Hydra sweep entry for bc_vip_policy (VIP representation baseline)."""
    return _build_rep_baseline_sweep_config(
        "bc_vip_policy",
        env_name=env_name,
        task_id=task_id,
        config_name=config_name,
        seeds=seeds,
        train_ratio=train_ratio,
        n_epochs=n_epochs,
        wandb_group=wandb_group,
        wandb_project=wandb_project,
        modalities=modalities,
        cameras=cameras,
        distract=distract,
        enable_rollout_during_train=enable_rollout_during_train,
        post_train_rollout=post_train_rollout,
        mixed_mode="vip",
        rep_loss_scale=rep_loss_scale,
    )


REP_BASELINE_MODALITY_SETS = [["image"]] #, ["image", "proprio"]]
REP_BASELINE_DISTRACT_VALUES = [False, True]
REP_BASELINE_SEEDS = [0, 1, 2, 3, 4]
VAE_BASELINE_MODALITY_SETS = REP_BASELINE_MODALITY_SETS
VAE_BASELINE_DISTRACT_VALUES = REP_BASELINE_DISTRACT_VALUES
VAE_BASELINE_SEEDS = REP_BASELINE_SEEDS
CURL_BASELINE_MODALITY_SETS = REP_BASELINE_MODALITY_SETS
CURL_BASELINE_DISTRACT_VALUES = REP_BASELINE_DISTRACT_VALUES
CURL_BASELINE_SEEDS = REP_BASELINE_SEEDS
VIP_BASELINE_MODALITY_SETS = REP_BASELINE_MODALITY_SETS
VIP_BASELINE_DISTRACT_VALUES = REP_BASELINE_DISTRACT_VALUES
VIP_BASELINE_SEEDS = REP_BASELINE_SEEDS


def build_bc_distract_sweep_config(
    *,
    env_name: str,
    policy: str,
    config_name: str,
    seeds: List[int],
    train_ratio: float,
    n_epochs: int,
    wandb_group: str,
    wandb_project: str,
    cameras: List[str] = ("agentview",),
    enable_rollout_during_train: bool = True,
    post_train_rollout: bool = True,
) -> Dict[str, Any]:
    """Hydra sweep entry for bc_policy / bc_ib_policy on distracted datasets."""
    cameras = list(cameras)
    distract_config = (
        config_name if config_name.endswith("_distract") else f"{config_name}_distract"
    )
    rgb_keys = [_CAMERA_RGB_KEY[c] for c in cameras]
    return {
        "--config-path": policy_config_path(policy),
        "--config-name": distract_config,
        "data.env_name": env_name,
        "data.train_ratio": train_ratio,
        "train.n_epochs": n_epochs,
        "train.seed": seeds,
        "train.train_gpus": "[0]",
        "data.obs.modality.rgb": "[" + ",".join(rgb_keys) + "]",
        "eval.enable_rollout": str(enable_rollout_during_train).lower(),
        "eval.post_train_rollout.enable": str(post_train_rollout).lower(),
        "wandb.project": wandb_project,
        "wandb.group": f"{wandb_group}_{env_name}_{distract_config}_cam-{camera_tag(cameras)}",
        "wandb.policy_arch": distract_config,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit LIBERO Hydra training jobs via sbatch.")
    parser.add_argument("--dry-run", action="store_true", help="Print Slurm scripts without submitting.")
    parser.add_argument("--num-runs-per-job", type=int, default=1, help="Sequential runs per Slurm job.")
    parser.add_argument("--job-name", type=str, default="libero-train")
    parser.add_argument("--partition", type=str, default="allnodes")
    parser.add_argument(
        "--exclude",
        type=str,
        default=DEFAULT_SLURM_EXCLUDE,
        help="Slurm --exclude node list (default: none).",
    )
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--mem", type=str, default="128GB")
    parser.add_argument("--time", dest="time_limit", type=str, default="24:00:00")
    parser.add_argument(
        "--wandb-group",
        "--group",
        dest="wandb_group",
        type=str,
        default="bc-cardpol-transformer",
        help="Base wandb run group (per-env/task/rep suffix added automatically).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="bc-cardpol-transformer",
        help="Wandb project name passed to Hydra as wandb.project.",
    )
    parser.add_argument(
        "--modalities",
        action="append",
        dest="modality_sets",
        metavar="image[,proprio][,language]",
        help=(
            "Comma-separated modalities to INCLUDE (e.g. 'image,proprio,language'). "
            "'image' is required. Repeat the flag to sweep multiple modality sets. "
            "If omitted, the modality_sets list defined in __main__ is used."
        ),
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        metavar="agentview[,wrist]",
        help=(
            "Comma-separated camera views to use as image inputs "
            "(options: agentview, wrist/eye_in_hand; aliases: agent, eih). "
            "Sets data.obs.modality.rgb. Default: agentview only (no wrist camera)."
        ),
    )
    parser.add_argument(
        "--distract",
        action="store_true",
        help="Use *_distract configs (datasets_distract + eval distractor rollouts).",
    )
    parser.add_argument(
        "--vae-baseline-sweep",
        action="store_true",
        help=(
            "Sweep bc_vae_policy baseline: modalities (image, image+proprio), "
            "distractions (yes/no), seeds 0-4."
        ),
    )
    parser.add_argument(
        "--curl-baseline-sweep",
        action="store_true",
        help=(
            "Sweep bc_curl_policy baseline: modalities (image, image+proprio), "
            "distractions (yes/no), seeds 0-4."
        ),
    )
    parser.add_argument(
        "--vip-baseline-sweep",
        action="store_true",
        help=(
            "Sweep bc_vip_policy baseline: modalities (image, image+proprio), "
            "distractions (yes/no), seeds 0-4."
        ),
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="bc_cardpol_policy",
        choices=[
            "bc_cardpol_policy",
            "bc_policy",
            "bc_ib_policy",
            "bc_vae_policy",
            "bc_curl_policy",
            "bc_vip_policy",
        ],
        help="Policy package (default: bc_cardpol_policy).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="vilt",
        help="Backbone config name; with --distract uses <backbone>_distract.yaml (default: vilt).",
    )
    parser.add_argument(
        "--rep-loss-scales",
        type=str,
        default=None,
        help=(
            "Comma-separated train.rep_loss_scale values to sweep for VAE/CURL/VIP "
            "baselines (e.g. '0.5,0.1,0.01,0.001'). Default: use the config value."
        ),
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated task ids to sweep (e.g. '0'). Default: all 10 tasks "
            "for baseline/cardpol sweeps."
        ),
    )
    parser.add_argument(
        "--distract-values",
        type=str,
        default=None,
        choices=["off", "on", "both"],
        help=(
            "For VAE/CURL/VIP baseline sweeps, choose distraction settings to sweep: "
            "'off' (no distractor), 'on' (distractor), or 'both'. "
            "Default: honor --distract (off unless --distract is set)."
        ),
    )
    parser.add_argument(
        "--envs",
        type=str,
        default=None,
        help=(
            "Comma-separated LIBERO env names to sweep "
            "(e.g. 'libero_goal,libero_object,libero_spatial'). "
            "Default: the libero_envs list hardcoded in __main__."
        ),
    )
    cli = parser.parse_args()

    distract_values_cli = {
        "off": [False],
        "on": [True],
        "both": [False, True],
    }.get(cli.distract_values)
    envs_cli = (
        [x.strip() for x in cli.envs.split(",") if x.strip()]
        if cli.envs
        else None
    )

    rep_loss_scales_cli = (
        [float(x) for x in cli.rep_loss_scales.split(",")]
        if cli.rep_loss_scales
        else None
    )
    task_ids_cli = (
        [int(x) for x in cli.task_ids.split(",")] if cli.task_ids else None
    )

    if sum([cli.vae_baseline_sweep, cli.curl_baseline_sweep, cli.vip_baseline_sweep]) > 1:
        parser.error(
            "Use at most one of --vae-baseline-sweep, --curl-baseline-sweep, "
            "and --vip-baseline-sweep."
        )

    rep_baseline_sweep = (
        cli.vae_baseline_sweep or cli.curl_baseline_sweep or cli.vip_baseline_sweep
    )
    if cli.vae_baseline_sweep:
        cli.policy = "bc_vae_policy"
    elif cli.curl_baseline_sweep:
        cli.policy = "bc_curl_policy"
    elif cli.vip_baseline_sweep:
        cli.policy = "bc_vip_policy"

    libero_envs = envs_cli if envs_cli is not None else [
        # "libero_spatial",
        # "libero_object",
        "libero_goal",
        # "libero_10",
    ]

    policies = [cli.policy] if cli.distract else [
        "bc_cardpol_policy",
        # "bc_policy",
        # "bc_ib_policy",
    ]
    config_names = [cli.backbone] if cli.distract else [
        # "transformer",
        "vilt",
        # "rnn",
        # "mlp",
    ]

    seeds = [0, 1, 2, 3, 4]
    task_ids = task_ids_cli if task_ids_cli is not None else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    rep_loss_scales = [0.0, 0.01] #, 0.001, 0.005]
    train_ratio = 0.9
    n_epochs = 50

    if rep_baseline_sweep:
        modality_sets = [parse_modality_set(",".join(m)) for m in REP_BASELINE_MODALITY_SETS]
        distract_values = (
            distract_values_cli
            if distract_values_cli is not None
            else ([True] if cli.distract else [False])
        )
        seeds = REP_BASELINE_SEEDS
    else:
        modality_sets = [
            # ["image"],
            ["image", "proprio"],
            # ["image", "proprio", "language"],
        ]
        if cli.modality_sets:
            modality_sets = [parse_modality_set(spec) for spec in cli.modality_sets]
        else:
            modality_sets = [parse_modality_set(",".join(m)) for m in modality_sets]
        distract_values = [True] if cli.distract else [False]

    cameras = parse_camera_set(cli.cameras) if cli.cameras else list(DEFAULT_CAMERAS)
    if (cli.distract or rep_baseline_sweep) and "agentview" not in cameras:
        parser.error(
            "--distract requires the 'agentview' camera: the distractor is baked "
            "into agentview_rgb (datasets_distract) and applied to agentview at "
            "rollout, and the shape check expects agentview_rgb=(3,128,256)."
        )

    if rep_baseline_sweep:
        if cli.vae_baseline_sweep:
            build_rep_baseline = build_vae_sweep_config
        elif cli.curl_baseline_sweep:
            build_rep_baseline = build_curl_sweep_config
        else:
            build_rep_baseline = build_vip_sweep_config
        sweep_configs = [
            build_rep_baseline(
                env_name=env_name,
                task_id=task_id,
                config_name=cli.backbone,
                seeds=seeds,
                train_ratio=train_ratio,
                n_epochs=n_epochs,
                wandb_group=cli.wandb_group,
                wandb_project=cli.wandb_project,
                modalities=modalities,
                cameras=cameras,
                distract=distract,
                rep_loss_scale=rep_loss_scales_cli,
            )
            for env_name in libero_envs
            for task_id in task_ids
            for modalities in modality_sets
            for distract in distract_values
        ]
    elif cli.distract and cli.policy in ("bc_policy", "bc_ib_policy"):
        sweep_configs = [
            build_bc_distract_sweep_config(
                env_name=env_name,
                policy=cli.policy,
                config_name=cli.backbone,
                seeds=seeds,
                train_ratio=train_ratio,
                n_epochs=n_epochs,
                wandb_group=cli.wandb_group,
                wandb_project=cli.wandb_project,
                cameras=cameras,
            )
            for env_name in libero_envs
        ]
    else:
        sweep_configs = [
            build_cardpol_sweep_config(
                env_name=env_name,
                task_id=task_id,
                rep_loss_scale=rep_loss_scale,
                policies=policies,
                config_names=config_names,
                seeds=seeds,
                train_ratio=train_ratio,
                n_epochs=n_epochs,
                wandb_group=cli.wandb_group,
                wandb_project=cli.wandb_project,
                modalities=modalities,
                cameras=cameras,
                distract=cli.distract,
            )
            for env_name in libero_envs
            for task_id in task_ids
            for rep_loss_scale in rep_loss_scales
            for modalities in modality_sets
        ]

    main(
        cli.dry_run,
        sweep_configs,
        num_runs_per_job=cli.num_runs_per_job,
        job_name=cli.job_name,
        partition=cli.partition,
        exclude=cli.exclude,
        gpus=cli.gpus,
        cpus=cli.cpus,
        mem=cli.mem,
        time_limit=cli.time_limit,
    )
    
#     python launch_scripts/mll/submit_libero.py \
#   --distract \
#   --job-name=libero-vilt-distract \
#   --wandb-project=bc-ib-vilt-distract \
#   --wandb-group=bc-vilt-distract --num-runs-per-job 6
#
#     python launch_scripts/mll/submit_libero.py \
#   --vae-baseline-sweep \
#   --dry-run \
#   --job-name=libero-vae-baseline \
#   --wandb-project=bc-vae-vilt \
#   --wandb-group=bc-vae-baseline --num-runs-per-job 5
#
#     python launch_scripts/mll/submit_libero.py \
#   --curl-baseline-sweep \
#   --dry-run \
#   --job-name=libero-curl-baseline \
#   --wandb-project=bc-curl-vilt \
#   --wandb-group=bc-curl-baseline --num-runs-per-job 5
#
#     python launch_scripts/mll/submit_libero.py \
#   --vip-baseline-sweep \
#   --dry-run \
#   --job-name=libero-vip-baseline \
#   --wandb-project=bc-vip-vilt \
#   --wandb-group=bc-vip-baseline --num-runs-per-job 5
