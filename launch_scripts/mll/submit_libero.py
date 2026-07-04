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
DEFAULT_SLURM_EXCLUDE = "slurm-node-[008-011]"

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_root}/job_%j/job_%j.out
#SBATCH --error={log_root}/job_%j/job_%j.err
#SBATCH --partition={partition}
#SBATCH --exclude={exclude}
#SBATCH --gres=gpu:{gpus}
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
    slurm_script = SLURM_TEMPLATE.format(
        job_name=job_name,
        log_root=SLURM_LOG_ROOT,
        partition=partition,
        exclude=exclude,
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
    wandb_group: str,
    wandb_project: str = "bc-cardpol-transformer",
    modalities: List[str] = ("image", "proprio", "language"),
) -> Dict[str, Any]:
    """One sweep entry: fixed env / task / rep scale; seeds expanded via product.

    `modalities` selects which inputs to include. 'image' is always required;
    'proprio' adds proprioceptive state, 'language' enables language conditioning.
    """
    modalities = list(modalities)
    use_proprio = "proprio" in modalities
    use_language = "language" in modalities

    group = wandb_run_group(wandb_group, env_name, task_id, rep_loss_scale)
    config: Dict[str, Any] = {
        "--config-path": [policy_config_path(p) for p in policies],
        "--config-name": config_names,
        "data.env_name": env_name,
        "data.train_ratio": train_ratio,
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
        "wandb.group": f"{group}_{modality_tag(modalities)}",
        "wandb.policy_arch": config_names,
    }

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
        help="Slurm --exclude node list (default: %(default)s).",
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
    cli = parser.parse_args()

    libero_envs = [
        # "libero_spatial",
        "libero_object",
        # "libero_goal",
        # "libero_10",
    ]
    policies = [
        "bc_cardpol_policy",
        # "bc_policy",
        # "bc_ib_policy",
    ]
    config_names = [
        "transformer",
        # "vilt",
        # "rnn",
        # "mlp",
    ]
    
    seeds = [0, 1, 2, 3, 4]
    task_ids = [0,1,2,3,]
    rep_loss_scales = [ 1.0, 0.1, 0.01 ]
    train_ratio = 0.9

    # Which input modalities to INCLUDE. 'image' is always required; add 'proprio'
    # and/or 'language'. Each entry is one run config; list multiple to sweep them.
    modality_sets = [
        ["image"],                            # vision only (no proprio, no language)
        # ["image", "proprio"],               # vision + proprio
        # ["image", "proprio", "language"],   # full (default LIBERO setup)
    ]

    # CLI --modalities (repeatable) overrides the list above.
    if cli.modality_sets:
        modality_sets = [parse_modality_set(spec) for spec in cli.modality_sets]
    else:
        modality_sets = [parse_modality_set(",".join(m)) for m in modality_sets]

    sweep_configs = [
        build_cardpol_sweep_config(
            env_name=env_name,
            task_id=task_id,
            rep_loss_scale=rep_loss_scale,
            policies=policies,
            config_names=config_names,
            seeds=seeds,
            train_ratio=train_ratio,
            wandb_group=cli.wandb_group,
            wandb_project=cli.wandb_project,
            modalities=modalities,
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
