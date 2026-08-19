#!/usr/bin/env python3
"""Submit LIBERO representation-loss *weight* sweeps to Slurm.

Defaults: 5 ``rep_loss_scale`` values, 4 seeds, 3 LIBERO suites, distractions
ON, 100 epochs. Choose the method with ``--baseline`` (start with cardpol).

    python launch_scripts/mll/submit_libero_weight_sweep.py --baseline cardpol --dry-run
    python launch_scripts/mll/submit_libero_weight_sweep.py --baseline cardpol --array
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path("/u/mrudolph/documents/BC-IB")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from launch_scripts.mll.submit_libero import (  # noqa: E402
    DEFAULT_CAMERAS,
    DEFAULT_SLURM_EXCLUDE,
    build_cardpol_sweep_config,
    build_curl_sweep_config,
    build_icvf_sweep_config,
    build_vae_sweep_config,
    build_vip_sweep_config,
    main,
    parse_camera_set,
    parse_modality_set,
)

DEFAULT_ENVS = ["libero_spatial", "libero_goal", "libero_object"]
DEFAULT_SEEDS = [0, 1, 2, 3]
DEFAULT_TASK_IDS = list(range(10))
DEFAULT_REP_LOSS_SCALES = [0.005, 0.01, 0.05, 0.1, 0.5]
DEFAULT_N_EPOCHS = 100
DEFAULT_MODALITIES = ["image"]

BASELINES = ("cardpol", "vae", "vqvae", "curl", "vip", "icvf")


def _parse_csv_floats(spec: str) -> List[float]:
    values = [float(x) for x in spec.split(",") if x.strip()]
    if not values:
        raise ValueError("expected at least one float")
    return values


def _parse_csv_ints(spec: str) -> List[int]:
    values = [int(x) for x in spec.split(",") if x.strip()]
    if not values:
        raise ValueError("expected at least one int")
    return values


def _parse_csv_strs(spec: str) -> List[str]:
    values = [x.strip() for x in spec.split(",") if x.strip()]
    if not values:
        raise ValueError("expected at least one value")
    return values


def build_weight_sweep_configs(
    *,
    baseline: str,
    envs: List[str],
    task_ids: List[int],
    seeds: List[int],
    rep_loss_scales: List[float],
    n_epochs: int,
    backbone: str,
    modalities: List[str],
    cameras: List[str],
    wandb_group: str,
    wandb_project: str,
    train_ratio: float = 0.9,
) -> List[Dict[str, Any]]:
    """One Hydra sweep entry per (env, task, weight); seeds expand via product."""
    common = dict(
        seeds=seeds,
        train_ratio=train_ratio,
        n_epochs=n_epochs,
        wandb_group=wandb_group,
        wandb_project=wandb_project,
        modalities=modalities,
        cameras=cameras,
        distract=True,
    )
    baseline_common = dict(config_name=backbone, **common)

    configs: List[Dict[str, Any]] = []
    for env_name in envs:
        for task_id in task_ids:
            for scale in rep_loss_scales:
                if baseline == "cardpol":
                    configs.append(
                        build_cardpol_sweep_config(
                            env_name=env_name,
                            task_id=task_id,
                            rep_loss_scale=scale,
                            policies=["bc_cardpol_policy"],
                            config_names=[backbone],
                            **common,
                        )
                    )
                elif baseline == "vae":
                    configs.append(
                        build_vae_sweep_config(
                            env_name=env_name,
                            task_id=task_id,
                            rep_loss_scale=scale,
                            vae_type="vae",
                            **baseline_common,
                        )
                    )
                elif baseline == "vqvae":
                    configs.append(
                        build_vae_sweep_config(
                            env_name=env_name,
                            task_id=task_id,
                            rep_loss_scale=scale,
                            vae_type="vqvae",
                            **baseline_common,
                        )
                    )
                elif baseline == "curl":
                    configs.append(
                        build_curl_sweep_config(
                            env_name=env_name,
                            task_id=task_id,
                            rep_loss_scale=scale,
                            **baseline_common,
                        )
                    )
                elif baseline == "vip":
                    configs.append(
                        build_vip_sweep_config(
                            env_name=env_name,
                            task_id=task_id,
                            rep_loss_scale=scale,
                            **baseline_common,
                        )
                    )
                elif baseline == "icvf":
                    configs.append(
                        build_icvf_sweep_config(
                            env_name=env_name,
                            task_id=task_id,
                            rep_loss_scale=scale,
                            **baseline_common,
                        )
                    )
                else:
                    raise ValueError(f"Unknown baseline {baseline!r}")
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Submit LIBERO rep_loss_scale weight sweeps via sbatch. "
            "Defaults: 5 weights, 4 seeds, 3 envs, distract ON, 100 epochs."
        )
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="cardpol",
        choices=list(BASELINES),
        help="Method to sweep (default: cardpol). Repeat later for other baselines.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print Slurm scripts without submitting.")
    parser.add_argument("--num-runs-per-job", type=int, default=1, help="Sequential runs per Slurm job/array task.")
    parser.add_argument(
        "--array",
        action="store_true",
        help=(
            "Submit one Slurm array job instead of many independent jobs. "
            "Each array index runs num-runs-per-job sequential trainings."
        ),
    )
    parser.add_argument(
        "--array-max-parallel",
        type=int,
        default=None,
        help="Optional concurrency cap for --array (Slurm %%N syntax).",
    )
    parser.add_argument("--job-name", type=str, default=None, help="Slurm job name (default: libero-weight-<baseline>).")
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
        default=None,
        help="Base wandb group (default: bc-<baseline>-weight-sweep).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="bc-libero-long",
        help="Wandb project name passed to Hydra as wandb.project.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="vilt",
        help="Backbone config name; distract ON uses <backbone>_distract.yaml (default: vilt).",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=",".join(DEFAULT_MODALITIES),
        metavar="image[,proprio][,language]",
        help="Comma-separated modalities to include (default: image).",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        metavar="agentview[,wrist]",
        help="Comma-separated camera views (default: agentview).",
    )
    parser.add_argument(
        "--rep-loss-scales",
        type=str,
        default=",".join(str(x) for x in DEFAULT_REP_LOSS_SCALES),
        help="Comma-separated train.rep_loss_scale values (default: 0.005,0.01,0.05,0.1,0.5).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds (default: 0,1,2,3).",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Comma-separated task ids (default: all 10 tasks).",
    )
    parser.add_argument(
        "--envs",
        type=str,
        default=",".join(DEFAULT_ENVS),
        help="Comma-separated LIBERO env names (default: spatial, goal, object).",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=DEFAULT_N_EPOCHS,
        help="train.n_epochs (default: 100).",
    )
    cli = parser.parse_args()

    if cli.array_max_parallel is not None and not cli.array:
        parser.error("--array-max-parallel requires --array")

    try:
        envs = _parse_csv_strs(cli.envs)
        seeds = _parse_csv_ints(cli.seeds)
        rep_loss_scales = _parse_csv_floats(cli.rep_loss_scales)
        modalities = parse_modality_set(cli.modalities)
        cameras = parse_camera_set(cli.cameras) if cli.cameras else list(DEFAULT_CAMERAS)
        task_ids = (
            _parse_csv_ints(cli.task_ids) if cli.task_ids is not None else list(DEFAULT_TASK_IDS)
        )
    except ValueError as exc:
        parser.error(str(exc))

    if "agentview" not in cameras:
        parser.error(
            "weight sweeps use distract ON and require the 'agentview' camera: "
            "the distractor is baked into agentview_rgb."
        )

    wandb_group = cli.wandb_group or f"bc-{cli.baseline}-weight-sweep"
    job_name = cli.job_name or f"libero-weight-{cli.baseline}"

    sweep_configs = build_weight_sweep_configs(
        baseline=cli.baseline,
        envs=envs,
        task_ids=task_ids,
        seeds=seeds,
        rep_loss_scales=rep_loss_scales,
        n_epochs=cli.n_epochs,
        backbone=cli.backbone,
        modalities=modalities,
        cameras=cameras,
        wandb_group=wandb_group,
        wandb_project=cli.wandb_project,
    )
    n_runs = len(sweep_configs) * len(seeds)
    print(
        f"# baseline={cli.baseline}  envs={envs}  tasks={len(task_ids)}  "
        f"weights={rep_loss_scales}  seeds={seeds}  epochs={cli.n_epochs}  "
        f"distract=on  runs={n_runs}"
    )

    main(
        cli.dry_run,
        sweep_configs,
        num_runs_per_job=cli.num_runs_per_job,
        job_name=job_name,
        partition=cli.partition,
        exclude=cli.exclude,
        gpus=cli.gpus,
        cpus=cli.cpus,
        mem=cli.mem,
        time_limit=cli.time_limit,
        use_array=cli.array,
        array_max_parallel=cli.array_max_parallel,
    )
