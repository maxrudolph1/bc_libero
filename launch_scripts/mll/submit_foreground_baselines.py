#!/usr/bin/env python
"""Foreground-video VIP/CURL/VAE pilot, submitted as one bounded Slurm array."""
import argparse
import csv
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from launch_scripts.mll.submit_libero import (
    build_vip_sweep_config, build_curl_sweep_config, build_vae_sweep_config,
    configure_foreground_video, from_sweep_config_to_config,
    main as submit,
)

BUILDERS = {"vip": build_vip_sweep_config, "curl": build_curl_sweep_config,
            "vae": build_vae_sweep_config}


def build_configs(tasks, seeds, weights):
    configs = []
    for method, builder in BUILDERS.items():
        for task in tasks:
            for weight in weights:
                sweep = builder(
                    env_name="libero_goal", task_id=task, config_name="vilt",
                    seeds=seeds, train_ratio=0.9, n_epochs=50,
                    wandb_group=f"fg-baseline-pilot_{method}",
                    wandb_project=f"bc-{method}-vilt", modalities=["image"],
                    cameras=["agentview"], rep_loss_scale=weight,
                    **({"vae_type": "vae"} if method == "vae" else {}),
                )
                configs.extend(configure_foreground_video(config)
                               for config in from_sweep_config_to_config(sweep))
    return configs


def smoke():
    """Two optimizer steps plus checkpoint/video/metric checks for each method."""
    directory = REPO / "slurm_jobs" / "libero" / (
        "baseline_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    directory.mkdir(parents=True)
    for config in build_configs([0], [0], [0.1]):
        method = Path(config["--config-path"]).name
        config.update({
            "train.n_epochs": 2, "train.debug": "true", "train.batch_size": 64,
            "train.num_workers": 2, "eval.num_workers": 2,
            "env.env_num": 2, "env.num_env_rollouts": 2,
            "train.log_freq": 1, "train.print_freq": 1,
            "wandb.group": f"fg-baseline-smoke_{method}",
        })
        command = [sys.executable, "-u", "train_libero.py"]
        command.extend(f"{key}={value}" for key, value in config.items())
        log_path = directory / f"{method}.log"
        print(f"SMOKE {method}: {log_path}", flush=True)
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=REPO, stdout=log,
                                    stderr=subprocess.STDOUT, env={
                **os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
            })
        result.check_returncode()
        text = log_path.read_text()
        artifact = Path(re.search(r"finished training in (.*)", text).group(1))
        with (artifact / "metrics.csv").open() as f:
            rows = list(csv.DictReader(f))
        train_rows = [r for r in rows if r.get("train/loss")]
        assert len(train_rows) >= 2
        assert all(math.isfinite(float(r[k])) for r in train_rows
                   for k in ["train/loss", "train/bc_loss", "train/rep_loss"])
        assert (artifact / "model_final.ckpt").is_file()
        assert any(r.get("post_train/model_final/success_env_avg") for r in rows)
        assert "Traceback" not in text
        print(f"PASS {method}: finite losses, checkpoint, rollout, metrics, clean exit", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run local GPU checks; do not submit jobs")
    parser.add_argument("--tasks", nargs="+", type=int, default=[0, 2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--weights", nargs="+", type=float, default=[0.1, 0.01, 0.001])
    parser.add_argument("--array-max-parallel", type=int, default=16)
    args = parser.parse_args()
    if any(task not in range(10) for task in args.tasks):
        parser.error("Task IDs must be between 0 and 9")
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.seeds)) != len(args.seeds) or len(set(args.weights)) != len(args.weights):
        parser.error("Tasks, seeds, and weights must not contain duplicates")
    if args.smoke:
        if args.dry_run:
            parser.error("--smoke and --dry-run are mutually exclusive")
        smoke()
    else:
        submit(
            args.dry_run, build_configs(args.tasks, args.seeds, args.weights),
            train_script="MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python train_libero.py",
            use_array=True, array_max_parallel=args.array_max_parallel,
            job_name="fg-baseline-pilot", partition="allnodes", gpus=1,
            cpus=16, mem="128GB", time_limit="24:00:00",
        )
