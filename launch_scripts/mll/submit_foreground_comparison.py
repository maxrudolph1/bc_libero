#!/usr/bin/env python
"""Prepare/submit matched foreground-video runs, gated on data and smoke checks."""
import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from hydra import compose, initialize_config_dir
from launch_scripts.mll.submit_foreground_baselines import BUILDERS
from launch_scripts.mll.submit_libero import (
    build_cardpol_sweep_config, configure_foreground_video,
    from_sweep_config_to_config,
)

WEIGHTS = {"bc": 0.0, "cardpol": 0.01, "vip": 0.001, "curl": 0.001, "vae": 0.001}
DATA_ROOT = Path("/datastor2/mrudolph/LIBERO/datasets_foreground_video")


def build_records(suites, tasks, seeds):
    records = []
    # Interleave methods/tasks so partial progress is not all from one method.
    for task in tasks:
        for method, weight in WEIGHTS.items():
            for suite in suites:
                common = dict(env_name=suite, task_id=task, seeds=seeds,
                              train_ratio=0.9, n_epochs=50,
                              wandb_group="fg-comparison_" + method,
                              wandb_project="bc-cardpol-vilt" if method in ("bc", "cardpol") else "bc-" + method + "-vilt",
                              modalities=["image"], cameras=["agentview"], rep_loss_scale=weight)
                if method in ("bc", "cardpol"):
                    sweep = build_cardpol_sweep_config(
                        **common, policies=["bc_cardpol_policy"], config_names=["vilt"],
                        stop_encoder_grad_from_bc=False)
                else:
                    sweep = BUILDERS[method](**common, config_name="vilt",
                                             **({"vae_type": "vae"} if method == "vae" else {}))
                for config in from_sweep_config_to_config(sweep):
                    records.append(dict(suite=suite, method=method, task=task,
                                        weight=weight, seed=config["train.seed"],
                                        config=configure_foreground_video(config)))
    return records


def preflight(records):
    seen = set()
    for r in records:
        key = (r["suite"], r["method"], r["task"], r["seed"])
        assert key not in seen, key
        seen.add(key)
        c = r["config"]
        with initialize_config_dir(config_dir=str(REPO / c["--config-path"]), version_base=None):
            cfg = compose(config_name=c["--config-name"], overrides=[
                "%s=%s" % (k, v) for k, v in c.items() if not k.startswith("--")])
        assert cfg.data.env_name == cfg.env.env_name == r["suite"]
        assert cfg.env.task_id == cfg.data.dual_task.focused_task_id == r["task"]
        assert cfg.train.seed == r["seed"] and cfg.train.rep_loss_scale == r["weight"]
        assert cfg.train.n_epochs == 50 and cfg.train.batch_size == 64
        assert cfg.train.save_final_only and not cfg.train.stop_encoder_grad_from_bc
        assert cfg.data.root_dir == str(DATA_ROOT)
        assert cfg.data.distractor.enable and cfg.data.distractor.layout == "keep_foreground_video"
        assert list(cfg.data.obs.modality.rgb) == ["agentview_rgb"]
        assert not list(cfg.data.obs.modality.low_dim) and not cfg.policy.use_language_conditioning
        assert cfg.env.num_env_rollouts == 100 and cfg.env.env_num == 10 and cfg.env.horizon == 600
        assert cfg.eval.post_train_rollout.enable and not cfg.eval.post_train_rollout.eval_all
        assert not cfg.eval.enable_rollout and cfg.wandb.mode == "offline"
        assert cfg.data.dual_task.mixed_mode == ("vip" if r["method"] == "vip" else "future_pair")
        if r["method"] == "vae":
            assert cfg.train.vae_type == "vae"
        assert r["suite"] in cfg.wandb.group
    print("PASS: %d unique Hydra configs and matched settings" % len(records), flush=True)


def train_command(config):
    return " ".join(shlex.quote(x) for x in [str(REPO / ".venv/bin/python"), "-u", "train_libero.py"] +
                    ["%s=%s" % (k, v) for k, v in config.items()])


def smoke(suite, log_dir):
    import h5py
    folder = DATA_ROOT / suite
    assert (folder / "VALIDATED_COMPLETE.txt").is_file(), "Dataset validation marker missing"
    files = sorted(folder.glob("*.hdf5"))
    assert len(files) == 10
    for path in files:
        with h5py.File(path) as f:
            assert f.attrs["foreground_video_complete"] and len(f["data"]) == 50
    subprocess.run([sys.executable, "scripts/validate_libero_foreground_env.py", "--suite", suite],
                   cwd=REPO, check=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for r in build_records([suite], [0], [0]):
        c = r["config"]
        c.update({"train.n_epochs": 2, "train.debug": "true", "train.save_freq": 1,
                  "train.num_workers": 2, "eval.num_workers": 2,
                  "env.env_num": 2, "env.num_env_rollouts": 2, "env.horizon": 2,
                  "train.log_freq": 1, "train.print_freq": 1,
                  "wandb.group": "fg-comparison-smoke_" + suite + "_" + r["method"]})
        log = log_dir / (r["method"] + ".log")
        with log.open("w") as f:
            subprocess.run(shlex.split(train_command(c)), cwd=REPO, stdout=f,
                           stderr=subprocess.STDOUT, check=True)
        text = log.read_text()
        artifact = Path(re.findall(r"finished training in (.+)", text)[-1].strip())
        with (artifact / "metrics.csv").open() as f:
            rows = list(csv.DictReader(f))
        train_rows = [row for row in rows if row.get("train/loss")]
        assert len(train_rows) >= 2
        assert all(math.isfinite(float(row[k])) for row in train_rows
                   for k in ["train/loss", "train/bc_loss", "train/rep_loss"])
        assert sorted(p.name for p in artifact.glob("*.ckpt")) == ["model_final.ckpt"]
        assert any(row.get("post_train/model_final/success_env_avg") and
                   math.isfinite(float(row["post_train/model_final/success_env_avg"])) for row in rows)
        assert "Traceback" not in text
        print("PASS smoke %s %s: losses, final-only checkpoint, reload, rollout, metrics" %
              (suite, r["method"]), flush=True)
    (log_dir / "PASSED.txt").write_text(datetime.now().isoformat() + "\n")


def batch_script(name, output, error, body, array=None, dependency=None, gate=False):
    lines = ["#!/bin/bash", "#SBATCH --job-name=" + name,
             "#SBATCH --output=" + str(output), "#SBATCH --error=" + str(error),
             "#SBATCH --partition=allnodes", "#SBATCH --gres=gpu:1",
             "#SBATCH --cpus-per-task=16", "#SBATCH --mem=128GB",
             "#SBATCH --time=" + ("02:00:00" if gate else "24:00:00")]
    if array:
        lines.append("#SBATCH --array=" + array)
    if dependency:
        lines.append("#SBATCH --dependency=afterok:" + ":".join(map(str, dependency)))
    return "\n".join(lines) + "\n" + "\n".join([
        "source ~/.bashrc", "set -euo pipefail", "cd " + shlex.quote(str(REPO)),
        "source .venv/bin/activate", "export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl",
        "export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        body, ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object"],
                        choices=["libero_spatial", "libero_object"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--tasks", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--array-max-parallel", type=int, default=16)
    parser.add_argument("--per-suite", action="store_true", help="Separate array throttle per suite")
    parser.add_argument("--data-job", action="append", default=[], metavar="SUITE=JOB_ID")
    parser.add_argument("--submit", action="store_true", help="Otherwise only prepare scripts and manifests")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.array_max_parallel < 1 or any(t not in range(10) for t in args.tasks) or any(s < 0 for s in args.seeds):
        parser.error("Invalid concurrency, task, or seed")
    if any(len(v) != len(set(v)) for v in [args.suites, args.tasks, args.seeds]):
        parser.error("Suites, tasks, and seeds must be unique")
    if args.smoke:
        if len(args.suites) != 1 or not args.output_dir:
            parser.error("--smoke requires one suite and --output-dir")
        smoke(args.suites[0], args.output_dir.resolve())
        return
    dependencies = dict(item.split("=", 1) for item in args.data_job)
    if any(not job.isdigit() for job in dependencies.values()):
        parser.error("Data job IDs must be numeric")
    records = build_records(args.suites, args.tasks, args.seeds)
    preflight(records)
    out = (args.output_dir or REPO / "slurm_jobs/libero" /
           ("comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "jobs.json").exists():
        parser.error("This directory already has submitted jobs; use a new directory")
    (out / "logs").mkdir(exist_ok=True)
    gates = {}
    for suite in args.suites:
        if suite not in dependencies and not (DATA_ROOT / suite / "VALIDATED_COMPLETE.txt").is_file():
            parser.error("Supply --data-job for unvalidated suite: " + suite)
        body = " ".join(shlex.quote(x) for x in [
            str(REPO / ".venv/bin/python"), "-u", str(Path(__file__).resolve()),
            "--smoke", "--suites", suite, "--output-dir", str(out / ("smoke_" + suite))])
        script = out / ("gate_" + suite + ".sh")
        script.write_text(batch_script("fg-check-" + suite[7:], out / "logs" / (suite + "_gate_%j.out"),
                                      out / "logs" / (suite + "_gate_%j.err"), body,
                                      dependency=[dependencies[suite]] if suite in dependencies else None, gate=True))
        gates[suite] = {"script": str(script), "data_job": dependencies.get(suite)}
    arrays = []
    for group in ([[suite] for suite in args.suites] if args.per_suite else [args.suites]):
        label = group[0] if args.per_suite else "both"
        selected = [r for r in records if r["suite"] in group]
        task_dir = out / ("tasks_" + label)
        task_dir.mkdir(exist_ok=True)
        for i, r in enumerate(selected):
            r.update(array=label, array_index=i, status="prepared")
            checks = ["test -f " + shlex.quote(str(DATA_ROOT / r["suite"] / "VALIDATED_COMPLETE.txt")),
                      "test -f " + shlex.quote(str(out / ("smoke_" + r["suite"]) / "PASSED.txt"))]
            (task_dir / (str(i) + ".sh")).write_text("#!/bin/bash\nset -euo pipefail\n" +
                                                  "\n".join(checks + [train_command(r["config"])]) + "\n")
        script = out / ("submission_" + label + ".sh")
        body = 'bash ' + shlex.quote(str(task_dir)) + '/"${SLURM_ARRAY_TASK_ID}.sh"'
        script.write_text(batch_script("fg-" + label + "-s3", out / "logs" / (label + "_%A_%a.out"),
                                      out / "logs" / (label + "_%A_%a.err"), body,
                                      array="0-%d%%%d" % (len(selected) - 1, args.array_max_parallel)))
        arrays.append(dict(label=label, suites=group, script=str(script), count=len(selected)))
    def save():
        (out / "experiment_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
        with (out / "experiment_manifest.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=["suite", "method", "task", "weight", "seed", "array", "array_index", "status", "job"], extrasaction="ignore")
            writer.writeheader(); writer.writerows(records)
    save()
    (out / "preflight.json").write_text(json.dumps(dict(
        checked_configs=len(records), suites=args.suites, seeds=args.seeds,
        tasks=args.tasks, weights=WEIGHTS, concurrency=args.array_max_parallel,
        concurrency_scope="per_suite" if args.per_suite else "total",
        checked_at=datetime.now().isoformat()), indent=2) + "\n")
    for script in list(out.glob("*.sh")) + list(out.glob("tasks_*/*.sh")):
        subprocess.run(["bash", "-n", str(script)], check=True)
    if args.submit:
        jobs = dict(gates=gates, arrays=arrays)
        def submit(script, dependencies=None):
            cmd = ["sbatch", "--parsable"]
            if dependencies:
                cmd.append("--dependency=afterok:" + ":".join(dependencies))
            cmd.append(str(script))
            job = subprocess.check_output(cmd, text=True).strip().split(";")[0]
            assert job.isdigit(), job
            return job
        for suite, gate in gates.items():
            gate["job"] = submit(gate["script"])
            (out / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
        for array in arrays:
            array["dependencies"] = [gates[s]["job"] for s in array["suites"]]
            script = Path(array["script"])
            script.write_text(script.read_text().replace(
                "#!/bin/bash\n", "#!/bin/bash\n#SBATCH --dependency=afterok:" +
                ":".join(array["dependencies"]) + "\n", 1))
            array["job"] = submit(array["script"])
            for r in records:
                if r["array"] == array["label"]:
                    r.update(job=array["job"] + "_" + str(r["array_index"]), status="submitted")
            save()
            (out / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
        print(json.dumps(jobs, indent=2), flush=True)
    print("Prepared %d runs in %s" % (len(records), out), flush=True)


if __name__ == "__main__":
    main()
