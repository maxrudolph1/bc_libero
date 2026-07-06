"""Run output directory layout and artifact persistence."""

from __future__ import annotations

import os
import re
import secrets
import shutil
from typing import Any, Optional

import pandas as pd
import wandb
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from .record_utils import WandbMetricsLogger


def _sanitize_path_component(value: Any) -> str:
    text = str(value).strip()
    text = text.replace(os.sep, "_")
    if os.altsep:
        text = text.replace(os.altsep, "_")
    text = re.sub(r"[^\w.\-+]", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "run"


def resolve_run_output_dir(cfg: DictConfig, *, suffix: Optional[str] = None) -> str:
    """Return `artifacts/<wandb-group>/<wandb-name>_<suffix>/` for this run."""
    group = OmegaConf.select(cfg, "wandb.group")
    name = OmegaConf.select(cfg, "wandb.name")
    if group is None or str(group) in ("", "None", "null", "???"):
        group = "ungrouped"
    if name is None or str(name) in ("", "None", "null", "???"):
        name = "run"
    if suffix is None:
        suffix = secrets.token_hex(3)
    run_name = f"{_sanitize_path_component(name)}_{suffix}"
    return os.path.join("artifacts", _sanitize_path_component(group), run_name)


def resolve_run_artifact_dir(cfg: DictConfig) -> str:
    """Absolute path to the run artifact directory (checkpoints, configs, metrics)."""
    artifact_dir = OmegaConf.select(cfg, "experiment_dir")
    if artifact_dir is None or str(artifact_dir) in ("", "None", "null", "???"):
        artifact_dir = resolve_run_output_dir(cfg)
    return os.path.abspath(str(artifact_dir))


def setup_run_output_dir(cfg: DictConfig) -> str:
    """Create the run directory and sync cfg paths to it."""
    suffix = secrets.token_hex(3)
    run_dir = os.path.abspath(resolve_run_output_dir(cfg, suffix=suffix))
    os.makedirs(run_dir, exist_ok=True)
    with open_dict(cfg):
        cfg.experiment_dir = run_dir
        if OmegaConf.select(cfg, "wandb") is not None:
            cfg.wandb.dir = run_dir
    return run_dir


def save_run_configs(cfg: DictConfig, output_dir: str) -> None:
    """Persist resolved config and Hydra overrides for the run."""
    os.makedirs(output_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))

    hydra_cfg = HydraConfig.get()
    overrides = {
        "job": {
            "name": OmegaConf.select(hydra_cfg, "job.name", default=None),
            "num": OmegaConf.select(hydra_cfg, "job.num", default=None),
            "override_dirname": OmegaConf.select(hydra_cfg, "job.override_dirname", default=None),
        },
        "runtime": {
            "output_dir": OmegaConf.select(hydra_cfg, "runtime.output_dir", default=None),
            "choices": dict(OmegaConf.select(hydra_cfg, "runtime.choices", default={})),
        },
        "overrides": {
            "task": list(OmegaConf.select(hydra_cfg, "overrides.task", default=[])),
            "config": list(OmegaConf.select(hydra_cfg, "overrides.config", default=[])),
        },
    }
    with open(os.path.join(output_dir, "hydra_overrides.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(overrides, f, sort_keys=False)

    hydra_output_dir = OmegaConf.select(hydra_cfg, "runtime.output_dir", default=None)
    if hydra_output_dir:
        hydra_config_src = os.path.join(hydra_output_dir, ".hydra", "config.yaml")
        if os.path.exists(hydra_config_src):
            shutil.copy2(
                hydra_config_src,
                os.path.join(output_dir, "hydra_config.yaml"),
            )


def _history_from_wandb_run() -> pd.DataFrame:
    if wandb.run is None:
        return pd.DataFrame()

    try:
        history = wandb.run.history(samples=1_000_000, pandas=True)
        if history is None or history.empty:
            return pd.DataFrame()
        non_scalar_cols = [
            col
            for col in history.columns
            if col.startswith("_") and col not in {"_step", "_timestamp"}
        ]
        drop_cols = non_scalar_cols + [
            col
            for col in history.columns
            if history[col].dtype == object
        ]
        return history.drop(columns=[col for col in drop_cols if col in history.columns], errors="ignore")
    except Exception as exc:
        print(f"[warn] failed to read wandb.run.history(): {exc}")
        return pd.DataFrame()


def _summary_from_wandb_run() -> pd.DataFrame:
    if wandb.run is None:
        return pd.DataFrame()

    summary = dict(wandb.run.summary)
    scalar_summary = {}
    for key, value in summary.items():
        if key.startswith("_"):
            continue
        if isinstance(value, (int, float, bool)):
            scalar_summary[key] = value
    if not scalar_summary:
        return pd.DataFrame()
    return pd.DataFrame([scalar_summary])


def _merge_metric_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True, sort=False)
    if "_step" not in merged.columns:
        return merged

    merged = merged.sort_values("_step", kind="mergesort")
    merged = merged.drop_duplicates(subset=["_step"], keep="last")
    preferred = ["_step", "_timestamp", "epoch"]
    other_cols = [col for col in merged.columns if col not in preferred]
    ordered = [col for col in preferred if col in merged.columns] + sorted(other_cols)
    return merged.reindex(columns=ordered)


def export_wandb_metrics_csv(
    output_dir: str,
    metrics_logger: Optional[WandbMetricsLogger] = None,
    filename: str = "metrics.csv",
) -> Optional[str]:
    """Export scalar wandb metrics into the run artifact directory."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    try:
        frames = []
        if metrics_logger is not None:
            frames.append(metrics_logger.to_dataframe())
        frames.append(_history_from_wandb_run())
        metrics_df = _merge_metric_frames(*frames)
        if metrics_df.empty:
            print(f"[warn] no wandb metrics available to export under {output_dir}")
            return None

        metrics_df.to_csv(out_path, index=False)

        summary_path = os.path.join(output_dir, "metrics_summary.csv")
        summary_df = _summary_from_wandb_run()
        if not summary_df.empty:
            summary_df.to_csv(summary_path, index=False)

        if wandb.run is not None:
            wandb.save(out_path, base_path=output_dir, policy="now")
            if os.path.exists(summary_path):
                wandb.save(summary_path, base_path=output_dir, policy="now")

        print(f"saved wandb metrics csv to {out_path}")
        return out_path
    except Exception as exc:
        print(f"[warn] failed to export wandb metrics csv to {output_dir}: {exc}")
        return None


def finalize_run_artifacts(
    cfg: DictConfig,
    metrics_logger: Optional[WandbMetricsLogger] = None,
) -> None:
    """Save configs and export wandb metrics alongside other run artifacts."""
    output_dir = resolve_run_artifact_dir(cfg)
    with open_dict(cfg):
        cfg.experiment_dir = output_dir
        if OmegaConf.select(cfg, "wandb") is not None:
            cfg.wandb.dir = output_dir
    save_run_configs(cfg, output_dir)
    export_wandb_metrics_csv(output_dir, metrics_logger=metrics_logger)
