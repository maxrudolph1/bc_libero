"""Run output directory layout and artifact persistence."""

from __future__ import annotations

import os
import re
import secrets
import shutil
from typing import Any, Optional

import wandb
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict


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


def setup_run_output_dir(cfg: DictConfig) -> str:
    """Create the run directory and sync cfg paths to it."""
    suffix = secrets.token_hex(3)
    run_dir = resolve_run_output_dir(cfg, suffix=suffix)
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


def export_wandb_metrics_csv(
    output_dir: str,
    filename: str = "metrics.csv",
) -> Optional[str]:
    """Export scalar wandb metrics for this run to CSV."""
    if wandb.run is None:
        return None

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    try:
        history = wandb.run.history(samples=100000)
        if history is None or history.empty:
            return None
        scalar_cols = [
            col
            for col in history.columns
            if col not in {"_step", "_timestamp"}
            and not col.startswith("_")
        ]
        cols = ["_step"] + [col for col in scalar_cols if col != "_step"]
        history = history.reindex(columns=[col for col in cols if col in history.columns])
        history.to_csv(out_path, index=False)
        return out_path
    except Exception as exc:
        print(f"[warn] failed to export wandb metrics csv: {exc}")
        return None


def finalize_run_artifacts(cfg: DictConfig) -> None:
    """Save configs and export wandb metrics at the end of training."""
    output_dir = cfg.experiment_dir
    save_run_configs(cfg, output_dir)
    export_wandb_metrics_csv(output_dir)
