#!/usr/bin/env python
"""Parse experiment artifacts into a single pandas DataFrame.

For each job directory under an experiment group (e.g.
``artifacts/07062026-distract-metrics_img+proprio_cam-agent_distract``) that
contains ``metrics_summary.csv``, read that file together with ``config.yaml``
and merge them into one row. All job rows are concatenated into a DataFrame.

Usage:
    python scripts/parse_experiment_results.py \\
        artifacts/07062026-distract-metrics_img+proprio_cam-agent_distract

    python scripts/parse_experiment_results.py \\
        artifacts/07062026-distract-metrics_img+proprio_cam-agent_distract \\
        --output results.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_FILENAME = "metrics_summary.csv"
CONFIG_FILENAME = "config.yaml"


def _flatten_mapping(
    value: Any,
    prefix: str = "",
    out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recursively flatten nested mappings/lists into scalar-friendly columns."""
    if out is None:
        out = {}

    if isinstance(value, Mapping):
        for key, nested in value.items():
            nested_key = f"{prefix}/{key}" if prefix else str(key)
            _flatten_mapping(nested, nested_key, out)
        return out

    if isinstance(value, (list, tuple)):
        if not value:
            out[prefix] = None
            return out
        if all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
            out[prefix] = ",".join("" if item is None else str(item) for item in value)
            return out
        out[prefix] = json.dumps(value, sort_keys=True)
        return out

    out[prefix] = value
    return out


def load_metrics_summary(metrics_path: Path) -> dict[str, Any]:
    """Load the final metrics summary row from a job directory."""
    df = pd.read_csv(metrics_path)
    if df.empty:
        return {}
    # Wandb summary export is one row; keep the last row if multiple exist.
    row = df.iloc[-1].to_dict()
    return {str(key): value for key, value in row.items()}


def load_run_config(config_path: Path, prefix: str = "cfg") -> dict[str, Any]:
    """Load and flatten a run ``config.yaml``."""
    if not config_path.exists():
        return {}

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    flat = _flatten_mapping(config)
    if prefix:
        return {f"{prefix}/{key}": value for key, value in flat.items()}
    return flat


def parse_job_dir(job_dir: Path) -> dict[str, Any] | None:
    """Build one result row from a single job artifact directory."""
    metrics_path = job_dir / METRICS_FILENAME
    if not metrics_path.is_file():
        return None

    row: dict[str, Any] = {
        "job_name": job_dir.name,
        "job_dir": str(job_dir.resolve()),
    }
    row.update(load_metrics_summary(metrics_path))
    row.update(load_run_config(job_dir / CONFIG_FILENAME))
    return row


def parse_experiment_dir(experiment_dir: Path) -> pd.DataFrame:
    """Parse all completed jobs under an experiment group directory."""
    experiment_dir = experiment_dir.resolve()
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"experiment directory not found: {experiment_dir}")

    rows: list[dict[str, Any]] = []
    for job_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
        row = parse_job_dir(job_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    preferred = ["job_name", "job_dir", "epoch", "rollout/success_env_avg"]
    front = [col for col in preferred if col in df.columns]
    rest = sorted(col for col in df.columns if col not in front)
    return df[front + rest]


REP_LOSS_SCALE_COL = "cfg/train/rep_loss_scale"
TASK_ID_COL = "cfg/env/task_id"
SUCCESS_COL = "rollout/success_env_avg"


def success_pivot_by_task_and_rep_loss_scale(
    df: pd.DataFrame,
    *,
    rep_loss_scale_col: str = REP_LOSS_SCALE_COL,
    task_id_col: str = TASK_ID_COL,
    success_col: str = SUCCESS_COL,
) -> pd.DataFrame:
    """Average success over seeds for each (task_id, rep_loss_scale) pair."""
    summary = (
        df.assign(**{success_col: pd.to_numeric(df[success_col], errors="coerce")})
        .groupby([task_id_col, rep_loss_scale_col], dropna=False)[success_col]
        .mean()
        .reset_index()
    )
    pivot = summary.pivot(
        index=task_id_col,
        columns=rep_loss_scale_col,
        values=success_col,
    ).sort_index()
    pivot.columns = [f"{col:g}" for col in pivot.columns]
    pivot.index.name = "task_id"
    return pivot


def _escape_latex_text(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
    }
    escaped = text
    for char, replacement in replacements.items():
        escaped = escaped.replace(char, replacement)
    return escaped


def _sanitize_latex_label(text: str) -> str:
    label = re.sub(r"[^\w]", "_", text)
    label = re.sub(r"_+", "_", label).strip("_").lower()
    return label or "experiment"


def _format_latex_cell(value: float, bold: bool) -> str:
    text = f"{value:.3f}"
    if bold:
        return f"\\textbf{{{text}}}"
    return text


def _format_latex_row(label: str, numeric_row: pd.Series) -> str:
    rounded = numeric_row.round(3)
    row_max = rounded.max(skipna=True)
    cells = [
        _format_latex_cell(value, rounded.loc[col] == row_max)
        for col, value in numeric_row.items()
    ]
    return f"{label} & " + " & ".join(cells) + " \\\\"


def format_success_latex_table(
    pivot: pd.DataFrame,
    *,
    experiment_name: str | None = None,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """Render a LaTeX table with the higher success rate bolded in each row."""
    base_caption = (
        "Average rollout success by task and rep\\_loss\\_scale (mean over seeds)."
    )
    if caption is None:
        if experiment_name:
            caption = (
                f"\\texttt{{{_escape_latex_text(experiment_name)}}}. {base_caption}"
            )
        else:
            caption = base_caption

    if label is None:
        if experiment_name:
            label = f"tab:{_sanitize_latex_label(experiment_name)}_rep_loss_success"
        else:
            label = "tab:rep_loss_success"

    column_spec = "l" + "c" * len(pivot.columns)
    header = " & ".join(["Task", *pivot.columns.tolist()])
    body_lines: list[str] = []

    for task_id, row in pivot.iterrows():
        body_lines.append(_format_latex_row(str(task_id), row.astype(float)))

    avg_row = pivot.astype(float).mean(numeric_only=True)
    body_lines.append("\\midrule")
    body_lines.append(_format_latex_row("Average", avg_row))

    body = "\n".join(body_lines)
    return f"""\\begin{{table}}
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{column_spec}}}
\\toprule
{header} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse experiment job artifacts into a pandas DataFrame.",
    )
    parser.add_argument(
        "-e", "--experiment-dir",
        type=Path,
        help="Experiment group directory under artifacts/ (contains job subdirs).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional path to write the parsed DataFrame as CSV.",
    )
    parser.add_argument(
        "--latex-success-table",
        action="store_true",
        help=(
            "Print a LaTeX table of mean rollout/success_env_avg by task (rows) "
            "and rep_loss_scale (columns), bolding the higher value in each row."
        ),
    )
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    if not experiment_dir.is_absolute():
        experiment_dir = REPO_ROOT / experiment_dir

    df = parse_experiment_dir(experiment_dir)
    if df.empty:
        print(
            f"No jobs with {METRICS_FILENAME} found under {experiment_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Parsed {len(df)} job(s) from {experiment_dir}")
    print(f"Columns: {len(df.columns)}")

    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Wrote {output_path}")

    if args.latex_success_table:
        pivot = success_pivot_by_task_and_rep_loss_scale(df)
        print(format_success_latex_table(pivot, experiment_name=experiment_dir.name))
        return

    with pd.option_context("display.max_columns", 12, "display.width", 200):
        print(df.head())


if __name__ == "__main__":
    main()
