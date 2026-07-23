#!/usr/bin/env python
"""Build a mean success table by method and environment setting.

Rows flatten environment (goal, object, spatial), observation (img,
img+proprio), and distraction (on/off). Columns are methods:
Vanilla (CardPol with rep_loss_scale=0), CardPol with rep_loss_scale=0.01,
and VAE.

Usage:
    python scripts/parse_method_env_table.py

    python scripts/parse_method_env_table.py --csv-output scripts/method_env_table.csv
    python scripts/parse_method_env_table.py --latex-output scripts/method_env_table.tex
    python scripts/parse_method_env_table.py --input-csv exports/.../all_runs.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_artifact_runs import _find_job_dirs, build_runs_dataframe
from parse_experiment_results import (
    METRICS_FILENAME,
    REPO_ROOT,
    SUCCESS_COL,
    TASK_ID_COL,
)

DEFAULT_LATEX_OUTPUT = SCRIPT_DIR / "method_env_table.tex"
DEFAULT_CSV_OUTPUT = SCRIPT_DIR / "method_env_table.csv"

ENV_NAME_COL = "cfg/env/env_name"
DISTRACTOR_COL = "cfg/data/distractor/enable"
LOW_DIM_COL = "cfg/data/obs/modality/low_dim"
ALGO_NAME_COL = "cfg/algo/algo_name"
REP_LOSS_COL = "cfg/train/rep_loss_scale"

ENV_LABELS = {
    "goal": "Goal",
    "object": "Object",
    "spatial": "Spatial",
}
ENV_ORDER = ["goal", "object", "spatial"]
OBS_LABELS = {
    "img": "img",
    "img+proprio": "img+proprio",
}
OBS_ORDER = ["img", "img+proprio"]
DISTRACT_LABELS = {
    False: "no",
    True: "yes",
}
DISTRACT_ORDER = [False, True]

METHOD_ORDER = ["cardpol_base", "cardpol_rl", "vae"]
METHOD_LABELS = {
    "cardpol_base": "Vanilla",
    "cardpol_rl": "CardPol (RL=0.01)",
    "vae": "VAE",
}

ENV_NAME_MAP = {
    "libero_goal": "goal",
    "libero_object": "object",
    "libero_spatial": "spatial",
}


def load_runs_dataframe(
    *,
    artifacts_dir: Path | None = None,
    input_csv: Path | None = None,
) -> pd.DataFrame:
    if input_csv is not None:
        return pd.read_csv(input_csv)
    if artifacts_dir is None:
        artifacts_dir = REPO_ROOT / "artifacts"
    jobs = _find_job_dirs(artifacts_dir)
    return build_runs_dataframe(jobs)


def infer_environment(env_name: str) -> str | None:
    return ENV_NAME_MAP.get(str(env_name))


def infer_observation(low_dim: object) -> str:
    low_dim_text = "" if pd.isna(low_dim) else str(low_dim)
    if "joint_states" in low_dim_text:
        return "img+proprio"
    return "img"


def infer_method(algo_name: str, rep_loss: float) -> str | None:
    if algo_name == "bc_cardpol_policy":
        if rep_loss == 0.0:
            return "cardpol_base"
        if rep_loss == 0.01:
            return "cardpol_rl"
        return None
    if algo_name == "bc_vae_policy" and rep_loss == 1.0:
        return "vae"
    return None


def prepare_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed condition columns and numeric success values."""
    runs = df.copy()
    runs["success"] = pd.to_numeric(runs[SUCCESS_COL], errors="coerce")
    runs["rep_loss"] = pd.to_numeric(runs[REP_LOSS_COL], errors="coerce")
    runs["env"] = runs[ENV_NAME_COL].map(infer_environment)
    runs["obs"] = runs[LOW_DIM_COL].map(infer_observation)
    runs["distract"] = runs[DISTRACTOR_COL].astype(bool)
    runs["method"] = [
        infer_method(algo, rep_loss)
        for algo, rep_loss in zip(runs[ALGO_NAME_COL], runs["rep_loss"])
    ]
    return runs[
        runs["env"].notna()
        & runs["method"].notna()
        & runs["success"].notna()
        & runs[TASK_ID_COL].notna()
    ]


def mean_success_over_tasks_and_seeds(group: pd.DataFrame) -> float:
    """Average success over seeds per task, then over tasks."""
    per_task = group.groupby(TASK_ID_COL, dropna=False)["success"].mean()
    return float(per_task.mean())


def build_method_env_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table indexed by env/obs/distract with method columns."""
    runs = prepare_runs(df)
    records: list[dict[str, object]] = []

    grouped = runs.groupby(["env", "obs", "distract", "method"], dropna=False)
    for (env, obs, distract, method), group in grouped:
        records.append(
            {
                "env": env,
                "obs": obs,
                "distract": distract,
                "method": method,
                "mean_success": mean_success_over_tasks_and_seeds(group),
                "num_runs": len(group),
            }
        )

    if not records:
        return pd.DataFrame()

    long_df = pd.DataFrame(records)
    table = long_df.pivot_table(
        index=["env", "obs", "distract"],
        columns="method",
        values="mean_success",
        aggfunc="first",
    )
    table = table.reindex(
        pd.MultiIndex.from_product(
            [ENV_ORDER, OBS_ORDER, DISTRACT_ORDER],
            names=["env", "obs", "distract"],
        ),
    )
    for method in METHOD_ORDER:
        if method not in table.columns:
            table[method] = pd.NA
    return table[METHOD_ORDER]


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


def _format_latex_cell(value: float | None, bold: bool = False) -> str:
    if value is None or pd.isna(value):
        return "--"
    text = f"{float(value):.3f}"
    if bold:
        return f"\\textbf{{{text}}}"
    return text


def format_method_env_latex_table(table: pd.DataFrame) -> str:
    """Render the method/environment summary as a LaTeX table."""
    method_columns = [col for col in METHOD_ORDER if col in table.columns]
    column_spec = "lll" + "c" * len(method_columns)
    header = " & ".join(
        [
            "Env",
            "Obs",
            "Distract",
            *[METHOD_LABELS[col] for col in method_columns],
        ]
    )

    body_lines: list[str] = []
    for (env, obs, distract), row in table.iterrows():
        rounded = row.round(3)
        row_max = rounded.max(skipna=True)
        cells = [
            _format_latex_cell(value, pd.notna(value) and rounded.loc[col] == row_max)
            for col, value in rounded.items()
        ]
        body_lines.append(
            " & ".join(
                [
                    ENV_LABELS[str(env)],
                    OBS_LABELS[str(obs)],
                    DISTRACT_LABELS[bool(distract)],
                    *cells,
                ]
            )
            + " \\\\"
        )

    body = "\n".join(body_lines)
    caption = (
        "Mean rollout success by environment setting (rows) and method (columns). "
        "Each cell averages over tasks and seeds."
    )
    return f"""\\begin{{table}}
\\small
\\caption{{{caption}}}
\\label{{tab:method_env_success}}
\\begin{{tabular}}{{{column_spec}}}
\\toprule
{header} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def table_to_csv(table: pd.DataFrame) -> pd.DataFrame:
    """Flatten the MultiIndex rows for CSV export."""
    flat = table.copy()
    flat.index = [
        f"{ENV_LABELS[str(env)]}|{OBS_LABELS[str(obs)]}|{DISTRACT_LABELS[bool(distract)]}"
        for env, obs, distract in flat.index
    ]
    flat.columns = [METHOD_LABELS[col] for col in flat.columns]
    flat.index.name = "env|obs|distract"
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a table of mean rollout success by method and flattened "
            "environment/observation/distractor setting."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts",
        help="Root artifacts directory to read from.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional pre-exported all_runs.csv instead of reading artifacts/.",
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=DEFAULT_LATEX_OUTPUT,
        help=f"Path to write LaTeX output (default: {DEFAULT_LATEX_OUTPUT.name}).",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help=f"Path to write CSV output (default: {DEFAULT_CSV_OUTPUT.name}).",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print LaTeX to stdout.",
    )
    parser.add_argument(
        "--no-latex-file",
        action="store_true",
        help="Do not write the LaTeX file.",
    )
    parser.add_argument(
        "--no-csv-file",
        action="store_true",
        help="Do not write the CSV file.",
    )
    args = parser.parse_args()

    input_csv = args.input_csv
    if input_csv is not None and not input_csv.is_absolute():
        input_csv = REPO_ROOT / input_csv

    artifacts_dir = args.artifacts_dir
    if not artifacts_dir.is_absolute():
        artifacts_dir = REPO_ROOT / artifacts_dir

    df = load_runs_dataframe(artifacts_dir=artifacts_dir, input_csv=input_csv)
    if df.empty:
        print("No runs found to summarize.", file=sys.stderr)
        sys.exit(1)

    table = build_method_env_table(df)
    if table.empty:
        print("No matching runs found for method/environment table.", file=sys.stderr)
        sys.exit(1)

    header = (
        f"% Generated by scripts/parse_method_env_table.py on "
        f"{datetime.now(timezone.utc).astimezone().isoformat()}\n"
    )
    latex = header + format_method_env_latex_table(table) + "\n"

    print(f"Built table with {len(table)} row(s) and {len(table.columns)} method column(s)", file=sys.stderr)

    if not args.no_csv_file:
        csv_path = args.csv_output
        if not csv_path.is_absolute():
            csv_path = REPO_ROOT / csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        table_to_csv(table).to_csv(csv_path)
        print(f"Wrote {csv_path}", file=sys.stderr)

    if args.print:
        print(latex, end="")
    elif not args.no_latex_file:
        latex_path = args.latex_output
        if not latex_path.is_absolute():
            latex_path = REPO_ROOT / latex_path
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        latex_path.write_text(latex, encoding="utf-8")
        print(f"Wrote {latex_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
