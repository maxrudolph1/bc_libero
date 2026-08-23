#!/usr/bin/env python
"""Parse the final img-cam-agent baseline folders into figures/final_results.

This is a scoped parse of the representation-baseline experiment groups
(CardPol / VIP / VAE / VQVAE / CURL / ICVF, with and without distractors).
It does not scan the rest of artifacts/.
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

from export_artifact_runs import build_runs_dataframe
from parse_experiment_results import (
    HISTORY_METRICS_FILENAME,
    METRICS_FILENAME,
    REPO_ROOT,
    SUCCESS_COL,
    TASK_ID_COL,
    _escape_latex_text,
    format_success_latex_table,
)
from parse_method_env_table import (
    ALGO_NAME_COL,
    DATA_MODALITY_COL,
    DISTRACTOR_COL,
    ENV_LABELS,
    ENV_NAME_COL,
    ENV_NAME_MAP,
    ENV_ORDER,
    EXPERIMENT_GROUP_COL,
    LOW_DIM_COL,
    OBS_LABELS,
    REP_LOSS_COL,
    infer_observation,
    success_stats_over_tasks_and_seeds,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures" / "final_results"
VAE_TYPE_COL = "cfg/train/vae_type"
SEED_COL = "cfg/train/seed"

# User-facing names -> actual artifacts/ directory names.
EXPERIMENT_GROUPS = [
    "bc-cardpol-baseline_img_cam-agent",
    "bc-cardpol-baseline_img_cam-agent_distract",
    "bc-vip-baseline-better-weight-rerun_img_cam-agent",
    "bc-vip-baseline-better-weight-rerun_img_cam-agent_distract",
    "bc-vae-baseline_img_cam-agent",
    "bc-vae-baseline_img_cam-agent_distract",
    "bc-vqvae-baseline_img_cam-agent_vqvae",
    "bc-vqvae-baseline_img_cam-agent_distract_vqvae",
    "bc-curl-baseline_img_cam-agent",
    "bc-curl-baseline_img_cam-agent_distract",
    "bc-icvf-weight-rerun_img_cam-agent",
    "bc-icvf-weight-rerun_img_cam-agent_distract",
]

METHOD_ORDER = ["cardpol", "vip", "vae", "vqvae", "curl", "icvf"]
METHOD_LABELS = {
    "cardpol": "CardPol",
    "vip": "VIP",
    "vae": "VAE",
    "vqvae": "VQVAE",
    "curl": "CURL",
    "icvf": "ICVF",
}
METHOD_LATEX_LABELS = {
    "cardpol": r"\method",
    "vip": "VIP",
    "vae": "VAE",
    "vqvae": "VQ-VAE",
    "curl": "CURL",
    "icvf": "ICVF",
}

DISTRACT_ORDER = [False, True]
DISTRACT_LABELS = {False: "no", True: "yes"}
DISTRACT_LATEX_LABELS = {False: "Clean", True: "Distract"}

HISTORY_SUCCESS_COL = "metrics_csv/rollout/success_env_avg"

SLIM_COLUMNS = [
    "experiment_group",
    "method",
    "env",
    "obs",
    "distract",
    TASK_ID_COL,
    SEED_COL,
    REP_LOSS_COL,
    VAE_TYPE_COL,
    SUCCESS_COL,
    HISTORY_SUCCESS_COL,
    "job_name",
]

METRIC_SOURCES = {
    "metrics_summary": {
        "success_col": SUCCESS_COL,
        "stem": "metrics_summary",
        "label": "tab:final_results_metrics_summary",
        "source_text": (
            "Scores are wandb \\texttt{run.summary} values from "
            "\\texttt{metrics\\_summary.csv}."
        ),
    },
    "metrics": {
        "success_col": HISTORY_SUCCESS_COL,
        "stem": "metrics",
        "label": "tab:final_results_metrics",
        "source_text": (
            "Scores are the last logged \\texttt{rollout/success\\_env\\_avg} "
            "from \\texttt{metrics.csv}."
        ),
    },
}


def _find_jobs(artifacts_dir: Path, groups: list[str]) -> list[tuple[str, Path]]:
    jobs: list[tuple[str, Path]] = []
    missing: list[str] = []
    empty: list[str] = []
    for group in groups:
        experiment_dir = artifacts_dir / group
        if not experiment_dir.is_dir():
            missing.append(group)
            continue
        n_before = len(jobs)
        for job_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
            if (job_dir / METRICS_FILENAME).is_file() or (
                job_dir / HISTORY_METRICS_FILENAME
            ).is_file():
                jobs.append((group, job_dir))
        if len(jobs) == n_before:
            empty.append(group)
    if missing:
        print(
            "warning: missing experiment folders (skipped): " + ", ".join(missing),
            file=sys.stderr,
        )
    if empty:
        print(
            "warning: folders with no metrics_summary.csv or metrics.csv yet (skipped): "
            + ", ".join(empty),
            file=sys.stderr,
        )
    return jobs


def infer_environment(env_name: object) -> str | None:
    text = str(env_name)
    for key, label in ENV_NAME_MAP.items():
        if key in text:
            return label
    return None


def infer_method(row: pd.Series) -> str | None:
    algo = str(row.get(ALGO_NAME_COL, ""))
    group = str(row.get(EXPERIMENT_GROUP_COL, ""))
    vae_type = str(row.get(VAE_TYPE_COL, "")).lower()
    if algo == "bc_cardpol_policy":
        return "cardpol"
    if algo == "bc_vip_policy":
        return "vip"
    if algo == "bc_curl_policy":
        return "curl"
    if algo == "bc_icvf_policy":
        return "icvf"
    if algo == "bc_vae_policy":
        if "vqvae" in group or vae_type == "vqvae":
            return "vqvae"
        return "vae"
    return None


def prepare_runs(df: pd.DataFrame, *, success_col: str = SUCCESS_COL) -> pd.DataFrame:
    runs = df.copy()
    if success_col not in runs.columns:
        runs[success_col] = pd.NA
    runs["success"] = pd.to_numeric(runs[success_col], errors="coerce")
    runs["rep_loss"] = pd.to_numeric(runs[REP_LOSS_COL], errors="coerce")
    runs["env"] = runs[ENV_NAME_COL].map(infer_environment)
    if DATA_MODALITY_COL in runs.columns:
        runs["obs"] = [
            infer_observation(low_dim, data_modality)
            for low_dim, data_modality in zip(
                runs[LOW_DIM_COL],
                runs[DATA_MODALITY_COL],
            )
        ]
    else:
        runs["obs"] = runs[LOW_DIM_COL].map(infer_observation)
    runs["distract"] = runs[DISTRACTOR_COL].astype(bool)
    runs["method"] = [infer_method(row) for _, row in runs.iterrows()]
    runs = runs[
        runs["env"].notna()
        & runs["method"].notna()
        & runs["success"].notna()
        & runs[TASK_ID_COL].notna()
    ]
    return runs


def build_method_env_tables(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    grouped = runs.groupby(["env", "obs", "distract", "method"], dropna=False)
    for (env, obs, distract, method), group in grouped:
        mean, stderr = success_stats_over_tasks_and_seeds(group)
        records.append(
            {
                "env": env,
                "obs": obs,
                "distract": distract,
                "method": method,
                "mean_success": mean,
                "stderr_success": stderr,
                "num_runs": len(group),
                "num_tasks": group[TASK_ID_COL].nunique(),
                "num_seeds": group[SEED_COL].nunique() if SEED_COL in group.columns else pd.NA,
            }
        )
    long_df = pd.DataFrame(records)
    row_index = pd.MultiIndex.from_product(
        [ENV_ORDER, ["img"], DISTRACT_ORDER],
        names=["env", "obs", "distract"],
    )
    mean_table = long_df.pivot_table(
        index=["env", "obs", "distract"],
        columns="method",
        values="mean_success",
        aggfunc="first",
    ).reindex(row_index)
    stderr_table = long_df.pivot_table(
        index=["env", "obs", "distract"],
        columns="method",
        values="stderr_success",
        aggfunc="first",
    ).reindex(row_index)
    for method in METHOD_ORDER:
        if method not in mean_table.columns:
            mean_table[method] = pd.NA
        if method not in stderr_table.columns:
            stderr_table[method] = pd.NA
    mean_table = mean_table[METHOD_ORDER].apply(pd.to_numeric, errors="coerce")
    stderr_table = stderr_table[METHOD_ORDER].apply(pd.to_numeric, errors="coerce")
    return mean_table, stderr_table, long_df


def _format_latex_cell(
    mean: float | None,
    stderr: float | None = None,
    *,
    bold: bool = False,
) -> str:
    if mean is None or pd.isna(mean):
        return "---"
    mean_text = f"{float(mean):.3f}"
    if bold:
        mean_text = f"\\textbf{{{mean_text}}}"
    if stderr is None or pd.isna(stderr):
        return mean_text
    return f"{mean_text}$\\,{{\\scriptstyle\\pm{float(stderr):.3f}}}$"


def format_method_env_latex_table(
    mean_table: pd.DataFrame,
    stderr_table: pd.DataFrame,
    *,
    source_text: str = "",
    label: str = "tab:final_results",
) -> str:
    method_columns = [col for col in METHOD_ORDER if col in mean_table.columns]
    column_spec = "ll" + "c" * len(method_columns)
    header = " & ".join(
        [
            "Suite",
            "Setting",
            *[METHOD_LATEX_LABELS[col] for col in method_columns],
        ]
    )
    body_lines: list[str] = []
    prev_env: str | None = None
    for (env, _obs, distract), mean_row in mean_table.iterrows():
        env_key = str(env)
        if prev_env is not None and env_key != prev_env:
            body_lines.append(f"\\cmidrule(lr){{1-{2 + len(method_columns)}}}")
        env_cell = ENV_LABELS[env_key] if env_key != prev_env else ""
        prev_env = env_key
        stderr_row = stderr_table.loc[(env, _obs, distract)]
        rounded_means = pd.to_numeric(mean_row, errors="coerce").round(3)
        row_max = rounded_means.max(skipna=True)
        cells = [
            _format_latex_cell(
                mean_row[col],
                stderr_row[col],
                bold=pd.notna(mean_row[col]) and rounded_means.loc[col] == row_max,
            )
            for col in method_columns
        ]
        body_lines.append(
            " & ".join(
                [
                    env_cell,
                    DISTRACT_LATEX_LABELS[bool(distract)],
                    *cells,
                ]
            )
            + " \\\\"
        )
    body = "\n".join(body_lines)
    caption = (
        "LIBERO image-only success (mean $\\pm$ SEM over 10 tasks; 5 seeds per task). "
        "Best result in each row is bold. "
        r"\method{} uses $\lambda{=}0.01$."
    )
    source_comment = f"% {source_text}\n" if source_text else ""
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{source_comment}\\small
\\setlength{{\\tabcolsep}}{{4.5pt}}
\\renewcommand{{\\arraystretch}}{{1.12}}
\\begin{{tabular}}{{{column_spec}}}
\\toprule
{header} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def table_to_csv(mean_table: pd.DataFrame, stderr_table: pd.DataFrame) -> pd.DataFrame:
    flat = mean_table.copy()
    flat.index = [
        f"{ENV_LABELS[str(env)]}|{OBS_LABELS[str(obs)]}|{DISTRACT_LABELS[bool(distract)]}"
        for env, obs, distract in flat.index
    ]
    flat.columns = [METHOD_LABELS[col] for col in flat.columns]
    flat.index.name = "env|obs|distract"
    stderr_flat = stderr_table.copy()
    stderr_flat.index = flat.index
    stderr_flat.columns = [f"{METHOD_LABELS[col]}_stderr" for col in stderr_flat.columns]
    return flat.join(stderr_flat)


def env_task_pivot(group: pd.DataFrame) -> pd.DataFrame:
    summary = (
        group.groupby(["env", TASK_ID_COL], dropna=False)["success"]
        .mean()
        .reset_index()
    )
    pivot = summary.pivot(index=TASK_ID_COL, columns="env", values="success").sort_index()
    pivot = pivot.reindex(columns=[env for env in ENV_ORDER if env in pivot.columns])
    pivot.columns = [ENV_LABELS[str(col)] for col in pivot.columns]
    pivot.index.name = "task_id"
    return pivot


def coverage_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, gdf in runs.groupby(EXPERIMENT_GROUP_COL, sort=True):
        envs = sorted(gdf["env"].dropna().unique(), key=lambda x: ENV_ORDER.index(x) if x in ENV_ORDER else 99)
        seeds = pd.to_numeric(gdf[SEED_COL], errors="coerce").dropna().astype(int)
        tasks = pd.to_numeric(gdf[TASK_ID_COL], errors="coerce").dropna().astype(int)
        rep_loss = pd.to_numeric(gdf[REP_LOSS_COL], errors="coerce").dropna()
        rows.append(
            {
                "experiment_group": group_name,
                "method": ",".join(sorted(gdf["method"].dropna().unique())),
                "runs": len(gdf),
                "envs": ",".join(envs),
                "distract": "yes" if bool(gdf["distract"].iloc[0]) else "no",
                "obs": ",".join(sorted(gdf["obs"].dropna().unique())),
                "rep_loss": ",".join(f"{v:g}" for v in sorted(rep_loss.unique())),
                "seeds": f"{seeds.min()}-{seeds.max()} ({seeds.nunique()})" if len(seeds) else "?",
                "tasks": f"{tasks.min()}-{tasks.max()} ({tasks.nunique()})" if len(tasks) else "?",
            }
        )
    return pd.DataFrame(rows)


def slim_runs(runs: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in SLIM_COLUMNS if col in runs.columns]
    extra = ["env", "obs", "distract", "method"]
    ordered = []
    for col in extra + cols:
        if col in runs.columns and col not in ordered:
            ordered.append(col)
    return runs[ordered].sort_values(
        ["experiment_group", "env", TASK_ID_COL, SEED_COL]
    )


def write_source_outputs(
    *,
    output_dir: Path,
    source_name: str,
    source_cfg: dict[str, str],
    df: pd.DataFrame,
    generated: str,
) -> pd.DataFrame:
    runs = prepare_runs(df, success_col=source_cfg["success_col"])
    print(
        f"Prepared {len(runs)} parseable runs from {source_name}",
        file=sys.stderr,
    )
    if runs.empty:
        print(f"No runs found for {source_name}.", file=sys.stderr)
        return runs

    mean_table, stderr_table, long_df = build_method_env_tables(runs)
    latex_table = format_method_env_latex_table(
        mean_table,
        stderr_table,
        source_text=source_cfg["source_text"],
        label=source_cfg["label"],
    )
    stem = source_cfg["stem"]
    header = generated + f"% Metric source: {source_name}\n"
    latex = header + latex_table + "\n"

    sections = [
        header.rstrip(),
        "",
        "% =============================================================================",
        f"% Method x environment table ({source_name})",
        "",
        latex_table,
        "",
    ]
    for group in EXPERIMENT_GROUPS:
        group_runs = runs[runs[EXPERIMENT_GROUP_COL] == group]
        if group_runs.empty:
            continue
        pivot = env_task_pivot(group_runs)
        sections.extend(
            [
                "% =============================================================================",
                f"% Experiment: {group}",
                f"% Path: artifacts/{group}",
                f"% Method: {', '.join(sorted(group_runs['method'].unique()))}",
                f"% Runs: {len(group_runs)}",
                f"% Metric source: {source_name}",
                "",
                format_success_latex_table(
                    pivot,
                    experiment_name=f"{group}_{stem}",
                    caption=(
                        f"\\texttt{{{_escape_latex_text(group)}}}. "
                        "Average rollout success by task and environment "
                        f"(mean over seeds; {source_name})."
                    ),
                ),
                "",
            ]
        )

    csv_path = output_dir / f"method_env_table_{stem}.csv"
    tex_path = output_dir / f"method_env_table_{stem}.tex"
    report_path = output_dir / f"experiment_results_latex_{stem}.txt"
    long_path = output_dir / f"method_env_long_{stem}.csv"
    runs_path = output_dir / f"runs_{stem}.csv"

    table_to_csv(mean_table, stderr_table).to_csv(csv_path)
    tex_path.write_text(latex, encoding="utf-8")
    report_path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    long_df.to_csv(long_path, index=False)
    slim_runs(runs).to_csv(runs_path, index=False)

    print(f"Wrote {tex_path}", file=sys.stderr)
    print(f"Wrote {csv_path}", file=sys.stderr)
    print(f"Wrote {report_path}", file=sys.stderr)
    print(f"Wrote {long_path}", file=sys.stderr)
    print(f"Wrote {runs_path}", file=sys.stderr)

    if source_name == "metrics_summary":
        # Keep the original un-suffixed names as the summary table.
        (output_dir / "method_env_table.csv").write_text(
            csv_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (output_dir / "method_env_table.tex").write_text(latex, encoding="utf-8")
        (output_dir / "experiment_results_latex.txt").write_text(
            report_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        long_df.to_csv(output_dir / "method_env_long.csv", index=False)
        slim_runs(runs).to_csv(output_dir / "runs.csv", index=False)

    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse final baseline folders into figures/final_results.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts",
        help="Root artifacts directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)}).",
    )
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir
    if not artifacts_dir.is_absolute():
        artifacts_dir = REPO_ROOT / artifacts_dir
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = _find_jobs(artifacts_dir, EXPERIMENT_GROUPS)
    print(f"Found {len(jobs)} jobs across {len(EXPERIMENT_GROUPS)} groups", file=sys.stderr)
    df = build_runs_dataframe(jobs)
    if df.empty:
        print("No runs found to summarize.", file=sys.stderr)
        sys.exit(1)

    generated = (
        f"% Generated by scripts/parse_final_results.py on "
        f"{datetime.now(timezone.utc).astimezone().isoformat()}\n"
        f"% Groups: {', '.join(EXPERIMENT_GROUPS)}\n"
    )

    source_runs: dict[str, pd.DataFrame] = {}
    for source_name, source_cfg in METRIC_SOURCES.items():
        source_runs[source_name] = write_source_outputs(
            output_dir=output_dir,
            source_name=source_name,
            source_cfg=source_cfg,
            df=df,
            generated=generated,
        )

    coverage_source = source_runs.get("metrics_summary")
    if coverage_source is None or coverage_source.empty:
        coverage_source = next(
            (runs for runs in source_runs.values() if not runs.empty),
            pd.DataFrame(),
        )
    if not coverage_source.empty:
        coverage = coverage_table(coverage_source)
        coverage_path = output_dir / "coverage.csv"
        coverage.to_csv(coverage_path, index=False)
        print(f"Wrote {coverage_path}", file=sys.stderr)
        print(coverage.to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
