#!/usr/bin/env python
"""Parse artifact experiment results and write LaTeX tables.

By default, writes a text file containing:
1. A cross-experiment summary table (2 rows x num experiments) comparing
   baseline ``rep_loss_scale=0`` vs ``rep_loss_scale=0.01`` mean task success.
2. Per-experiment task breakdown tables for every artifact directory.

Usage:
    python scripts/parse_experiment_summary.py

    python scripts/parse_experiment_summary.py --latex-output out.tex
    python scripts/parse_experiment_summary.py --summary-only --print
    python scripts/parse_experiment_summary.py --csv-output summary.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parse_experiment_results import (
    METRICS_FILENAME,
    REPO_ROOT,
    build_cross_experiment_summary,
    format_success_latex_table,
    parse_experiment_dir,
    success_pivot_by_task_and_rep_loss_scale,
)

DEFAULT_LATEX_OUTPUT = SCRIPT_DIR / "experiment_results_latex.txt"

TASK_SHORT_NAMES = {
    "distract-metrics": "DM",
    "object": "Obj",
    "goal-rep-sweep": "GS",
    "object-rep-sweep": "OS",
    "spatial-rep-sweep": "SS",
    "goal": "Goal",
    "spatial": "Spat",
    "spatial-long": "SpL",
    "libero10": "L10",
}
OBS_SHORT_NAMES = {
    "img": "I",
    "img+proprio": "I+P",
}


def shorten_experiment_title(name: str) -> str:
    """Map an artifact directory name to a compact LaTeX column label."""
    body = re.sub(r"^\d{8}-", "", name)
    distract = body.endswith("_cam-agent_distract")
    body = re.sub(r"_cam-agent(_distract)?$", "", body)

    if "_" in body:
        task, obs = body.rsplit("_", 1)
    else:
        task, obs = body, "img"

    task_short = TASK_SHORT_NAMES.get(task, task.replace("-", "")[:4])
    obs_short = OBS_SHORT_NAMES.get(obs, obs.replace("+", "")[:3])
    suffix = "+D" if distract else ""
    return f"{task_short}/{obs_short}{suffix}"


def _find_experiment_dirs(artifacts_dir: Path) -> list[Path]:
    experiment_dirs: list[Path] = []
    for path in sorted(artifacts_dir.iterdir()):
        if not path.is_dir():
            continue
        has_metrics = any(
            (job_dir / METRICS_FILENAME).is_file()
            for job_dir in path.iterdir()
            if job_dir.is_dir()
        )
        if has_metrics:
            experiment_dirs.append(path)
    return experiment_dirs


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


def _format_latex_cell(value: float, bold: bool) -> str:
    text = f"{value:.3f}"
    if bold:
        return f"\\textbf{{{text}}}"
    return text


def format_summary_latex_table(
    summary: pd.DataFrame,
    *,
    column_mapping: dict[str, str] | None = None,
) -> str:
    """Render the cross-experiment summary as a LaTeX table."""
    display_summary = summary
    if column_mapping is not None:
        display_summary = summary.rename(columns=column_mapping)

    column_spec = "l" + "c" * len(display_summary.columns)
    header = " & ".join(
        [
            "Cond.",
            *[
                f"\\texttt{{{_escape_latex_text(str(col))}}}"
                for col in display_summary.columns
            ],
        ]
    )

    body_lines: list[str] = []
    for condition, row in display_summary.iterrows():
        rounded = row.round(3)
        row_max = rounded.max(skipna=True)
        cells = [
            _format_latex_cell(value, rounded.loc[col] == row_max)
            for col, value in rounded.items()
        ]
        condition_label = str(condition)
        if condition_label.startswith("Baseline"):
            condition_label = "Base"
        elif condition_label.startswith("rep"):
            condition_label = "RL=0.01"
        body_lines.append(f"{condition_label} & " + " & ".join(cells) + " \\\\")

    body = "\n".join(body_lines)
    caption = (
        "Mean rollout success averaged over tasks and seeds. "
        "Column abbreviations: task/observation (+D = distract setting)."
    )
    return f"""\\begin{{table}}
\\small
\\caption{{{caption}}}
\\label{{tab:experiment_summary_rep_loss}}
\\begin{{tabular}}{{{column_spec}}}
\\toprule
{header} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def build_latex_report(
    artifacts_dir: Path,
    *,
    include_per_experiment: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Build the full LaTeX report and return the summary table plus text."""
    summary = build_cross_experiment_summary(artifacts_dir)
    if summary.empty:
        return summary, ""

    column_mapping = {col: shorten_experiment_title(str(col)) for col in summary.columns}

    sections: list[str] = [
        f"% Generated by scripts/parse_experiment_summary.py on "
        f"{datetime.now(timezone.utc).astimezone().isoformat()}",
        f"% Repository: {REPO_ROOT}",
        "",
        "% =============================================================================",
        "% Cross-experiment summary: baseline vs rep_loss_scale=0.01",
        "% Column key (short -> full experiment directory):",
    ]
    for full_name, short_name in column_mapping.items():
        sections.append(f"%   {short_name}: {full_name}")
    sections.extend(
        [
            "",
            format_summary_latex_table(summary, column_mapping=column_mapping),
            "",
        ]
    )

    if include_per_experiment:
        for experiment_dir in _find_experiment_dirs(artifacts_dir):
            df = parse_experiment_dir(experiment_dir)
            if df.empty:
                continue

            sections.extend(
                [
                    "% =============================================================================",
                    f"% Experiment: {experiment_dir.name}",
                    f"% Path: artifacts/{experiment_dir.name}",
                    "",
                ]
            )
            pivot = success_pivot_by_task_and_rep_loss_scale(df)
            sections.append(
                format_success_latex_table(pivot, experiment_name=experiment_dir.name)
            )
            sections.append("")

    return summary, "\n".join(sections).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse artifact experiment results and write LaTeX tables comparing "
            "baseline vs rep_loss_scale=0.01 across all experiment directories."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts",
        help="Root artifacts directory containing experiment group folders.",
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=DEFAULT_LATEX_OUTPUT,
        help=(
            "Path to write the LaTeX report. "
            f"Default: {DEFAULT_LATEX_OUTPUT.relative_to(REPO_ROOT)}"
        ),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional path to write the summary table as CSV.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only include the cross-experiment summary table in the LaTeX report.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the LaTeX report to stdout instead of writing a file.",
    )
    parser.add_argument(
        "--no-latex-file",
        action="store_true",
        help="Do not write the LaTeX report to disk.",
    )
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir
    if not artifacts_dir.is_absolute():
        artifacts_dir = REPO_ROOT / artifacts_dir

    summary, latex_report = build_latex_report(
        artifacts_dir,
        include_per_experiment=not args.summary_only,
    )
    if summary.empty:
        print(
            f"No experiment directories with {METRICS_FILENAME} found under {artifacts_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Built summary for {len(summary.columns)} experiment(s) under {artifacts_dir}",
        file=sys.stderr,
    )

    if args.csv_output is not None:
        csv_path = args.csv_output
        if not csv_path.is_absolute():
            csv_path = REPO_ROOT / csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(csv_path)
        print(f"Wrote {csv_path}", file=sys.stderr)

    if args.print:
        print(latex_report, end="")
    elif not args.no_latex_file:
        latex_path = args.latex_output
        if not latex_path.is_absolute():
            latex_path = REPO_ROOT / latex_path
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        latex_path.write_text(latex_report, encoding="utf-8")
        print(f"Wrote {latex_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
