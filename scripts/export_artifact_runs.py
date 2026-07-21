#!/usr/bin/env python
"""Export metrics and configs from artifact runs into a downloadable archive.

Reads job directories under ``artifacts/`` and writes copies of each run's
metrics and config files to a new zip archive. The artifacts directory is
never modified.

By default, exports:
- ``metrics_summary.csv``
- ``metrics.csv`` (if present)
- ``config.yaml`` (if present)
- ``hydra_config.yaml`` (if present)
- ``hydra_overrides.yaml`` (if present)
- ``best_epoch.txt`` (if present)

Also writes a consolidated ``all_runs.csv`` with one row per job.

Usage:
    python scripts/export_artifact_runs.py

    python scripts/export_artifact_runs.py --output exports/my_runs.zip
    python scripts/export_artifact_runs.py --output-dir exports/my_runs
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parse_experiment_results import (
    CONFIG_FILENAME,
    METRICS_FILENAME,
    REPO_ROOT,
    parse_job_dir,
)

DEFAULT_EXPORT_FILES = (
    METRICS_FILENAME,
    "metrics.csv",
    CONFIG_FILENAME,
    "hydra_config.yaml",
    "hydra_overrides.yaml",
    "best_epoch.txt",
)
DEFAULT_OUTPUT = REPO_ROOT / "exports" / "artifact_runs_export.zip"


def _find_job_dirs(artifacts_dir: Path) -> list[tuple[str, Path]]:
    """Return (experiment_group, job_dir) pairs that contain metrics summaries."""
    jobs: list[tuple[str, Path]] = []
    for experiment_dir in sorted(path for path in artifacts_dir.iterdir() if path.is_dir()):
        for job_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
            if (job_dir / METRICS_FILENAME).is_file():
                jobs.append((experiment_dir.name, job_dir))
    return jobs


def build_runs_dataframe(jobs: list[tuple[str, Path]]) -> pd.DataFrame:
    """Build a consolidated DataFrame with one row per job."""
    rows = []
    for experiment_group, job_dir in jobs:
        row = parse_job_dir(job_dir)
        if row is None:
            continue
        row["experiment_group"] = experiment_group
        row["experiment_group_dir"] = f"artifacts/{experiment_group}"
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    preferred = [
        "experiment_group",
        "experiment_group_dir",
        "job_name",
        "job_dir",
        "epoch",
        "rollout/success_env_avg",
    ]
    front = [col for col in preferred if col in df.columns]
    rest = sorted(col for col in df.columns if col not in front)
    return df[front + rest]


def _files_to_export(job_dir: Path, filenames: tuple[str, ...]) -> list[Path]:
    return [job_dir / name for name in filenames if (job_dir / name).is_file()]


def _write_readme(path: Path, *, artifacts_dir: Path, num_jobs: int, num_files: int) -> None:
    path.write_text(
        "\n".join(
            [
                "BC-IB artifact run export",
                "=======================",
                "",
                f"Generated: {datetime.now(timezone.utc).astimezone().isoformat()}",
                f"Source artifacts directory: {artifacts_dir.resolve()}",
                f"Jobs exported: {num_jobs}",
                f"Files copied: {num_files}",
                "",
                "Contents:",
                "- all_runs.csv: consolidated metrics + flattened configs",
                "- runs/<experiment_group>/<job_name>/: copied metrics/config files",
                "",
                "Note: this export is a read-only copy. The source artifacts directory",
                "was not modified.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def export_runs_to_directory(
    artifacts_dir: Path,
    output_dir: Path,
    *,
    export_files: tuple[str, ...] = DEFAULT_EXPORT_FILES,
) -> tuple[int, int, pd.DataFrame]:
    """Copy run files and consolidated CSV into ``output_dir``."""
    jobs = _find_job_dirs(artifacts_dir)
    df = build_runs_dataframe(jobs)

    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    runs_root = output_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    copied_files = 0
    for experiment_group, job_dir in jobs:
        dest_dir = runs_root / experiment_group / job_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src_path in _files_to_export(job_dir, export_files):
            shutil.copy2(src_path, dest_dir / src_path.name)
            copied_files += 1

    if not df.empty:
        df.to_csv(output_dir / "all_runs.csv", index=False)

    _write_readme(
        output_dir / "README.txt",
        artifacts_dir=artifacts_dir,
        num_jobs=len(jobs),
        num_files=copied_files,
    )
    return len(jobs), copied_files, df


def export_runs_to_zip(
    artifacts_dir: Path,
    output_zip: Path,
    *,
    export_files: tuple[str, ...] = DEFAULT_EXPORT_FILES,
) -> tuple[int, int, pd.DataFrame]:
    """Copy run files and consolidated CSV into a zip archive."""
    jobs = _find_job_dirs(artifacts_dir)
    df = build_runs_dataframe(jobs)

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        raise FileExistsError(f"output file already exists: {output_zip}")

    copied_files = 0
    with zipfile.ZipFile(output_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        readme_lines = [
            "BC-IB artifact run export",
            "=======================",
            "",
            f"Generated: {datetime.now(timezone.utc).astimezone().isoformat()}",
            f"Source artifacts directory: {artifacts_dir.resolve()}",
            f"Jobs exported: {len(jobs)}",
            "",
            "Contents:",
            "- all_runs.csv: consolidated metrics + flattened configs",
            "- runs/<experiment_group>/<job_name>/: copied metrics/config files",
            "",
            "Note: this export is a read-only copy. The source artifacts directory",
            "was not modified.",
            "",
        ]
        zf.writestr("README.txt", "\n".join(readme_lines))

        if not df.empty:
            zf.writestr("all_runs.csv", df.to_csv(index=False))

        for experiment_group, job_dir in jobs:
            for src_path in _files_to_export(job_dir, export_files):
                archive_path = (
                    f"runs/{experiment_group}/{job_dir.name}/{src_path.name}"
                )
                zf.write(src_path, arcname=archive_path)
                copied_files += 1

    return len(jobs), copied_files, df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export metrics and config files from artifact runs into a new "
            "downloadable archive without modifying artifacts/."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts",
        help="Root artifacts directory to read from.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Zip file to create (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write an export directory instead of a zip file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file or directory.",
    )
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir
    if not artifacts_dir.is_absolute():
        artifacts_dir = REPO_ROOT / artifacts_dir

    if not artifacts_dir.is_dir():
        print(f"Artifacts directory not found: {artifacts_dir}", file=sys.stderr)
        sys.exit(1)

    if args.output_dir is not None:
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
        if output_dir.exists():
            if not args.force:
                print(
                    f"Output directory already exists: {output_dir}. "
                    "Use --force to overwrite.",
                    file=sys.stderr,
                )
                sys.exit(1)
            shutil.rmtree(output_dir)
        num_jobs, num_files, df = export_runs_to_directory(artifacts_dir, output_dir)
        destination = output_dir
    else:
        output_zip = args.output
        if not output_zip.is_absolute():
            output_zip = REPO_ROOT / output_zip
        if output_zip.exists():
            if not args.force:
                print(
                    f"Output file already exists: {output_zip}. "
                    "Use --force to overwrite.",
                    file=sys.stderr,
                )
                sys.exit(1)
            output_zip.unlink()
        num_jobs, num_files, df = export_runs_to_zip(artifacts_dir, output_zip)
        destination = output_zip

    if num_jobs == 0:
        print(
            f"No jobs with {METRICS_FILENAME} found under {artifacts_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Exported {num_jobs} job(s)", file=sys.stderr)
    print(f"Copied {num_files} file(s)", file=sys.stderr)
    if not df.empty:
        print(f"Consolidated CSV rows: {len(df)}", file=sys.stderr)
        print(f"Consolidated CSV columns: {len(df.columns)}", file=sys.stderr)
    print(f"Wrote {destination}", file=sys.stderr)


if __name__ == "__main__":
    main()
