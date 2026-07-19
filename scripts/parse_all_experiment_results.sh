#!/bin/bash
# Wrapper around scripts/parse_experiment_summary.py for backwards compatibility.
#
# Usage:
#   bash scripts/parse_all_experiment_results.sh
#   bash scripts/parse_all_experiment_results.sh path/to/output.tex

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_FILE="${1:-${SCRIPT_DIR}/experiment_results_latex.txt}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${REPO_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    PYTHON="python"
fi

exec "${PYTHON}" "${SCRIPT_DIR}/parse_experiment_summary.py" --latex-output "${OUTPUT_FILE}"
