# Experiment Results Parsing Guide

Guide for agents (and humans) parsing LIBERO artifact runs into summary tables and LaTeX.

**Last updated:** 2026-08-02 (generated from live `artifacts/` scan: 4,123 runs across 27 experiment groups)

---

## Quick start

Use the repo virtualenv and run from the repo root:

```bash
# Method × environment table (main paper table)
.venv/bin/python scripts/parse_method_env_table.py

# Cross-experiment CardPol summary + per-experiment LaTeX tables
.venv/bin/python scripts/parse_experiment_summary.py

# Optional: export all runs to a zip (read-only; does not modify artifacts)
.venv/bin/python scripts/export_artifact_runs.py \
  --output exports/artifact_runs_export_$(date +%Y-%m-%d).zip --force
```

Default outputs:

| Script | Default output |
|--------|----------------|
| `parse_method_env_table.py` | `scripts/method_env_table.tex`, `scripts/method_env_table.csv` |
| `parse_experiment_summary.py` | `scripts/experiment_results_latex.txt` |
| `export_artifact_runs.py` | `exports/artifact_runs_export.zip` (+ `all_runs.csv` inside) |

Bash wrapper (calls `parse_experiment_summary.py`):

```bash
bash scripts/parse_all_experiment_results.sh
```

---

## Artifacts layout

```
artifacts/                          # symlink → /datastor2/mrudolph/BC-IB-artifacts
  <experiment_group>/               # wandb.group name, e.g. 07112026-goal_img_cam-agent
    <job_name>/                     # one Hydra job / seed / task / config combo
      metrics_summary.csv           # required for parsing (final rollout metrics)
      config.yaml                   # flattened into cfg/* columns
      metrics.csv                   # optional
      hydra_config.yaml             # optional
      best_epoch.txt                # optional
```

A job is **parseable** iff it contains `metrics_summary.csv`. Scripts skip directories without it.

Key flattened config columns used by parsers:

| Column | Meaning |
|--------|---------|
| `cfg/algo/algo_name` | e.g. `bc_cardpol_policy`, `bc_vae_policy` |
| `cfg/env/env_name` | `libero_goal`, `libero_object`, `libero_spatial`, `libero_10` |
| `cfg/env/task_id` | LIBERO task index (usually 0–9) |
| `cfg/train/seed` | Random seed |
| `cfg/train/rep_loss_scale` | Representation loss weight |
| `cfg/train/n_epochs` | Training epochs |
| `cfg/data/distractor/enable` | `true` / `false` |
| `cfg/data/data_modality` | `image` or `image,proprio` |
| `rollout/success_env_avg` | Rollout success rate (primary metric) |

---

## Scripts overview

### `parse_experiment_results.py` (library + CLI)

Core parsing utilities. Other scripts import from here.

- `parse_job_dir(job_dir)` — one row per job from `metrics_summary.csv` + `config.yaml`
- `parse_experiment_dir(experiment_dir)` — all jobs in one experiment group
- `build_cross_experiment_summary(artifacts_dir)` — 2×N table: CardPol baseline (`rep_loss_scale=0`) vs `0.01`, averaged over tasks and seeds per experiment folder

Single-experiment CLI:

```bash
.venv/bin/python scripts/parse_experiment_results.py artifacts/07112026-goal_img_cam-agent
```

### `parse_experiment_summary.py`

Produces `scripts/experiment_results_latex.txt` containing:

1. **Cross-experiment summary** — one column per experiment folder; rows compare `rep_loss_scale=0` vs `0.01` (CardPol sweeps only; baseline folders with other rep_loss values show whatever scales exist).
2. **Per-experiment tables** — task × rep_loss_scale success grids for every folder under `artifacts/`.

Useful flags:

```bash
.venv/bin/python scripts/parse_experiment_summary.py --summary-only   # skip per-experiment tables
.venv/bin/python scripts/parse_experiment_summary.py --print            # stdout instead of file
.venv/bin/python scripts/parse_experiment_summary.py --csv-output scripts/summary.csv
```

### `parse_method_env_table.py` (main comparison table)

Builds the **method × environment** table used for paper-style comparisons.

**Rows (12):** `goal | object | spatial` × `img | img+proprio` × `distract no | yes`

**Columns (5 methods):**

| Column label | Selection rule |
|--------------|----------------|
| Vanilla | `bc_cardpol_policy`, `rep_loss_scale=0` |
| CardPol (RL=0.01) | `bc_cardpol_policy`, `rep_loss_scale=0.01` |
| VAE | `bc_vae_policy`, `rep_loss_scale=1.0` |
| CURL | `bc_curl_policy`, `rep_loss_scale=1.0` |
| VIP | `bc_vip_policy`, `rep_loss_scale=0.1`, experiment group must match `bc-vip-alltasks_img_cam-agent*` and **not** contain `long-train` or `repscale` |

**Aggregation per cell:**

1. Mean success over seeds for each `task_id`
2. Report mean ± **SEM across tasks** (std of per-task means / √n_tasks)

LaTeX cells look like `0.688 $\pm$ 0.088`. Best mean in each row is bolded. Missing data → `--`.

**Condition inference:**

- **Environment:** `cfg/env/env_name` → goal / object / spatial
- **Observation:** `proprio` in `cfg/data/data_modality` → `img+proprio`, else `img` (falls back to `joint_states` in `cfg/data/obs/modality/low_dim`)
- **Distraction:** `cfg/data/distractor/enable`

Parse from an exported CSV instead of live artifacts:

```bash
.venv/bin/python scripts/parse_method_env_table.py \
  --input-csv exports/artifact_runs_export_2026-07-22/all_runs.csv
```

### `export_artifact_runs.py`

Read-only export: copies per-job metrics/config files into a zip and writes consolidated `all_runs.csv`. Never modifies `artifacts/`.

```bash
.venv/bin/python scripts/export_artifact_runs.py \
  --output exports/artifact_runs_export_$(date +%Y-%m-%d).zip --force
```

---

## Typical workflow after new runs land

1. Confirm new jobs appear under the expected `artifacts/<experiment_group>/` folder with `metrics_summary.csv`.
2. Re-run both parsers (method table takes ~2–3 min on full artifacts):

   ```bash
   .venv/bin/python scripts/parse_method_env_table.py
   .venv/bin/python scripts/parse_experiment_summary.py
   ```

3. Check `scripts/method_env_table.tex` for `--` cells (missing condition coverage).
4. Optionally re-export artifacts zip if sharing results off-cluster.

---

## Experiment folder naming

Folders follow wandb group names from `launch_scripts/mll/submit_libero.py`:

```
<date>-<description>_<modality>_cam-agent[_distract]
bc-<method>-baseline_<modality>_cam-agent[_distract]
bc-vip-alltasks[_-long-train]_img_cam-agent[_distract]
```

Examples:

- `07112026-goal_img_cam-agent` — CardPol goal env, image only, no distraction
- `bc-curl-baseline_img_cam-agent_distract` — CURL baseline, all LIBERO envs, distracted
- `07222026-goal_img+proprio_cam-agent` — CardPol goal, image+proprio, no distraction

Expected run count for a **complete** sweep:

```
runs = (# envs) × (# tasks) × (# seeds) × (# rep_loss values)
```

CardPol rep sweeps with 4 rep_loss values, 10 tasks, 5 seeds, 1 env → **200 runs**.  
VAE/CURL img baseline: 3 envs × 10 tasks × 5 seeds → **150 runs** (155 if a few extra jobs exist).

---

## Artifact inventory

Total: **4,123 runs** in **27** experiment groups (snapshot 2026-08-02).

| Experiment folder | Runs | Algo | Environments | Modality | Distract | rep_loss scales | Seeds | Tasks | Epochs |
|-------------------|-----:|------|--------------|----------|----------|-----------------|-------|-------|-------|
| `07062026-distract-metrics_img+proprio_cam-agent_distract` | 200 | cardpol | goal | img+proprio | yes | 0, 0.01 | 10 (0–9) | 10 (0–9) | 50 |
| `07062026-object_img+proprio_cam-agent_distract` | 200 | cardpol | object | img+proprio | yes | 0, 0.01 | 10 (0–9) | 10 (0–9) | 50 |
| `07072026-goal-rep-sweep_img_cam-agent_distract` | 200 | cardpol | goal | img | yes | 0, 0.001, 0.005, 0.01 | 5 (0–4) | 10 (0–9) | 50 |
| `07072026-object-rep-sweep_img_cam-agent_distract` | 200 | cardpol | object | img | yes | 0, 0.001, 0.005, 0.01 | 5 (0–4) | 10 (0–9) | 50 |
| `07082026-spatial-rep-sweep_img+proprio_cam-agent_distract` | 200 | cardpol | spatial | img+proprio | yes | 0, 0.001, 0.005, 0.01 | 5 (0–4) | 10 (0–9) | 50 |
| `07082026-spatial-rep-sweep_img_cam-agent_distract` | 200 | cardpol | spatial | img | yes | 0, 0.001, 0.005, 0.01 | 5 (0–4) | 10 (0–9) | 50 |
| `07092026-goal-rep-sweep_img+proprio_cam-agent_distract` | 200 | cardpol | goal | img+proprio | yes | 0, 0.001, 0.005, 0.01 | 5 (0–4) | 10 (0–9) | 50 |
| `07092026-object-rep-sweep_img+proprio_cam-agent_distract` | 200 | cardpol | object | img+proprio | yes | 0, 0.001, 0.005, 0.01 | 5 (0–4) | 10 (0–9) | 50 |
| `07102026-libero10_img_cam-agent_distract` | 200 | cardpol | libero_10 | img | yes | 0, 0.001, 0.005, 0.01 | 5 (0–4) | 10 (0–9) | 50 |
| `07112026-goal_img_cam-agent` | 200 | cardpol | goal | img | no | 0, 0.001, 0.005, 0.01 | 5 (0–4) | 10 (0–9) | 50 |
| `07122026-object_img_cam-agent` | 200 | cardpol | object | img | no | 0, 0.001, 0.005, 0.01 | 5 (0–4) | 10 (0–9) | 50 |
| `07132026-spatial_img_cam-agent` | 200 | cardpol | spatial | img | no | 0, 0.001, 0.005, 0.01 | 5 (0–4) | 10 (0–9) | 50 |
| `07142026-spatial-long_img_cam-agent` | 200 | cardpol | spatial | img | no | 0, 0.001, 0.005, 0.01 | 5 (0–4) | 10 (0–9) | 100 |
| `07222026-goal_img+proprio_cam-agent` | 100 | cardpol | goal | img+proprio | no | 0, 0.01 | 5 (0–4) | 10 (0–9) | 50 |
| `07222026-object_img+proprio_cam-agent` | 100 | cardpol | object | img+proprio | no | 0, 0.01 | 5 (0–4) | 10 (0–9) | 50 |
| `07222026-spatial_img+proprio_cam-agent` | 100 | cardpol | spatial | img+proprio | no | 0, 0.01 | 5 (0–4) | 10 (0–9) | 50 |
| `bc-curl-baseline_img_cam-agent` | 150 | curl | goal, object, spatial | img | no | 1.0 | 5 (0–4) | 10 (0–9) | 50 |
| `bc-curl-baseline_img_cam-agent_distract` | 150 | curl | goal, object, spatial | img | yes | 1.0 | 5 (0–4) | 10 (0–9) | 50 |
| `bc-vae-baseline_img+proprio_cam-agent` | 150 | vae | goal, object, spatial | img+proprio | no | 1.0 | 5 (0–4) | 10 (0–9) | 50 |
| `bc-vae-baseline_img+proprio_cam-agent_distract` | 155 | vae | goal, object, spatial | img+proprio | yes | 1.0 | 5 (0–4) | 10 (0–9) | 50 |
| `bc-vae-baseline_img_cam-agent` | 150 | vae | goal, object, spatial | img | no | 1.0 | 5 (0–4) | 10 (0–9) | 50 |
| `bc-vae-baseline_img_cam-agent_distract` | 150 | vae | goal, object, spatial | img | yes | 1.0 | 5 (0–4) | 10 (0–9) | 50 |
| `bc-vip-alltasks-long-train_img_cam-agent` | 50 | vip | goal | img | no | 0.1 | 5 (0–4) | 10 (0–9) | 100 |
| `bc-vip-alltasks-long-train_img_cam-agent_distract` | 48 | vip | goal | img | yes | 0.1 | 5 (0–4) | 10 (0–9) | 100 |
| `bc-vip-alltasks_img_cam-agent` | 100 | vip | goal, object | img | no | 0.1 | 5 (0–4) | 10 (0–9) | 50 |
| `bc-vip-alltasks_img_cam-agent_distract` | 100 | vip | goal, object | img | yes | 0.1 | 5 (0–4) | 10 (0–9) | 50 |
| `bc-vip-repscale_img_cam-agent` | 20 | vip | goal | img | no | 0.001, 0.01, 0.1, 0.5 | 5 (0–4) | 1 (0) | 50 |

**Notes on the inventory:**

- **cardpol** = `bc_cardpol_policy`; **vae** / **curl** / **vip** = `bc_vae_policy` / `bc_curl_policy` / `bc_vip_policy`
- Run counts slightly below the formula (e.g. 155 vs 150, 48 vs 50) mean some jobs are still missing or failed
- `parse_method_env_table.py` pulls CardPol from **all** matching folders; VAE/CURL from any folder with the right algo + rep_loss; VIP only from `bc-vip-alltasks_img_cam-agent*`
- VIP `long-train` and `repscale` folders are listed here but **excluded** from the method table

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `--` in method table cell | No runs match that env/obs/distract/method combo | Check artifact inventory; confirm algo, rep_loss, and experiment group filters |
| VIP column empty / 0.000 | Wrong `rep_loss_scale` filter (VIP uses **0.1**, not 1.0) | See `VIP_REP_LOSS` in `parse_method_env_table.py` |
| `TypeError` on `.round()` | Mixed dtypes in pivot (all-NA method column) | Script coerces with `pd.to_numeric`; re-pull latest `parse_method_env_table.py` |
| Parser very slow (~2–3 min) | Reads every job's YAML under `artifacts/` | Normal; use `--input-csv` from a prior export for faster iteration |
| New method not appearing | `infer_method()` doesn't map algo + rep_loss | Add mapping in `parse_method_env_table.py` and regenerate |

---

## Regenerating the inventory table

To refresh the artifact inventory section after new runs:

```bash
.venv/bin/python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from export_artifact_runs import _find_job_dirs, build_runs_dataframe

df = build_runs_dataframe(_find_job_dirs(Path("artifacts")))
print(f"Total runs: {len(df)}, groups: {df['experiment_group'].nunique()}")
for group, gdf in sorted(df.groupby("experiment_group"), key=lambda x: x[0]):
    def fmt(col, fn=str):
        if col not in gdf.columns: return "?"
        vals = sorted(set(gdf[col].dropna()), key=lambda x: (str(type(x)), x))
        return ", ".join(fn(v) for v in vals)
    envs = fmt("cfg/env/env_name", lambda x: str(x).replace("libero_", ""))
    mod = "img+proprio" if any("proprio" in str(x) for x in gdf.get("cfg/data/data_modality", [])) else "img"
    distract = fmt("cfg/data/distractor/enable", lambda x: "yes" if x in (True, "true", "True", 1) else "no")
    seeds = gdf["cfg/train/seed"].dropna().astype(int)
    tasks = gdf["cfg/env/task_id"].dropna().astype(int)
    print(f"| `{group}` | {len(gdf)} | ... |")
PY
```

Paste updated rows into the inventory table above.

---

## Files in `scripts/` (parsing-related)

| File | Role |
|------|------|
| `parse_experiment_results.py` | Core job/experiment parsing; cross-experiment CardPol summary |
| `parse_experiment_summary.py` | LaTeX report: summary + per-experiment tables |
| `parse_method_env_table.py` | Method × env table with mean ± SEM |
| `parse_all_experiment_results.sh` | Thin wrapper → `parse_experiment_summary.py` |
| `export_artifact_runs.py` | Zip export + `all_runs.csv` |
| `method_env_table.tex` / `.csv` | Generated method comparison outputs |
| `experiment_results_latex.txt` | Generated LaTeX report |
| `PARSING_GUIDE.md` | This file |
