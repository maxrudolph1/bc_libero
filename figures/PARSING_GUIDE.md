# Final baseline table parsing

How the paper comparison table in `figures/final_results/` is built from `artifacts/`.

The older, repo-wide parsers (all experiment groups, Vanilla vs CardPol sweeps, img+proprio, etc.) are documented in [`scripts/PARSING_GUIDE.md`](../scripts/PARSING_GUIDE.md). **This file is only for the scoped baseline table.**

**Last updated:** 2026-08-24

---

## Quick start

From the repo root, with the project venv:

```bash
.venv/bin/python scripts/parse_final_results.py
```

Writes (or overwrites) everything under `figures/final_results/`. Takes about a minute. Missing or still-running folders are skipped with a warning; `--` cells mean that condition has no parseable jobs yet.

Optional flags:

```bash
.venv/bin/python scripts/parse_final_results.py \
  --artifacts-dir artifacts \
  --output-dir figures/final_results
```

---

## Source folders

Every number in `figures/final_results/` comes from **exactly these 12 wandb groups** under `artifacts/`. The list is `EXPERIMENT_GROUPS` in `scripts/parse_final_results.py`. Nothing else under `artifacts/` is read.

ICVF uses **`bc-icvf-weight-rerun2_*` for Goal, Object, and Spatial** (clean and distract). It does not mix in `bc-icvf-weight-rerun_*` or `bc-icvf-baseline_*`.

| Method | Setting | Artifact folder | Suites | Runs | `rep_loss` |
|--------|---------|-----------------|--------|-----:|------------|
| CardPol (`\method`) | Clean | `artifacts/bc-cardpol-baseline_img_cam-agent` | goal, object, spatial | 150 | 0.01 |
| CardPol (`\method`) | Distract | `artifacts/bc-cardpol-baseline_img_cam-agent_distract` | goal, object, spatial | 150 | 0.01 |
| VIP | Clean | `artifacts/bc-vip-baseline-better-weight-rerun_img_cam-agent` | goal, object, spatial | 139 | 0.01 |
| VIP | Distract | `artifacts/bc-vip-baseline-better-weight-rerun_img_cam-agent_distract` | goal, object, spatial | 138 | 0.01 |
| VAE | Clean | `artifacts/bc-vae-baseline_img_cam-agent` | goal, object, spatial | 150 | 1.0 |
| VAE | Distract | `artifacts/bc-vae-baseline_img_cam-agent_distract` | goal, object, spatial | 150 | 1.0 |
| VQ-VAE | Clean | `artifacts/bc-vqvae-baseline_img_cam-agent_vqvae` | goal, object, spatial | 150 | 1.0 |
| VQ-VAE | Distract | `artifacts/bc-vqvae-baseline_img_cam-agent_distract_vqvae` | goal, object, spatial | 150 | 1.0 |
| CURL | Clean | `artifacts/bc-curl-baseline_img_cam-agent` | goal, object, spatial | 150 | 1.0 |
| CURL | Distract | `artifacts/bc-curl-baseline_img_cam-agent_distract` | goal, object, spatial | 150 | 1.0 |
| ICVF | Clean | `artifacts/bc-icvf-weight-rerun2_img_cam-agent` | goal, object, spatial | 150 | 0.01 |
| ICVF | Distract | `artifacts/bc-icvf-weight-rerun2_img_cam-agent_distract` | goal, object, spatial | 150 | 0.01 |

Run counts are parseable jobs (`metrics_summary.csv` and/or `metrics.csv`). A complete folder is **3 suites × 10 tasks × 5 seeds = 150**. Live counts are in `figures/final_results/coverage.csv`. VIP is still a few jobs short.

### Not used (do not confuse with the table)

These related groups exist under `artifacts/` but are **excluded**:

| Folder | Why it is not in the table |
|--------|----------------------------|
| `bc-icvf-weight-rerun_img_cam-agent[_distract]` | First ICVF weight rerun; incomplete; replaced by `rerun2` |
| `bc-icvf-baseline_img_cam-agent[_distract]` | Earlier ICVF baseline |
| `bc-vip-alltasks_img_cam-agent[_distract]` | Older VIP (goal+object only) |
| `bc-vip-alltasks-long-train_*` | Longer train, not the paper VIP column |
| `bc-vip-best-baseline_*` | Intermediate VIP group; not used |
| `bc-vip-baseline-better-rep-loss_*` | Different VIP loss setting |
| `bc-vip-repscale_*` / `bc-vip-weight-sweep_*` | Sweeps, not the main column |
| `bc-cardpol-weight-sweep_*` | CardPol sweep, not the baseline column |
| `bc-vae-baseline_img+proprio_*` | Image+proprio; this table is image-only |
| Dated CardPol folders (`07112026-goal_*`, `07132026-spatial_*`, …) | Older CardPol sweeps |

To change sources, edit `EXPERIMENT_GROUPS` and re-run the parser.

---

## Two score sources (read this)

Wandb `run.summary` and the last logged history point can disagree. CardPol LIBERO-Spatial *clean* is the example: summary **0.526**, last `metrics.csv` value **0.596** (what the wandb chart shows). Distracted CardPol matches on both.

The parser therefore writes **two full tables**:

| Source | File on disk | Score used |
|--------|----------------|------------|
| `metrics_summary` | last row of `metrics_summary.csv` = wandb `run.summary` | `rollout/success_env_avg` |
| `metrics` | last **non-null** `rollout/success_env_avg` in `metrics.csv` | training/eval history |

`method_env_table.tex` / `.csv` (no suffix) is a copy of the **metrics_summary** table, for convenience.

Jobs are included if they have `metrics_summary.csv` **or** `metrics.csv` (plus `config.yaml` for history-only jobs).

---

## Aggregation

Same rule as `scripts/parse_method_env_table.py`:

1. Mean success over seeds for each `task_id`.
2. Report **mean ± SEM across tasks** (std of the 10 task means / √n_tasks).
3. Bold the highest mean in each row (ties at 3 decimals: every max). Missing → `---`.

Rows are LIBERO suite × setting: Goal / Object / Spatial × Clean / Distract. All of these folders are image-only, so there is no obs column in the LaTeX table.

Method identity is inferred from algo + folder, not from a global `rep_loss_scale` filter:

- `bc_cardpol_policy` → CardPol (`\method`), these folders use `rep_loss_scale=0.01`
- `bc_vip_policy` → VIP
- `bc_vae_policy` + `vqvae` in the group name or `cfg/train/vae_type` → VQ-VAE, else VAE
- `bc_curl_policy` → CURL
- `bc_icvf_policy` → ICVF

---

## Outputs (`figures/final_results/`)

| File | What it is |
|------|------------|
| `method_env_table.tex` / `.csv` | Paper table, **metrics_summary** scores (unsuffixed copy) |
| `method_env_table_metrics_summary.tex` / `.csv` | Same table, explicitly named |
| `method_env_table_metrics.tex` / `.csv` | Same layout, **last `metrics.csv`** scores |
| `method_task_table.tex` / `.csv` | Per-task table (no average across tasks), summary scores |
| `method_task_table_metrics_summary.tex` / `.csv` | Same per-task table, explicitly named |
| `method_task_table_metrics.tex` / `.csv` | Per-task table from last `metrics.csv` |
| `experiment_results_latex.txt` | Per-folder task × env grids (summary scores) |
| `experiment_results_latex_metrics_summary.txt` | Same |
| `experiment_results_latex_metrics.txt` | Per-folder grids from `metrics.csv` |
| `runs.csv` / `runs_metrics_summary.csv` | One row per job (summary scores) |
| `runs_metrics.csv` | One row per job (history scores) |
| `method_env_long*.csv` | Long-form cell stats (`mean`, `sem`, `n_runs`, `n_tasks`) |
| `coverage.csv` | Per-folder run counts, envs, seeds, tasks, `rep_loss` |

LaTeX table details:

- Header uses `\method` (your paper macro), not the string `CardPol`.
- Caption: LIBERO image-only success, mean ± SEM, `\method{}` uses $\lambda=0.01$.
- Suite table: suites are grouped with `\cmidrule`; Clean/Distract share a suite name on the first row.
- Per-task table: within each suite, clean tasks 0–9 then distract tasks 0–9 (same task ids stacked for C vs D). `\cmidrule` between suites and between the C and D blocks.
- SEM is `\scriptstyle\pm`.
- Score-source note is a `%` comment under `\label`, not in the caption.
- Needs `booktabs` (and whatever defines `\method`). The per-task table also needs `longtable`.

---

## Artifacts layout (same as the scripts guide)

```
artifacts/                          # symlink → cluster artifacts store
  <experiment_group>/               # wandb.group, listed in EXPERIMENT_GROUPS
    <job_name>/
      metrics_summary.csv           # wandb summary (one row)
      metrics.csv                   # full history
      config.yaml                   # flattened to cfg/* columns
```

Primary metric column: `rollout/success_env_avg`.

---

## Workflow after new baseline jobs finish

1. Confirm jobs under the folders in `EXPERIMENT_GROUPS` have `metrics_summary.csv` and/or `metrics.csv`.
2. Re-run `.venv/bin/python scripts/parse_final_results.py`.
3. Check `coverage.csv` for run counts (150 = complete; VIP may still be short).
4. Compare `method_env_table.tex` vs `method_env_table_metrics.tex` if a wandb number disagrees with the summary table.
5. Copy the `.tex` you want into the paper. Use the `metrics` table if you want last-logged history (wandb charts); use `metrics_summary` if you want `run.summary`.

---

## Troubleshooting

| Symptom | Cause | What to do |
|---------|--------|------------|
| `--` / `---` in a cell | No parseable jobs for that suite × distract × method | `coverage.csv`; folder still running or not in `EXPERIMENT_GROUPS` |
| Table 0.526 vs wandb 0.596 (CardPol Spatial clean) | Summary vs last history | Use `method_env_table_metrics.tex` |
| Wrong VIP/ICVF numbers | Older folders (`bc-vip-alltasks_*`, `bc-icvf-baseline_*`, `bc-icvf-weight-rerun_*` without `2`, `bc-vip-best-baseline_*`) | Parser uses the folders in **Source folders** above |
| VQ-VAE missing / merged into VAE | Both algos are `bc_vae_policy` | Distinguished by `_vqvae` in the group name / `vae_type` |
| Parser skipped a folder | No `metrics_summary.csv` or `metrics.csv` yet | Warning on stderr; re-run when jobs land |

---

## Code map

| File | Role |
|------|------|
| `scripts/parse_final_results.py` | This table: folder list, dual metric sources, LaTeX/CSV writers |
| `scripts/parse_experiment_results.py` | Job parse; `load_last_rollout_success()` for `metrics.csv` |
| `scripts/export_artifact_runs.py` | `build_runs_dataframe()` |
| `scripts/parse_method_env_table.py` | Shared SEM helper (`success_stats_over_tasks_and_seeds`) |
| `scripts/PARSING_GUIDE.md` | General artifact parsing (all groups, not this table) |
| `figures/final_results/` | Generated outputs |
| `figures/PARSING_GUIDE.md` | This file |
