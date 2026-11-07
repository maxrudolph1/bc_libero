# LIBERO foreground video dataset

Convert existing clean `libero_goal` demonstrations into 128x128 observations
with moving EPIC-Kitchens video behind the robot, scene objects, and fixtures.
This uses the `keep_foreground` definition from the four-panel previews: replace
floor, walls, and all table geometry, while retaining all other visible geometry.
Unrelated objects remain visible, as do task-critical cabinets, drawers, and stoves.

## Recovered build status (2026-11-06)

The full `libero_goal` build completed and its validator succeeded at
2026-11-06 18:28:57 UTC (12:28:57 America/Chicago). The saved build log reports
10 validated task HDF5 files, 50 demos per task, 500 demos total, and 63,728
frames. Output task files occupy approximately 5.2 GiB. All 10 task manifests
are present and account for 500 demos. `build_failures.json` contains `[]`.

Evidence and artifacts under `/datastor2/mrudolph/LIBERO/datasets_foreground_video/`:

- `libero_goal/`: completed HDF5 files and per-task `.manifest.json` QA records
- `VALIDATED_COMPLETE.txt`: successful build-and-validation completion timestamp
- `build_active.log`: conversion progress and validation results for every task
- `build_launch.json`: launch details and original build scope
- `previews/libero_goal/`: original / mask / composite PNG and MP4 per task

The recorded validation checked all original fields and all composited pixels.
This recovery inspected the saved logs, completion marker, and manifests; it did
not rerun the full pixel validator or a policy rollout. No further dataset build
is pending for this suite. Other LIBERO suites were not part of this build.

The implementation is in `scripts/build_libero_foreground_video.py`,
`scripts/validate_libero_foreground_video.py`,
`libero_exp/utils/foreground_utils.py`, and
`launch_scripts/mll/build_foreground_video.slurm`. At recovery, these files and
this document were untracked in Git; the data live outside the repository.

## Build

From the repository root, using its existing Python environment:

```bash
.venv/bin/python scripts/build_libero_foreground_video.py
```

Defaults:

- Source: `/datastor2/mrudolph/LIBERO/datasets/libero_goal`
- Destination: `/datastor2/mrudolph/LIBERO/datasets_foreground_video/libero_goal`
- Video: `epickitchens/P01/rgb_frames/p01_01` (99,135 extracted JPEG frames)
- Seed: 0; one deterministically sampled contiguous video window per demo
- Video advances one frame per observation; wraps only if a demo exceeds the clip

Use `--files /path/to/task_demo.hdf5 --max-demos 2` for a pilot. The script keeps
unfinished tasks as `.hdf5.partial`; rerun without the demo limit to resume.
Completed task files are published atomically as `.hdf5` and skipped on later runs.
Per-task file locks prevent simultaneous writers. A build fingerprint prevents
resuming with different source files, seeds, or video directories/frame lists.
The source data and existing side-by-side distractor datasets are never overwritten.

```bash
sbatch --export=ALL,REPO_ROOT="$PWD" \
    --output=/datastor2/mrudolph/LIBERO/datasets_foreground_video/build_%j.log \
    launch_scripts/mll/build_foreground_video.slurm
```

Choose a node with access to all data paths. The batch job converts all goal tasks,
then checks every output frame and all preserved fields. It creates
`VALIDATED_COMPLETE.txt` only after both phases succeed. The batch job finishes
other tasks if one task fails, records failures in `build_failures.json`, and exits
nonzero without marking the suite complete. Completed demos remain available for
resuming. Direct CLI builds stop immediately unless `--keep-going` is supplied.

## Alignment and pixels

The source generation code saves observations after executing an action, while
the corresponding `states` entry can be the state before that action. For each
demo, the converter checks all stored arm and gripper joint positions to choose
state offset 0 or 1. It requires maximum absolute joint error at most 1e-5.

With offset 1, the last observation has no following saved state. The converter
replays the last 5 actions with saved-state restoration; it retries with 20 or all
actions when necessary. If replay differs by more than 1e-5 but at most 1e-3
(radians for arm joints, meters for gripper joints), it restores the best replay
and sets the robot joints to the exact saved observation; larger differences
stop the build. This correction and the replay error are recorded. Five sampled
RGB frames and the terminal frame must have mean
absolute RGB error at most 5/255 against the original image. These checks are
recorded per demo; they do not guarantee exact object-pose identity at the final
frame, so inspect previews when adding unfamiliar scenes.

Segmentation and RGB must share the source camera orientation. The original
foreground RGB is copied byte-for-byte; only background pixels are replaced.
Video is vertically oriented to match the saved RGB convention. Preview videos
are shown upright. Physics, actions, timestamps, and original source metadata
remain unchanged.

## Outputs

Each task HDF5 retains its existing dataset structure and metadata, with:

- `data/demo_N/obs/agentview_rgb`: new uint8 RGB, unchanged `(T,128,128,3)` shape
- `data/demo_N/foreground_mask`: new uint8 `(T,128,128)` mask; 1 means preserved
- `foreground_video` demo attribute: video window, geometry names, alignment QA
- `foreground_video_build` root attribute: source/video identity, seed, build version
- `foreground_video_complete` root attribute: true only when all demos are finished

Actions, wrist RGB, proprioception, states, rewards, dones, and original attributes
are retained. Sidecar `.manifest.json` files report completed demos and QA.
`previews/libero_goal/` contains a demo-0 video and PNG per task, arranged as
original / foreground mask / composited output. Masks can be reused to generate
other video variants without rerendering the simulator.

Validate completed files or stopped pilot `.partial` files with:

```bash
.venv/bin/python scripts/validate_libero_foreground_video.py \
    /datastor2/mrudolph/LIBERO/datasets_foreground_video/libero_goal/*.hdf5
```

The validator checks every foreground pixel against source RGB, every background
pixel against the recorded video window, and all original datasets and attributes.

## Training and evaluation

### Foreground-video evaluation mode

Use `data.distractor.enable=true` and
`data.distractor.layout=keep_foreground_video` together with the new dataset root.
The `foreground_video` data configuration sets these automatically. Training
config `vilt_foreground_video` and evaluation config
`vilt_eval_foreground_video` select this data configuration.

To evaluate an existing run, retain its saved model/training configuration and
explicitly select the new observation mode:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python eval_libero.py \
    --config-path=/absolute/path/to/training/run \
    --config-name=config.yaml \
    eval.load_path=/absolute/path/to/training/run \
    data.root_dir=/datastor2/mrudolph/LIBERO/datasets_foreground_video \
    data.distractor.enable=true \
    data.distractor.layout=keep_foreground_video \
    'env.task_id=[0,1,2,3,4,5,6,7,8,9]'
```

Use the model configuration from the checkpoint's own training run. The example
assumes its data/environment suite is `libero_goal` and its agentview image size
is 128. The default task selection is `[2]`, so specify the task IDs explicitly
for a full goal-suite evaluation. Existing `*_distract` configs still select the
older side-by-side mode; they do not select foreground-video compositing.

| Configuration | Rollout observation | Dataset |
| --- | --- | --- |
| `data.distractor.enable=false` | Clean scene, normally 128x128 | `datasets` |
| Enabled, `layout=hconcat_right` | Scene on left, video on right, 128x256 | `datasets_distract` |
| Enabled, `layout=keep_foreground_video` | Preserved foreground over video, 128x128 | `datasets_foreground_video` |

`ForegroundMaskEnv` wraps each individual LIBERO environment inside the vector
worker. After reset, initial-state restoration, and every step, it renders a
geometry segmentation mask at the live simulator state. Geometry IDs are
resolved again to handle hard resets. Masks use the same robosuite image
convention as RGB. Both `DummyVectorEnv` and the subprocess vector environment
carry the mask and video-orientation flag with the observation. Foreground-video subprocess
workers use `spawn` through a small LIBERO worker adapter so they do not inherit
EGL/CUDA state from the parent. Existing side-by-side workers are unchanged.
RGB is freshly rendered alongside segmentation because cached camera
observables can lag the live physics state; the mask and RGB must agree.

The rollout's `DistractorAugmentor` composites a deterministic video window for
each episode/environment slot using the same `background_geometry`,
`foreground_mask`, and `composite_foreground` functions as dataset generation.
Only floor, walls, table, and renderer void pixels are replaced. Original
foreground RGB, wrist images, and nonvisual observations are preserved. One
video frame is consumed for the initial observation and each subsequent step;
window sampling reserves `horizon + 1` frames. Videos wrap only if necessary.
Training and evaluation windows use different deterministic seed identities;
they are not a held-out video split (both use the configured clip by default).

Shape validation requires `(3,128,128)` for foreground video and `(3,128,256)`
for side-by-side. Shape alone cannot distinguish clean and foreground-video
HDF5 files, so select the correct data root as well as the layout.

Run the environment validation without a policy checkpoint:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    .venv/bin/python scripts/validate_libero_foreground_env.py
```

It exercises all 10 goal tasks with one single-process and two subprocess
environments, checking foreground/background pixels, untouched observations,
video progression, and resets. Task 0 also exercises the real rollout, policy
observation conversion, and output-video path with a probe policy in both modes.
The single-process path additionally compares
observation RGB and masks with independent live simulator renders. Hard resets
may change scene properties outside the saved simulator state; reset validation
therefore checks the fresh foreground and the restarted video independently.

Validation completed on 2026-11-06: all 20 task/vector-mode combinations passed
(10 tasks each with one and two environments), including three successive
observations and a reset per combination. Probe-policy rollouts passed for task
0 with both vector modes. The new training/evaluation configs and existing
side-by-side configs composed successfully; matching shapes were accepted and
invalid shapes rejected. Routing checks covered Hydra task lists, scalar task
IDs, and selecting all tasks. These are short environment/integration checks;
no trained checkpoint success-rate evaluation was run.

The original recovery audit found no foreground-video evaluation path. The
implementation described here replaces that gap; dataset build completion and
policy success-rate evaluation remain separate checks.

## CARDPol foreground-video pilot

The pilot uses target tasks 0 and 2, representation-loss weights 0, 0.01, and
0.1, and training seeds 0, 1, and 2: 18 independent runs. Weight 0 is the BC-only
control within the CARDPol pipeline. Each policy uses ViLT, agentview RGB only,
no proprioception or language, and joint BC/auxiliary encoder gradients.
Training uses 50 epochs, batch size 64, AdamW at 1e-4, 45 training demos and
5 validation demos per task. BC trains on the target task and the auxiliary
sampler draws from all ten tasks, with future offsets 1 through 10.

The launcher now accepts `--foreground-video`, which sets the data root and
evaluation layout explicitly, uses EGL, and evaluates only the final checkpoint.
Final evaluation uses 100 episodes (10 environments x 10 rollout batches),
600 steps maximum per episode, and fixed background seed 0 across runs. The
10 initial simulator states are reused across batches with different video
windows. Groups separate target task and auxiliary weight; seeds share a group.
The training initializer now respects configured environment/episode counts
instead of overwriting both with 10.

```bash
.venv/bin/python launch_scripts/mll/submit_libero.py \
    --foreground-video --task-ids 0,2 --seeds 0,1,2 \
    --rep-loss-scales 0,0.01,0.1 --modalities image --cameras agentview \
    --wandb-project bc-cardpol-vilt --wandb-group fg-video-pilot \
    --array --array-max-parallel 3 --job-name cardpol-fg-pilot
```

Each array task requests one GPU, 16 CPUs, 128 GB RAM, and 24 hours in `allnodes`.
Three runs may execute concurrently. Add `--dry-run` to inspect scripts without
submitting. Submission scripts, per-index commands, and logs are retained in
`slurm_jobs/libero/array_<timestamp>/`; run configs, checkpoints, and metrics are
retained in `artifacts/<wandb.group>/` and logged to W&B.

Foreground-video submissions now explicitly set `wandb.mode=offline`. The
installed W&B 0.13.1 client fails during online server-settings introspection
(`ServerSettingsType` is null), before training starts. Offline mode avoids
that failing API call while retaining metrics CSVs, videos, checkpoints, and
offline W&B bundles locally. These runs do not update cloud dashboards.
`init_wandb` honors the configured mode. Training also closes rollout workers
in a `finally` block so renderer contexts are released before interpreter
teardown, including when training raises an exception.

Initially submitted on 2026-11-06 at 14:28 America/Chicago as Slurm array **119667**,
indices 0-17 with concurrency limit 3. Submission directory:
`slurm_jobs/libero/array_20261106_142803/`. At submission verification all indices
were pending with reason `Priority`; training had not started. All 18 later
failed in online W&B initialization before completing any training steps.
All 18 expanded
Hydra configurations passed composition and dual-task consistency checks.

| Array indices | Target task | Auxiliary weight | Seeds in index order |
| --- | --- | --- | --- |
| 0-2 | 0 | 0 | 0, 1, 2 |
| 3-5 | 0 | 0.01 | 0, 1, 2 |
| 6-8 | 0 | 0.1 | 0, 1, 2 |
| 9-11 | 2 | 0 | 0, 1, 2 |
| 12-14 | 2 | 0.01 | 0, 1, 2 |
| 15-17 | 2 | 0.1 | 0, 1, 2 |

Check status with `squeue -j 119667`. Once running, inspect per-index `.out` and
`.err` files in the submission directory's `logs/` folder.

### Startup repair and relaunch

After the online W&B failure, local checks on `slurm-node-006` completed with
offline logging for all three auxiliary weights (0, 0.01, 0.1). Each small-batch
check performed training and validation, saved/reloaded a checkpoint, ran a
two-environment, ten-step foreground-video rollout, and exported finite metrics.
A separate check used the production batch size 64 and ten renderer workers.
Finally, a full one-epoch run with weight 0.1 completed all 100 training batches,
all validation batches, checkpoint save/reload, and a ten-environment, ten-step
rollout. It exited successfully without EGL cleanup tracebacks. The epoch took
about 67 seconds including validation on this node. These are execution checks,
not trained-policy performance results; short rollouts do not estimate success.

Full-epoch smoke artifacts:
`artifacts/fg-video-full-epoch-smoke_img_cam-agent_foreground_video_task0_rep0.1/libero_goal_bc_cardpol_policy_vilt_task0_1106_1738_seed0_ratio0.9_a78380/`.
The directory contains the exact config, checkpoint, and `metrics.csv` for
reproducing/inspecting the test. Small-batch and full-batch checks are under
the corresponding `fg-video-smoke*` and `fg-video-full-batch-smoke*` groups.

Relaunched the same 18-run pilot on 2026-11-06 at 17:40 America/Chicago as Slurm
array **119776**, indices 0-17 with maximum concurrency 3. Submission directory:
`slurm_jobs/libero/array_20261106_174044/`. The index mapping above is unchanged.
Production runs use 50 full epochs and 100 final-checkpoint evaluation episodes
of up to 600 steps, with `train.debug=false` and explicit offline W&B logging.
Use `squeue -j 119776` and the new submission directory's `logs/` folder for
current status; the failed original array 119667 is retained for diagnosis.
At 17:41:57, replacement indices 0, 1, and 2 had each reached batch 50/100
of epoch 1/50 on `slurm-node-011`, with finite BC losses and no startup
tracebacks. Indices 3-17 were pending behind the array concurrency limit.

## All-ten-task comparison: weights 0 and 0.01

Expanded on 2026-11-07 to all ten `libero_goal` target tasks, auxiliary weights
0 and 0.01, and seeds 0, 1, and 2: **60 unique training/evaluation runs**.
Reused the 12 completed runs for tasks 0 and 2 from array 119776; verified their
checkpoints, final metrics, and matching training/evaluation settings. The
weight-0.1 pilot runs are excluded from this comparison.

Submitted the remaining 48 runs as Slurm array **120497**, indices 0-47,
with maximum concurrency 3. Submission directory:
`slurm_jobs/libero/array_20261107_091837/`. Each group of six indices covers one
task in order `[1,3,4,5,6,7,8,9]`: first weight 0 with seeds 0/1/2, then weight
0.01 with seeds 0/1/2. All 48 expanded Hydra configurations passed preflight
checks; the generated task scripts were checked against the experiment manifest.

`experiment_manifest.json` and `experiment_manifest.csv` in that directory map
all 60 combinations to their jobs, including the reused runs' artifact paths
and completed success rates. Submission-time status fields are snapshots; use
Slurm and per-run metrics for current status.

```bash
.venv/bin/python launch_scripts/mll/submit_libero.py \
    --foreground-video --task-ids 1,3,4,5,6,7,8,9 --seeds 0,1,2 \
    --rep-loss-scales 0,0.01 --modalities image --cameras agentview \
    --wandb-project bc-cardpol-vilt --wandb-group fg-video-pilot \
    --array --array-max-parallel 3 --job-name cardpol-fg-all10
```

Training and evaluation settings remain those of the repaired pilot: ViLT,
agentview only, joint BC/auxiliary encoder gradients, batch 64, 50 epochs,
and 100 foreground-video evaluation episodes per final checkpoint. W&B logging
is offline. Keeping the same group prefix combines pilot and expansion results
under the same per-task/per-weight groups. Do not rerun this submission command
unless intentionally repeating those 48 experiments.

On 2026-11-07, increased the live concurrency limit for array 120497 from 3
to **16** with `scontrol update JobId=120497 ArrayTaskThrottle=16`. Slurm
confirmed `ArrayTaskThrottle=16`; existing runs were retained. The original
submission script records the initial limit of 3. For future submissions,
use `--array-max-parallel 16` to match the updated limit.

## VIP/CURL/VAE auxiliary-weight pilot

The baseline pilot covers the same target tasks 0 and 2 with training seeds
0, 1, and 2. For each of VIP, CURL, and VAE, test auxiliary weights 0.1, 0.01,
and 0.001: **54 runs total**, 18 per method. VAE explicitly selects
`train.vae_type=vae` (the repository's representation VAE); VQ-VAE is not included.
VIP uses its `vip` frame sampler; CURL and VAE use `future_pair`. Existing
auxiliary objectives and their internal hyperparameters are retained.

The common recipe matches the CARDPol foreground-video experiment: ViLT,
agentview only, no proprioception/language, joint BC/auxiliary encoder gradients,
batch 64, 50 epochs, a 90/10 demo split, pooled auxiliary data from all ten tasks,
and final-checkpoint evaluation on the target task for 100 episodes of up to
600 steps. Logging is offline, with per-method/task/weight groups and per-seed
artifacts. Use rollout success averaged over the two tasks and three seeds to
select a weight for each method before expanding to all ten tasks. Raw
auxiliary-loss values have different scales across methods.

The launcher creates one combined Slurm array with a concurrency limit of 16
for the baseline pilot. Each run requests one GPU, 16 CPUs, 128 GB RAM, and
24 hours. The CARDPol array's concurrency is controlled separately.

```bash
# Inspect all 54 runs without submitting.
.venv/bin/python launch_scripts/mll/submit_foreground_baselines.py --dry-run

# Local GPU checks: two optimizer steps, validation, checkpoint reload,
# short foreground-video rollout, metric export, and clean shutdown per method.
.venv/bin/python launch_scripts/mll/submit_foreground_baselines.py --smoke

# Submit the pilot after checks pass.
.venv/bin/python launch_scripts/mll/submit_foreground_baselines.py
```

All 54 Hydra configurations passed preflight composition and consistency checks.
The generic `submit_libero.py --foreground-video` path now expands list-valued
baseline weights before constructing scalar per-weight logging groups; this
also permits using its individual VIP/CURL/VAE sweep flags.

Local batch-64 smoke tests passed on `slurm-node-006` for all three methods,
with two optimizer steps, finite losses, saved/reloaded checkpoints, short
two-environment foreground-video evaluation, exported metrics, and clean exit.
Logs are under `slurm_jobs/libero/baseline_smoke_20261107_111748/`; saved configs,
checkpoints, and metrics are under `artifacts/fg-baseline-smoke_bc_*_policy/`.

Submitted on 2026-11-07 at 11:20 America/Chicago as Slurm array **120572**,
indices 0-53, concurrency 16. Submission directory:
`slurm_jobs/libero/array_20261107_112053/`. `experiment_manifest.json` and
`experiment_manifest.csv` map all 54 runs to method, task, weight, seed, group,
and array index; JSON also retains the exact command configuration.

| Array indices | Method | Within each method |
| --- | --- | --- |
| 0-17 | VIP | Task 0 first, then task 2 |
| 18-35 | CURL | Task 0 first, then task 2 |
| 36-53 | VAE | Task 0 first, then task 2 |

Within each task's nine indices, weights are ordered 0.1, 0.01, 0.001;
each weight has seeds 0, 1, 2. Check current state with `squeue -j 120572`
and inspect the submission directory's `logs/` folder. Existing CARDPol/BC
control runs on these tasks remain available for comparison; this submission
does not repeat them or expand the baselines to the other eight tasks yet.

On 2026-11-07, reduced both live array throttles to **8 per array**:
`scontrol update JobId=120497 ArrayTaskThrottle=8` and
`scontrol update JobId=120572 ArrayTaskThrottle=8`. Slurm confirmed both limits.
Already-running jobs were retained; an array above eight running tasks must
drain below its limit before additional tasks start. The original submission
scripts and earlier concurrency notes record historical settings.

## Progress audit — 2026-11-07 14:34 America/Chicago

Read live `squeue` and `sacct` accounting for arrays 119776, 120497, and
120572, checked per-index output/error logs, and matched completed runs to
`model_final.ckpt` and `post_train/model_final/success_env_avg` in their
`metrics.csv`. Counts below are this audit's snapshot; jobs continue running.
Submission manifests retain their historical status fields.

| Experiment | Completed | Running | Pending | Failed |
| --- | ---: | ---: | ---: | ---: |
| Original repaired CARDPol pilot, 119776 | 18 | 0 | 0 | 0 |
| All-task expansion, 120497 | 37 | 8 | 3 | 0 |
| All-task comparison including 12 reused pilot runs | 49 / 60 | 8 | 3 | 0 |
| VIP/CURL/VAE pilot, 120572 | 4 / 54 | 8 | 42 | 0 |

All completed jobs in these arrays exited `0:0`. Both active arrays retain
throttle 8, and pending indices are waiting on `JobArrayTaskLimit`.
Expansion indices 0–36 completed, 37–44 are running, and 45–47 are pending.
Task 8 indices 37–41 have reached final evaluation (roughly rollout batches
1–8 of 10); task 9 indices 42–44 are training around epochs 17–27 of 50.
The remaining three task 9 runs have not started.

### CARDPol/BC results available so far

Success percentages below average seeds 0, 1, and 2, with 100 final-checkpoint
rollout episodes per seed. All 49 completed comparison runs have a final
checkpoint and exported success metric. Tasks 0–7 have all six results;
exclude partial task 8 and unstarted task 9 from the aggregate comparison.
Weight 0 is the BC-only control within the CARDPol pipeline.

| Target task | BC, weight 0 | CARDPol, weight 0.01 |
| --- | ---: | ---: |
| 0 | 68.33% | 77.00% |
| 1 | 89.33% | 92.33% |
| 2 | 66.33% | 78.00% |
| 3 | 28.67% | 38.00% |
| 4 | 53.67% | 51.67% |
| 5 | 42.67% | 52.33% |
| 6 | 57.00% | 49.67% |
| 7 | 87.67% | 87.33% |
| Mean over completed tasks 0–7 | 61.71% | 65.79% |

The preliminary difference is +4.08 percentage points on these eight tasks;
this is not the final ten-task result or a statistical significance claim.
Task 8's only completed run at this snapshot is BC seed 0, with 75% success.

### Baseline pilot progress

Completed VIP runs are `120572_2` (task 0, weight 0.1, seed 2: 51%),
`120572_3` (task 0, weight 0.01, seed 0: 64%), `120572_4` (same weight,
seed 1: 64%), and `120572_6` (weight 0.001, seed 0: 47%). All four have
verified final checkpoints and metrics. These incomplete seed sets do not yet
support selecting a weight or comparing methods.

VIP indices 0, 1, 5, and 7–11 are running; all but the newly started index 11
have reached final evaluation. Index 7 is at rollout batch 10/10, and indices
0/1 at roughly 8–9/10, so their long elapsed times include ongoing evaluation.
Indices 12–53 are pending, including every CURL and VAE run. No traceback,
CUDA out-of-memory error, or time-limit termination was found in the inspected
error logs for the two active arrays.

Completed output logs contain a nonfatal warning that the installed W&B `Run`
object has no `history` attribute; local metrics CSV export nevertheless
succeeded and the final metrics above were read directly from those CSVs.
Logging remains offline, so cloud dashboards do not reflect this progress.

This save also includes the existing foreground launcher, VIP/CURL/VAE pilot
launcher, configured rollout-count handling, offline W&B mode, and renderer
cleanup changes described earlier in this logbook. This audit ran Python syntax
checks and `git diff --check`; it did not launch new experiments, rerun the
previous smoke tests, or change scheduler settings.
