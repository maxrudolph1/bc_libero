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
