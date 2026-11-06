#!/usr/bin/env python
"""Build resumable LIBERO video-background datasets, preserving source foreground RGB."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import h5py
import imageio.v2 as imageio
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from libero_exp.utils.distractor_utils import DistractorSource, demo_seed
from libero_exp.utils.foreground_utils import (
    background_geometry, composite_foreground, foreground_mask, localize_model_xml,
)

VERSION = 1


def write_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def render(env, segmentation=False):
    value = env.sim.render(camera_name="agentview", width=128, height=128,
                           **({"segmentation": True} if segmentation else {}))
    return value[..., 1] if segmentation else value


def restore(env, state):
    env.set_state(state)
    env.sim.forward()


def alignment(env, demo, states, max_rgb_mae):
    """Detect state offset using all observed arm/gripper positions, then check RGB."""
    robot = env.env.robots[0]
    indices = np.asarray(list(robot._ref_joint_pos_indexes) + list(robot._ref_gripper_joint_pos_indexes))
    observed = np.concatenate([demo["obs/joint_states"][()], demo["obs/gripper_states"][()]], axis=1)
    errors = {}
    for offset in (0, 1):
        n = len(states) - offset
        errors[offset] = float(np.max(np.abs(states[offset:, 1 + indices] - observed[:n])))
    offset = min(errors, key=errors.get)
    if errors[offset] > 1e-5:
        raise ValueError("No aligned saved state found: joint errors %s" % errors)
    samples = np.unique(np.linspace(0, len(states) - 1 - offset, 5).astype(int))
    maes = [[], []]
    for t in samples:
        restore(env, states[t + offset])
        rgb = render(env)
        original = demo["obs/agentview_rgb"][t].astype(float)
        for flip in (0, 1):
            candidate = rgb[::-1] if flip else rgb
            maes[flip].append(float(np.abs(candidate.astype(float) - original).mean()))
    flip = int(np.argmin([np.mean(values) for values in maes]))
    if max(maes[flip]) > max_rgb_mae:
        raise ValueError("Replayed RGB differs from source (MAEs=%s); refusing misaligned masks" % maes[flip])
    return offset, flip, dict(state_offset=offset, joint_max_error=errors[offset],
                             render_vertical_flip=bool(flip), sampled_rgb_mae=maes[flip])


def final_state(env, demo, states):
    """Recover missing post-action terminal state; fail if observed joints disagree."""
    robot = env.env.robots[0]
    indices = list(robot._ref_joint_pos_indexes) + list(robot._ref_gripper_joint_pos_indexes)
    expected = np.concatenate([demo["obs/joint_states"][-1], demo["obs/gripper_states"][-1]])
    best_error, best_state, best_warmup = float("inf"), None, None
    for warmup in sorted(set([min(5, len(states)), min(20, len(states)), len(states)])):
        for t in range(len(states) - warmup, len(states)):
            env.regenerate_obs_from_state(states[t])
            env.step(demo["actions"][t])
        error = float(np.max(np.abs(env.sim.data.qpos[indices] - expected)))
        if error < best_error:
            best_error, best_state, best_warmup = error, env.get_sim_state().copy(), warmup
        if error <= 1e-5:
            return dict(warmup_actions=warmup, joint_max_error=error, robot_pose_corrected=False)
    if best_error > 1e-3:
        raise ValueError("Terminal replay does not match observed joints: %g" % best_error)
    # Minor solver drift is possible when restoring demonstrations collected with
    # a different MuJoCo version. Use the measured final robot pose, never move
    # the source RGB. Object geometry still comes from the best physics replay.
    restore(env, best_state)
    env.sim.data.qpos[indices] = expected
    env.sim.forward()
    return dict(warmup_actions=best_warmup, replay_joint_max_error=best_error,
                joint_max_error=0.0, robot_pose_corrected=True)


def convert_demo(env, source, target, video, suite, task, demo_name, seed, max_rgb_mae):
    started = time.monotonic()
    env.reset_from_xml_string(localize_model_xml(source.attrs["model_file"]))
    ids, names = background_geometry(env.sim.model)
    states = source["states"][()]
    count = len(states)
    if source["obs/agentview_rgb"].shape != (count, 128, 128, 3) or len(source["actions"]) != count:
        raise ValueError("Expected equally sized states, actions, and 128x128 RGB observations")
    offset, flip, qa = alignment(env, source, states, max_rgb_mae)
    rng = np.random.RandomState(demo_seed(suite, task, demo_name, seed))
    start = video.sample_start(rng, count)
    obs = target["obs"]
    old_attrs = dict(obs["agentview_rgb"].attrs)
    del obs["agentview_rgb"]
    output = obs.create_dataset("agentview_rgb", shape=(count, 128, 128, 3), dtype="uint8",
                                chunks=(1, 128, 128, 3), compression="gzip", compression_opts=4)
    for key, value in old_attrs.items():
        output.attrs[key] = value
    masks = target.create_dataset("foreground_mask", shape=(count, 128, 128), dtype="uint8",
                                  chunks=(1, 128, 128), compression="gzip", compression_opts=4)
    fractions = []
    preview_frames = []
    for t in range(count):
        if t + offset < count:
            restore(env, states[t + offset])
        else:
            qa["terminal_replay"] = final_state(env, source, states)
            rgb = render(env)
            rgb = rgb[::-1] if flip else rgb
            error = float(np.abs(rgb.astype(float) - source["obs/agentview_rgb"][t]).mean())
            qa["terminal_rgb_mae"] = error
            if error > max_rgb_mae:
                raise ValueError("Terminal RGB alignment failed: MAE=%g" % error)
        seg = render(env, segmentation=True)
        seg = seg[::-1] if flip else seg
        keep = foreground_mask(seg, ids)
        original = source["obs/agentview_rgb"][t]
        # Raw MuJoCo is bottom-up: orient the video to match source RGB.
        distractor = video.load_frame(start + t)
        if not flip:
            distractor = distractor[::-1]
        composed = composite_foreground(original, keep, distractor)
        output[t] = composed
        masks[t] = keep.astype(np.uint8)
        fractions.append(float(keep.mean()))
        if demo_name == "demo_0":
            frame = np.concatenate([original, np.repeat(keep[..., None], 3, axis=2).astype(np.uint8) * 255, composed], axis=1)
            preview_frames.append(np.ascontiguousarray(frame[::-1] if not flip else frame))
    if not (0.001 < np.mean(fractions) < 0.999):
        raise ValueError("Degenerate foreground mask")
    record = dict(frames=count, video_start=start, video_end=start + count - 1,
                  video_wrapped=start + count > video.num_frames, video_flip_vertical=not bool(flip),
                  background_geometries=names, foreground_fraction_mean=float(np.mean(fractions)),
                  alignment=qa, seconds=round(time.monotonic() - started, 3))
    target.attrs["foreground_video"] = json.dumps(record)
    return record, preview_frames


def convert_file(path, args, video):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = path.parent.name
    destination = args.dst_root / suite / path.name
    if path.resolve() == destination.resolve():
        raise ValueError("Source and destination must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".hdf5.partial")
    manifest_path = destination.with_suffix(".manifest.json")
    fingerprint = dict(version=VERSION, source=str(path.resolve()), source_bytes=path.stat().st_size,
                       source_mtime_ns=path.stat().st_mtime_ns, video_dir=str(Path(video.frames_dir).resolve()),
                       video_frames=video.num_frames,
                       video_frame_names_sha256=hashlib.sha256("\n".join(Path(p).name for p in video.frame_paths).encode()).hexdigest(),
                       seed=args.seed, layout="keep_foreground_video", rgb_source="original_hdf5",
                       max_rgb_mae=args.max_rgb_mae)
    with destination.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = destination if destination.exists() else partial
        if existing.exists():
            with h5py.File(existing, "r") as f:
                if json.loads(f.attrs["foreground_video_build"]) != fingerprint:
                    raise ValueError("Existing output belongs to a different build: %s" % existing)
                if existing == destination:
                    if not f.attrs.get("foreground_video_complete", False):
                        raise ValueError("Final output is not marked complete")
                    print("SKIP completed", destination, flush=True)
                    return
        with h5py.File(path, "r") as src, h5py.File(partial, "a") as dst:
            if "foreground_video_build" not in dst.attrs:
                for key, value in src.attrs.items():
                    dst.attrs[key] = value
                for key in src:
                    if key != "data":
                        src.copy(key, dst)
                dst.create_group("data")
                for key, value in src["data"].attrs.items():
                    dst["data"].attrs[key] = value
                dst.attrs["foreground_video_build"] = json.dumps(fingerprint, sort_keys=True)
            if "_pending_demo" in dst:
                del dst["_pending_demo"]
            demos = sorted(src["data"], key=lambda name: int(name.split("_")[-1]))
            report = dict(build=fingerprint, complete=False, demos={})
            for name in dst["data"]:
                report["demos"][name] = json.loads(dst["data"][name].attrs["foreground_video"])
            if args.max_demos and len(dst["data"]) >= args.max_demos and len(dst["data"]) < len(demos):
                print("PAUSED at requested demo limit:", partial, flush=True)
                return
            bddl = Path(get_libero_path("bddl_files")) / suite / Path(src["data"].attrs["bddl_file_name"]).name
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_names=["agentview"],
                                     camera_heights=128, camera_widths=128, hard_reset=False)
            try:
                env.reset()
                for name in demos:
                    if name in dst["data"]:
                        continue
                    src.copy(src["data"][name], dst, name="_pending_demo")
                    record, preview = convert_demo(env, src["data"][name], dst["_pending_demo"], video,
                                                   suite, path.name, name, args.seed, args.max_rgb_mae)
                    dst.flush()
                    dst.move("_pending_demo", "data/" + name)
                    dst.flush()
                    report["demos"][name] = record
                    write_json(manifest_path, report)
                    print("%s/%s %s: %d frames, %.1fs (%d/%d demos)" %
                          (suite, path.stem, name, record["frames"], record["seconds"], len(dst["data"]), len(demos)), flush=True)
                    if preview:
                        preview_dir = args.dst_root / "previews" / suite
                        preview_dir.mkdir(parents=True, exist_ok=True)
                        imageio.mimsave(str(preview_dir / (path.stem + ".mp4")), preview, fps=20)
                        imageio.imwrite(str(preview_dir / (path.stem + ".png")), preview[len(preview) // 2])
                    if args.max_demos and len(dst["data"]) >= args.max_demos and len(dst["data"]) < len(demos):
                        print("PAUSED at requested demo limit:", partial, flush=True)
                        return
                dst.attrs["foreground_video_complete"] = True
                report["complete"] = True
            finally:
                env.close()
        partial.replace(destination)
        write_json(manifest_path, report)
        print("COMPLETE", destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, default=Path("/datastor2/mrudolph/LIBERO/datasets"))
    parser.add_argument("--dst-root", type=Path, default=Path("/datastor2/mrudolph/LIBERO/datasets_foreground_video"))
    parser.add_argument("--suites", nargs="+", default=["libero_goal"])
    parser.add_argument("--files", nargs="+", type=Path)
    parser.add_argument("--frames-dir", default=str(REPO / "epickitchens/P01/rgb_frames/p01_01"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-demos", type=int, help="Stop each task after this many demos; resume by omitting")
    parser.add_argument("--max-rgb-mae", type=float, default=5.0)
    parser.add_argument("--keep-going", action="store_true", help="Finish other tasks if a task fails; still exit nonzero")
    args = parser.parse_args()
    paths = args.files or [path for suite in args.suites for path in sorted((args.src_root / suite).glob("*.hdf5"))]
    if not paths:
        parser.error("No input datasets found")
    args.dst_root.mkdir(parents=True, exist_ok=True)
    video = DistractorSource(args.frames_dir)
    print("Building %d task files using %d video frames" % (len(paths), video.num_frames), flush=True)
    failures = []
    for path in paths:
        try:
            convert_file(path, args, video)
        except Exception as exc:
            if not args.keep_going:
                raise
            traceback.print_exc()
            failures.append(dict(source=str(path), error=str(exc)))
            write_json(args.dst_root / "build_failures.json", failures)
    if failures:
        raise SystemExit("%d task(s) failed; see build_failures.json. Completed demos were retained." % len(failures))
    write_json(args.dst_root / "build_failures.json", [])


if __name__ == "__main__":
    main()
