"""Replay one LIBERO demo and preview two segmentation-based video backgrounds."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import robosuite
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libero_exp.utils.distractor_utils import DistractorSource


def local_xml(xml):
    root = ET.fromstring(xml)
    for element in root.iter():
        filename = element.get("file")
        if not filename:
            continue
        if "/robosuite/" in filename:
            filename = str(Path(robosuite.__file__).parent / filename.split("/robosuite/", 1)[1])
        elif "/assets/" in filename:
            filename = str(Path(get_libero_path("assets")) / filename.split("/assets/", 1)[1])
        if not Path(filename).exists():
            raise FileNotFoundError(filename)
        element.set("file", filename)
    return ET.tostring(root, encoding="unicode")


def panel(images, frame):
    canvas = Image.new("RGB", (1024, 584), (24, 27, 33))
    draw = ImageDraw.Draw(canvas)
    titles = ["1. Original (replayed simulator)", "2. Segmentation: room / table / foreground",
              "3. Keep workspace; replace room", "4. Keep robot + objects; replace room + table"]
    for i, (array, title) in enumerate(zip(images, titles)):
        x, y = (i % 2) * 512, (i // 2) * 280
        draw.text((x + 12, y + 6), title, fill="white")
        im = Image.fromarray(array).resize((256, 256), Image.Resampling.NEAREST)
        canvas.paste(im, (x + 128, y + 23))
    draw.text((12, 568), "Frame %d | segmentation: blue=room, gold=table, green=robot/objects/fixtures" % frame, fill="white")
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=next(Path("/datastor2/mrudolph/LIBERO/datasets/libero_spatial").glob("*.hdf5")))
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--frames-dir", default="epickitchens/P01/rgb_frames/p01_01")
    parser.add_argument("--output", type=Path, default=Path("exports/background_distractor_preview"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = DistractorSource(args.frames_dir)
    with h5py.File(args.dataset, "r") as f:
        demo = f["data"][args.demo]
        bddl = Path(get_libero_path("bddl_files")) / Path(f["data"].attrs["bddl_file_name"]).parent.name / Path(f["data"].attrs["bddl_file_name"]).name
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_names=["agentview"], camera_heights=128, camera_widths=128)
        try:
            env.reset()
            env.reset_from_xml_string(local_xml(demo.attrs["model_file"]))
            model = env.sim.model
            names = [model.geom_id2name(i) or "" for i in range(model.ngeom)]
            room_ids = [i for i, n in enumerate(names) if n == "floor" or n.startswith("wall_")]
            table_ids = [i for i, n in enumerate(names) if n.startswith("table")]
            print("Room geoms:", [names[i] for i in room_ids], flush=True)
            print("Table geoms:", [names[i] for i in table_ids], flush=True)
            print("Objects:", list(env.env.model.instances_to_ids), flush=True)
            count = len(demo["states"])
            frames, errors = [], []
            for t in range(count):
                env.set_state(demo["states"][t])
                env.sim.forward()
                rgb_raw = env.sim.render(camera_name="agentview", width=128, height=128)
                seg_raw = env.sim.render(camera_name="agentview", width=128, height=128, segmentation=True)[..., 1]
                stored = demo["obs/agentview_rgb"][t]
                # Identify the saved image convention directly rather than assuming a flip.
                candidates = [rgb_raw, rgb_raw[::-1]]
                errs = [float(np.abs(c.astype(float) - stored.astype(float)).mean()) for c in candidates]
                flip = int(np.argmin(errs))
                # MuJoCo's raw render is bottom-up. Display upright, independently
                # of the stored dataset convention used for the alignment check.
                rgb = rgb_raw[::-1]
                seg = seg_raw[::-1]
                errors.append(min(errs))
                room = np.isin(seg, room_ids) | (seg < 0)
                table = np.isin(seg, table_ids)
                labels = np.full_like(rgb, (75, 190, 115))
                labels[room] = (65, 115, 215)
                labels[table] = (220, 175, 65)
                video = source.load_frame(1000 + t)
                workspace = np.where(room[..., None], video, rgb)
                foreground = np.where((room | table)[..., None], video, rgb)
                frames.append(panel([rgb, labels, workspace, foreground], t))
                if t in (0, count // 2, count - 1):
                    imageio.imwrite(str(args.output / ("comparison_%03d.png" % t)), frames[-1])
                    for name, array in zip(("original", "segmentation", "keep_workspace", "keep_foreground"), (rgb, labels, workspace, foreground)):
                        imageio.imwrite(str(args.output / ("%s_%03d.png" % (name, t))), array)
                if t % 30 == 0:
                    print("Rendered", t, "/", count, "RGB MAE", errors[-1], flush=True)
            imageio.mimsave(str(args.output / "comparison.mp4"), frames, fps=20, macro_block_size=2)
            imageio.mimsave(str(args.output / "comparison.gif"), frames[::3], duration=0.15, loop=0)
            metadata = dict(dataset=str(args.dataset), demo=args.demo, frames=count,
                            room_geoms=[names[i] for i in room_ids], table_geoms=[names[i] for i in table_ids],
                            mean_replay_rgb_mae=float(np.mean(errors)), display_orientation="upright (raw MuJoCo render flipped vertically)",
                            note="Foreground preserves all non-room, non-table geometry, including distractor objects and fixtures.")
            (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
            print(json.dumps(metadata, indent=2), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
