#!/usr/bin/env python
"""Check preserved source data and every foreground/background pixel without MuJoCo."""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libero_exp.utils.distractor_utils import DistractorSource


def same_attributes(source, output):
    for key in source.attrs:
        if key not in output.attrs or not np.array_equal(source.attrs[key], output.attrs[key]):
            raise AssertionError("Changed attribute: %s:%s" % (source.name, key))


def validate(path):
    checked = 0
    with h5py.File(path, "r") as output:
        build = json.loads(output.attrs["foreground_video_build"])
        video = DistractorSource(build["video_dir"])
        with h5py.File(build["source"], "r") as source:
            same_attributes(source, output)
            same_attributes(source["data"], output["data"])
            if output.attrs.get("foreground_video_complete", False):
                assert set(source["data"]) == set(output["data"]), "Missing demos in complete output"
            for name, dst in output["data"].items():
                src = source["data"][name]
                record = json.loads(dst.attrs["foreground_video"])

                def compare(key, obj):
                    candidate = dst[key]
                    same_attributes(obj, candidate)
                    if isinstance(obj, h5py.Dataset):
                        assert obj.shape == candidate.shape and obj.dtype == candidate.dtype, key
                        if key == "obs/agentview_rgb":
                            return
                        for start in range(0, len(obj), 64):
                            assert np.array_equal(obj[start:start + 64], candidate[start:start + 64]), key

                same_attributes(src, dst)
                src.visititems(compare)
                assert dst["foreground_mask"].shape == src["obs/agentview_rgb"].shape[:3]
                for t in range(record["frames"]):
                    mask = dst["foreground_mask"][t]
                    assert np.isin(mask, [0, 1]).all(), "Nonbinary mask"
                    keep = mask.astype(bool)
                    original = src["obs/agentview_rgb"][t]
                    composed = dst["obs/agentview_rgb"][t]
                    frame = video.load_frame(record["video_start"] + t)
                    if record["video_flip_vertical"]:
                        frame = frame[::-1]
                    assert np.array_equal(original[keep], composed[keep]), "Foreground changed"
                    assert np.array_equal(frame[~keep], composed[~keep]), "Wrong video background"
                    checked += 1
            print("VALIDATED %s: %d demos, %d frames; all source fields and composited pixels checked" %
                  (path, len(output["data"]), checked), flush=True)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.files:
        validate(path)


if __name__ == "__main__":
    main()
