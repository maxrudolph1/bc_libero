"""Geometry-based foreground masks shared by LIBERO dataset tools."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def localize_model_xml(xml):
    """Resolve demonstration asset paths against the installed local packages."""
    import robosuite
    from libero.libero import get_libero_path

    root = ET.fromstring(xml)
    for element in root.iter():
        filename = element.get("file")
        if not filename:
            continue
        if "/robosuite/" in filename:
            filename = str(Path(robosuite.__file__).parent / filename.split("/robosuite/", 1)[1])
        elif "/assets/" in filename:
            filename = str(Path(get_libero_path("assets")) / filename.split("/assets/", 1)[1])
        if not Path(filename).is_file():
            raise FileNotFoundError(filename)
        element.set("file", filename)
    return ET.tostring(root, encoding="unicode")


def background_geometry(model):
    """Remove room and any table; floor-workspace scenes need no table."""
    names = [model.geom_id2name(i) or "" for i in range(model.ngeom)]
    room = [i for i, name in enumerate(names) if name == "floor" or name.startswith("wall_")]
    table = [i for i, name in enumerate(names) if name.startswith("table")]
    if not room or "floor" not in names:
        raise ValueError("Expected floor/room geoms; inspect this scene before converting it")
    ids = room + table
    return ids, [names[i] for i in ids]


def foreground_mask(segmentation, background_ids):
    """True for retained geometry; renderer void pixels belong to background."""
    return (segmentation >= 0) & ~np.isin(segmentation, background_ids)


def composite_foreground(rgb, keep, video):
    if rgb.shape != video.shape or keep.shape != rgb.shape[:2]:
        raise ValueError("RGB, video, and foreground-mask dimensions must agree")
    return np.ascontiguousarray(np.where(keep[..., None], rgb, video), dtype=np.uint8)
