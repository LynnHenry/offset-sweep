"""Optional render step: tifxyz surface + CT volume -> surface-volume Zarr with vc_render_tifxyz (VC3D)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import tifffile

from .sweep import SweepError


def check_tifxyz(path: Path) -> dict:
    path = Path(path)
    missing = [f for f in ("x.tif", "y.tif", "z.tif", "meta.json") if not (path / f).exists()]
    if missing:
        raise SweepError(f"{path} is not a tifxyz folder (missing {', '.join(missing)})")
    meta = json.loads((path / "meta.json").read_text())
    scale = meta.get("scale")
    if not scale or len(scale) != 2 or min(scale) <= 0:
        raise SweepError(f"{path}/meta.json has no valid 'scale' ({scale!r}); vc_render_tifxyz needs it")
    return meta


def expected_size(path: Path, meta: dict) -> tuple[int, int]:
    """Render size at --scale 1: grid size divided by the grid's scale (pixels per voxel)."""
    h, w = tifffile.imread(Path(path) / "x.tif").shape
    return int(round(h / meta["scale"][1])), int(round(w / meta["scale"][0]))


def render(vc: str, tifxyz: Path, volume: str, out: Path, layers: int, *, linear: bool = False, cache_gb: int = 8,
           crop: tuple[int, int, int, int] | None = None, log=print) -> Path:
    """vc: path to the VC3D AppImage (or its extracted AppRun); the first argument selects the tool."""
    meta = check_tifxyz(tifxyz)
    if shutil.which(vc) is None and not Path(vc).exists():
        raise SweepError(f"VC3D not found at {vc!r}; download the AppImage (README, step 2) and pass its path with --vc3d")
    out = Path(out)
    if (out / ".done").exists():
        log(f"render exists, reusing {out}")
        return out
    shutil.rmtree(out, ignore_errors=True)
    cmd = [vc, "vc_render_tifxyz", "--volume", volume, "--group-idx", "0", "--scale", "1", "--segmentation", str(tifxyz),
           "--num-slices", str(layers), "--slice-step", "1", "--cache-gb", str(cache_gb)]
    if linear:
        cmd += ["--surface-interpolation", "linear"]
    if crop:
        cmd += ["--crop-x", str(crop[0]), "--crop-y", str(crop[1]), "--crop-width", str(crop[2]), "--crop-height", str(crop[3])]
    cmd += ["--zarr-output", str(out), "--pyramid", "0"]
    log("render: " + " ".join(cmd))
    with open(out.parent / f"{out.name}.log", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise SweepError(f"vc_render_tifxyz failed (exit {rc}); see {out.parent / (out.name + '.log')}")
    from .sweep import open_stack
    st = open_stack(out)
    exp = (crop[3], crop[2]) if crop else expected_size(tifxyz, meta)
    if st.shape[0] != layers:
        raise SweepError(f"render has {st.shape[0]} layers, asked for {layers}")
    if np.max(np.abs(np.array(st.shape[1:]) - np.array(exp)) / np.array(exp)) > 0.02:
        raise SweepError(f"render is {st.shape[2]} x {st.shape[1]} px, expected about {exp[1]} x {exp[0]} from the "
                         f"crop or the tifxyz grid and its scale {meta['scale']}; check meta.json 'scale'")
    (out / ".done").write_text("ok")
    return out
