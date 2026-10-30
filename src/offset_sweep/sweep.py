"""Shift a fixed-depth input window along the surface normal and score an ink model at every offset."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile
import zarr
from PIL import Image
from scipy.stats import rankdata

Image.MAX_IMAGE_PIXELS = None

# Same determinism settings as the runs this tool was validated against.
DETERMINISTIC_ENV = {"PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
DEFAULT_INFER_ARGS = ("--overlap", "0.5", "--blend-mode", "hann", "--batch-size", "4")


class SweepError(Exception):
    """A problem with the inputs that the user can fix; printed without a traceback."""


@dataclass(frozen=True)
class Window:
    offset: int
    start: int
    stop: int


def window_start(depth: int, window: int) -> int:
    """First layer of the offset-0 window: villa's centre crop (upper centre for odd excess), start = D//2 - W//2."""
    return depth // 2 - window // 2


def max_offset(depth: int, window: int) -> int:
    """Largest |offset| that keeps a `window`-layer window inside a `depth`-layer stack on both sides."""
    base = window_start(depth, window)
    return min(base, depth - window - base)


def fit_offsets(depth: int, window: int, offsets: Sequence[int]) -> tuple[list[int], str | None]:
    """Keep the requested offsets that fit; a note when the range had to shrink. Errors only when no shift fits."""
    if window <= 0:
        raise SweepError(f"window must be positive, got {window}")
    if depth < window:
        raise SweepError(f"the surface volume has {depth} layers but the model reads {window}; render at least "
                         f"{window} layers (see README, 'Two ways in')")
    n = max_offset(depth, window)
    kept = [o for o in offsets if abs(o) <= n]
    if len(offsets) > 1 and len(kept) < 2:
        raise SweepError(f"no offset is possible: {depth} layers and a {window}-layer window leave ±{n} voxels"
                         + (" (none of the requested offsets fits)" if n else "")
                         + f". Render at least {window + 2 * max(abs(o) for o in offsets)} layers yourself "
                           f"(see README, 'Two ways in').")
    note = None
    if len(kept) < len(offsets):
        note = (f"with these data only ±{n} voxels are possible ({depth} layers, model window {window}); running "
                f"offsets {kept[0]:+d}…{kept[-1]:+d}. For a larger range render more layers yourself "
                f"(see README, 'Two ways in').")
    return kept, note


def plan_windows(depth: int, window: int, offsets: Sequence[int]) -> list[Window]:
    """Window at offset 0 is exactly the centre crop villa's inference takes from the full stack."""
    base = window_start(depth, window)
    out = [Window(o, base + o, base + o + window) for o in offsets]
    if any(w.start < 0 or w.stop > depth for w in out):
        raise SweepError(f"offsets {list(offsets)} do not fit {depth} layers with a {window}-layer window; "
                         f"at most ±{max(0, max_offset(depth, window))}")
    return out


def model_depth(checkpoint: Path) -> int:
    """Input depth (layers) a villa ink checkpoint reads, from the training config stored in it."""
    try:
        from vesuvius.ink_detection.models.checkpoint import config_from_checkpoint, load_checkpoint
        return int(config_from_checkpoint(load_checkpoint(checkpoint), source=checkpoint).model.crop_size[0])
    except Exception as e:  # noqa: BLE001 - reported to the user as one line
        raise SweepError(f"cannot read the input depth from {checkpoint} ({type(e).__name__}: {e}); pass --window") from None


def parse_offsets(text: str) -> list[int]:
    """'-12:12:2' (start:stop:step, stop included) or a comma list '-8,-4,0,4,8'."""
    try:
        if ":" in text:
            a, b, s = (int(x) for x in text.split(":"))
            if s <= 0 or b < a:
                raise ValueError
            vals = list(range(a, b + 1, s))
        else:
            vals = sorted({int(x) for x in text.split(",") if x.strip()})
    except ValueError:
        raise SweepError(f"cannot read --offsets {text!r}; use start:stop:step such as -12:12:2, or -8,-4,0,4,8") from None
    if 0 not in vals:
        raise SweepError("--offsets must include 0 (the reference window)")
    return vals


def _is_zarr(path: Path) -> bool:
    return any((Path(path) / f).exists() for f in (".zarray", ".zgroup", "zarr.json"))


def layer_tiffs(path: Path) -> list[Path]:
    """Layer files of a TIFF surface volume (00.tif, 01.tif, ...), in numeric order."""
    files = [f for f in Path(path).iterdir() if f.suffix.lower() in (".tif", ".tiff")]
    try:
        return sorted(files, key=lambda f: int(f.stem))
    except ValueError:
        raise SweepError(f"{path}: layer TIFFs must be named by layer number (00.tif, 01.tif, ...)") from None


def open_stack(path: Path):
    """(layers, H, W) surface volume: a Zarr (bare array, or group with array '0' = OME-Zarr level 0) or a folder of
    layer TIFFs. TIFF layers are read into memory."""
    path = Path(path)
    if not path.exists():
        raise SweepError(f"surface volume not found: {path}")
    if path.is_dir() and not _is_zarr(path):
        files = layer_tiffs(path)
        if not files:
            raise SweepError(f"{path} is neither a Zarr nor a folder of layer TIFFs")
        layers = [tifffile.imread(f) for f in files]
        if len({a.shape for a in layers}) != 1 or layers[0].ndim != 2:
            raise SweepError(f"{path}: layer TIFFs must all be 2-D images of the same size")
        if layers[0].dtype != np.uint8:
            raise SweepError(f"{path}: layer TIFFs are {layers[0].dtype}; villa's ink models expect 8-bit layers")
        return np.stack(layers)
    node = zarr.open(str(path), mode="r")
    if isinstance(node, zarr.Group):
        if "0" not in node:
            raise SweepError(f"{path} is a Zarr group without array '0'; pass the full-resolution array")
        node = node["0"]
    if node.ndim != 3:
        raise SweepError(f"{path} has shape {node.shape}; expected (layers, height, width)")
    return node


def read_image(path: Path) -> np.ndarray:
    if not Path(path).exists():
        raise SweepError(f"file not found: {path}")
    if str(path).lower().endswith((".tif", ".tiff")):
        a = tifffile.imread(path)
    else:
        a = np.asarray(Image.open(path).convert("L"))
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] in (3, 4) else a[0]
    return a


def binarize(a: np.ndarray) -> np.ndarray:
    """uint8/uint16 labels: > half of the type range; float labels: > 0.5; bool as is."""
    if a.dtype == bool:
        return a
    if np.issubdtype(a.dtype, np.integer):
        return a > np.iinfo(a.dtype).max // 2
    return a > 0.5


def load_labels(ink_path: Path, mask_path: Path | None, shape: tuple[int, int]):
    ink = binarize(read_image(ink_path))
    mask = binarize(read_image(mask_path)) if mask_path else np.ones_like(ink)
    for name, a in (("ink labels", ink), ("label mask", mask)):
        if a.shape != shape:
            r = (a.shape[0] / shape[0], a.shape[1] / shape[1])
            hint = ""
            if abs(r[0] - r[1]) < 0.01 and abs(r[0] - 1) > 0.01:
                hint = (f" The ratio is {r[0]:.3f} in both axes: the labels were probably made at another resolution or "
                        f"render scale. Resample them onto the render grid, or render with the matching --scale.")
            raise SweepError(f"{name} are {a.shape[1]} x {a.shape[0]} px but the surface volume is "
                             f"{shape[1]} x {shape[0]} px.{hint}")
    return ink, mask


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC (Mann-Whitney, ties averaged); equal to sklearn.metrics.roc_auc_score."""
    y = np.asarray(y, bool)
    n1 = int(y.sum()); n0 = y.size - n1
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def write_window(stack, w: Window, path: Path, reverse: bool) -> None:
    a = np.ascontiguousarray(np.asarray(stack[w.start:w.stop]))
    if reverse:
        a = np.ascontiguousarray(a[::-1])
    shutil.rmtree(path, ignore_errors=True)
    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    g.create_array("0", data=a, chunks=(a.shape[0], 128, 128), compressors=None)


def villa_command(volume: Path, checkpoint: Path, out: Path, w: Window, direction: str, extra: Sequence[str]):
    return [sys.executable, "-m", "vesuvius.ink_detection.inference.infer", str(volume), str(checkpoint), str(out),
            "--layer-start", str(w.start), "--layer-end", str(w.stop), "--direction", direction, *extra]


# Packages villa's model code imports but installs only with its heavy [models] extra (versions within villa's bounds).
MODEL_PACKAGES = "einops==0.8.2 timm==1.0.27 imagecodecs==2026.6.26"


def check_villa(checkpoint=None):
    """Fail early, in one line, if villa's inference is missing, too old, cannot write its output or build the model."""
    try:
        from vesuvius.ink_detection.inference import infer
    except Exception as e:  # noqa: BLE001 - any import failure means the same thing to the user
        raise SweepError(f"villa's ink inference is not importable ({type(e).__name__}: {e}). Install it as described "
                         f"in the README, or use --command for a model outside villa.") from None
    if "layer_start" not in infer.parse_args(["x", "y", "z"]).__dict__:
        raise SweepError("the installed villa inference has no --layer-start; install the villa commit named in the README")
    try:                                   # villa writes its probability maps LZW-compressed, which tifffile does via imagecodecs
        tifffile.imwrite(io.BytesIO(), np.zeros((1, 1), np.uint8), compression="lzw")
    except (KeyError, ValueError) as e:
        raise SweepError(f"villa cannot write its output TIFF ({e.args[0] if e.args else e}). "
                         f"Install the model packages with: uv pip install {MODEL_PACKAGES} (see README, 'Install')") from None
    if checkpoint is None:
        return
    try:                                   # builds on the CPU; the model is only constructed, not run
        with contextlib.redirect_stdout(io.StringIO()):          # villa prints its whole network config
            infer.configure_single_model(Path(checkpoint), amp_dtype="default")
    except ModuleNotFoundError as e:
        raise SweepError(f"villa cannot build the model in {checkpoint}: Python package '{e.name}' is missing. "
                         f"Install the model packages with: uv pip install {MODEL_PACKAGES} (see README, 'Install')") from None
    except Exception as e:  # noqa: BLE001
        raise SweepError(f"villa cannot load {checkpoint} ({type(e).__name__}: {e})") from None


LOW_AUC = 0.6          # AUC at offset 0 below this: warn
ORDER_MARGIN = 0.05    # opposite layer order better by more than this: warn


def run_sweep(*, volume: Path, checkpoint: Path | None, labels: Path, mask: Path | None, out: Path,
              offsets: Sequence[int], window: int | None = None, direction: str = "forward", command: str | None = None,
              reverse_layers: bool = False, infer_args: Sequence[str] = DEFAULT_INFER_ARGS, outward: str = "unknown",
              model_name: str | None = None, order_check: bool = True, log=print, warn=None) -> list[dict]:
    warn = warn or (lambda m: print(f"offset-sweep: WARNING: {m}", file=sys.stderr))
    stack = open_stack(volume)
    depth, H, W = stack.shape
    if window is None:
        if command is not None:
            raise SweepError("with --command, give --window (the number of layers your model reads)")
        window = model_depth(checkpoint)
    offsets, note = fit_offsets(depth, window, offsets)
    if note:
        warn(note)
    wins = plan_windows(depth, window, offsets)
    ink, mask_arr = load_labels(labels, mask, (H, W))
    if command is None:
        if checkpoint is None or not Path(checkpoint).exists():
            raise SweepError(f"checkpoint not found: {checkpoint}")
        check_villa(checkpoint)
    elif "{input}" not in command or "{output}" not in command:
        raise SweepError("--command must contain {input} and {output}, e.g. 'python predict.py {input} {output}'")
    out.mkdir(parents=True, exist_ok=True)
    model_name = model_name or (Path(checkpoint).stem if checkpoint else "model")
    env = {**os.environ, **DETERMINISTIC_ENV}
    staged = None
    if command is None and isinstance(stack, np.ndarray):          # layer TIFFs: villa inference reads Zarr
        staged = out / "_stack.zarr"
        shutil.rmtree(staged, ignore_errors=True)
        g = zarr.open_group(str(staged), mode="w", zarr_format=2)
        g.create_array("0", data=stack, chunks=(depth, 128, 128), compressors=None)
        volume = staged

    def infer(w: Window, tif: Path, flip: bool) -> float | str:
        """Run the model on window w; flip = feed the opposite layer order to the one requested."""
        if tif.exists():
            return ""
        t0 = time.time()
        logf = tif.with_suffix(".log")
        tmp = None
        if command is None:
            d = direction if not flip else ("reverse" if direction == "forward" else "forward")
            cmd = villa_command(volume, checkpoint, tif, w, d, infer_args)
        else:
            tmp = out / f"_window_{tif.stem}.zarr"
            write_window(stack, w, tmp, reverse_layers != flip)
            cmd = [a.format(input=tmp, output=tif, checkpoint=checkpoint or "") for a in shlex.split(command)]
        log(f"offset {w.offset:+d}: layers {w.start}-{w.stop - 1}" + (" (opposite layer order)" if flip else ""))
        with open(logf, "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
        if rc != 0 or not tif.exists():
            raise SweepError(f"inference failed at offset {w.offset:+d} (exit {rc}); see {logf}")
        return round(time.time() - t0, 1)

    def load(tif: Path, w: Window) -> np.ndarray:
        p = tifffile.imread(tif).astype(np.float32)
        if p.shape != (H, W):
            raise SweepError(f"prediction at offset {w.offset:+d} is {p.shape[1]} x {p.shape[0]} px, surface volume "
                             f"{W} x {H}; the model must predict on the input grid")
        return p

    try:
        preds, valid, times = {}, np.ones((H, W), bool), {}
        for w in wins:
            tif = out / f"pred_o{w.offset:+d}.tif"
            # Valid = every layer of this window rendered (nonzero somewhere along the normal).
            valid &= np.asarray(stack[w.start:w.stop]).max(0) > 0
            times[w.offset] = infer(w, tif, False)
            preds[w.offset] = load(tif, w)
        w0 = next(w for w in wins if w.offset == 0)
        other = None
        if order_check:
            tif = out / "pred_o+0_opposite_order.tif"
            infer(w0, tif, True)
            other = load(tif, w0)
    finally:
        if staged is not None:
            shutil.rmtree(staged, ignore_errors=True)

    sel = valid & mask_arr
    y = ink[sel]
    if sel.sum() == 0:
        raise SweepError("no labelled pixel lies inside the rendered area (label mask ∩ valid render is empty)")
    if y.all() or not y.any():
        raise SweepError(f"the evaluated area has {'only ink' if y.all() else 'no ink'} pixels; AUC needs both. "
                         f"Check the label file and threshold (labels > half range count as ink).")
    rows = []
    a0 = auc(y, preds[0][sel])
    for w in wins:
        a = auc(y, preds[w.offset][sel])
        rows.append({"model": model_name, "offset": w.offset, "side": side(w.offset, outward),
                     "outward_voxels": outward_voxels(w.offset, outward),
                     "layer_start": w.start, "layer_stop": w.stop, "auc": round(a, 6), "delta_auc": round(a - a0, 6),
                     "pixels": int(sel.sum()), "ink_pixels": int(y.sum()), "seconds": times.get(w.offset, "")})
    warnings = plausibility(a0, auc(y, other[sel]) if other is not None else None)
    for m in warnings:
        warn(m)
    write_csv(out / "curve.csv", rows)
    (out / "run.json").write_text(json.dumps({
        "volume": str(volume if staged is None else Path(volume).parent), "depth": depth, "window": window,
        "offsets": list(offsets), "direction": direction, "command": command, "reverse_layers": reverse_layers,
        "infer_args": list(infer_args), "outward": outward, "checkpoint": str(checkpoint) if checkpoint else None,
        "labels": str(labels), "mask": str(mask) if mask else None, "auc_offset0": round(a0, 6),
        "auc_offset0_opposite_order": round(auc(y, other[sel]), 6) if other is not None else None,
        "warnings": warnings, "offset_range_note": note}, indent=1))
    return rows


def plausibility(a0: float, a_other: float | None) -> list[str]:
    """Warnings, not errors: a curve from a misconfigured run is still written, but should not be trusted."""
    hint = "check layer order (--direction / --reverse-layers) and scale; see README, 'Plausibility warnings'"
    out = []
    if a0 < LOW_AUC:
        out.append(f"AUC at offset 0 is {a0:.3f}, close to chance (< {LOW_AUC}); {hint}")
    if a_other is not None and a_other > a0 + ORDER_MARGIN:
        out.append(f"the opposite layer order scores higher at offset 0 (AUC {a_other:.3f} vs {a0:.3f}); {hint}")
    return out


def side(offset: int, outward: str) -> str:
    if offset == 0:
        return "on surface"
    if outward == "unknown":
        return "higher layers" if offset > 0 else "lower layers"
    up = "outside" if outward == "higher" else "inside"
    down = "inside" if outward == "higher" else "outside"
    return up if offset > 0 else down


def outward_voxels(offset: int, outward: str):
    """Offset re-signed so that positive always points away from the scroll centre; empty when unknown."""
    return "" if outward == "unknown" else (offset if outward == "higher" else -offset)


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))
