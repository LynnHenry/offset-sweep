"""offset-sweep command line: `offset-sweep run ...` and `offset-sweep plot ...`."""
from __future__ import annotations

import argparse
import shlex
import shutil
import sys
from pathlib import Path

from .sweep import DEFAULT_INFER_ARGS, SweepError, parse_offsets, read_csv, run_sweep


def crop_labels(labels: Path, mask: Path | None, crop, out: Path):
    """Cut label images to the render crop; full-render label size is checked against the crop box."""
    import numpy as np
    from PIL import Image
    from .sweep import read_image
    x, y, w, h = crop
    res = []
    for p in (labels, mask):
        if p is None:
            res.append(None); continue
        a = read_image(p)
        if x + w > a.shape[1] or y + h > a.shape[0]:
            raise SweepError(f"--crop {crop} reaches beyond {p} ({a.shape[1]} x {a.shape[0]} px)")
        dst = out / f"_crop_{Path(p).stem}.png"
        Image.fromarray(np.ascontiguousarray(a[y:y + h, x:x + w]).astype(np.uint8) if a.dtype != bool else
                        (a[y:y + h, x:x + w] * 255).astype(np.uint8)).save(dst)
        res.append(dst)
    return res[0], res[1]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="offset-sweep", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="compute the offset curve of one model on one labelled surface")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--surface-volume", type=Path,
                     help="existing surface volume with enough layers: a folder of layer TIFFs (00.tif, 01.tif, ...) "
                          "or a Zarr (layers, H, W); no VC3D needed")
    src.add_argument("--tifxyz", type=Path, help="tifxyz surface folder; rendered first with vc_render_tifxyz")
    r.add_argument("--volume", help="CT volume URL or path for --tifxyz (the scan the mesh was traced on)")
    r.add_argument("--vc3d", help="VC3D AppImage for --tifxyz rendering")
    r.add_argument("--linear", action="store_true", help="render with --surface-interpolation linear")
    r.add_argument("--crop", help="X,Y,W,H in pixels of the full-surface grid: with --tifxyz only this region is rendered; "
                                 "with --surface-volume the volume is that region. Full-size labels are cut the same way")
    r.add_argument("--keep-render", action="store_true", help="keep the rendered Zarr (deleted by default)")
    r.add_argument("--labels", type=Path, required=True, help="ink labels on the render grid (png/tif; > half range = ink)")
    r.add_argument("--label-mask", type=Path, help="pixels to evaluate (png/tif); default: all labelled pixels")
    r.add_argument("--checkpoint", type=Path, help="villa ink checkpoint (.pth); not needed with --command")
    r.add_argument("--command", help="run a model outside villa: shell-style template with {input} (window Zarr) "
                                     "and {output} (prediction tif), optionally {checkpoint}")
    r.add_argument("--offsets", default="-12:12:2", help="start:stop:step (stop included) or a list; default -12:12:2")
    r.add_argument("--window", type=int, help="layers the model reads; villa checkpoints: read from the checkpoint "
                                             "(ink_9um and Reader V3: 17); required with --command (v8-in: 24)")
    r.add_argument("--direction", choices=("forward", "reverse"), default="forward",
                   help="villa: layer order fed to the model (reverse = high layer index first)")
    r.add_argument("--reverse-layers", action="store_true",
                   help="--command models: flip each window before writing it (for models trained on reversed order)")
    r.add_argument("--outward", choices=("higher", "lower", "unknown"), required=True,
                   help="which way the layer index points away from the scroll centre (vc_render_tifxyz: along the "
                        "mesh normal); 'unknown' labels the axis by layer index only")
    r.add_argument("--infer-args", default=" ".join(DEFAULT_INFER_ARGS),
                   help=f"extra arguments for villa inference; default '{' '.join(DEFAULT_INFER_ARGS)}'")
    r.add_argument("--no-order-check", dest="order_check", action="store_false",
                   help="skip the extra offset-0 run with the opposite layer order (plausibility warning)")
    r.add_argument("--model-name", help="name in CSV and legend; default: checkpoint file name")
    r.add_argument("--out", type=Path, required=True, help="output folder: curve.csv, curve.png, predictions, logs")

    p = sub.add_parser("plot", help="draw one figure from one or more curve.csv files")
    p.add_argument("csv", type=Path, nargs="+")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--title")
    p.add_argument("--xlim", help="e.g. -8:8")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "plot":
            from .plot import plot
            xl = tuple(int(v) for v in a.xlim.split(":")) if a.xlim else None
            plot({str(c): read_csv(c) for c in a.csv}, a.output, a.title, xl)
            print(a.output)
            return 0
        offsets = parse_offsets(a.offsets)
        if a.command and a.checkpoint is None and "{checkpoint}" in a.command:
            raise SweepError("--command uses {checkpoint} but --checkpoint is not set")
        if not a.command and a.checkpoint is None:
            raise SweepError("give --checkpoint (villa model) or --command (other model)")
        a.out.mkdir(parents=True, exist_ok=True)
        crop = None
        if a.crop:
            try:
                crop = tuple(int(v) for v in a.crop.split(","))
                assert len(crop) == 4 and min(crop) >= 0 and crop[2] > 0 and crop[3] > 0
            except (ValueError, AssertionError):
                raise SweepError(f"cannot read --crop {a.crop!r}; use X,Y,W,H such as 2592,768,3000,3000") from None
            a.labels, a.label_mask = crop_labels(a.labels, a.label_mask, crop, a.out)
        volume = a.surface_volume
        if a.tifxyz:
            if not a.volume or not a.vc3d:
                raise SweepError("--tifxyz needs --volume (CT scan) and --vc3d (VC3D AppImage)")
            if a.window is None and a.command:
                raise SweepError("with --command, give --window (the number of layers your model reads)")
            from .render import check_tifxyz, render
            from .sweep import model_depth
            check_tifxyz(a.tifxyz)
            a.window = a.window or model_depth(a.checkpoint)
            layers = a.window + 2 * max(abs(o) for o in offsets)
            # An even layer count puts the layers at half-integer distances from the mesh (±0.5, ±1.5, ...), as in
            # the 28- and 64-layer renders the published curves come from; the offset-0 window is then the same.
            layers += layers % 2
            volume = render(a.vc3d, a.tifxyz, a.volume, a.out / "_render.zarr", layers, linear=a.linear, crop=crop)
        rows = run_sweep(volume=volume, checkpoint=a.checkpoint, labels=a.labels, mask=a.label_mask, out=a.out,
                         offsets=offsets, window=a.window, direction=a.direction, command=a.command,
                         reverse_layers=a.reverse_layers, infer_args=shlex.split(a.infer_args), outward=a.outward,
                         model_name=a.model_name, order_check=a.order_check)
        from .plot import plot
        plot({"run": rows}, a.out / "curve.png", a.model_name or "offset curve")
        if a.tifxyz and not a.keep_render:
            shutil.rmtree(a.out / "_render.zarr")
        for r in rows:
            print(f"{r['offset']:+4d}  {r['side']:<12} AUC {r['auc']:.4f}  Δ {r['delta_auc']:+.4f}")
        print(f"wrote {a.out / 'curve.csv'} and {a.out / 'curve.png'}")
        return 0
    except SweepError as e:
        print(f"offset-sweep: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
