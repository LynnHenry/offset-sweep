import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr
from PIL import Image

from offset_sweep import cli
from offset_sweep.sweep import SweepError, auc, fit_offsets, parse_offsets, plan_windows, run_sweep


def make_volume(path: Path, depth=40, h=48, w=64, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 255, (depth, h, w), dtype=np.uint8)
    a[:, :, :4] = 0                                   # unrendered strip
    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    g.create_array("0", data=a, chunks=(depth, 32, 32))
    return a


def make_labels(d: Path, h=48, w=64):
    ink = np.zeros((h, w), np.uint8); ink[10:30, 20:40] = 255
    mask = np.zeros((h, w), np.uint8); mask[4:44, 8:60] = 255
    Image.fromarray(ink).save(d / "ink.png"); Image.fromarray(mask).save(d / "mask.png")
    return ink > 127, mask > 127


# A stand-in model: "ink" = brightness of the first layer of the window it receives.
FAKE = """import sys, numpy as np, tifffile, zarr
a = np.asarray(zarr.open_group(sys.argv[1], mode="r")["0"][:]).astype(np.float32)
tifffile.imwrite(sys.argv[2], a[0] / 255)
"""


def test_offsets_parse():
    assert parse_offsets("-4:4:2") == [-4, -2, 0, 2, 4]
    assert parse_offsets("4,-4,0") == [-4, 0, 4]
    with pytest.raises(SweepError, match="include 0"):
        parse_offsets("1,2")
    with pytest.raises(SweepError, match="cannot read"):
        parse_offsets("a:b")


def test_window_zero_is_villa_centre_crop():
    w = plan_windows(64, 28, [-12, 0, 12])
    assert [(x.start, x.stop) for x in w] == [(6, 34), (18, 46), (30, 58)]
    assert [(x.start, x.stop) for x in plan_windows(64, 17, [0])] == [(24, 41)]
    infer = pytest.importorskip("vesuvius.ink_detection.inference.infer")
    for depth, window in ((64, 28), (64, 17), (42, 17), (28, 17), (24, 17), (41, 24)):
        full = infer.select_layer_indices(depth, layer_start=None, layer_end=None, output_depth=window, direction="forward")
        (w0,) = plan_windows(depth, window, [0])
        assert list(full) == list(range(w0.start, w0.stop)), (depth, window)
    # villa centre-crops a 28-layer window for a 17-layer model to exactly the 17-layer window used here
    for o in (-12, 0, 12):
        (w28,) = plan_windows(64, 28, [o]); (w17,) = plan_windows(64, 17, [o])
        crop = infer.select_layer_indices(64, layer_start=w28.start, layer_end=w28.stop, output_depth=17, direction="forward")
        assert list(crop) == list(range(w17.start, w17.stop))


def test_offset_range_shrinks_to_what_fits():
    kept, note = fit_offsets(28, 17, list(range(-12, 13, 2)))      # official PHerc0841 surface volume, ink_9um
    assert kept == [-4, -2, 0, 2, 4] and "only ±5 voxels" in note and "render more layers" in note
    kept, note = fit_offsets(24, 17, list(range(-12, 13, 2)))      # PHerc1447 layers on Hugging Face
    assert kept == [-2, 0, 2] and "only ±3" in note
    kept, note = fit_offsets(42, 17, list(range(-12, 13, 2)))
    assert kept == list(range(-12, 13, 2)) and note is None


def test_no_offset_possible_is_an_error():
    with pytest.raises(SweepError, match="no offset is possible"):
        fit_offsets(17, 17, [-2, 0, 2])
    with pytest.raises(SweepError, match="model reads 28"):
        fit_offsets(24, 28, [0, 2])
    assert fit_offsets(17, 17, [0]) == ([0], None)                 # asking for offset 0 alone is allowed


def test_sweep_runs_on_the_range_that_fits(tmp_path):
    make_volume(tmp_path / "v.zarr", depth=12); make_labels(tmp_path)
    script = tmp_path / "fake.py"; script.write_text(FAKE)
    msgs = []
    rows = run_sweep(volume=tmp_path / "v.zarr", checkpoint=None, labels=tmp_path / "ink.png", mask=tmp_path / "mask.png",
                     out=tmp_path / "o", offsets=[-4, -2, 0, 2, 4], window=8, order_check=False, warn=msgs.append,
                     command=f"{sys.executable} {script} {{input}} {{output}}")
    assert [r["offset"] for r in rows] == [-2, 0, 2]
    assert any("only ±2 voxels" in m for m in msgs)
    assert "only ±2 voxels" in json.loads((tmp_path / "o" / "run.json").read_text())["offset_range_note"]


def test_command_needs_window(tmp_path):
    make_volume(tmp_path / "v.zarr"); make_labels(tmp_path)
    with pytest.raises(SweepError, match="give --window"):
        run_sweep(volume=tmp_path / "v.zarr", checkpoint=None, labels=tmp_path / "ink.png", mask=None,
                  out=tmp_path / "o", offsets=[0, 2], command="x {input} {output}")


def test_auc_matches_sklearn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    y = rng.random(5000) < 0.3
    s = np.round(rng.random(5000) + y * 0.3, 2)          # many ties
    assert auc(y, s) == pytest.approx(metrics.roc_auc_score(y, s), abs=1e-12)


def test_label_shape_error_names_ratio(tmp_path):
    make_volume(tmp_path / "v.zarr")
    Image.fromarray(np.zeros((96, 128), np.uint8)).save(tmp_path / "ink.png")
    with pytest.raises(SweepError, match="ratio is 2.000"):
        run_sweep(volume=tmp_path / "v.zarr", checkpoint=None, labels=tmp_path / "ink.png", mask=None,
                  out=tmp_path / "o", offsets=[0], window=8, command="x {input} {output}")


def test_no_ink_error(tmp_path):
    make_volume(tmp_path / "v.zarr")
    Image.fromarray(np.zeros((48, 64), np.uint8)).save(tmp_path / "ink.png")
    script = tmp_path / "fake.py"; script.write_text(FAKE)
    with pytest.raises(SweepError, match="no ink"):
        run_sweep(volume=tmp_path / "v.zarr", checkpoint=None, labels=tmp_path / "ink.png", mask=None,
                  out=tmp_path / "o", offsets=[0], window=8, command=f"{sys.executable} {script} {{input}} {{output}}")


def test_sweep_end_to_end_with_command_model(tmp_path):
    vol = make_volume(tmp_path / "v.zarr")
    ink, mask = make_labels(tmp_path)
    script = tmp_path / "fake.py"; script.write_text(FAKE)
    cmd = f"{sys.executable} {script} {{input}} {{output}}"
    rows = run_sweep(volume=tmp_path / "v.zarr", checkpoint=None, labels=tmp_path / "ink.png", mask=tmp_path / "mask.png",
                     out=tmp_path / "o", offsets=[-4, 0, 4], window=8, command=cmd, outward="lower", model_name="fake")
    sel = mask & (vol[12:28].max(0) > 0)                 # common valid over windows 12-19, 16-23, 20-27
    for r, first in zip(rows, (12, 16, 20)):
        assert r["auc"] == pytest.approx(auc(ink[sel], vol[first][sel]), abs=1e-6)
    assert [r["outward_voxels"] for r in rows] == [4, 0, -4]
    assert [r["side"] for r in rows] == ["outside", "on surface", "inside"]
    # --reverse-layers hands the model the window back to front: its first layer is the window's last.
    rev = run_sweep(volume=tmp_path / "v.zarr", checkpoint=None, labels=tmp_path / "ink.png", mask=tmp_path / "mask.png",
                    out=tmp_path / "r", offsets=[0], window=8, command=cmd, reverse_layers=True)
    assert rev[0]["auc"] == pytest.approx(auc(ink[mask & (vol[16:24].max(0) > 0)], vol[23][mask & (vol[16:24].max(0) > 0)]), abs=1e-6)
    assert not any(p.name.startswith("_window") for p in (tmp_path / "o").iterdir())


def test_cli_writes_csv_and_png(tmp_path, capsys):
    make_volume(tmp_path / "v.zarr"); make_labels(tmp_path)
    script = tmp_path / "fake.py"; script.write_text(FAKE)
    rc = cli.main(["run", "--surface-volume", str(tmp_path / "v.zarr"), "--labels", str(tmp_path / "ink.png"),
                   "--command", f"{sys.executable} {script} {{input}} {{output}}", "--offsets", "-2:2:2",
                   "--window", "8", "--outward", "higher", "--out", str(tmp_path / "o")])
    assert rc == 0
    rows = list(csv.DictReader(open(tmp_path / "o" / "curve.csv")))
    assert [r["side"] for r in rows] == ["inside", "on surface", "outside"]
    assert (tmp_path / "o" / "curve.png").stat().st_size > 10_000
    assert cli.main(["plot", str(tmp_path / "o" / "curve.csv"), "-o", str(tmp_path / "p.png")]) == 0


def test_cli_errors_are_short(tmp_path, capsys):
    make_volume(tmp_path / "v.zarr", depth=20); make_labels(tmp_path)
    rc = cli.main(["run", "--surface-volume", str(tmp_path / "v.zarr"), "--labels", str(tmp_path / "ink.png"),
                   "--command", "x {input} {output}", "--outward", "unknown", "--out", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert rc == 2 and "give --window" in err and "Traceback" not in err
    rc = cli.main(["run", "--surface-volume", str(tmp_path / "v.zarr"), "--labels", str(tmp_path / "ink.png"),
                   "--command", "x {input} {output}", "--window", "20", "--outward", "unknown", "--out", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert rc == 2 and "no offset is possible" in err and "Traceback" not in err
    rc = cli.main(["run", "--surface-volume", str(tmp_path / "v.zarr"), "--labels", str(tmp_path / "ink.png"),
                   "--outward", "unknown", "--out", str(tmp_path / "o")])
    assert rc == 2 and "--checkpoint" in capsys.readouterr().err
    tif = tmp_path / "t"; tif.mkdir()
    rc = cli.main(["run", "--tifxyz", str(tif), "--volume", "v", "--vc3d", "/nonexistent/vc", "--labels",
                   str(tmp_path / "ink.png"), "--checkpoint", str(tmp_path / "ink.png"), "--outward", "higher",
                   "--out", str(tmp_path / "o2")])
    assert rc == 2 and "not a tifxyz folder" in capsys.readouterr().err


def test_layer_tiff_folder_input(tmp_path):
    vol = make_volume(tmp_path / "v.zarr")
    (tmp_path / "layers").mkdir()
    for i, a in enumerate(vol):
        tifffile.imwrite(tmp_path / "layers" / f"{i:02d}.tif", a)
    make_labels(tmp_path)
    script = tmp_path / "fake.py"; script.write_text(FAKE)
    kw = dict(checkpoint=None, labels=tmp_path / "ink.png", mask=tmp_path / "mask.png", offsets=[-4, 0, 4], window=8,
              command=f"{sys.executable} {script} {{input}} {{output}}", order_check=False)
    a = run_sweep(volume=tmp_path / "layers", out=tmp_path / "a", **kw)
    b = run_sweep(volume=tmp_path / "v.zarr", out=tmp_path / "b", **kw)
    assert [r["auc"] for r in a] == [r["auc"] for r in b]


def test_layer_tiff_names_must_be_numbers(tmp_path):
    (tmp_path / "layers").mkdir()
    tifffile.imwrite(tmp_path / "layers" / "first.tif", np.zeros((4, 4), np.uint8))
    with pytest.raises(SweepError, match="named by layer number"):
        run_sweep(volume=tmp_path / "layers", checkpoint=None, labels=tmp_path / "x.png", mask=None,
                  out=tmp_path / "o", offsets=[0], window=1, command="x {input} {output}")


# Stand-in model whose "ink" is the LAST layer it receives: a wrong layer order for the labels below.
LAST = FAKE.replace("a[0]", "a[-1]")


def test_plausibility_warnings(tmp_path):
    vol = make_volume(tmp_path / "v.zarr", depth=8)
    ink = vol[0] > 128                                   # ink visible in layer 0 only
    Image.fromarray((ink * 255).astype(np.uint8)).save(tmp_path / "ink.png")
    script = tmp_path / "last.py"; script.write_text(LAST)
    msgs = []
    rows = run_sweep(volume=tmp_path / "v.zarr", checkpoint=None, labels=tmp_path / "ink.png", mask=None,
                     out=tmp_path / "o", offsets=[0], window=8, command=f"{sys.executable} {script} {{input}} {{output}}",
                     warn=msgs.append)
    assert rows[0]["auc"] < 0.6
    assert any("close to chance" in m for m in msgs) and any("opposite layer order" in m for m in msgs)
    assert all("README" in m for m in msgs)
    assert json.loads((tmp_path / "o" / "run.json").read_text())["warnings"] == msgs


def test_crop_cuts_full_size_labels_for_a_cropped_volume(tmp_path):
    vol = make_volume(tmp_path / "v.zarr", depth=16)                 # the "cropped" volume, 48 x 64 px
    ink, mask = make_labels(tmp_path)
    big_ink = np.zeros((100, 120), np.uint8); big_ink[20:68, 30:94] = ink * 255
    big_mask = np.zeros((100, 120), np.uint8); big_mask[20:68, 30:94] = mask * 255
    Image.fromarray(big_ink).save(tmp_path / "big_ink.png"); Image.fromarray(big_mask).save(tmp_path / "big_mask.png")
    script = tmp_path / "fake.py"; script.write_text(FAKE)
    base = ["run", "--surface-volume", str(tmp_path / "v.zarr"), "--command", f"{sys.executable} {script} {{input}} {{output}}",
            "--window", "8", "--offsets", "-2:2:2", "--outward", "higher", "--no-order-check"]
    assert cli.main(base + ["--labels", str(tmp_path / "ink.png"), "--label-mask", str(tmp_path / "mask.png"),
                            "--out", str(tmp_path / "a")]) == 0
    assert cli.main(base + ["--labels", str(tmp_path / "big_ink.png"), "--label-mask", str(tmp_path / "big_mask.png"),
                            "--crop", "30,20,64,48", "--out", str(tmp_path / "b")]) == 0
    a = list(csv.DictReader(open(tmp_path / "a" / "curve.csv"))); b = list(csv.DictReader(open(tmp_path / "b" / "curve.csv")))
    assert [r["auc"] for r in a] == [r["auc"] for r in b]
    assert cli.main(base + ["--labels", str(tmp_path / "big_ink.png"), "--crop", "100,20,64,48",
                            "--out", str(tmp_path / "c")]) == 2                 # crop reaches beyond the labels


def _fake_villa(monkeypatch, build):
    """Install a stand-in for villa's infer module whose model build runs `build`."""
    import types
    infer = types.ModuleType("vesuvius.ink_detection.inference.infer")
    infer.parse_args = lambda argv: types.SimpleNamespace(layer_start=None, layer_end=None)
    infer.configure_single_model = lambda ckpt, amp_dtype: build()
    names = ["vesuvius", "vesuvius.ink_detection", "vesuvius.ink_detection.inference"]
    for n in names:
        monkeypatch.setitem(sys.modules, n, types.ModuleType(n))
    sys.modules["vesuvius.ink_detection.inference"].infer = infer
    monkeypatch.setitem(sys.modules, "vesuvius.ink_detection.inference.infer", infer)


def test_check_villa_names_missing_model_package(monkeypatch, tmp_path):
    from offset_sweep.sweep import MODEL_PACKAGES, check_villa

    def build():
        raise ModuleNotFoundError("No module named 'einops'", name="einops")
    _fake_villa(monkeypatch, build)
    with pytest.raises(SweepError) as e:
        check_villa(tmp_path / "step.pth")
    msg = str(e.value)
    assert "'einops' is missing" in msg and f"uv pip install {MODEL_PACKAGES}" in msg and "\n" not in msg


def test_check_villa_passes_when_model_builds(monkeypatch, tmp_path):
    from offset_sweep.sweep import check_villa
    _fake_villa(monkeypatch, lambda: object())
    check_villa(tmp_path / "step.pth")
    check_villa()                                     # no checkpoint: import and --layer-start check only


def test_check_villa_names_missing_tiff_codec(monkeypatch):
    import offset_sweep.sweep as sw

    def imwrite(*a, **k):
        raise KeyError("<COMPRESSION.LZW: 5> requires the 'imagecodecs' package")
    _fake_villa(monkeypatch, lambda: object())
    monkeypatch.setattr(sw.tifffile, "imwrite", imwrite)
    with pytest.raises(SweepError) as e:
        sw.check_villa()
    msg = str(e.value)
    assert "'imagecodecs'" in msg and f"uv pip install {sw.MODEL_PACKAGES}" in msg and "\n" not in msg
    assert "imagecodecs==" in sw.MODEL_PACKAGES
