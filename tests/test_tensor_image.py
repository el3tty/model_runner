"""tensor_from_image.py and tensor_to_image.py, run as commands on small generated images (no data files needed)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DTYPES = {"u8": "<u1", "u16": "<u2", "f16": "<f2", "f32": "<f4"}


def run(script, *args, ok=True):
    r = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], cwd=ROOT, capture_output=True,
                       text=True)
    if ok:
        assert r.returncode == 0, r.stdout + r.stderr
    return r


def from_image(*args, ok=True):
    return run("tensor_from_image.py", *args, ok=ok)


def to_image(*args, ok=True):
    return run("tensor_to_image.py", *args, ok=ok)


def random_image(path, w, h, seed=0):
    """An RGB image of random pixels (HxWx3 u8 returned), saved losslessly at path."""
    hwc = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    Image.fromarray(hwc, "RGB").save(path)
    return hwc


def read(path, dtype, shape):
    return np.fromfile(path, dtype=DTYPES[dtype]).reshape(shape)


# ---- tensor_from_image.py ------------------------------------------------------------------------------------------

def test_from_image_nchw_rgb_u8(tmp_path):
    hwc = random_image(tmp_path / "a.png", 7, 5)
    from_image(tmp_path / "a.png", "7x5", "--output", tmp_path / "a.bin")
    assert np.array_equal(read(tmp_path / "a.bin", "u8", (1, 3, 5, 7))[0], hwc.transpose(2, 0, 1))


def test_from_image_bgr(tmp_path):
    hwc = random_image(tmp_path / "a.png", 7, 5)
    from_image(tmp_path / "a.png", "7x5", "--bgr", "--output", tmp_path / "a.bin")
    assert np.array_equal(read(tmp_path / "a.bin", "u8", (1, 3, 5, 7))[0], hwc[:, :, ::-1].transpose(2, 0, 1))


@pytest.mark.parametrize("dtype", ["u16", "f16", "f32"])
def test_from_image_types_keep_0_255(tmp_path, dtype):
    hwc = random_image(tmp_path / "a.png", 7, 5)
    from_image(tmp_path / "a.png", "7x5", "--type", dtype, "--output", tmp_path / "a.bin")
    out = read(tmp_path / "a.bin", dtype, (1, 3, 5, 7))[0]  # little-endian, as documented
    assert np.array_equal(out.astype(np.float64), hwc.transpose(2, 0, 1).astype(np.float64))


@pytest.mark.parametrize("dtype", ["f16", "f32"])
def test_from_image_range_1(tmp_path, dtype):
    hwc = random_image(tmp_path / "a.png", 7, 5)
    from_image(tmp_path / "a.png", "7x5", "--range", "1", "--type", dtype, "--output", tmp_path / "a.bin")
    out = read(tmp_path / "a.bin", dtype, (1, 3, 5, 7))[0].astype(np.float64)
    assert np.allclose(out, hwc.transpose(2, 0, 1) / 255.0, atol=1e-3 if dtype == "f16" else 1e-7)


def test_from_image_default_output_path(tmp_path):
    random_image(tmp_path / "pic.png", 4, 4)
    from_image(tmp_path / "pic.png", "4x4")
    assert (tmp_path / "pic.bin").stat().st_size == 3 * 4 * 4


def test_from_image_stretch(tmp_path):
    Image.new("RGB", (40, 10), (10, 20, 30)).save(tmp_path / "a.png")
    from_image(tmp_path / "a.png", "16x16", "--output", tmp_path / "a.bin")
    out = read(tmp_path / "a.bin", "u8", (1, 3, 16, 16))[0]
    assert np.array_equal(out, np.broadcast_to(np.array([10, 20, 30], np.uint8)[:, None, None], (3, 16, 16)))


@pytest.mark.parametrize("pad", [114, 128, 0])
def test_from_image_letterbox_centered(tmp_path, pad):
    """A 2:1 image in a square: scaled to the full width, centered, padding rows above and below."""
    Image.new("RGB", (200, 100), (10, 20, 30)).save(tmp_path / "a.png")
    from_image(tmp_path / "a.png", "100x100", "--letterbox", "--pad", pad, "--output", tmp_path / "a.bin")
    out = read(tmp_path / "a.bin", "u8", (1, 3, 100, 100))[0]
    assert (out[:, :25] == pad).all() and (out[:, 75:] == pad).all()
    assert np.array_equal(out[:, 25:75], np.broadcast_to(np.array([10, 20, 30], np.uint8)[:, None, None],
                                                         (3, 50, 100)))


def test_from_image_transparency_on_white(tmp_path):
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[:2] = (200, 0, 0, 255)  # top half opaque red, bottom half fully transparent
    Image.fromarray(rgba, "RGBA").save(tmp_path / "a.png")
    from_image(tmp_path / "a.png", "4x4", "--output", tmp_path / "a.bin")
    out = read(tmp_path / "a.bin", "u8", (1, 3, 4, 4))[0]
    assert (out[:, :2] == np.array([200, 0, 0])[:, None, None]).all()
    assert (out[:, 2:] == 255).all()


def test_from_image_exif_orientation(tmp_path):
    """EXIF orientation 6 (rotate 90 degrees clockwise to display): the stored 20x10 image is used as 10x20."""
    hwc = np.zeros((10, 20, 3), np.uint8)
    hwc[:, :10] = (255, 0, 0)  # left half red, right half blue
    hwc[:, 10:] = (0, 0, 255)
    img = Image.fromarray(hwc, "RGB")
    exif = img.getexif()
    exif[0x0112] = 6
    img.save(tmp_path / "a.png", exif=exif)
    from_image(tmp_path / "a.png", "10x20", "--output", tmp_path / "a.bin")
    out = read(tmp_path / "a.bin", "u8", (1, 3, 20, 10))[0]
    assert (out[:, :10] == np.array([255, 0, 0])[:, None, None]).all()  # the left half is now on top
    assert (out[:, 10:] == np.array([0, 0, 255])[:, None, None]).all()


@pytest.mark.parametrize("args", [["4x4", "--range", "1"], ["4x4", "--range", "1", "--type", "u16"],
                                  ["4x4", "--pad", "256"], ["0x4"], ["4by4"]])
def test_from_image_rejects_bad_arguments(tmp_path, args):
    random_image(tmp_path / "a.png", 4, 4)
    r = from_image(tmp_path / "a.png", *args, "--output", tmp_path / "a.bin", ok=False)
    assert r.returncode != 0 and "error" in r.stderr
    assert not (tmp_path / "a.bin").exists()


# ---- tensor_to_image.py --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype,extra", [("u8", []), ("u8", ["--bgr"]), ("u16", []), ("f16", []), ("f32", []),
                                         ("f16", ["--range", "1"]), ("f32", ["--range", "1"]),
                                         ("f32", ["--range", "1", "--bgr"])])
def test_round_trip(tmp_path, dtype, extra):
    """tensor_from_image -> tensor_to_image (png) -> tensor_from_image gives the same bytes."""
    random_image(tmp_path / "a.png", 9, 6)
    from_image(tmp_path / "a.png", "9x6", "--type", dtype, *extra, "--output", tmp_path / "a.bin")
    to_image(tmp_path / "a.bin", "9x6", dtype, *extra, "--output", tmp_path / "b.png")
    from_image(tmp_path / "b.png", "9x6", "--type", dtype, *extra, "--output", tmp_path / "b.bin")
    assert (tmp_path / "a.bin").read_bytes() == (tmp_path / "b.bin").read_bytes()


def test_to_image_pixels(tmp_path):
    rng = np.random.default_rng(1)
    nchw = rng.integers(0, 256, (1, 3, 6, 9), dtype=np.uint8)
    nchw.tofile(tmp_path / "a.rgb")
    to_image(tmp_path / "a.rgb", "9x6", "--output", tmp_path / "a.png")
    assert np.array_equal(np.asarray(Image.open(tmp_path / "a.png")), nchw[0].transpose(1, 2, 0))


def test_to_image_nhwc(tmp_path):
    nhwc = np.random.default_rng(2).integers(0, 256, (1, 6, 9, 3), dtype=np.uint8)
    nhwc.tofile(tmp_path / "a.rgb")
    to_image(tmp_path / "a.rgb", "9x6", "--layout", "nhwc", "--output", tmp_path / "a.png")
    assert np.array_equal(np.asarray(Image.open(tmp_path / "a.png")), nhwc[0])


def test_to_image_rounds_and_clips(tmp_path):
    values = np.array([-5.0, 0.4, 0.6, 254.5, 300.0, 127.0], np.float32)  # rint: 254.5 -> 254 (half to even)
    np.broadcast_to(values[None, None, None, :], (1, 3, 1, 6)).astype("<f4").tofile(tmp_path / "a.rgb")
    to_image(tmp_path / "a.rgb", "6x1", "f32", "--output", tmp_path / "a.png")
    assert np.asarray(Image.open(tmp_path / "a.png"))[0, :, 0].tolist() == [0, 0, 1, 254, 255, 127]


def test_to_image_nan_replaced(tmp_path):
    t = np.full((1, 3, 1, 2), 100.0, np.float32)
    t[0, :, 0, 0] = np.nan
    t.astype("<f4").tofile(tmp_path / "a.rgb")
    r = to_image(tmp_path / "a.rgb", "2x1", "f32", "--output", tmp_path / "a.png")
    assert "NaN" in r.stdout
    assert np.asarray(Image.open(tmp_path / "a.png"))[0, :, 0].tolist() == [0, 100]


def test_to_image_wrong_size(tmp_path):
    np.zeros((1, 3, 4, 4), np.uint8).tofile(tmp_path / "a.rgb")
    r = to_image(tmp_path / "a.rgb", "4x4", "f32", "--output", tmp_path / "a.png", ok=False)
    assert r.returncode != 0 and "needs 192" in r.stderr
    assert not (tmp_path / "a.png").exists()


@pytest.mark.parametrize("args,path,fmt", [(["--output", "x.bmp"], "x.bmp", "BMP"),
                                           (["--output", "x.jpg"], "x.jpg", "JPEG"),
                                           (["--format", "jpg"], "a.jpg", "JPEG"),
                                           ([], "a.png", "PNG")])
def test_to_image_format(tmp_path, args, path, fmt):
    np.zeros((1, 3, 4, 4), np.uint8).tofile(tmp_path / "a.rgb")
    args = [tmp_path / a if a.startswith("x.") else a for a in args]
    to_image(tmp_path / "a.rgb", "4x4", *args)
    assert Image.open(tmp_path / path).format == fmt


def test_to_image_warns_about_0_1_range(tmp_path):
    np.full((1, 3, 2, 2), 0.5, np.float32).tofile(tmp_path / "a.rgb")
    r = to_image(tmp_path / "a.rgb", "2x2", "f32", "--output", tmp_path / "a.png")
    assert "--range 1" in r.stdout
