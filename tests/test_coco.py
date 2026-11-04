"""util/coco.py on synthesized inputs: generated images, raw outputs and COCO annotations, no dataset or model.

The geometry of preprocess() is checked against sizes worked out by hand, not against the code's own formulas. The
pipeline test goes the way of coco_prepare.py, coco_run.py and coco_eval.py: a white rectangle in a generated image is
found again in the preprocessed tensor, encoded as a detection (the inverse of the decoder, test_yolo_post.py), cut
into cells, stored as float16, decoded, and must land on the rectangle in the original image.
"""
import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image
from pycocotools.coco import COCO

from test_yolo_post import VARIANTS, box
from util import coco as cu
from util import yolo_post

MODELS = sorted(cu.PREP)
# the COCO category ids of the 80 classes: 1..90 with gaps
CAT_IDS = [i for i in range(1, 91) if i not in (12, 26, 29, 30, 45, 66, 68, 69, 71, 83)]
COLOR = (200, 50, 10)  # R, G, B: all different, none the padding gray


def save_image(path, w, h, color=COLOR):
    Image.fromarray(np.full((h, w, 3), color, np.uint8)).save(path)
    return str(path)


# ----------------------------------------------------------------------------
# preprocess

# the input size (W, H) and padding gray of each model's own evaluation
INPUT = {"yolox": (416, 416, 114), "yolov3": (416, 416, 128), "yolo-fastest": (320, 320, 0), "yolo26": (640, 640, 114)}

# (model, image width, height): where the image lands in the input, x, y, w, h, and the scale x, y
GEOMETRY = {
    ("yolox", 640, 480): (0, 0, 416, 312, 0.65, 0.65),
    ("yolox", 480, 640): (0, 0, 312, 416, 0.65, 0.65),
    ("yolox", 100, 50): (0, 0, 416, 208, 4.16, 4.16),
    ("yolox", 333, 500): (0, 0, 277, 416, 0.832, 0.832),
    ("yolov3", 640, 480): (0, 52, 416, 312, 0.65, 0.65),
    ("yolov3", 480, 640): (52, 0, 312, 416, 0.65, 0.65),
    ("yolov3", 100, 50): (0, 104, 416, 208, 4.16, 4.16),
    ("yolov3", 333, 500): (69, 0, 277, 416, 0.832, 0.832),
    ("yolo-fastest", 640, 480): (0, 0, 320, 320, 0.5, 2 / 3),
    ("yolo-fastest", 100, 50): (0, 0, 320, 320, 3.2, 6.4),
    ("yolo26", 640, 480): (0, 80, 640, 480, 1.0, 1.0),
    ("yolo26", 480, 640): (80, 0, 480, 640, 1.0, 1.0),
    ("yolo26", 100, 50): (0, 160, 640, 320, 6.4, 6.4),
    ("yolo26", 333, 500): (106, 0, 427, 640, 427 / 333, 1.28),
}


@pytest.mark.parametrize("model, iw, ih", GEOMETRY)
def test_preprocess_geometry(model, iw, ih, tmp_path):
    """The image fills the expected rectangle, in the model's channel order; the rest is the padding gray."""
    x0, y0, w, h, sx, sy = GEOMETRY[model, iw, ih]
    x, geo = cu.preprocess(save_image(tmp_path / "i.png", iw, ih), model)
    size_w, size_h, pad = INPUT[model]
    assert x.shape == (1, 3, size_h, size_w) and x.dtype == np.uint8 and x.flags.c_contiguous
    want = np.full((3, size_h, size_w), pad, np.uint8)
    channels = COLOR[::-1] if model == "yolox" else COLOR  # YOLOX takes BGR
    want[:, y0:y0 + h, x0:x0 + w] = np.array(channels, np.uint8)[:, None, None]
    assert np.array_equal(x[0], want)
    assert geo == pytest.approx(dict(sx=sx, sy=sy, dx=float(x0), dy=float(y0)))


@pytest.mark.parametrize("model", MODELS)
def test_preprocess_gray_image(model, tmp_path):
    """val2017 has grayscale JPEGs: they become three equal channels."""
    path = tmp_path / "g.png"
    Image.fromarray(np.full((48, 64), 77, np.uint8)).save(path)
    x, _ = cu.preprocess(str(path), model)
    assert np.array_equal(x[0, 0], x[0, 1]) and np.array_equal(x[0, 1], x[0, 2]) and (x == 77).any()


def test_preprocess_of_a_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="cannot read"):
        cu.preprocess(str(tmp_path / "none.png"), "yolox")


def test_prepared_input_round_trip(tmp_path):
    """What coco_prepare.py writes comes back through load_index and read_input."""
    x = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(1, 3, 4, 5)
    x.tofile(tmp_path / "7.u8")
    (tmp_path / "index.json").write_text(json.dumps({"model": "yolox", "images": [{"id": 7, "file": "7.u8"}]}))
    meta = cu.load_index(str(tmp_path))
    assert meta["model"] == "yolox"
    assert np.array_equal(cu.read_input(str(tmp_path), meta["images"][0]).reshape(x.shape), x)


def test_model_input():
    u8 = np.array([0, 51, 255], np.uint8)
    same = cu.model_input(u8, np.float32, 255.0)
    assert same.dtype == np.float32 and same.tolist() == [0.0, 51.0, 255.0]
    assert cu.model_input(u8, np.uint8, 255.0).dtype == np.uint8
    scaled = cu.model_input(u8, np.float16, 1.0)
    assert scaled.dtype == np.float16 and scaled.tolist() == pytest.approx([0.0, 0.2, 1.0], abs=1e-3)


# ----------------------------------------------------------------------------
# cells and candidates


def test_kind():
    assert [cu.kind(m) for m in ("yolox", "yolov3", "yolo-fastest", "yolo26")] == ["rows", "grid", "grid", "dense"]


def test_cells_layout():
    """The cell of each position, written out by index."""
    rng = np.random.default_rng(0)
    rows = rng.standard_normal((1, 6, 85)).astype(np.float32)
    assert np.array_equal(cu.cells(rows, "rows")[4], rows[0, 4])
    grid = rng.standard_normal((1, 2 * 85, 3, 4)).astype(np.float32)  # 2 anchors, 3 x 4 grid
    c = cu.cells(grid, "grid")
    assert c.shape == (2 * 3 * 4, 85)
    assert np.array_equal(c[(1 * 3 + 2) * 4 + 3], grid[0, 85:, 2, 3])  # anchor 1, row 2, column 3
    dense = rng.standard_normal((1, 84, 3, 4)).astype(np.float32)
    c = cu.cells(dense, "dense")
    assert c.shape == (3 * 4, 84)
    assert np.array_equal(c[2 * 4 + 3], dense[0, :, 2, 3])


@pytest.mark.parametrize("k, shape", [("rows", (1, 6, 85)), ("grid", (1, 170, 3, 4)), ("dense", (1, 84, 3, 4))])
def test_uncells_restores_the_kept_cells(k, shape):
    out = np.random.default_rng(1).standard_normal(shape).astype(np.float32)
    c = cu.cells(out, k)
    keep = np.array([0, 2, 5])
    back = cu.uncells(keep, c[keep], shape, k)
    assert back.shape == shape and back.dtype == np.float32
    b = cu.cells(back, k)
    assert np.array_equal(b[keep], c[keep])
    dropped = np.setdiff1d(np.arange(len(c)), keep)
    assert (b[dropped] == cu.EMPTY[k]).all()


def score_of(c, k):
    """objectness * best class score of one cell, in plain Python."""
    def sig(v):
        return 1.0 / (1.0 + np.exp(-float(v)))

    if k == "rows":
        return float(c[4]) * max(float(v) for v in c[5:])
    if k == "dense":
        return max(sig(v) for v in c[4:])
    return sig(c[4]) * max(sig(v) for v in c[5:])


@pytest.mark.parametrize("conf", [0.05, 0.3, 0.8])
@pytest.mark.parametrize("k, width", [("rows", 85), ("grid", 85), ("dense", 84)])
def test_candidates(k, width, conf):
    rng = np.random.default_rng(2)
    if k == "rows":
        c = rng.random((500, width), dtype=np.float32)
    else:  # the best of 80 logits is high: lower the dense cells by up to 6
        low = 6 * rng.random((500, 1)) if k == "dense" else 0
        c = (rng.standard_normal((500, width)) * 2 - 2 - low).astype(np.float32)
    want = [i for i in range(len(c)) if score_of(c[i], k) >= conf]
    assert 0 < len(want) < len(c)
    assert cu.candidates(c, k, conf).tolist() == want


def test_candidates_threshold_is_inclusive():
    c = np.zeros((2, 85), np.float32)
    c[0, 4], c[0, 5] = 0.5, 0.5  # 0.25
    c[1, 4], c[1, 5] = 0.5, 0.25  # 0.125
    assert cu.candidates(c, "rows", 0.25).tolist() == [0]


@pytest.mark.parametrize("model", MODELS)
def test_dropped_cells_cannot_become_detections(model):
    """Filtering into cells and filling the gaps with EMPTY gives the detections of the complete outputs."""
    w, h = 96, 64
    m, p, k = yolo_post.MODELS[model], cu.PREP[model], cu.kind(model)
    rng = np.random.default_rng(3)
    shapes = [o.shape for o in cu.group(VARIANTS[model][1]([], w, h), model)]
    outs = [rng.random(s, dtype=np.float32) if k == "rows" else
            (rng.standard_normal(s) * 2 - 1).astype(np.float32) for s in shapes]
    if k == "dense":  # the best of 80 logits is high: lower the class logits of some positions, by up to 10
        for o in outs:
            o[:, 4:] -= 10 * rng.random((1, 1, *o.shape[2:]), dtype=np.float32)
    conf = 0.2

    def detections(raw):
        return cu.postprocess(m["decode"](raw, h, w, m["anchors"]), conf, p["nms"], p["multi_label"], 0)

    kept = []
    for o in outs:
        c = cu.cells(o, k)
        i = cu.candidates(c, k, conf)
        assert 0 < len(i) < len(c)
        kept.append(cu.uncells(i, c[i], o.shape, k))
    want = detections(outs)
    assert len(want) and np.allclose(detections(kept), want, rtol=1e-5, atol=1e-3)


# ----------------------------------------------------------------------------
# postprocess, to_coco


def det(cx, cy, w, h, obj, **classes):
    """A decoded row; classes as c<index>=score."""
    row = np.zeros(85, np.float32)
    row[:5] = cx, cy, w, h, obj
    for name, score in classes.items():
        row[5 + int(name[1:])] = score
    return row


def rows(*r):
    return np.stack(r)


def test_postprocess_corners_and_score():
    out = cu.postprocess(rows(det(50, 40, 20, 10, 0.5, c3=0.8)), 0.1, 0.5, False, 0)
    assert out.dtype == np.float32
    assert out.tolist() == [pytest.approx([40, 35, 60, 45, 0.4, 3])]


def test_postprocess_threshold():
    d = rows(det(10, 10, 4, 4, 0.5, c0=0.5), det(50, 50, 4, 4, 0.5, c0=0.25))
    assert cu.postprocess(d, 0.25, 0.5, False, 0)[:, 4].tolist() == [0.25]
    assert cu.postprocess(d, 0.25, 0.5, True, 0)[:, 4].tolist() == [0.25]


@pytest.mark.parametrize("multi", [False, True])
def test_postprocess_nothing_above_the_threshold(multi):
    out = cu.postprocess(rows(det(10, 10, 4, 4, 0.1, c0=0.1)), 0.5, 0.5, multi, 0)
    assert out.shape == (0, 6) and out.dtype == np.float32


def test_postprocess_best_class_or_every_class():
    d = rows(det(50, 50, 20, 20, 1.0, c2=0.6, c7=0.9, c9=0.01))
    assert cu.postprocess(d, 0.1, 0.5, False, 0)[:, 4:].tolist() == [pytest.approx([0.9, 7])]
    assert cu.postprocess(d, 0.1, 0.5, True, 0)[:, 4:].tolist() == [pytest.approx([0.9, 7]), pytest.approx([0.6, 2])]


def test_postprocess_nms_is_per_class():
    same = rows(det(50, 50, 20, 20, 1.0, c1=0.9), det(51, 50, 20, 20, 1.0, c1=0.8))
    assert cu.postprocess(same, 0.1, 0.5, False, 0)[:, 4].tolist() == [pytest.approx(0.9)]
    other = rows(det(50, 50, 20, 20, 1.0, c1=0.9), det(51, 50, 20, 20, 1.0, c2=0.8))
    assert cu.postprocess(other, 0.1, 0.5, False, 0)[:, 5].tolist() == [1, 2]


def test_postprocess_nms_threshold():
    """Two 20 x 20 boxes 10 apart: intersection 200, union 600, IoU 1/3."""
    d = rows(det(50, 50, 20, 20, 1.0, c1=0.9), det(60, 50, 20, 20, 1.0, c1=0.8))
    assert len(cu.postprocess(d, 0.1, 0.4, False, 0)) == 2
    assert len(cu.postprocess(d, 0.1, 0.3, False, 0)) == 1


def test_postprocess_order_and_max_dets():
    """Best score first over all the classes; max_dets keeps the best ones, 0 keeps all."""
    d = rows(det(10, 10, 4, 4, 1.0, c5=0.3), det(30, 30, 4, 4, 1.0, c1=0.5), det(50, 50, 4, 4, 1.0, c9=0.9),
             det(70, 70, 4, 4, 1.0, c0=0.7))
    assert cu.postprocess(d, 0.1, 0.5, False, 0)[:, 5].tolist() == [9, 0, 1, 5]
    assert cu.postprocess(d, 0.1, 0.5, False, 2)[:, 5].tolist() == [9, 0]


def test_to_coco():
    """Input pixels back to the original image: (x - dx) / sx, (y - dy) / sy, as x, y, width, height."""
    entry = dict(id=42, sx=0.5, sy=0.25, dx=10.0, dy=20.0)
    dets = np.array([[20, 30, 60, 50, 0.75, 79], [10, 20, 12, 24, 0.5, 0]], np.float32)
    assert cu.to_coco(dets, entry, CAT_IDS) == [
        {"image_id": 42, "category_id": 90, "bbox": [20.0, 40.0, 80.0, 80.0], "score": 0.75},
        {"image_id": 42, "category_id": 1, "bbox": [0.0, 0.0, 4.0, 16.0], "score": 0.5}]
    assert cu.to_coco(np.zeros((0, 6), np.float32), entry, CAT_IDS) == []


# ----------------------------------------------------------------------------
# AP


def ground_truth(tmp_path, images):
    """A COCO object for images = {id: [(class index, x, y, w, h), ...]}."""
    anns = [{"id": n + 1, "image_id": i, "category_id": CAT_IDS[c], "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0}
            for n, (i, c, x, y, w, h) in enumerate((i, *a) for i, boxes in images.items() for a in boxes)]
    path = tmp_path / "instances.json"
    path.write_text(json.dumps({"images": [{"id": i, "width": 640, "height": 480} for i in images],
                                "annotations": anns, "categories": [{"id": c, "name": str(c)} for c in CAT_IDS]}))
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(path))


def result(image, c, x, y, w, h, score=0.9):
    return {"image_id": image, "category_id": CAT_IDS[c], "bbox": [x, y, w, h], "score": score}


FOUR = {1: [(0, 100, 100, 100, 100), (0, 300, 100, 100, 100)], 2: [(0, 100, 100, 100, 100), (0, 300, 300, 100, 100)]}


def test_coco_ap_perfect(tmp_path):
    gt = ground_truth(tmp_path, FOUR)
    dets = [result(i, c, x, y, w, h) for i, boxes in FOUR.items() for c, x, y, w, h in boxes]
    stats = cu.coco_ap(gt, dets, [1, 2])
    assert len(stats) == 12 and stats[0] == pytest.approx(1.0) and stats[1] == pytest.approx(1.0)


def test_coco_ap_half_found(tmp_path):
    """Recall 0.5 at precision 1: 51 of the 101 recall points of COCOeval."""
    gt = ground_truth(tmp_path, FOUR)
    stats = cu.coco_ap(gt, [result(1, *FOUR[1][0]), result(2, *FOUR[2][0])], [1, 2])
    assert stats[0] == pytest.approx(51 / 101)


def test_coco_ap_iou_thresholds(tmp_path):
    """A 100 x 100 box shifted by 20: IoU 80 / 120 = 2/3, a hit at 0.50..0.65 (4 of the 10 thresholds), not at 0.75."""
    gt = ground_truth(tmp_path, {1: [(0, 100, 100, 100, 100)]})
    stats = cu.coco_ap(gt, [result(1, 0, 120, 100, 100, 100)], [1])
    assert stats[1] == pytest.approx(1.0) and stats[2] == 0.0 and stats[0] == pytest.approx(0.4)


def test_coco_ap_only_of_the_given_images(tmp_path):
    gt = ground_truth(tmp_path, FOUR)
    stats = cu.coco_ap(gt, [result(1, *b) for b in FOUR[1]], [1])
    assert stats[0] == pytest.approx(1.0)


def test_coco_ap_without_detections(tmp_path):
    assert cu.coco_ap(ground_truth(tmp_path, FOUR), [], [1, 2]).tolist() == [0.0] * 12


# ----------------------------------------------------------------------------
# the whole way


def white_rectangle(tensor):
    """Corners of the white rectangle in a preprocessed tensor, in input pixels."""
    ys, xs = np.nonzero(tensor[0].min(0) > 190)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def through_the_pipeline(path, model, cls):
    """Image -> COCO results: the rectangle found in the tensor is what the model detects."""
    x, geo = cu.preprocess(path, model)
    size_w, size_h = cu.PREP[model]["size"]
    x1, y1, x2, y2 = white_rectangle(x)
    # the small shift keeps the center off the cell edge, where the inverse sigmoid of the encoder is infinite
    b = box((x1 + x2) / 2 + 0.01, (y1 + y2) / 2 + 0.01, float(x2 - x1), float(y2 - y1), 0.9, cls, 32)
    m, p, k = yolo_post.MODELS[model], cu.PREP[model], cu.kind(model)
    outs = cu.group(VARIANTS[model][1]([b], size_w, size_h), model)
    stored = []
    for o in outs:  # coco_run.py, then coco_eval.load_candidates
        c = cu.cells(o, k)
        i = cu.candidates(c, k, p["conf"])
        stored.append(cu.uncells(i, c[i].astype(np.float16).astype(np.float32), o.shape, k))
    decoded = m["decode"](stored, size_h, size_w, m["anchors"])
    dets = cu.postprocess(decoded, p["conf"], p["nms"], p["multi_label"], 0)
    return cu.to_coco(dets, dict(id=1, **geo), CAT_IDS), geo


@pytest.mark.parametrize("iw, ih, rect", [(640, 480, (200, 120, 240, 200)), (480, 640, (100, 300, 200, 260)),
                                          (150, 100, (40, 20, 70, 50))])
@pytest.mark.parametrize("model", MODELS)
def test_rectangle_comes_back_in_image_pixels(model, iw, ih, rect, tmp_path):
    x, y, w, h = rect
    pixels = np.full((ih, iw, 3), 20, np.uint8)
    pixels[y:y + h, x:x + w] = 255
    path = tmp_path / "r.png"
    Image.fromarray(pixels).save(path)
    res, geo = through_the_pipeline(str(path), model, 79)
    assert len(res) == 1 and res[0]["category_id"] == 90 and res[0]["score"] == pytest.approx(0.9, abs=0.01)
    assert res[0]["bbox"] == pytest.approx([x, y, w, h], abs=1 + 2.5 / min(geo["sx"], geo["sy"]))
    gt = ground_truth(tmp_path, {1: [(79, x, y, w, h)]})
    assert cu.coco_ap(gt, res, [1])[1] == pytest.approx(1.0)


def test_board_needs_numpy_only():
    """coco_run.py runs on the board: importing util.coco must not pull in the PC packages."""
    code = ("import sys\n"
            "for m in ('cv2', 'PIL', 'pycocotools', 'torch', 'onnx'):\n"
            "    sys.modules[m] = None\n"
            "import util.coco\n")
    subprocess.run([sys.executable, "-c", code], check=True)
