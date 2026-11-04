"""util/yolo_post.py through --postprocess (util/postprocess.py): known boxes are encoded into raw head outputs (the
inverse of each decoder), post-processed, and the printed detection list is compared with the boxes.

Boxes have whole-pixel corners, even sizes and centers in the middle of a grid cell, so the decoded values print
exactly (no rounding edges); scores have 3 decimals.
"""
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

from util import postprocess, yolo_post

NUM_CLASSES = 80
BG_LOGIT = -20.0  # sigmoid ~ 2e-9: empty cells never pass the score threshold
OBJ = 0.95  # objectness of a box (YOLOX, anchor heads); its class score is score / OBJ


def logit(p):
    return math.log(p / (1.0 - p))


def box(cx, cy, w, h, score, cls, stride, anchor=1):
    """A detection to encode: center, size in input pixels, at the cell of (cx, cy) on the grid of `stride`."""
    return dict(cx=cx, cy=cy, w=w, h=h, score=score, cls=cls, stride=stride, anchor=anchor)


def cell(b):
    s = b["stride"]
    gx, gy = int(b["cx"] // s), int(b["cy"] // s)
    return gx, gy, b["cx"] / s - gx, b["cy"] / s - gy


def encode_yolox(boxes, w, h):
    """[1, N, 85]: (dx, dy, log w, log h in strides, objectness, classes after sigmoid), strides 8, 16, 32."""
    offsets, n = {}, 0
    for s in (8, 16, 32):
        offsets[s] = n
        n += (h // s) * (w // s)
    out = np.zeros((1, n, 5 + NUM_CLASSES), np.float32)
    for b in boxes:
        s = b["stride"]
        gx, gy, fx, fy = cell(b)
        row = out[0, offsets[s] + gy * (w // s) + gx]
        row[:4] = fx, fy, math.log(b["w"] / s), math.log(b["h"] / s)
        row[4] = OBJ
        row[5 + b["cls"]] = b["score"] / OBJ
    return [out]


def encode_anchors(boxes, w, h, anchors):
    """One [1, A*85, H/s, W/s] per stride: darknet logits (sigmoid tx, ty + cell; exp tw, th * anchor)."""
    outs = {}
    for s, anc in anchors.items():
        t = np.zeros((len(anc), 5 + NUM_CLASSES, h // s, w // s), np.float32)
        t[:, 4] = BG_LOGIT
        outs[s] = t
    for b in boxes:
        s, a = b["stride"], b["anchor"]
        gx, gy, fx, fy = cell(b)
        aw, ah = anchors[s][a]
        t = outs[s][a, :, gy, gx]
        t[:4] = logit(fx), logit(fy), math.log(b["w"] / aw), math.log(b["h"] / ah)
        t[4] = logit(OBJ)
        t[5:] = BG_LOGIT
        t[5 + b["cls"]] = logit(b["score"] / OBJ)
    return [t.reshape(1, -1, *t.shape[2:]) for t in outs.values()]


def dfl_logits(d):
    """16 bin logits whose softmax-weighted mean of 0..15 is d (the two bins around d)."""
    i, f = int(d), d - int(d)
    p = np.full(yolo_post.DFL_BINS, 1e-30)
    p[i] += 1.0 - f
    if f:
        p[i + 1] += f
    return np.log(p)


def encode_yolo26(boxes, w, h, joined=False, dfl=False):
    """Per stride 8, 16, 32 a box map [1, 4, H/s, W/s] (l, t, r, b from the cell center in strides; DFL: [1, 64, ...])
    and a class logit map [1, 80, H/s, W/s]; joined: one [1, 84, H/s, W/s] per stride."""
    outs = []
    for s in (8, 16, 32):
        gh, gw = h // s, w // s
        dist = np.ones((4, gh, gw), np.float32)
        cls = np.full((NUM_CLASSES, gh, gw), BG_LOGIT, np.float32)
        for b in (b for b in boxes if b["stride"] == s):
            gx, gy, _, _ = cell(b)
            ax, ay = gx + 0.5, gy + 0.5
            x1, y1 = (b["cx"] - b["w"] / 2) / s, (b["cy"] - b["h"] / 2) / s
            x2, y2 = (b["cx"] + b["w"] / 2) / s, (b["cy"] + b["h"] / 2) / s
            dist[:, gy, gx] = ax - x1, ay - y1, x2 - ax, y2 - ay
            cls[b["cls"], gy, gx] = logit(b["score"])
        if dfl:
            bins = np.stack([np.stack([dfl_logits(v) for v in side.reshape(-1)], 1).reshape(-1, gh, gw)
                             for side in dist])  # [4, 16, gh, gw]
            boxmap = bins.reshape(4 * yolo_post.DFL_BINS, gh, gw).astype(np.float32)
        else:
            boxmap = dist
        if joined:
            outs.append(np.concatenate((boxmap, cls))[None])
        else:
            outs += [boxmap[None], cls[None]]
    return outs


# variant -> (--postprocess name, encoder)
VARIANTS = {
    "yolox": ("yolox", encode_yolox),
    "yolov3": ("yolov3", lambda b, w, h: encode_anchors(b, w, h, yolo_post.MODELS["yolov3"]["anchors"])),
    "yolo-fastest": ("yolo-fastest",
                     lambda b, w, h: encode_anchors(b, w, h, yolo_post.MODELS["yolo-fastest"]["anchors"])),
    "yolo26": ("yolo26", encode_yolo26),
    "yolo26-joined": ("yolo26", lambda b, w, h: encode_yolo26(b, w, h, joined=True)),
    "yolo26-dfl": ("yolo26", lambda b, w, h: encode_yolo26(b, w, h, dfl=True)),  # YOLO12 / YOLO11 / YOLOv8
}
STRIDES = {"yolo-fastest": (16, 32)}  # the others: 8, 16, 32
SIZES = {"yolox": [(416, 416), (640, 480)], "yolov3": [(416, 416), (416, 320)],
         "yolo-fastest": [(320, 320), (320, 256)]}  # yolo26*: 640x640, 640x480


def strides(variant):
    return STRIDES.get(variant, (8, 16, 32))


def sizes(variant):
    return SIZES.get(variant, [(640, 640), (640, 480)])


def image(variant, w, h, seed=0):
    """A random input in the model's value range: u8 0..255, or f32 0..1 for the 0..1 models."""
    rng = np.random.default_rng(seed)
    if yolo_post.MODELS[VARIANTS[variant][0]]["white"] == 1.0:
        return rng.random((1, 3, h, w), dtype=np.float32)
    return rng.integers(0, 256, (1, 3, h, w), dtype=np.uint8)


def run(variant, boxes, w, h, tmp_path, capsys, outs=None, data=None, score_thr=0.25):
    name, encode = VARIANTS[variant]
    args = SimpleNamespace(postprocess=name, score_thr=score_thr, nms_thr=0.45, output_prefix=str(tmp_path / "out-"))
    data = image(variant, w, h) if data is None else data
    code = postprocess.run(args, data, encode(boxes, w, h) if outs is None else outs)
    return code, capsys.readouterr().out, data


def expected(boxes, score_thr=0.25):
    lines = [f"{len(boxes)} detection(s) with score >= {score_thr}:"]
    for b in sorted(boxes, key=lambda b: -b["score"]):
        x1, y1 = b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2
        x2, y2 = b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2
        lines.append(f"  {b['score']:.3f}  class {b['cls']:2d} {yolo_post.COCO[b['cls']]:<14s} "
                     f"box ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")
    return lines


def printed(out):
    """The detection lines of the output: the header and the lines under it, up to the 'saved' line."""
    lines = out.splitlines()
    start = next(i for i, l in enumerate(lines) if "detection(s) with score" in l)
    return [l for l in lines[start:] if not l.startswith("saved ")]


def boxes_on_strides(variant, w, h):
    """One box per stride, in different places, classes and scores."""
    sizes_px = {8: (40, 24), 16: (96, 64), 32: (200, 160)}
    spots = [(0.25, 0.3), (0.6, 0.5), (0.4, 0.75)]
    scores = [0.931, 0.833, 0.702]  # <= OBJ
    out = []
    for k, s in enumerate(strides(variant)):
        bw, bh = sizes_px[s]
        cx = (int(w * spots[k][0] / s) + 0.5) * s
        cy = (int(h * spots[k][1] / s) + 0.5) * s
        out.append(box(cx, cy, bw, bh, scores[k], cls=[0, 2, 16][k], stride=s))
    return out


@pytest.mark.parametrize("variant", VARIANTS)
def test_single_box(variant, tmp_path, capsys):
    w, h = sizes(variant)[0]
    boxes = [box(104, 120, 96, 64, 0.876, cls=0, stride=16)]
    code, out, data = run(variant, boxes, w, h, tmp_path, capsys)
    assert code == 0
    assert printed(out) == expected(boxes)
    rgb = tmp_path / f"out-{VARIANTS[variant][0]}.rgb"  # the drawing: same shape and element type as the input
    assert rgb.stat().st_size == data.nbytes


@pytest.mark.parametrize("variant,size", [(v, s) for v in VARIANTS for s in sizes(v)])
def test_boxes_on_every_stride(variant, size, tmp_path, capsys):
    w, h = size
    boxes = boxes_on_strides(variant, w, h)
    code, out, _ = run(variant, boxes, w, h, tmp_path, capsys)
    assert code == 0
    assert printed(out) == expected(boxes)


@pytest.mark.parametrize("variant", VARIANTS)
def test_nms_is_per_class(variant, tmp_path, capsys):
    """A lower-scored box of the same class overlapping by IoU 0.71 is dropped, one of another class is kept."""
    w, h = sizes(variant)[0]
    best = box(104, 104, 96, 96, 0.912, cls=0, stride=16)
    duplicate = box(120, 104, 96, 96, 0.805, cls=0, stride=16)  # the next cell: IoU 0.714 > 0.45
    other = box(104, 120, 96, 96, 0.654, cls=1, stride=16)
    code, out, _ = run(variant, [best, duplicate, other], w, h, tmp_path, capsys)
    assert code == 0
    assert printed(out) == expected([best, other])


@pytest.mark.parametrize("variant", VARIANTS)
def test_score_threshold(variant, tmp_path, capsys):
    w, h = sizes(variant)[0]
    above = box(104, 104, 64, 64, 0.251, cls=3, stride=16)
    below = box(296, 200, 64, 64, 0.249, cls=3, stride=16)
    code, out, _ = run(variant, [above, below], w, h, tmp_path, capsys)
    assert code == 0
    assert printed(out) == expected([above])
    code, out, _ = run(variant, [above, below], w, h, tmp_path, capsys, score_thr=0.5)
    assert printed(out) == expected([], score_thr=0.5)


@pytest.mark.parametrize("variant", [v for v in VARIANTS if v != "yolox"])
def test_output_order_does_not_matter(variant, tmp_path, capsys):
    """The outputs are matched to their strides by grid size, not by position (the Yolo-Fastest names are swapped)."""
    w, h = sizes(variant)[0]
    boxes = boxes_on_strides(variant, w, h)
    outs = VARIANTS[variant][1](boxes, w, h)
    for seed in range(3):
        shuffled = outs[:]
        random.Random(seed).shuffle(shuffled)
        code, out, _ = run(variant, boxes, w, h, tmp_path, capsys, outs=shuffled)
        assert code == 0
        assert printed(out) == expected(boxes)


def test_yolox_wrong_input_size(tmp_path, capsys):
    outs = encode_yolox([], 416, 416)
    code, out, _ = run("yolox", [], 320, 320, tmp_path, capsys, outs=outs)
    assert code == 1
    assert "error: yolox: output 0 has 3549 rows, a 320x320 YOLOX input gives 2100" in out


def test_anchor_heads_of_another_model(tmp_path, capsys):
    """Yolo-Fastest's two heads (strides 16, 32) decoded as YOLOv3 (8, 16, 32) on the same input."""
    outs = VARIANTS["yolo-fastest"][1]([], 320, 320)
    code, out, _ = run("yolov3", [], 320, 320, tmp_path, capsys, outs=outs)
    assert code == 1
    assert "the outputs have strides [16, 32], the model has [8, 16, 32]" in out


def test_yolo26_missing_class_map(tmp_path, capsys):
    outs = encode_yolo26([], 640, 640)[:-1]  # the stride 32 class map dropped
    code, out, _ = run("yolo26", [], 640, 640, tmp_path, capsys, outs=outs)
    assert code == 1
    assert "expected one box map (4 or 64 channels) and one class map" in out


@pytest.mark.parametrize("shape", [(1, 80, 32, 320), (1, 3, 416), (3, 416, 416)])
def test_not_an_image_input(shape, tmp_path, capsys):
    data = np.zeros(shape, np.uint8)
    code, out, _ = run("yolox", [], 416, 416, tmp_path, capsys, data=data)
    assert code == 1
    assert "needs an image input" in out


def test_nhwc_input(tmp_path, capsys):
    boxes = [box(104, 120, 96, 64, 0.876, cls=0, stride=16)]
    data = np.ascontiguousarray(image("yolox", 416, 416).transpose(0, 2, 3, 1))
    code, out, _ = run("yolox", boxes, 416, 416, tmp_path, capsys, data=data)
    assert code == 0
    assert printed(out) == expected(boxes)
    assert (tmp_path / "out-yolox.rgb").stat().st_size == data.nbytes
