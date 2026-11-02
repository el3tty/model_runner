"""
YOLO post-processing shared by run-model.py (K230 board / simulator) and run-onnx.py (onnxruntime): decoding, NMS and
drawing, for the detector heads selected with --detect:

  yolox         the raw YOLOX head output [1, N, 85] = (dx, dy, log w, log h per grid cell, strides 8/16/32,
                objectness, 80 class scores), as exported without box decoding (pth_to_onnx.py). Input BGR, 0..255.
  yolov3        the three raw heads of yolov3-12-int8-heads.onnx (ONNX model zoo YOLOv3, keras-yolo3, with the
                decode / NMS part cut off): [1, 255, H/s, W/s] for strides 8, 16, 32. Input RGB, 0..1 (letterbox,
                gray 128 padding).
  yolo-fastest  the two raw heads of Yolo-Fastest 1.1 (darknet, yolo-fastest-1.1.cfg): [1, 255, H/s, W/s] for strides
                16 and 32. Input RGB, 0..1 (plain resize).
  yolo26        the raw one-to-many head of Ultralytics YOLO26 (yolo_to_onnx.py): per stride 8, 16, 32 a box map
                [1, 4, H/s, W/s] = (l, t, r, b) distances from the cell center in strides (no DFL) and a class map
                [1, 80, H/s, W/s] of logits (no objectness; decoded with objectness 1). Input RGB, 0..255 (letterbox
                centered, pad 114). Also the other Ultralytics detectors exported by yolo_to_onnx.py (YOLO12,
                YOLO11, YOLOv8): their box maps are DFL logits [1, 64, H/s, W/s], reduced to the 4 distances first.

The anchor heads hold 3 anchors x (tx, ty, tw, th, objectness, 80 class logits) per cell and are decoded as in darknet:
cx = (sigmoid(tx) + col) * stride, w = exp(tw) * anchor_w (likewise y, h), sigmoid on objectness and classes. Every
output is matched to its stride by its grid size, not by its name or position (the Yolo-Fastest output names are
swapped: out_stride16 is the 10x10 grid, stride 32).

score = objectness * best class score. draw() draws the boxes that survive --score-thr and class-aware NMS on the input
image (colored by score, with the score to 3 decimals above each box) and writes <prefix><detect>.rgb (e.g.
outputyolox.rgb): a raw tensor with exactly the layout, element type and size of the input. Input may be u8, u16, f16
or f32 as 1x3xHxW or 1xHxWx3, in the channel order and value range of the model (see above).
tensor_to_image.py turns the result into an image (--bgr for yolox, --scale 1 for the 0..1 models).
"""
import os

import numpy as np


COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
        "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
        "tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
        "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
        "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote "
        "keyboard cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear "
        "hair_drier toothbrush").split()

# ----------------------------------------------------------------------------
# decoding: every decoder returns [N, 85] = (cx, cy, w, h in input pixels, objectness, 80 class scores)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_yolox(outs, h, w, _anchors):
    """Raw YOLOX head output (reg = dx, dy, log w, log h; objectness and classes already through sigmoid)."""
    out = outs[0].reshape(-1, outs[0].shape[-1]).astype(np.float32).copy()
    grids, strides = [], []
    for stride in (8, 16, 32):
        ys, xs = np.meshgrid(np.arange(h // stride), np.arange(w // stride), indexing="ij")
        grids.append(np.stack((xs, ys), -1).reshape(-1, 2))
        strides.append(np.full((xs.size, 1), stride))
    grid, stride = np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)
    if len(out) != len(grid):
        raise ValueError(f"output 0 has {len(out)} rows, a {w}x{h} YOLOX input gives {len(grid)}")
    out[:, :2] = (out[:, :2] + grid) * stride
    out[:, 2:4] = np.exp(out[:, 2:4]) * stride
    return out


DFL_BINS = 16


def dfl(box):
    """DFL box logits [1, 4 * 16, h, w] -> distances [1, 4, h, w]: per side the softmax-weighted mean of 0..15."""
    t = box.astype(np.float32).reshape(box.shape[0], 4, DFL_BINS, *box.shape[2:])
    p = np.exp(t - t.max(2, keepdims=True))
    return (p * np.arange(DFL_BINS, dtype=np.float32).reshape(1, 1, -1, 1, 1)).sum(2) / p.sum(2)


def join_split(outs):
    """Ultralytics box [1, 4, h, w] (or DFL logits [1, 64, h, w]) and class [1, 80, h, w] maps (any order) -> one
    [1, 84, h, w] per stride, finest grid first. Already joined outputs are returned as they are."""
    if all(o.shape[1] not in (4, 4 * DFL_BINS) for o in outs):
        return list(outs)
    by_grid = {}
    for o in outs:
        if o.shape[1] == 4 * DFL_BINS:
            by_grid.setdefault(tuple(o.shape[2:]), {})["box"] = dfl(o)
        else:
            by_grid.setdefault(tuple(o.shape[2:]), {})["box" if o.shape[1] == 4 else "cls"] = o
    joined = []
    for g in sorted(by_grid, key=lambda g: -g[0] * g[1]):
        pair = by_grid[g]
        if set(pair) != {"box", "cls"}:
            raise ValueError(f"grid {g}: expected one box map (4 or 64 channels) and one class map, got {sorted(pair)}")
        joined.append(np.concatenate((pair["box"], pair["cls"]), 1))
    return joined


def decode_yolo26(outs, h, w, _anchors):
    """Raw YOLO26 one-to-many head (6 box / class maps, or the 3 joined [1, 84, H/s, W/s] of join_split)."""
    rows = []
    for out in join_split(outs):
        gh, gw = out.shape[2:]
        stride = h // gh
        if stride != w // gw or h % gh or w % gw:
            raise ValueError(f"output {list(out.shape)} does not fit a {w}x{h} input")
        t = out[0].astype(np.float32).reshape(out.shape[1], -1).T  # [gh*gw, 84]
        ys, xs = np.meshgrid(np.arange(gh) + 0.5, np.arange(gw) + 0.5, indexing="ij")
        ax, ay = xs.reshape(-1), ys.reshape(-1)
        x1, y1, x2, y2 = ax - t[:, 0], ay - t[:, 1], ax + t[:, 2], ay + t[:, 3]
        d = np.empty((len(t), 85), np.float32)
        d[:, 0], d[:, 1] = (x1 + x2) / 2 * stride, (y1 + y2) / 2 * stride
        d[:, 2], d[:, 3] = (x2 - x1) * stride, (y2 - y1) * stride
        d[:, 4] = 1.0
        with np.errstate(over="ignore"):
            d[:, 5:] = sigmoid(t[:, 4:])
        rows.append(d)
    return np.concatenate(rows)


def decode_anchors(outs, h, w, anchors):
    """Raw darknet / keras-yolo3 heads [1, A*85, H/s, W/s], logits everywhere; anchors: {stride: [(aw, ah), ...]}."""
    rows, strides = [], []
    for out in outs:
        if out.ndim != 4 or out.shape[0] != 1 or h % out.shape[2] or w % out.shape[3]:
            raise ValueError(f"expected [1, A*85, H/s, W/s] head outputs, got {list(out.shape)}")
        gh, gw = out.shape[2:]
        stride = h // gh
        if stride != w // gw or stride not in anchors:
            raise ValueError(f"output {list(out.shape)}: stride {stride} for a {w}x{h} input, the model has "
                             f"strides {sorted(anchors)}")
        strides.append(stride)
        anc =np.asarray(anchors[stride], np.float32)  # [A, 2]
        t = out.astype(np.float32).reshape(len(anc), -1, gh, gw).transpose(0, 2, 3, 1)  # [A, gh, gw, 85]
        ys, xs = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        d = np.empty_like(t)
        d[..., 0] = (sigmoid(t[..., 0]) + xs) * stride
        d[..., 1] = (sigmoid(t[..., 1]) + ys) * stride
        with np.errstate(over="ignore"):  # huge tw/th of empty cells: inf boxes that never pass the score threshold
            d[..., 2] = np.exp(t[..., 2]) * anc[:, 0, None, None]
            d[..., 3] = np.exp(t[..., 3]) * anc[:, 1, None, None]
        d[..., 4:] = sigmoid(t[..., 4:])
        rows.append(d.reshape(-1, d.shape[-1]))
    if sorted(strides) != sorted(anchors):  # e.g. a 2-head model decoded with the anchors of a 3-head one
        raise ValueError(f"the outputs have strides {sorted(strides)}, the model has {sorted(anchors)}")
    return np.concatenate(rows)


# decode, input channel order (bgr), input value of a white pixel (white), anchors {stride: [(w, h) ...]}
MODELS = {
    "yolox": dict(decode=decode_yolox, bgr=True, white=255.0, anchors=None),
    # anchors from the decode part of the ONNX model zoo yolov3-12-int8.onnx
    "yolov3": dict(decode=decode_anchors, bgr=False, white=1.0,
                   anchors={8: [(10, 13), (16, 30), (33, 23)], 16: [(30, 61), (62, 45), (59, 119)],
                            32: [(116, 90), (156, 198), (373, 326)]}),
    # yolo-fastest-1.1.cfg: anchors 12,18 37,49 52,132 115,73 119,199 242,238; masks 3,4,5 (10x10) and 0,1,2 (20x20)
    "yolo-fastest": dict(decode=decode_anchors, bgr=False, white=1.0,
                         anchors={16: [(12, 18), (37, 49), (52, 132)], 32: [(115, 73), (119, 199), (242, 238)]}),
    "yolo26": dict(decode=decode_yolo26, bgr=False, white=255.0, anchors=None),
}

# ----------------------------------------------------------------------------
# NMS and drawing (numpy only)

# 3x5 pixel font for the characters of "0.123": rows top to bottom, 3 bits each.
FONT = {
    "0": (7, 5, 5, 5, 7), "1": (2, 6, 2, 2, 7), "2": (7, 1, 7, 4, 7),
    "3": (7, 1, 7, 1, 7), "4": (5, 5, 7, 1, 1), "5": (7, 4, 7, 1, 7),
    "6": (7, 4, 7, 5, 7), "7": (7, 1, 1, 1, 1), "8": (7, 5, 7, 5, 7),
    "9": (7, 5, 7, 1, 7), ".": (0, 0, 0, 0, 2),
}

# (minimum score, RGB color); the highest matching tier wins.
SCORE_COLORS = [
    (0.0, (255, 0, 0)),      # red:    low confidence
    (0.4, (255, 140, 0)),    # orange
    (0.6, (255, 255, 0)),    # yellow
    (0.8, (0, 255, 0)),      # green:  high confidence
]


def score_color(score):
    color = SCORE_COLORS[0][1]
    for thr, c in SCORE_COLORS:
        if score >= thr:
            color = c
    return color


def nms(boxes, scores, thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        order = rest[inter / (area_i + area_r - inter + 1e-9) <= thr]
    return keep


def detect(det, score_thr, nms_thr):
    """det: [N, 85] decoded (cx, cy, w, h, obj, cls...). Returns list of (x1, y1, x2, y2, score, class)."""
    cls = det[:, 5:]
    cls_id = cls.argmax(1)
    scores = det[:, 4] * cls[np.arange(len(det)), cls_id]
    m = scores >= score_thr
    det, cls_id, scores = det[m], cls_id[m], scores[m]
    if not len(det):
        return []
    boxes = np.stack((det[:, 0] - det[:, 2] / 2, det[:, 1] - det[:, 3] / 2,
                      det[:, 0] + det[:, 2] / 2, det[:, 1] + det[:, 3] / 2), 1)
    res = []
    for c in np.unique(cls_id):  # class-aware NMS
        idx = np.where(cls_id == c)[0]
        for k in nms(boxes[idx], scores[idx], nms_thr):
            j = idx[k]
            res.append((*boxes[j], float(scores[j]), int(c)))
    return sorted(res, key=lambda r: -r[4])


def draw_rect(img, x1, y1, x2, y2, color, thick=2):
    """img: HxWx3 uint8, modified in place."""
    h, w = img.shape[:2]
    x1, x2 = int(round(max(0, min(w - 1, x1)))), int(round(max(0, min(w - 1, x2))))
    y1, y2 = int(round(max(0, min(h - 1, y1)))), int(round(max(0, min(h - 1, y2))))
    img[y1:y1 + thick, x1:x2 + 1] = color
    img[max(y2 - thick + 1, 0):y2 + 1, x1:x2 + 1] = color
    img[y1:y2 + 1, x1:x1 + thick] = color
    img[y1:y2 + 1, max(x2 - thick + 1, 0):x2 + 1] = color


def draw_text(img, x, y, text, color, bg=(0, 0, 0), scale=2):
    """Draw text with its top-left corner at (x, y) on a solid background strip."""
    h, w = img.shape[:2]
    cw, ch = 4 * scale, 5 * scale  # 3 px glyph + 1 px gap
    x0, y0 = max(0, min(w - 1, x)), max(0, min(h - 1, y))
    x1, y1 = min(w, x0 + cw * len(text) + scale), min(h, y0 + ch + 2 * scale)
    img[y0:y1, x0:x1] = bg
    for n, c in enumerate(text):
        for r, bits in enumerate(FONT.get(c, ())):
            for b in range(3):
                if bits >> (2 - b) & 1:
                    px, py = x0 + scale // 2 + n * cw + b * scale, y0 + scale + r * scale
                    img[py:py + scale, px:px + scale] = color  # numpy clips slices at the border


def draw_detections(hwc, dets, bgr):
    """hwc: u8 image in the model's channel order; with bgr the RGB drawing colors are swapped."""
    img = hwc.copy()
    for x1, y1, x2, y2, score, _ in dets:
        color = score_color(score)[::-1] if bgr else score_color(score)
        draw_rect(img, x1, y1, x2, y2, color)
        label_h = 5 * 2 + 4
        ty = int(round(y1)) - label_h
        if ty < 0:  # no room above the box: put the label just inside it
            ty = int(round(y1)) + 2
        draw_text(img, int(round(x1)), ty, f"{score:.3f}", color)
    return img


def add_arguments(parser):
    """Add the detection options shared by run-model.py and run-onnx.py to an argparse parser."""
    parser.add_argument("--detect", choices=sorted(MODELS), default=None,
                        help="decode the outputs as raw heads of this detector, draw the boxes on the input and save "
                             "<prefix><detect>.rgb (raw, same layout, channel order, value range and element type as "
                             "the input); yolox: BGR 0..255 input, yolov3 / yolo-fastest: RGB 0..1, yolo26: RGB "
                             "0..255")
    parser.add_argument("--score-thr", type=float, default=0.25,
                        help="with --detect: minimum objectness*class score (default 0.25)")
    parser.add_argument("--nms-thr", type=float, default=0.45,
                        help="with --detect: NMS IoU threshold (default 0.45)")


def draw(args, data, outs):
    """Decode the raw outputs `outs` of the --detect model, draw the detections on the input `data`, save raw."""
    model = MODELS[args.detect]
    if data.dtype.kind not in "uf" or data.ndim != 4 or 3 not in (data.shape[1], data.shape[3]):
        print(f"error: --detect needs an image input (1x3xHxW or 1xHxWx3, u8/u16/f16/f32), "
              f"got {data.dtype} {list(data.shape)}")
        return 1
    nchw = data.shape[1] == 3
    orig = np.ascontiguousarray(data[0].transpose(1, 2, 0) if nchw else data[0])  # HxWx3, input dtype
    white = model["white"]
    # draw on a u8 copy (0..255)
    hwc = np.clip(np.rint(orig.astype(np.float32) * (255.0 / white)), 0, 255).astype(np.uint8)
    if hwc.max() <= 1 and white == 255.0:
        print(f"warning: all input values are <= 1, but {args.detect} expects 0..255")
    elif orig.max() > 1.5 and white == 1.0:
        print(f"warning: input values up to {float(orig.max()):g}, but {args.detect} expects 0..1")
    try:
        det = model["decode"]([np.asarray(o) for o in outs], hwc.shape[0], hwc.shape[1], model["anchors"])
    except ValueError as e:
        print(f"error: {args.detect}: {e}")
        return 1
    dets = detect(det, args.score_thr, args.nms_thr)
    print(f"\n{len(dets)} detection(s) with score >= {args.score_thr}:")
    for x1, y1, x2, y2, score, c in dets:
        name = COCO[c] if c < len(COCO) else str(c)
        print(f"  {score:.3f}  class {c:2d} {name:<14s} box ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")
    img = draw_detections(hwc, dets, model["bgr"])
    if data.dtype != np.uint8 or white != 255.0:
        # keep the untouched pixels exactly as they were in the input; only the drawn ones are replaced
        drawn = (img != hwc).any(axis=2)
        merged = orig.copy()
        merged[drawn] = (img[drawn].astype(np.float32) * (white / 255.0)).astype(data.dtype)
        img = merged
    res = np.ascontiguousarray(img.transpose(2, 0, 1)[None] if nchw else img[None])
    path = f"{args.output_prefix}{args.detect}.rgb"
    res.tofile(path)
    print(f"saved {os.path.abspath(path)}: raw {res.dtype} {list(res.shape)} ({res.nbytes} bytes), "
          f"colors by score: <0.4 red, <0.6 orange, <0.8 yellow, >=0.8 green")
    return 0
