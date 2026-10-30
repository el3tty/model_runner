"""
YOLOX specifics shared by run-model.py (K230 board) and run-onnx.py (onnxruntime): decoding, NMS and drawing.

Expected model output: [1, N, 85] = (cx, cy, w, h, objectness, 80 class scores) per row; score = objectness * best
class score. With --yolox-raw the box columns are the raw head values (dx, dy, log w, log h per grid cell, strides
8/16/32) and are decoded here; otherwise they must already be pixels of the input image.

draw_yolox() draws the boxes that survive --score-thr and class-aware NMS on the input image (colored by score, with
the score to 3 decimals above each box) and writes <prefix>yolox.rgb: a raw tensor with exactly the layout, element
type and size of the input. Input may be u8, u16, f16 or f32 as 1x3xHxW or 1xHxWx3; pixel values are assumed to be in
0..255 for every type. tensor_to_image.py turns the result into an image.
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
# YOLOX post-processing and drawing (numpy only)

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


def decode_raw(det, size):
    """Decode raw YOLOX head output (reg = dx, dy, log w, log h) of shape [N, 85] for a square input."""
    out = det.astype(np.float32).copy()
    grids, strides = [], []
    for stride in (8, 16, 32):
        n = size // stride
        ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        grids.append(np.stack((xs, ys), -1).reshape(-1, 2))
        strides.append(np.full((n * n, 1), stride))
    grid, stride = np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)
    out[:, :2] = (out[:, :2] + grid) * stride
    out[:, 2:4] = np.exp(out[:, 2:4]) * stride
    return out


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
    """det: [N, 85] fused (cx, cy, w, h, obj, cls...). Returns list of (x1, y1, x2, y2, score, class)."""
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


def draw_detections(hwc, dets, bgr=False):
    img = hwc.copy()
    for x1, y1, x2, y2, score, _ in dets:
        color = score_color(score)[::-1] if bgr else score_color(score)  # BGR input: swap the drawing colors too
        draw_rect(img, x1, y1, x2, y2, color)
        label_h = 5 * 2 + 4
        ty = int(round(y1)) - label_h
        if ty < 0:  # no room above the box: put the label just inside it
            ty = int(round(y1)) + 2
        draw_text(img, int(round(x1)), ty, f"{score:.3f}", color)
    return img


def add_arguments(parser):
    """Add the YOLOX options shared by run-model.py and run-onnx.py to an argparse parser."""
    parser.add_argument("--yolox", action="store_true",
                        help="treat output 0 as YOLOX detections [1, N, 85], draw boxes on the input and save "
                             "<prefix>yolox.rgb (raw, same layout and element type as the input)")
    parser.add_argument("--yolox-raw", action="store_true",
                        help="with --yolox: output is NOT decoded (dx, dy, log w, log h per grid cell)")
    parser.add_argument("--bgr", action="store_true",
                        help="with --yolox: the input tensor is in BGR order; draw the box colors in BGR as well")
    parser.add_argument("--score-thr", type=float, default=0.25,
                        help="with --yolox: minimum objectness*class score (default 0.25)")
    parser.add_argument("--nms-thr", type=float, default=0.45,
                        help="with --yolox: NMS IoU threshold (default 0.45)")


def draw_yolox(args, data, out):
    """Draw detections of the fused YOLOX output `out` on the input `data` and save raw RGB."""
    if data.dtype.kind not in "uf" or data.ndim != 4 or 3 not in (data.shape[1], data.shape[3]):
        print(f"error: --yolox needs an image input (1x3xHxW or 1xHxWx3, u8/u16/f16/f32), "
              f"got {data.dtype} {list(data.shape)}")
        return 1
    nchw = data.shape[1] == 3
    orig = np.ascontiguousarray(data[0].transpose(1, 2, 0) if nchw else data[0])  # HxWx3, input dtype
    # Pixel values are 0..255 in every input type (no normalisation); draw on a u8 copy.
    hwc = np.clip(np.rint(orig.astype(np.float32)), 0, 255).astype(np.uint8)
    det = out.reshape(-1, out.shape[-1]).astype(np.float32)
    if det.shape[1] != 85:
        print(f"error: --yolox expects [N, 85] detections, output 0 has shape {list(out.shape)}")
        return 1
    if args.yolox_raw:
        det = decode_raw(det, hwc.shape[0])
    dets = detect(det, args.score_thr, args.nms_thr)
    print(f"\n{len(dets)} detection(s) with score >= {args.score_thr}:")
    for x1, y1, x2, y2, score, c in dets:
        name = COCO[c] if c < len(COCO) else str(c)
        print(f"  {score:.3f}  class {c:2d} {name:<14s} box ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")
    img = draw_detections(hwc, dets, args.bgr)
    if data.dtype != np.uint8:
        # keep the untouched pixels exactly as they were in the input; only the drawn ones are replaced
        drawn = (img != hwc).any(axis=2)
        merged = orig.copy()
        merged[drawn] = img[drawn].astype(data.dtype)
        img = merged
    res = np.ascontiguousarray(img.transpose(2, 0, 1)[None] if nchw else img[None])
    path = f"{args.output_prefix}yolox.rgb"
    res.tofile(path)
    print(f"saved {os.path.abspath(path)}: raw {res.dtype} {list(res.shape)} ({res.nbytes} bytes), "
          f"colors by score: <0.4 red, <0.6 orange, <0.8 yellow, >=0.8 green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
