"""
COCO val2017 evaluation helpers shared by coco_prepare.py (PC), coco_run.py (board / simulator) and coco_eval.py (PC).

Only numpy is imported at module level, so coco_run.py runs on the board with numpy and nncaseruntime alone; the
preprocessing needs OpenCV / Pillow and the AP needs pycocotools, imported where they are used.

Prepared input folder (coco_prepare.py):
    <id>.u8        raw uint8 1x3xHxW tensor per image, in the channel order of the model, 0..255
    index.json     {"model": <detect name>, "size": [W, H], "images": [{"id", "file", "width", "height",
                    "sx", "sy", "dx", "dy"}, ...]}; a box (x, y) in input pixels maps back to the original image as
                    ((x - dx) / sx, (y - dy) / sy)

Candidate file (coco_run.py, <name>.npz): the raw outputs are cut into cells of 85 values (a YOLOX row, or one anchor
of an anchor-head grid cell, as cells() lays them out; for YOLO26 the box and class maps of a stride are first joined
into one [1, 84, H, W] output, yolo_post.join_split, and a cell is one grid position) and only the cells whose score
objectness * best class score can reach --conf are kept:
    model, conf    the --detect name and the threshold
    ids [M]        image ids
    shape<o>       shape of output o
    rows<o> [M]    number of cells of output o kept per image
    index<o> [R]   cell index within output o (int32)
    values<o> [R, 85]  the raw cell values (float16; [R, 84] for YOLO26)
"""
import json
import os

import numpy as np

from util import yolo_post

# Input geometry and evaluation defaults per --detect model:
#   size       model input (W, H)
#   letterbox  "topleft" (YOLOX: resized image in the top-left corner), "center" (keras-yolo3) or None (stretch)
#   pad        padding gray value
#   resample   resize method: "cv2-linear" (cv2.INTER_LINEAR, BGR image as read by cv2), "cv2-ultralytics"
#              (Ultralytics val: long side to size with INTER_AREA when shrinking, INTER_LINEAR when enlarging) or
#              "pil-bicubic"
#   conf, nms  candidate score threshold and NMS IoU of the reference evaluations (YOLOX test_conf / nmsthre,
#              darknet detector map)
#   multi_label  one detection per class above conf (darknet, keras-yolo3) instead of the best class only (YOLOX)
PREP = {
    "yolox": dict(size=(416, 416), letterbox="topleft", pad=114, resample="cv2-linear", conf=0.01, nms=0.65,
                  multi_label=False),
    "yolov3": dict(size=(416, 416), letterbox="center", pad=128, resample="pil-bicubic", conf=0.005, nms=0.45,
                   multi_label=True),
    "yolo-fastest": dict(size=(320, 320), letterbox=None, pad=0, resample="cv2-linear", conf=0.005, nms=0.45,
                         multi_label=True),
    # Ultralytics val defaults (conf 0.001, iou 0.7, multi-label), but square 640 input instead of rect batches
    "yolo26": dict(size=(640, 640), letterbox="center", pad=114, resample="cv2-ultralytics", conf=0.001, nms=0.7,
                   multi_label=True),
}

# Fill value of the cells that coco_run.py dropped: YOLOX scores are already probabilities (0 -> score 0), anchor
# heads and YOLO26 are logits (-80 -> sigmoid ~ 0, exp ~ 0, and exp(80) still fits in float32).
EMPTY = {"rows": 0.0, "grid": -80.0, "dense": -80.0}


def kind(model):
    """'rows' for a [1, N, 85] head (YOLOX), 'grid' for [1, A*85, H/s, W/s] anchor heads, 'dense' for the joined
    [1, 84, H/s, W/s] YOLO26 outputs (4 box distances + 80 class logits per grid position, no objectness)."""
    if model == "yolo26":
        return "dense"
    return "rows" if yolo_post.MODELS[model]["anchors"] is None else "grid"


def group(outs, model):
    """The raw model outputs as cut into cells: YOLO26 box / class maps joined per stride, others unchanged."""
    return yolo_post.join_split(outs) if kind(model) == "dense" else outs


# ----------------------------------------------------------------------------
# preprocessing (PC)


def preprocess(path, model):
    """Image file -> (uint8 1x3xHxW tensor in the model's channel order, geometry dict sx, sy, dx, dy)."""
    p = PREP[model]
    w, h = p["size"]
    if p["resample"].startswith("cv2-"):
        import cv2

        img = cv2.imread(path)  # BGR
        if img is None:
            raise SystemExit(f"cannot read {path}")
        ih, iw = img.shape[:2]
        if p["letterbox"] == "topleft":
            # exactly YOLOX: COCODataset.load_resized_img, then ValTransform (preproc) resizes once more
            r0 = min(h / ih, w / iw)
            img = cv2.resize(img, (int(iw * r0), int(ih * r0)), interpolation=cv2.INTER_LINEAR)
            r = min(h / img.shape[0], w / img.shape[1])
            img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR)
            out = np.full((h, w, 3), p["pad"], np.uint8)
            out[:img.shape[0], :img.shape[1]] = img
            geo = dict(sx=r0, sy=r0, dx=0.0, dy=0.0)  # YOLOX maps back with min(H / ih, W / iw)
        elif p["resample"] == "cv2-ultralytics" and p["letterbox"] == "center":
            # Ultralytics BaseDataset.load_image (long side -> size), then LetterBox(center=True), no further resize
            r = min(w, h) / max(ih, iw)
            nw, nh = min(int(np.ceil(iw * r)), w), min(int(np.ceil(ih * r)), h)
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            img = cv2.resize(img, (nw, nh), interpolation=interp) if (nw, nh) != (iw, ih) else img
            top, left = round((h - nh) / 2 - 0.1), round((w - nw) / 2 - 0.1)
            out = np.full((h, w, 3), p["pad"], np.uint8)
            out[top:top + nh, left:left + nw] = img
            geo = dict(sx=nw / iw, sy=nh / ih, dx=float(left), dy=float(top))
        elif p["letterbox"] is None:
            out = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            geo = dict(sx=w / iw, sy=h / ih, dx=0.0, dy=0.0)
        else:
            raise ValueError(f"{model}: letterbox {p['letterbox']!r} with cv2 is not implemented")
        if not yolo_post.MODELS[model]["bgr"]:
            out = out[:, :, ::-1]
    else:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        iw, ih = img.size
        if p["letterbox"] != "center":
            raise ValueError(f"{model}: letterbox {p['letterbox']!r} with Pillow is not implemented")
        # keras-yolo3 letterbox_image (as in the ONNX model zoo YOLOv3 preprocessing)
        s = min(w / iw, h / ih)
        nw, nh = int(iw * s), int(ih * s)
        canvas = Image.new("RGB", (w, h), (p["pad"],) * 3)
        canvas.paste(img.resize((nw, nh), Image.BICUBIC), ((w - nw) // 2, (h - nh) // 2))
        out = np.asarray(canvas, np.uint8)
        geo = dict(sx=s, sy=s, dx=float((w - nw) // 2), dy=float((h - nh) // 2))
        if yolo_post.MODELS[model]["bgr"]:
            out = out[:, :, ::-1]
    return np.ascontiguousarray(out.transpose(2, 0, 1)[None]), geo


def load_index(folder):
    with open(os.path.join(folder, "index.json")) as f:
        return json.load(f)


def read_input(folder, entry):
    """The uint8 1x3xHxW tensor of an index entry."""
    return np.fromfile(os.path.join(folder, entry["file"]), np.uint8)


def model_input(u8, dtype, white):
    """uint8 0..255 -> the model input: dtype with a white pixel = `white` (255: unchanged values, 1: / 255)."""
    if white == 255.0:
        return u8.astype(dtype)
    return (u8.astype(np.float32) / 255.0).astype(dtype)  # = tensor_from_image.py --range 1


# ----------------------------------------------------------------------------
# cells and candidate filter (board and PC, numpy only)


def cells(out, k):
    """Raw output -> [C, 85] cells (a copy for 'grid'; [C, 84] per grid position for 'dense')."""
    if k == "rows":
        return out.reshape(-1, out.shape[-1])
    if k == "dense":
        return out.reshape(out.shape[1], -1).T.copy()
    a = out.shape[1] // 85
    return out.reshape(a, 85, out.shape[2], out.shape[3]).transpose(0, 2, 3, 1).reshape(-1, 85)


def uncells(index, values, shape, k):
    """Inverse of cells() for the kept cells; the others get EMPTY[k]."""
    if k == "rows":
        out = np.full((int(np.prod(shape[:-1])), shape[-1]), EMPTY[k], np.float32)
        out[index] = values
        return out.reshape(shape)
    if k == "dense":
        c = np.full((int(np.prod(shape[2:])), shape[1]), EMPTY[k], np.float32)
        c[index] = values
        return np.ascontiguousarray(c.T).reshape(shape)
    a, gh, gw = shape[1] // 85, shape[2], shape[3]
    c = np.full((a * gh * gw, 85), EMPTY[k], np.float32)
    c[index] = values
    return np.ascontiguousarray(c.reshape(a, gh, gw, 85).transpose(0, 3, 1, 2)).reshape(shape)


def sigmoid(x):
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def candidates(c, k, conf):
    """Indices of the cells whose objectness * best class score >= conf."""
    if k == "rows":
        return np.nonzero(c[:, 4] * c[:, 5:].max(1) >= conf)[0]
    if k == "dense":  # no objectness: sigmoid(best class logit) >= conf
        return np.nonzero(c[:, 4:].max(1) >= np.log(conf / (1.0 - conf)))[0]
    obj = sigmoid(c[:, 4].astype(np.float32))
    idx = np.nonzero(obj >= conf)[0]  # class scores are <= 1: cheap pre-filter on the objectness alone
    score = obj[idx] * sigmoid(c[idx, 5:].max(1).astype(np.float32))
    return idx[score >= conf]


# ----------------------------------------------------------------------------
# post-processing and AP (PC)


def postprocess(det, conf, nms_thr, multi_label, max_dets):
    """det: decoded [N, 85] (cx, cy, w, h in input pixels, objectness, class scores).
    Returns [K, 6] (x1, y1, x2, y2, score, class) after class-aware NMS, best score first."""
    obj, cls = det[:, 4:5], det[:, 5:]
    if multi_label:
        i, c = np.nonzero(obj * cls >= conf)
        score = obj[i, 0] * cls[i, c]
    else:
        c = cls.argmax(1)
        score = obj[:, 0] * cls[np.arange(len(det)), c]
        i = np.nonzero(score >= conf)[0]
        c, score = c[i], score[i]
    if not len(i):
        return np.zeros((0, 6), np.float32)
    d = det[i]
    boxes = np.stack((d[:, 0] - d[:, 2] / 2, d[:, 1] - d[:, 3] / 2, d[:, 0] + d[:, 2] / 2, d[:, 1] + d[:, 3] / 2), 1)
    keep = []
    for cl in np.unique(c):
        m = np.nonzero(c == cl)[0]
        keep += [m[j] for j in yolo_post.nms(boxes[m], score[m], nms_thr)]
    keep = np.array(keep)
    keep = keep[np.argsort(-score[keep], kind="stable")]
    if max_dets > 0:
        keep = keep[:max_dets]
    return np.concatenate((boxes[keep], score[keep, None], c[keep, None]), 1).astype(np.float32)


def to_coco(dets, entry, cat_ids):
    """[K, 6] detections in input pixels -> COCO result dicts in original image pixels."""
    res = []
    for x1, y1, x2, y2, score, c in dets:
        x1, x2 = (x1 - entry["dx"]) / entry["sx"], (x2 - entry["dx"]) / entry["sx"]
        y1, y2 = (y1 - entry["dy"]) / entry["sy"], (y2 - entry["dy"]) / entry["sy"]
        res.append({"image_id": entry["id"], "category_id": cat_ids[int(c)],
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(score)})
    return res


def coco_ap(coco, dets, img_ids):
    """COCOeval bbox stats (12 values: AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl)."""
    import contextlib
    import io

    from pycocotools.cocoeval import COCOeval

    if not dets:
        return np.zeros(12)
    with contextlib.redirect_stdout(io.StringIO()):
        e = COCOeval(coco, coco.loadRes(dets), "bbox")
        e.params.imgIds = img_ids
        e.evaluate()
        e.accumulate()
        e.summarize()
    return e.stats
