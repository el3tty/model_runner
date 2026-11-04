#!/usr/bin/env python3
"""
COCO val2017 AP of detector outputs on the images written by coco_prepare.py: the candidate files written by
coco_run.py (kmodel on the board or in the simulator) and, with --onnx, ONNX models run here with onnxruntime on
exactly the same input tensors (the float / QDQ reference).

Every output is decoded with the model's head (util/yolo_post.py), filtered by objectness * class score >= --conf,
reduced by class-aware NMS (--nms), mapped back to the original image and scored with pycocotools COCOeval. The
defaults are the model's own evaluation settings (util/coco.py PREP): yolox conf 0.01, NMS 0.65, best class only
(YOLOX); yolov3 and yolo-fastest conf 0.005, NMS 0.45, one detection per class above conf (darknet).

Usage:
    python coco_eval.py --input coco-yolox dets-yolox.npz [more.npz ...] [--onnx yolox.onnx ...]
                        --coco <COCO> (or $COCO_DIR) [--conf C] [--nms T] [--max-dets 100] [--limit N]
"""
import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np

from util import coco as cu
from util import yolo_post



def load_candidates(path):
    """Candidate file -> (model, conf, {image id: [raw output arrays]})."""
    with np.load(path) as f:  # NpzFile re-reads an array on every key access: load each one once
        z = {k: f[k] for k in f.files}
    model, k = str(z["model"]), cu.kind(str(z["model"]))
    n_out = sum(1 for key in z if key.startswith("shape"))
    starts = [np.concatenate([[0], np.cumsum(z[f"rows{o}"])]) for o in range(n_out)]

    def outputs(m):
        return [cu.uncells(z[f"index{o}"][starts[o][m]:starts[o][m + 1]],
                           z[f"values{o}"][starts[o][m]:starts[o][m + 1]].astype(np.float32),
                           tuple(z[f"shape{o}"]), k) for o in range(n_out)]

    return model, float(z["conf"]), {int(i): (lambda m=m: outputs(m)) for m, i in enumerate(z["ids"])}


def onnx_outputs(path, folder, size, white):
    """{image id: lazy [raw outputs]} of an ONNX model run on the prepared inputs."""
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    dtype = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(uint8)": np.uint8}[inp.type]
    w, h = size

    def run(e):
        x = cu.read_input(folder, e).reshape(1, 3, h, w)
        return sess.run(None, {inp.name: cu.model_input(x, dtype, white)})

    return run


def evaluate(label, get_outputs, images, model, args, coco, cat_ids, size):
    p = cu.PREP[model]
    conf = p["conf"] if args.conf is None else args.conf
    nms_thr = p["nms"] if args.nms is None else args.nms
    multi = p["multi_label"] if args.multi_label is None else args.multi_label == "on"
    m = yolo_post.MODELS[model]
    w, h = size
    dets = []
    t0 = time.time()
    for n, e in enumerate(images):
        det = m["decode"]([np.asarray(o) for o in get_outputs(e)], h, w, m["anchors"])
        dets += cu.to_coco(cu.postprocess(det, conf, nms_thr, multi, args.max_dets), e, cat_ids)
        if (n + 1) % 1000 == 0:
            print(f"  {label}: {n + 1}/{len(images)} images ({time.time() - t0:.0f} s)", flush=True)
    stats = cu.coco_ap(coco, dets, [e["id"] for e in images])
    return stats, f"conf {conf}, NMS {nms_thr}, {'multi-label' if multi else 'best class'}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dets", nargs="*", help="candidate files written by coco_run.py (.npz)")
    ap.add_argument("--input", required=True, help="folder written by coco_prepare.py (index.json, <id>.u8)")
    ap.add_argument("--onnx", action="append", default=[], help="ONNX model to evaluate on the same inputs "
                                                                "(repeatable)")
    # the COCO 2017 dataset root (cocodataset.org: val2017/ images, annotations/instances_val2017.json); the
    # ground truth of the evaluation
    ap.add_argument("--coco", default=os.environ.get("COCO_DIR"), required="COCO_DIR" not in os.environ,
                    help="COCO root with annotations/ (default: $COCO_DIR)")
    ap.add_argument("--conf", type=float, default=None, help="score threshold (default: the model's, see above)")
    ap.add_argument("--nms", type=float, default=None, help="NMS IoU threshold (default: the model's)")
    ap.add_argument("--multi-label", choices=("on", "off"), default=None,
                    help="one detection per class above conf (on) or the best class only (off); default: the model's")
    ap.add_argument("--max-dets", type=int, default=0,
                    help="keep the N best detections per image (default: 0 = all; COCOeval scores the best 100 "
                         "per class)")
    ap.add_argument("--limit", type=int, default=0, help="first N prepared images only (default: 0 = all)")
    args = ap.parse_args()
    if not args.dets and not args.onnx:
        ap.error("nothing to evaluate: give candidate files and/or --onnx models")

    from pycocotools.coco import COCO

    meta = cu.load_index(args.input)
    model, size = meta["model"], meta["size"]
    images = meta["images"][:args.limit] if args.limit > 0 else meta["images"]
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(os.path.join(args.coco, "annotations", "instances_val2017.json"))
    cat_ids = sorted(coco.getCatIds())  # class index 0..79 -> COCO category id

    results = []
    for path in args.dets:
        det_model, det_conf, outs = load_candidates(path)
        if det_model != model:
            raise SystemExit(f"{path} holds {det_model} outputs, {args.input} was prepared for {model}")
        conf = cu.PREP[model]["conf"] if args.conf is None else args.conf
        if conf < det_conf:
            print(f"warning: {path} only kept cells with score >= {det_conf}; --conf {conf} cannot see lower ones")
        sel = [e for e in images if e["id"] in outs]
        if len(sel) < len(images):
            print(f"note: {path} has outputs for {len(sel)} of the {len(images)} images; evaluating those")
        stats, how = evaluate(os.path.basename(path), lambda e: outs[e["id"]](), sel, model, args, coco, cat_ids,
                              size)
        results.append((os.path.basename(path), len(sel), stats, how))
    for path in args.onnx:
        run = onnx_outputs(path, args.input, size, yolo_post.MODELS[model]["white"])
        stats, how = evaluate(os.path.basename(path), run, images, model, args, coco, cat_ids, size)
        results.append((os.path.basename(path), len(images), stats, how))

    print(f"\nCOCO val2017, {model} inputs from {args.input}:")
    print(f"  {'':36s} {'images':>6s}  {'AP':>6s} {'AP50':>6s} {'AP75':>6s} {'APs':>6s} {'APm':>6s} {'APl':>6s}")
    for label, n, s, how in results:
        print(f"  {label:36s} {n:6d}  {s[0]:6.4f} {s[1]:6.4f} {s[2]:6.4f} {s[3]:6.4f} {s[4]:6.4f} {s[5]:6.4f}"
              f"   ({how})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
