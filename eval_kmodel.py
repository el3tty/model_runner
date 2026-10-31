#!/usr/bin/env python3
"""COCO val2017 AP of a YOLOX kmodel run on the K230 (or the nncase simulator), next to the PyTorch evaluation of the
QAT checkpoint on exactly the same input tensors.

1. prepare: write the first N val2017 images, preprocessed by YOLOX itself (ValTransform: letterbox to the test size,
   pad 114, BGR, no normalization), as raw uint8 1x3xHxW files <dir>/<image id>.u8 plus <dir>/index.json:
       python eval_kmodel.py prepare coco-val --images 5000
2. on the board (run-coco.py, numpy + nncaseruntime only):
       python run-coco.py yolox-qat.kmodel coco-val dets-yolox-qat
3. eval: decode the kmodel's candidate rows with the YOLOX head, the YOLOX postprocess (test_conf, NMS) and COCOeval
   restricted to the prepared images; --ckpt also evaluates the QAT checkpoint on the same tensors in PyTorch, with
   the quantizers off (float) and/or on (qdq, = the exported QDQ ONNX model):
       python eval_kmodel.py eval coco-val dets-yolox-qat.npz --ckpt ../qat-run/best.pth --modes float,qdq

Needs torch, pycocotools and the yolox package with COCO in YOLOX_DATADIR (or yolox's datasets/COCO); --ckpt needs a
CUDA device.
"""
import argparse
import contextlib
import io
import json
import os

import numpy as np
import torch

from yolox.exp import get_exp
from yolox.utils import postprocess


def val_dataset(exp):
    return exp.get_evaluator(1, False).dataloader.dataset


def prepare(args):
    exp = get_exp(args.exp_file, args.name)
    ds = val_dataset(exp)
    n = len(ds) if args.images <= 0 else min(args.images, len(ds))
    os.makedirs(args.dir, exist_ok=True)
    index = []
    for i in range(n):
        img, _, info, img_id = ds[i]
        img_id = int(np.asarray(img_id).reshape(-1)[0])
        if img.min() < 0 or img.max() > 255 or not np.array_equal(img, np.round(img)):
            raise SystemExit(f"image {img_id}: the preprocessed values are not integers in 0..255")
        img.astype(np.uint8).tofile(os.path.join(args.dir, f"{img_id}.u8"))
        index.append({"id": img_id, "height": int(info[0]), "width": int(info[1])})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    with open(os.path.join(args.dir, "index.json"), "w") as f:
        json.dump({"test_size": list(exp.test_size), "images": index}, f)
    size = sum(os.path.getsize(os.path.join(args.dir, f"{e['id']}.u8")) for e in index)
    print(f"wrote {n} images ({size / 2**20:.0f} MB) to {args.dir}")


def kmodel_outputs(npz_path, ids, n_anchors):
    """Raw [n_anchors, 85] output per image id, rebuilt from the candidate rows (other rows: score 0)."""
    with np.load(npz_path) as f:  # NpzFile re-reads an array on every key access: load each one once
        z = {k: f[k] for k in f.files}
    starts = np.concatenate([[0], np.cumsum(z["rows"])])
    by_id = {int(i): k for k, i in enumerate(z["ids"])}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"{npz_path} has no output for {len(missing)} of the images (e.g. {missing[:3]})")
    for img_id in ids:
        k = by_id[img_id]
        out = np.zeros((n_anchors, 85), np.float32)
        out[z["index"][starts[k]:starts[k + 1]]] = z["values"][starts[k]:starts[k + 1]].astype(np.float32)
        yield out


def coco_ap(ds, dets, img_ids):
    from pycocotools.cocoeval import COCOeval

    if not dets:
        return 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        e = COCOeval(ds.coco, ds.coco.loadRes(dets), "bbox")
        e.params.imgIds = img_ids
        e.evaluate()
        e.accumulate()
        e.summarize()
    return e.stats[0], e.stats[1]


@torch.no_grad()
def evaluate(args):
    torch.set_num_threads(4)  # the per-image tensors are tiny: with every CPU thread most of the time is spent syncing
    exp = get_exp(args.exp_file, args.name)
    evaluator = exp.get_evaluator(1, False)
    ds = evaluator.dataloader.dataset
    with open(os.path.join(args.dir, "index.json")) as f:
        meta = json.load(f)
    images = meta["images"]
    h, w = meta["test_size"]
    ids = [e["id"] for e in images]

    head = exp.get_model().head  # only for decode_outputs: the anchor grid of the test size
    head.hw = [(h // s, w // s) for s in head.strides]
    n_anchors = sum(a * b for a, b in head.hw)

    def finish(raw_or_decoded, decoded, info, img_id):
        out = raw_or_decoded if decoded else head.decode_outputs(raw_or_decoded, raw_or_decoded.type())
        out = postprocess(out, exp.num_classes, exp.test_conf, exp.nmsthre)
        return evaluator.convert_to_coco_format(out, ([info["height"]], [info["width"]]), [img_id])

    results = {}
    for path in args.dets:
        label = os.path.basename(path)
        dets = []
        for n, (info, raw) in enumerate(zip(images, kmodel_outputs(path, ids, n_anchors))):
            dets += finish(torch.from_numpy(raw)[None], False, info, info["id"])
            if (n + 1) % 500 == 0:
                print(f"  {label}: {n + 1}/{len(ids)} images decoded", flush=True)
        results[label] = coco_ap(ds, dets, ids)

    if args.ckpt:
        from util.qat_generic import set_state
        from util.qat_yolox import load_qat_checkpoint

        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        _, model, _, _ = load_qat_checkpoint(args.ckpt, args.name, args.exp_file)
        model.cuda().eval()
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        for m in modes:
            if m not in ("float", "qdq"):
                raise SystemExit(f"unknown mode {m!r} (use float, qdq)")
        dets = {m: [] for m in modes}
        for info in images:
            x = np.fromfile(os.path.join(args.dir, f"{info['id']}.u8"), np.uint8).reshape(1, 3, h, w)
            x = torch.from_numpy(x.astype(np.float32)).cuda()
            for m in modes:
                set_state(model, weights=m == "qdq", acts=m == "qdq", observe=False)
                dets[m] += finish(model(x).float().cpu(), True, info, info["id"])
        for m in modes:
            results[f"{os.path.basename(args.ckpt)} {m}"] = coco_ap(ds, dets[m], ids)

    print(f"\nCOCO val2017, {len(ids)} images (same input tensors):")
    for label, (ap, ap50) in results.items():
        print(f"  {label:32s} AP(0.50:0.95) {ap:.4f}   AP50 {ap50:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--name", default="yolox-nano", help="YOLOX experiment name (default: yolox-nano)")
    ap.add_argument("-f", "--exp-file", default=None, help="YOLOX experiment file (instead of --name)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="write preprocessed val2017 images for run-coco.py")
    p.add_argument("dir")
    p.add_argument("--images", type=int, default=5000, help="first N val2017 images (default: 5000 = all)")
    e = sub.add_parser("eval", help="COCO AP of kmodel outputs (and of a QAT checkpoint on the same tensors)")
    e.add_argument("dir", help="the directory written by prepare (index.json, *.u8)")
    e.add_argument("dets", nargs="*", help="<out>.npz files written by run-coco.py")
    e.add_argument("--ckpt", default=None, help="QAT checkpoint to evaluate on the same tensors (needs CUDA)")
    e.add_argument("--modes", default="qdq", help="PyTorch modes for --ckpt, comma separated: float, qdq "
                                                  "(default: qdq)")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
