#!/usr/bin/env python3
"""COCO val2017 AP of a QAT checkpoint (written by yolox_qat_train.py) evaluated in several ways, in one pass over the
images:

    float  quantizers off (the QAT weights in float)
    qdq    fake quantization as in training (= the exported QDQ ONNX model, yolox-qat-qdq.onnx)
    k230   fake quantization plus the K230 arithmetic between the quantization points (util/k230_emul.py: PWL SiLU
           and Sigmoid, output quantization folded into f16 coefficients, f16 intermediates), close to what the board
           and the nncase simulator compute

    python eval_k230.py ../qat-run/best.pth
    python eval_k230.py ../qat-run/best.pth --images 500 --modes float,qdq,k230
    python eval_k230.py ../qat-run/best.pth --check yolox-f32-bgr.in output0.txt --images 0

The difference between the qdq and k230 AP is what the K230 arithmetic costs this checkpoint. --images N evaluates the
first N val2017 images only (COCOeval restricted to them), 0 skips the COCO evaluation.

--check INPUT REFERENCE runs one raw f32 1x3xHxW tensor (tensor_from_image.py --type f32) through every mode and
compares the raw head output [1, N, 85] (no box decoding, as exported by pth_to_onnx.py) with a reference output of
the kmodel: output0.txt of run-model.py (board or simulator) or a runtime dump; lines with a ':' (dump headers) are
skipped.

Needs a CUDA device (the YOLOX evaluator data pipeline), torch, pycocotools and the yolox package with COCO in
YOLOX_DATADIR (or yolox's datasets/COCO).
"""
import argparse
import contextlib
import io
import time

import numpy as np
import torch

from util.qat_generic import set_state
from util.qat_yolox import load_qat_checkpoint, set_k230

MODES = ("float", "qdq", "k230")


def set_mode(model, mode):
    set_state(model, weights=mode != "float", acts=mode != "float", observe=False)
    set_k230(model, mode == "k230")


def load_reference(path):
    with open(path) as f:
        return np.array([float(s) for s in f if s.strip() and ":" not in s], dtype=np.float64)


@torch.no_grad()
def check(exp, model, modes, input_path, ref_path):
    x = np.fromfile(input_path, dtype="<f4").reshape(1, 3, *exp.test_size)
    ref = load_reference(ref_path)
    head = model.head
    old = head.decode_in_inference
    head.decode_in_inference = False
    try:
        print(f"check: {input_path} -> raw head output vs {ref_path}")
        for mode in modes:
            set_mode(model, mode)
            out = model(torch.from_numpy(x).cuda()).double().cpu().numpy().reshape(-1)
            if out.size != ref.size:
                raise SystemExit(f"{ref_path} has {ref.size} values, the model output has {out.size}")
            d = out - ref
            o, r = out.reshape(-1, 85), ref.reshape(-1, 85)
            score = lambda a: a[:, 4] * a[:, 5:].max(1)
            print(f"  {mode:6s} rms {np.sqrt((d ** 2).mean()):.5f}  max|d| {np.abs(d).max():.4f}  "
                  f"score rms {np.sqrt(((score(o) - score(r)) ** 2).mean()):.5f}")
    finally:
        head.decode_in_inference = old


@torch.no_grad()
def coco_eval(exp, model, modes, n_images, batch_size, workers):
    from pycocotools.cocoeval import COCOeval
    from yolox.utils import postprocess

    evaluator = exp.get_evaluator(batch_size, False)
    ds = evaluator.dataloader.dataset
    idx = range(len(ds)) if n_images <= 0 or n_images >= len(ds) else range(n_images)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, list(idx)), batch_size=batch_size,
                                         shuffle=False, num_workers=workers, pin_memory=True)
    dets = {m: [] for m in modes}
    img_ids = []
    t0 = time.time()
    for i, (imgs, _, info_imgs, ids) in enumerate(loader):
        imgs = imgs.float().cuda()
        img_ids += [int(v) for v in ids]
        for mode in modes:
            set_mode(model, mode)
            out = postprocess(model(imgs), exp.num_classes, exp.test_conf, exp.nmsthre)
            dets[mode] += evaluator.convert_to_coco_format(out, info_imgs, ids)
        if (i + 1) % 10 == 0:
            print(f"  {len(img_ids)}/{len(idx)} images, {time.time() - t0:.0f} s", flush=True)

    results = {}
    for mode in modes:
        if not dets[mode]:
            results[mode] = (0.0, 0.0)
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            e = COCOeval(ds.coco, ds.coco.loadRes(dets[mode]), "bbox")
            e.params.imgIds = img_ids
            e.evaluate()
            e.accumulate()
            e.summarize()
        results[mode] = (e.stats[0], e.stats[1])
    print(f"\nCOCO val2017, {len(img_ids)} images:")
    for mode in modes:
        ap, ap50 = results[mode]
        print(f"  {mode:6s} AP(0.50:0.95) {ap:.4f}   AP50 {ap50:.4f}")
    if "qdq" in results and "k230" in results:
        print(f"  k230 - qdq: AP {results['k230'][0] - results['qdq'][0]:+.4f}   "
              f"AP50 {results['k230'][1] - results['qdq'][1]:+.4f}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="checkpoint written by yolox_qat_train.py (e.g. ../qat-run/best.pth)")
    ap.add_argument("-n", "--name", default="yolox-nano", help="YOLOX experiment name (default: yolox-nano)")
    ap.add_argument("-f", "--exp-file", default=None, help="YOLOX experiment file (instead of --name)")
    ap.add_argument("--modes", default="qdq,k230", help=f"comma separated, of {', '.join(MODES)} (default: qdq,k230)")
    ap.add_argument("--images", type=int, default=5000, help="first N val2017 images (default: all 5000; 0: none)")
    ap.add_argument("-b", "--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check", nargs=2, metavar=("INPUT", "REFERENCE"),
                    help="compare the raw output for one input tensor with a kmodel output (board / simulator)")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"unknown mode {m!r} (use {', '.join(MODES)})")

    # cuDNN uses TF32 (10 bit mantissa, coarser than f16) for convolutions by default: it would add its own rounding
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    exp, model, cfg, ckpt = load_qat_checkpoint(args.ckpt, args.name, args.exp_file)
    print(f"{args.ckpt}: epoch {ckpt.get('epoch', -1) + 1}, training AP(0.50:0.95) "
          f"{ckpt.get('ap50_95', float('nan')):.4f}")
    model.cuda().eval()
    if args.check:
        check(exp, model, modes, *args.check)
    if args.images != 0:
        coco_eval(exp, model, modes, args.images, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
