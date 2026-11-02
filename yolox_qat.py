#!/usr/bin/env python3
"""Calibrate the fake-quant YOLOX (nano) model: the starting point of QAT (yolox_qat_train.py) and a PTQ baseline.

Needs util/qat_generic.py (model independent fake-quant code) and util/qat_yolox.py (YOLOX specific code) next to
this script, and the yolox package installed:

    python yolox_qat.py -c yolox_nano.pth

What it does:
  1. loads the FP32 checkpoint and folds BatchNorm into the convolutions (yolox.utils.fuse_model);
  2. replaces every folded BaseConv by a quantizable one: per-output-channel fake-quant on the weights,
     per-tensor fake-quant on the output (after SiLU), on the residual adds and on concat inputs;
  3. calibrates the activation ranges on N training images like nncase's PTQ (--calib-method kld: KL divergence
     of 8192-bin histograms, nncase's default; noclip: plain min / max);
  4. evaluates on val2017: weights-only fake-quant, then weights + activations;
  5. saves the calibrated model (qat_init.pth) and the ranges (ranges.json).

The range / scale / zero-point rules copy nncase (QuantUtility.GetQuantParam, FixupRange,
GetWeightsRangesByChannel, Quantizer.GetOptRanges), so the numbers should match what the K230 compile path derives.
Weights and activations are uint8 (the nncase defaults). pth_to_onnx.py exports qat_init.pth like a trained checkpoint.
"""
import argparse
import os

import torch

from yolox.exp import get_exp

from util.qat_generic import calibrate, describe, dump_ranges, set_state
from util.qat_yolox import evaluate, fold, make_calib_loader, make_quantizable


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--name", default="yolox-nano", help="YOLOX experiment name")
    ap.add_argument("-c", "--ckpt", required=True, help="FP32 YOLOX checkpoint (e.g. yolox_nano.pth)")
    ap.add_argument("-b", "--batch-size", type=int, default=32, help="calibration and evaluation batch size")
    ap.add_argument("--calib-images", type=int, default=512, help="random train2017 images to calibrate on")
    ap.add_argument("--calib-method", choices=("kld", "noclip"), default="kld",
                    help="activation range selection, as nncase's --calib-method (default: kld)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--evals", default="w,wa", help="comma list: 'w' weights only, 'wa' weights + activations ('' = none)")
    ap.add_argument("--out", default="qat_out")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (the YOLOX evaluator uses CUDA tensors)")
    exp = get_exp(None, args.name)
    exp.data_num_workers = args.workers
    os.makedirs(args.out, exist_ok=True)

    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()

    qat = make_quantizable(fold(model)).cuda().eval()
    print(describe(qat))
    calibrate(qat, make_calib_loader(exp, args.calib_images, args.batch_size, args.workers), args.calib_method)

    evals = [e for e in args.evals.split(",") if e]
    if "w" in evals:
        set_state(qat, weights=True, acts=False)
        evaluate(exp, qat, args.batch_size, "weights only")
    if "wa" in evals:
        set_state(qat, weights=True, acts=True)
        evaluate(exp, qat, args.batch_size, "weights + activations")

    torch.save({"model": qat.state_dict(), "calib_method": args.calib_method}, os.path.join(args.out, "qat_init.pth"))
    dump_ranges(qat, os.path.join(args.out, "ranges.json"))
    print("saved", os.path.join(args.out, "qat_init.pth"), "and ranges.json")


if __name__ == "__main__":
    main()
