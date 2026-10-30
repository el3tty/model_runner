#!/usr/bin/env python3
"""Fake-quant building blocks for QAT of YOLOX (nano) + "quantize without training" baseline.

Needs util\qat_generic.py (model independent fake-quant code) and util\qat_yolox.py (YOLOX specific code) next to
this script, and the yolox package installed:

    python yolox_qat.py -c yolox_nano.pth --check-fold

What it does:
  1. loads the FP32 checkpoint and folds BatchNorm into the convolutions (yolox.utils.fuse_model);
  2. (--check-fold) compares FP32 and folded outputs on a few images;
  3. replaces every folded BaseConv by a quantizable one: per-output-channel fake-quant on the weights,
     per-tensor fake-quant on the output (after SiLU), on the residual adds and on concat inputs;
  4. calibrates the activation ranges on N training images (no gradients);
  5. evaluates on val2017: weights-only fake-quant, then weights + activations;
  6. saves the calibrated model (qat_init.pth) and the ranges (ranges.json).

The range / scale / zero-point rules copy nncase (QuantUtility.GetQuantParam, FixupRange,
GetWeightsRangesByChannel), so the numbers should match what the K230 compile path derives.
Weights and activations are uint8 (the nncase defaults; the int8 mode was removed).
"""
import argparse
import os
from dataclasses import asdict

import torch

from yolox.exp import get_exp

from util.qat_generic import QuantConfig, calibrate, check_fold, describe, dump_ranges, set_state
from util.qat_yolox import evaluate, export_onnx, fold, make_calib_loader, make_quantizable


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--name", default="yolox-nano")
    ap.add_argument("-f", "--exp-file", default=None)
    ap.add_argument("-c", "--ckpt", required=True)
    ap.add_argument("-b", "--batch-size", type=int, default=32, help="evaluation batch size")
    ap.add_argument("--calib-images", type=int, default=512)
    ap.add_argument("--calib-batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-tie", action="store_true", help="do not share ranges at concats")
    ap.add_argument("--quant-head-outputs", action="store_true")
    ap.add_argument("--check-fold", action="store_true", help="compare FP32 and folded outputs")
    ap.add_argument("--evals", default="w,wa", help="comma list: 'w' weights only, 'wa' weights + activations ('' = none)")
    ap.add_argument("--out", default="qat_out")
    ap.add_argument("--stop-before-calib", action="store_true",
                    help="exit after building the fake-quant model, before calibration (no dataset needed unless "
                         "--check-fold); with --export-onnx the Q/DQ nodes get placeholder ranges")
    ap.add_argument("--export-onnx", default=None, metavar="PATH",
                    help="export the model to ONNX with Q/DQ pairs (calibrated ranges, or placeholders with "
                         "--stop-before-calib)")
    ap.add_argument("--export-float", action="store_true",
                    help="with --stop-before-calib --export-onnx: export the plain float graph, without Q/DQ nodes")
    ap.add_argument("--opset", type=int, default=13, help="ONNX opset (per-channel Q/DQ needs >= 13)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (the YOLOX evaluator uses CUDA tensors)")
    exp = get_exp(args.exp_file, args.name)
    exp.data_num_workers = args.workers
    cfg = QuantConfig(quantize_head_outputs=args.quant_head_outputs, tie_concats=not args.no_tie)
    print("config:", asdict(cfg))
    os.makedirs(args.out, exist_ok=True)

    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()

    loader = first = None
    if args.check_fold or not args.stop_before_calib:
        loader = make_calib_loader(exp, args.calib_images, args.calib_batch_size, args.workers)
        first = next(iter(loader))[0].float().cuda()

    folded = fold(model)
    if args.check_fold:
        check_fold(model, folded, first)

    qat = make_quantizable(fold(model), cfg).cuda().eval()
    set_state(qat, weights=False, acts=False, observe=False)
    if args.check_fold:
        with torch.no_grad():
            d = (qat(first) - folded(first)).abs().max().item()
        print(f"fake-quant model with quantizers disabled vs folded model: max |diff| = {d:.3e}")
    print(describe(qat))

    if args.stop_before_calib:
        if args.export_onnx:
            if not args.export_float:
                # Q/DQ nodes are emitted, but the activation ranges are placeholders (nothing was observed)
                set_state(qat, weights=True, acts=True, observe=False)
            export_onnx(exp, qat, args.export_onnx, args.opset)
        return

    calibrate(qat, loader)

    evals = [e for e in args.evals.split(",") if e]
    if "w" in evals:
        set_state(qat, weights=True, acts=False, observe=False)
        evaluate(exp, qat, args.batch_size, "weights only")
    if "wa" in evals:
        set_state(qat, weights=True, acts=True, observe=False)
        evaluate(exp, qat, args.batch_size, "weights + activations")

    set_state(qat, weights=True, acts=True, observe=False)
    torch.save({"model": qat.state_dict(), "cfg": asdict(cfg)}, os.path.join(args.out, "qat_init.pth"))
    dump_ranges(qat, os.path.join(args.out, "ranges.json"))
    print("saved", os.path.join(args.out, "qat_init.pth"), "and ranges.json")
    if args.export_onnx:
        export_onnx(exp, qat, args.export_onnx, args.opset)  # quantizers enabled: Q/DQ graph


if __name__ == "__main__":
    main()