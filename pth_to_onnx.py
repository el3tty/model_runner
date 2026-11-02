#!/usr/bin/env python3
"""Export a QAT-trained YOLOX checkpoint (qat-run/best.pth or latest.pth, written by yolox_qat_train.py) to ONNX.

    python pth_to_onnx.py qat-run/best.pth yolox-qat-best.onnx
    python pth_to_onnx.py qat-run/best.pth yolox-qat-best.float.onnx --float
    python pth_to_onnx.py qat-run/best.pth yolox-qat-new.onnx --new-exporter --opset 18 --example yolox-f32-bgr.in

Needs util/qat_generic.py and util/qat_yolox.py (the shared QAT code) next to this script, and torch + the yolox
package. Runs
on the CPU.

The exported graph has per-channel quantized weights; input 1x3x416x416 float32 (raw 0..255 BGR); output
[1, 3549, 85] without box decoding (raw dx, dy, log w, log h, sigmoid objectness, sigmoid class scores). --float
exports the same trained weights without any weight quantization, a float reference of the QAT result. The warnings
that are expected during the export are hidden.

By default the per-channel weight quantization is baked into the Conv weights (they are stored as the fake-quantized
float values, like the weights of yolov5s_scheme.onnx), so every Conv is a plain Conv node. --weight-qdq keeps the
Q/DQ chain on the weights instead. The activations get QuantizeLinear/DequantizeLinear pairs (the learned ranges)
only with --act-qdq.

Unless --float is given, an nncase quant scheme is written next to the model: output + ".qs.json" (e.g.
yolox-qat.onnx.qs.json), in the format of nncase's QuantScheme.json, read back from the Q/DQ nodes of the exported
model (see util/quant_scheme.py; no ranges are calibrated). Without --act-qdq the scheme is read from a temporary
export that has the activation Q/DQ pairs, so it still holds the learned activation ranges. Use --no-scheme to skip
it.
"""
import argparse
import os
import tempfile

import numpy as np

from util.qat_generic import describe, set_state
from util.qat_yolox import export_onnx, load_qat_checkpoint
from util.quant_scheme import write_scheme


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="checkpoint written by yolox_qat_train.py (e.g. qat-run/best.pth)")
    ap.add_argument("output", help="output .onnx file")
    ap.add_argument("-n", "--name", default="yolox-nano", help="YOLOX experiment name (default: yolox-nano)")
    ap.add_argument("-f", "--exp-file", default=None, help="YOLOX experiment file (instead of --name)")
    ap.add_argument("--opset", type=int, default=13, help="ONNX opset (per-channel Q/DQ needs >= 13)")
    ap.add_argument("--new-exporter", action="store_true",
                    help="use the new torch.export based exporter (PyTorch >= 2.5; untested with fake-quant, "
                         "try --opset 18; check the result with run-onnx.py)")
    ap.add_argument("--example", default=None, metavar="FILE",
                    help="raw f32 1x3xHxW tensor file used as the example input of the export instead of zeros, "
                         "e.g. yolox-f32-bgr.in")
    ap.add_argument("--weight-qdq", action="store_true",
                    help="keep Constant -> QuantizeLinear -> DequantizeLinear in front of every Conv weight; by "
                         "default the weight quantization is baked into the Conv weights (fake-quantized values) "
                         "and only the activations get Q/DQ pairs (with --act-qdq)")
    ap.add_argument("--act-qdq", action="store_true",
                    help="put QuantizeLinear -> DequantizeLinear pairs (the learned ranges) on the activations; by "
                         "default the activations stay float")
    ap.add_argument("--no-scheme", action="store_true", help="do not write the nncase quant scheme (output.qs.json)")
    ap.add_argument("--float", action="store_true", help="export without Q/DQ nodes (float graph, QAT weights)")
    args = ap.parse_args()

    exp, qat, ckpt = load_qat_checkpoint(args.ckpt, args.name, args.exp_file)
    print(f"checkpoint: epoch {ckpt.get('epoch', -1) + 1}, AP(0.50:0.95) {ckpt.get('ap50_95', float('nan')):.4f}, "
          f"AP50 {ckpt.get('ap50', float('nan')):.4f}")
    # the quantizer on/off flags are not part of the state_dict
    set_state(qat, weights=not args.float, acts=args.act_qdq and not args.float)
    print(describe(qat))

    example = None
    if args.example:
        example = np.fromfile(args.example, dtype="<f4").reshape(1, 3, *exp.test_size)
    export_onnx(exp, qat, args.output, args.opset, args.new_exporter, example, quiet=True,
                bake_conv_weights=not args.weight_qdq)
    if not args.float and not args.no_scheme:
        if args.act_qdq:
            write_scheme(args.output, args.output + ".qs.json")
        else:
            # the activation ranges are only in the graph as Q/DQ pairs: read them from a temporary export that has them
            set_state(qat, weights=True, acts=True)
            with tempfile.TemporaryDirectory() as tmp:
                tmp_onnx = os.path.join(tmp, "act-qdq.onnx")
                export_onnx(exp, qat, tmp_onnx, args.opset, args.new_exporter, example, quiet=True,
                            bake_conv_weights=not args.weight_qdq)
                write_scheme(tmp_onnx, args.output + ".qs.json")


if __name__ == "__main__":
    main()
