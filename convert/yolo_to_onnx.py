#!/usr/bin/env python3
"""
Export an Ultralytics detector checkpoint (YOLO26 yolo26n.pt, YOLO12 yolo12n.pt, ...) to a float ONNX model with the
raw head outputs, for coco_eval.py --onnx and nncase (YOLOX has its own exporter, qat_export.py yolox).

The graph ends at the last conv of every head branch, before the box/class concat (end2end=False, no decoding, no
top-k), so box and class logits keep their own quantization scales on the K230:
    box8  [1, 4, H/8, W/8]    cls8  [1, 80, H/8, W/8]     (likewise 16 and 32)
box = (l, t, r, b) distances from the cell center in units of the stride; cls = class logits (no objectness). That is
YOLO26 (no DFL, reg_max 1); with a DFL head (reg_max 16: YOLOv8, YOLO11, YOLO12) box is [1, 4 * 16, H/s, W/s]: per
side 16 logits, the distance is their softmax-weighted mean of 0..15 (left to the decoder). BatchNorm is folded and
the 1/255 input scaling is folded into the first conv, so the input is RGB 0..255 float32 1x3xHxW (letterbox
centered, pad 114: coco_prepare.py --detect yolo26). convert/onnx_opt.py then replaces the Slice / Split of the C3k2,
C2PSA and attention blocks by separate convs, because nncase runs those on the CPU.

Usage:
    python convert/yolo_to_onnx.py build/weights/yolo26n.pt build/onnx/yolo26n-640.onnx [--size 640] [--opset 13]
    python convert/yolo_to_onnx.py build/weights/yolo12n.pt build/onnx/yolo12n-640.onnx
"""
import argparse
import inspect

import torch
import torch.nn as nn


class RawHead(nn.Module):
    """The network up to the raw (one-to-many) head convs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y = []
        layers = self.model.model
        for m in layers[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.model.save else None)
        det = layers[-1]
        feats = [y[j] for j in det.f]
        outs = []
        for i, f in enumerate(feats):
            outs += [det.cv2[i](f), det.cv3[i](f)]
        return tuple(outs)


def build(weights):
    """Ultralytics checkpoint -> RawHead (eval, BatchNorm folded, input 0..255)."""
    from ultralytics import YOLO

    model = YOLO(weights).model.float().eval()
    model = model.fuse(verbose=False)
    first = model.model[0].conv
    with torch.no_grad():
        first.weight.mul_(1.0 / 255.0)  # zero padding: conv(x / 255, w) == conv(x, w / 255) exactly
    return RawHead(model).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("output")
    ap.add_argument("--size", type=int, default=640, help="square input size (default 640)")
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    net = build(args.weights)
    det = net.model.model[-1]
    names = [f"{k}{int(s)}" for s in det.stride for k in ("box", "cls")]
    dummy = torch.zeros(1, 3, args.size, args.size)
    kwargs = dict(input_names=["images"], output_names=names, opset_version=args.opset, do_constant_folding=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    with torch.no_grad():
        outs = net(dummy)
        torch.onnx.export(net, dummy, args.output, **kwargs)
    print("exported", args.output, {n: list(o.shape) for n, o in zip(names, outs)})


if __name__ == "__main__":
    main()
