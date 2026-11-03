#!/usr/bin/env python3
"""
Export an Ultralytics detector checkpoint (YOLO26 yolo26n.pt, YOLO12 yolo12n.pt, ...) to a float ONNX model with the
raw head outputs, for coco_eval.py --onnx and nncase (YOLOX has its own exporter, pth_to_onnx.py).

The graph ends at the last conv of every head branch, before the box/class concat (end2end=False, no decoding, no
top-k), so box and class logits keep their own quantization scales on the K230:
    box8  [1, 4, H/8, W/8]    cls8  [1, 80, H/8, W/8]     (likewise 16 and 32)
box = (l, t, r, b) distances from the cell center in units of the stride; cls = class logits (no objectness). That is
YOLO26 (no DFL, reg_max 1); with a DFL head (reg_max 16: YOLOv8, YOLO11, YOLO12) box is [1, 4 * 16, H/s, W/s]: per
side 16 logits, the distance is their softmax-weighted mean of 0..15 (left to the decoder). BatchNorm is folded and
the 1/255 input scaling is folded into the first conv, so the input is RGB 0..255 float32 1x3xHxW (letterbox
centered, pad 114: coco_prepare.py --detect yolo26). The cv1(x).chunk(2) of every C3k2 / C2PSA block becomes two
convs and the qkv conv + split of every attention block three (q, k, v): no Slice/Split, which nncase would run on
the CPU. --keep-split exports both as they are.

Usage:
    python yolo_to_onnx.py weights/yolo26n.pt onnx/yolo26n-640.onnx [--size 640] [--opset 13] [--keep-split]
    python yolo_to_onnx.py weights/yolo12n.pt onnx/yolo12n-640.onnx
"""
import argparse
import copy
import inspect
import types

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


def _pick(conv, channels, scale=1.0):
    """Folded Conv (conv + act) -> a Conv with only the given output channels, weight and bias times scale."""
    h = copy.deepcopy(conv)
    h.conv.weight = nn.Parameter(conv.conv.weight[channels].detach().clone() * scale)
    h.conv.bias = nn.Parameter(conv.conv.bias[channels].detach().clone() * scale)
    h.conv.out_channels = len(channels)
    return h


def _halves(conv):
    """Folded Conv (conv + act) with 2c outputs -> two Convs with the first / second c outputs."""
    c = conv.conv.out_channels // 2
    return [_pick(conv, list(range(0, c))), _pick(conv, list(range(c, 2 * c)))]


def _c2f_forward(self, x):
    y = [self.cv1a(x), self.cv1b(x)]
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))


def _c2psa_forward(self, x):
    return self.cv2(torch.cat((self.cv1a(x), self.m(self.cv1b(x))), 1))


def split_cv1(model):
    """Replace cv1(x).chunk(2) / .split() of C3k2 (C2f) and C2PSA by two convs: same result, but no Slice/Split in
    the graph (nncase runs those on the CPU, cutting the KPU graph at every C3k2)."""
    from ultralytics.nn.modules.block import C2PSA, C2f

    n = 0
    for m in model.modules():
        if isinstance(m, (C2f, C2PSA)):
            m.cv1a, m.cv1b = _halves(m.cv1)
            del m.cv1
            m.forward = types.MethodType(_c2psa_forward if isinstance(m, C2PSA) else _c2f_forward, m)
            n += 1
    return n


def _attention_forward(self, x):
    B, C, H, W = x.shape
    N = H * W
    q = self.q(x).view(B, self.num_heads, self.key_dim, N)
    k = self.k(x).view(B, self.num_heads, self.key_dim, N)
    v = self.v(x)
    attn = (q.transpose(-2, -1) @ k).softmax(dim=-1)
    x = (v.view(B, self.num_heads, self.head_dim, N) @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v)
    return self.proj(x)


def split_qkv(model):
    """Replace the qkv conv + Split of every Attention (C2PSA / PSABlock) by separate q, k, v convs: same result
    (the 1/sqrt(key_dim) scale is folded into q), but no Split, whose outputs nncase fetches on the CPU."""
    from ultralytics.nn.modules.block import Attention

    n = 0
    for m in model.modules():
        if isinstance(m, Attention):
            per_head = 2 * m.key_dim + m.head_dim  # qkv channels: per head [q (key_dim), k (key_dim), v (head_dim)]
            heads = [h * per_head for h in range(m.num_heads)]
            m.q = _pick(m.qkv, [b + i for b in heads for i in range(m.key_dim)], m.scale)
            m.k = _pick(m.qkv, [b + m.key_dim + i for b in heads for i in range(m.key_dim)])
            m.v = _pick(m.qkv, [b + 2 * m.key_dim + i for b in heads for i in range(m.head_dim)])
            del m.qkv
            m.forward = types.MethodType(_attention_forward, m)
            n += 1
    return n


def build(weights, split=True):
    """Ultralytics checkpoint -> RawHead (eval, BatchNorm folded, input 0..255; split: see split_cv1, split_qkv)."""
    from ultralytics import YOLO

    model = YOLO(weights).model.float().eval()
    model = model.fuse(verbose=False)
    first = model.model[0].conv
    with torch.no_grad():
        first.weight.mul_(1.0 / 255.0)  # zero padding: conv(x / 255, w) == conv(x, w / 255) exactly
    if split:
        print(f"split cv1 of {split_cv1(model)} C3k2 / C2PSA blocks into two convs")
        print(f"split qkv of {split_qkv(model)} attention blocks into q, k, v convs")
    return RawHead(model).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("output")
    ap.add_argument("--size", type=int, default=640, help="square input size (default 640)")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--keep-split", action="store_true", help="keep the cv1 chunk of C3k2 / C2PSA and the qkv split of attention")
    args = ap.parse_args()

    net = build(args.weights, split=not args.keep_split)
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
