#!/usr/bin/env python3
"""YOLOX specific QAT code, shared by yolox_qat.py (calibration / PTQ baseline), yolox_qat_train.py and
pth_to_onnx_newexp.py: where the fake quantizers go in a YOLOX network, how to rebuild the quantized model, load a
checkpoint, calibrate / evaluate with the YOLOX data pipeline and export it.

The model independent parts (fake-quant modules, nncase range rules, calibration loop, ONNX export) live in
qat_generic.py. Needs the yolox package (and torch).
"""
import copy
import random

import torch
import torch.nn as nn

from yolox.exp import get_exp
from yolox.models.network_blocks import BaseConv, Bottleneck, CSPLayer, DWConv, SPPBottleneck
from yolox.utils import fuse_model

from util import qat_generic
from util.qat_generic import ActFakeQuant, QConv2d, QuantConfig, tie_ranges


# ----------------------------------------------------------------------------------------------------------------------
# modules
# ----------------------------------------------------------------------------------------------------------------------
class QBaseConv(nn.Module):
    """Folded BaseConv: fake-quant conv -> SiLU -> fake-quant of the result (the tensor the hardware stores)."""

    def __init__(self, base, cfg):
        super().__init__()
        assert not hasattr(base, "bn"), "fold BatchNorm first (fuse_model)"
        self.conv = QConv2d(base.conv, cfg)
        self.act = base.act
        self.out_q = ActFakeQuant()

    def forward(self, x):
        return self.out_q(self.act(self.conv(x)))


class QBottleneck(nn.Module):
    def __init__(self, b, cfg):
        super().__init__()
        self.conv1 = b.conv1
        self.conv2 = b.conv2
        self.use_add = b.use_add
        self.add_q = ActFakeQuant() if b.use_add else None

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = self.add_q(y + x)
        return y


# ----------------------------------------------------------------------------------------------------------------------
# model surgery
# ----------------------------------------------------------------------------------------------------------------------
def fold(model):
    """BatchNorm folding with YOLOX's own function. NOTE: it freezes the fused convs (requires_grad False)."""
    with torch.no_grad():  # otherwise the fused weights/biases keep a graph to the BN parameters (non-leaf)
        return fuse_model(copy.deepcopy(model))


def _convert(module, cfg):
    for name, child in list(module.named_children()):
        if type(child) is BaseConv:
            setattr(module, name, QBaseConv(child, cfg))
        elif isinstance(child, Bottleneck):
            _convert(child, cfg)
            setattr(module, name, QBottleneck(child, cfg))
        else:
            _convert(child, cfg)


def last_q(m):
    """The ActFakeQuant that quantizes the tensor produced by module m."""
    if isinstance(m, QBaseConv):
        return m.out_q
    if isinstance(m, DWConv):
        return last_q(m.pconv)
    if isinstance(m, QBottleneck):
        return m.add_q if m.use_add else last_q(m.conv2)
    if isinstance(m, CSPLayer):
        return last_q(m.conv3)
    if isinstance(m, SPPBottleneck):
        return last_q(m.conv2)
    if isinstance(m, nn.Sequential):
        return last_q(m[-1])
    raise TypeError(type(m))


def make_quantizable(folded, cfg):
    """Replace the folded model's blocks by fake-quant ones (in place) and return it."""
    _convert(folded, cfg)
    head = folded.head
    for lst in (head.cls_preds, head.reg_preds, head.obj_preds):
        for i, conv in enumerate(lst):
            lst[i] = QConv2d(conv, cfg, out_q=cfg.quantize_head_outputs)

    if cfg.tie_concats:
        quantizers = [m for m in folded.modules() if isinstance(m, ActFakeQuant)]
        # CSPLayer: cat(m(conv1(x)), conv2(x))
        for csp in [m for m in folded.modules() if isinstance(m, CSPLayer)]:
            tie_ranges(quantizers, last_q(csp.m), last_q(csp.conv2))
        # YOLOPAFPN.forward
        p = folded.backbone
        dark = p.backbone
        tie_ranges(quantizers, last_q(p.lateral_conv0), last_q(dark.dark4))  # cat([upsample(fpn_out0), x1])
        tie_ranges(quantizers, last_q(p.reduce_conv1), last_q(dark.dark3))  # cat([upsample(fpn_out1), x2])
        tie_ranges(quantizers, last_q(p.bu_conv2), last_q(p.reduce_conv1))  # cat([bu_conv2(...), fpn_out1])
        tie_ranges(quantizers, last_q(p.bu_conv1), last_q(p.lateral_conv0))  # cat([bu_conv1(...), fpn_out0])
        # SPPBottleneck: the max-pools keep the grid of their input, nothing to tie.

    for m in folded.modules():  # make sure every parameter is a leaf (the optimizer needs that)
        for name, prm in list(m._parameters.items()):
            if prm is not None and not prm.is_leaf:
                m._parameters[name] = nn.Parameter(prm.detach().clone())
    for prm in folded.parameters():  # fuse_model froze the fused convs
        prm.requires_grad_(True)
    return folded


# ----------------------------------------------------------------------------------------------------------------------
# model building, data, evaluation, export
# ----------------------------------------------------------------------------------------------------------------------
def build_qat_model(exp, cfg):
    """Fresh YOLOX model of the experiment, BatchNorm folded and fake quantizers inserted (weights NOT loaded: they
    come from a checkpoint). The quantizers start disabled."""
    model = exp.get_model()
    model.eval()
    return make_quantizable(fold(model), cfg)


def load_qat_checkpoint(path, name="yolox-nano", exp_file=None):
    """Rebuild the model of a checkpoint written by yolox_qat.py / yolox_qat_train.py and load it (CPU, eval mode).
    Returns (exp, qat, cfg, ckpt). The quantizer on/off flags are not part of the state_dict: the quantizers are left
    disabled, use qat_generic.set_state() to switch them on."""
    exp = get_exp(exp_file, name)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = QuantConfig.from_dict(ckpt["cfg"])
    qat = build_qat_model(exp, cfg)
    qat.load_state_dict(ckpt["model"])
    qat.eval()
    return exp, qat, cfg, ckpt


def export_onnx(exp, model, path, opset=13, new_exporter=False, example=None, quiet=False,
                bake_conv_weights=False):
    """qat_generic.export_onnx for a YOLOX model. The head is exported without box decoding
    (decode_in_inference=False); input 1x3xHxW float32, raw 0..255 BGR; output [1, N, 85]."""
    head = model.head
    old_decode = head.decode_in_inference
    head.decode_in_inference = False
    try:
        qat_generic.export_onnx(model, (1, 3, *exp.test_size), path, opset, new_exporter, example, quiet,
                                bake_conv_weights)
    finally:
        head.decode_in_inference = old_decode


def make_calib_loader(exp, n_images, batch_size, workers, seed=0):
    from yolox.data import COCODataset, ValTransform

    ds = COCODataset(
        data_dir=exp.data_dir,
        json_file=exp.train_ann,
        name="train2017",
        img_size=exp.test_size,
        preproc=ValTransform(legacy=False),  # same preprocessing as evaluation: raw 0..255 BGR, no normalization
    )
    idx = random.Random(seed).sample(range(len(ds)), min(n_images, len(ds)))
    sub = torch.utils.data.Subset(ds, idx)
    return torch.utils.data.DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=workers)


def evaluate(exp, model, batch_size, label):
    evaluator = exp.get_evaluator(batch_size, False)
    ap50_95, ap50, summary = evaluator.evaluate(model, False, False, None, None, exp.test_size)
    print(f"[{label}] AP(0.50:0.95) = {ap50_95:.4f}   AP50 = {ap50:.4f}")
    return ap50_95, ap50
