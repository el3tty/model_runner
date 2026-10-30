#!/usr/bin/env python3
"""Model independent quantization-aware-training building blocks (fake quantization, nncase rules).

Nothing here knows about a particular network: it works on any torch model made of nn.Conv2d layers.
  * QuantConfig, act_qparams, weight_qparams: the range / scale / zero-point rules copied from nncase
    (QuantUtility.GetQuantParam, FixupRange, GetWeightsRangesByChannel); uint8 (nncase UnsignedMode) only.
  * RangeObserver, ActFakeQuant, WeightFakeQuant, QConv2d: the fake-quant modules (activation ranges can be shared
    between several quantizers, see tie_ranges).
  * set_state, set_observer_momentum, describe, dump_ranges, calibrate, check_fold, bake_weights, export_onnx.
A model specific module (see qat_yolox.py) decides where to put the quantizers.
"""
import inspect
import json
import os
import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------------------------------
# configuration and quantization parameters
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class QuantConfig:
    quantize_head_outputs: bool = False  # also fake-quantize the outputs of cls/reg/obj prediction convs
    tie_concats: bool = True  # concat inputs share one range (nncase UniteConcatRange; verify on your model)

    @classmethod
    def from_dict(cls, d):
        """QuantConfig of a checkpoint's "cfg". Older checkpoints also have act_signed / w_signed (the removed int8
        mode): accepted when False, refused when True."""
        d = dict(d)
        signed = [k for k in ("act_signed", "w_signed") if d.pop(k, False)]
        if signed:
            raise ValueError(f"checkpoint uses int8 quantization ({', '.join(signed)}), which is no longer supported")
        return cls(**d)


QMIN, QMAX = 0, 255  # uint8 for activations and weights (the nncase default)


def act_qparams(min_val, max_val):
    """nncase: Quantizer.FixUpRange + QuantUtility.GetQuantParam. Returns (scale[1] float32, zero_point[1] int32)."""
    lo = min_val.clamp(max=0.0)  # the range always contains zero
    hi = max_val.clamp(min=0.0)
    r = hi - lo
    r = torch.where(r < 1e-6, torch.full_like(r, 0.1), r.clamp(min=0.01))
    scale = r / (QMAX - QMIN)
    zp = torch.round(-lo / scale).clamp(QMIN, QMAX)
    return scale.reshape(1).float(), zp.reshape(1).to(torch.int32)


def weight_qparams(w):
    """nncase: QuantUtility.GetWeightsRangesByChannel + GetQuantParam, per output channel (dim 0)."""
    flat = w.detach().reshape(w.shape[0], -1)
    lo = flat.amin(dim=1).clamp(max=0.0)
    hi = flat.amax(dim=1).clamp(min=0.0)
    r = hi - lo
    r = torch.where(r == 0, torch.full_like(r, 0.1), r.clamp(min=0.01))
    scale = r / (QMAX - QMIN)
    zp = torch.round(-lo / scale).clamp(QMIN, QMAX)
    return scale.float(), zp.to(torch.int32)


# ----------------------------------------------------------------------------------------------------------------------
# modules
# ----------------------------------------------------------------------------------------------------------------------
class RangeObserver(nn.Module):
    """Running min / max of a tensor. momentum None: cumulative mean of per-batch min/max, else EMA."""

    def __init__(self):
        super().__init__()
        self.register_buffer("min_val", torch.zeros(()))
        self.register_buffer("max_val", torch.zeros(()))
        self.register_buffer("seen", torch.zeros((), dtype=torch.long))
        self.momentum = None

    @torch.no_grad()
    def update(self, x):
        mn = x.detach().amin().float()
        mx = x.detach().amax().float()
        n = int(self.seen)
        if n == 0:
            self.min_val.copy_(mn)
            self.max_val.copy_(mx)
        else:
            m = 1.0 / (n + 1) if self.momentum is None else self.momentum
            self.min_val.add_((mn - self.min_val) * m)
            self.max_val.add_((mx - self.max_val) * m)
        self.seen += 1


class ActFakeQuant(nn.Module):
    """Per-tensor activation fake-quant. `params` can be shared with other ActFakeQuant (see tie())."""

    def __init__(self):
        super().__init__()
        self.params = RangeObserver()
        self.enabled = False
        self.observe = False

    def forward(self, x):
        if self.observe:
            self.params.update(x)
        if not self.enabled:
            return x
        scale, zp = act_qparams(self.params.min_val, self.params.max_val)
        if torch.onnx.is_in_onnx_export():
            # constants, not traced ops
            return torch.fake_quantize_per_tensor_affine(x, float(scale), int(zp), QMIN, QMAX)
        return torch.fake_quantize_per_tensor_affine(x, scale, zp, QMIN, QMAX)


class WeightFakeQuant(nn.Module):
    """Per-output-channel weight fake-quant; the ranges are recomputed from the weights on every call."""

    def __init__(self):
        super().__init__()
        self.enabled = False

    def forward(self, w):
        if not self.enabled:
            return w
        scale, zp = weight_qparams(w)
        if torch.onnx.is_in_onnx_export():
            scale = torch.tensor(scale.tolist(), dtype=torch.float32, device=w.device)  # constants, not traced ops
            zp = torch.tensor(zp.tolist(), dtype=torch.int32, device=w.device)
        return torch.fake_quantize_per_channel_affine(w, scale, zp, 0, QMIN, QMAX)


class QConv2d(nn.Module):
    """Conv2d with fake-quantized weights (and optionally a fake-quantized output)."""

    def __init__(self, conv, cfg, out_q=False):
        super().__init__()
        self.conv = conv
        self.wq = WeightFakeQuant()
        self.out_q = ActFakeQuant() if out_q else None
        self.k230 = None  # K230 arithmetic emulation (k230_emul): None (off), "f16" or "sigmoid" output

    def forward(self, x):
        if self.k230:
            from util import k230_emul
            y = k230_emul.conv_f16(self, x)
            if self.out_q is not None:
                y = self.out_q(y)
            return k230_emul.sigmoid_logit(y) if self.k230 == "sigmoid" else y
        c = self.conv
        y = F.conv2d(x, self.wq(c.weight), c.bias, c.stride, c.padding, c.dilation, c.groups)
        return self.out_q(y) if self.out_q is not None else y


# ----------------------------------------------------------------------------------------------------------------------
# sharing, state, calibration, export
# ----------------------------------------------------------------------------------------------------------------------
def tie_ranges(quantizers, a, b):
    """Make the quantizers a and b (and everything already sharing with b) use one range."""
    pa, pb = a.params, b.params
    if pa is pb:
        return
    for q in quantizers:
        if q.params is pb:
            q.params = pa


def set_state(model, weights, acts, observe):
    for m in model.modules():
        if isinstance(m, ActFakeQuant):
            m.enabled = acts
            m.observe = observe
        elif isinstance(m, WeightFakeQuant):
            m.enabled = weights


def set_observer_momentum(model, momentum):
    for m in model.modules():
        if isinstance(m, RangeObserver):
            m.momentum = momentum


def describe(model):
    qs = [m for m in model.modules() if isinstance(m, ActFakeQuant)]
    groups = {id(q.params) for q in qs}
    nconv = sum(isinstance(m, WeightFakeQuant) for m in model.modules())
    return f"{nconv} weight quantizers, {len(qs)} activation quantizers in {len(groups)} range groups"


@torch.no_grad()
def check_fold(fp32, folded, batch):
    fp32.eval()
    folded.eval()
    a = fp32(batch)
    b = folded(batch)
    d = (a - b).abs()
    n_bn = sum(isinstance(m, nn.BatchNorm2d) for m in folded.modules())
    print(f"fold check: BatchNorm layers left = {n_bn}; output max |diff| = {d.max().item():.3e}, "
          f"mean |diff| = {d.mean().item():.3e}, max |fp32 output| = {a.abs().max().item():.3e}")


@torch.no_grad()
def calibrate(model, loader):
    set_observer_momentum(model, None)
    set_state(model, weights=True, acts=False, observe=True)
    model.eval()
    device = next(model.parameters()).device
    n = 0
    for batch in loader:
        model(batch[0].float().to(device))
        n += batch[0].shape[0]
    set_state(model, weights=True, acts=True, observe=False)
    print(f"calibrated on {n} images")
    unseen = [name for name, m in model.named_modules() if isinstance(m, ActFakeQuant) and int(m.params.seen) == 0]
    if unseen:
        print("WARNING: quantizers that saw no data:", unseen)


def bake_weights(model):
    """Replace every conv weight by its fake-quantized (per-output-channel) value and switch the weight quantizer off,
    so a following ONNX export has plain Conv nodes whose weights already lie on the quantization grid, with no
    Constant -> QuantizeLinear -> DequantizeLinear chain in front of them. Returns an undo() function that restores
    the original float weights and the quantizer state."""
    saved = []
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, QConv2d) and m.wq.enabled:
                w = m.conv.weight
                saved.append((m, w.detach().clone()))
                scale, zp = weight_qparams(w)
                w.copy_(torch.fake_quantize_per_channel_affine(w, scale, zp, 0, QMIN, QMAX))
                m.wq.enabled = False

    def undo():
        with torch.no_grad():
            for m, w in saved:
                m.conv.weight.copy_(w)
                m.wq.enabled = True

    return undo


def export_onnx(model, input_shape, path, opset=13, new_exporter=False, example=None, quiet=False,
                bake_conv_weights=False):
    """Export a (fake-quant) model to ONNX in its current quantizer state: quantizers enabled -> QuantizeLinear/
    DequantizeLinear pairs (per-tensor on activations, per-channel on weights); disabled -> a plain float graph.

    input_shape: e.g. (1, 3, 416, 416); the input is named "images", the output "output".
    new_exporter: use the torch.export based exporter (PyTorch >= 2.5) instead of the legacy TorchScript one, which
        has the fake-quant -> Q/DQ symbolics.
    example: numpy array of input_shape (float32) used as the example input of the trace instead of zeros.
    quiet: hide the warnings that are expected here (the scales / zero points are meant to be constants of the
        exported graph, and the legacy exporter is chosen on purpose).
    bake_conv_weights: fold the weight quantization into the conv weights (see bake_weights): the graph keeps
        Q/DQ pairs on the activations only. The model is restored afterwards."""
    model.eval()
    device = next(model.parameters()).device
    input_shape = tuple(input_shape)
    if example is None:
        dummy = torch.zeros(*input_shape, device=device)
    else:
        if tuple(example.shape) != input_shape:
            raise SystemExit(f"example input has shape {tuple(example.shape)}, the model needs {input_shape}")
        dummy = torch.from_numpy(example).float().to(device)
    kwargs = dict(input_names=["images"], output_names=["output"], opset_version=opset, do_constant_folding=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = new_exporter  # the legacy TorchScript exporter has the fake-quant -> Q/DQ symbolics
    elif new_exporter:
        raise SystemExit("this PyTorch version has no torch.export based ONNX exporter (needs >= 2.5)")
    undo = bake_weights(model) if bake_conv_weights else None
    try:
        with torch.no_grad(), warnings.catch_warnings():
            if quiet:
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*legacy TorchScript.*")
            program = torch.onnx.export(model, dummy, path, **kwargs)
            if program is not None and hasattr(program, "save") and not os.path.exists(path):
                program.save(path)
    finally:
        if undo is not None:
            undo()
    print("exported", path)


def dump_ranges(model, path):
    group_ids = {}
    rows = []
    for name, m in model.named_modules():
        if isinstance(m, ActFakeQuant):
            gid = group_ids.setdefault(id(m.params), len(group_ids))
            rows.append({"module": name, "group": gid, "min": float(m.params.min_val), "max": float(m.params.max_val),
                         "batches_seen": int(m.params.seen)})
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)
