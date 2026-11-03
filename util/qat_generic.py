#!/usr/bin/env python3
"""Model independent quantization-aware-training building blocks (fake quantization, nncase rules).

Nothing here knows about a particular network: it works on any torch model made of convs, linears and LSTMs.
  * act_qparams, weight_qparams: the range / scale / zero-point rules copied from nncase (QuantUtility.GetQuantParam,
    FixupRange, GetWeightsRangesByChannel); uint8 (nncase UnsignedMode) only.
  * ActRange, ActFakeQuant, WeightFakeQuant, QConv2d, QLinear, QLSTM: the fake-quant modules (several ActFakeQuant
    can share one ActRange).
  * calibrate: the activation ranges, chosen like nncase's PTQ (KLD or plain min/max); they stay fixed in training.
  * set_state, describe, dump_ranges, bake_weights, export_onnx.
The graph rules of qat_graph.py decide where to put the quantizers; qat_train.py is the training driver.
"""
import inspect
import json
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------------------------------
# quantization parameters
# ----------------------------------------------------------------------------------------------------------------------
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
class ActRange(nn.Module):
    """The [min_val, max_val] range of an activation, set by calibrate(); seen = number of calibration batches."""

    def __init__(self):
        super().__init__()
        self.register_buffer("min_val", torch.zeros(()))
        self.register_buffer("max_val", torch.zeros(()))
        self.register_buffer("seen", torch.zeros((), dtype=torch.long))


class ActFakeQuant(nn.Module):
    """Per-tensor activation fake-quant. `params` can be shared with other ActFakeQuant (one range group)."""

    def __init__(self):
        super().__init__()
        self.params = ActRange()
        self.enabled = False

    def forward(self, x):
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

    def __init__(self, conv, out_q=False):
        super().__init__()
        self.conv = conv
        self.wq = WeightFakeQuant()
        self.out_q = ActFakeQuant() if out_q else None

    def forward(self, x):
        c = self.conv
        y = F.conv2d(x, self.wq(c.weight), c.bias, c.stride, c.padding, c.dilation, c.groups)
        return self.out_q(y) if self.out_q is not None else y


class QLinear(nn.Module):
    """Linear (ONNX Gemm) with fake-quantized weights, per output feature."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.wq = WeightFakeQuant()

    def forward(self, x):
        return F.linear(x, self.wq(self.linear.weight), self.linear.bias)


class QLSTM(nn.Module):
    """ONNX LSTM (gates i, o, f, c; no peepholes; default activations) with the K230 fake quantization: W and R
    per output row (direction x gate x hidden unit), and the hidden state h quantized (out_q, one range for both
    directions) every time step: that is the tensor stored as Y and fed back into R * h. The cell state c and the
    gates stay float (fp16 on the KPU). The input x is quantized by its producer.

    w [D, 4H, I], r [D, 4H, H], b [D, 8H] as in ONNX; forward(x [T, B, I]) -> (Y [T, D, B, H], Y_h, Y_c [D, B, H])."""

    def __init__(self, w, r, b, hidden_size, direction="forward"):
        super().__init__()
        if direction not in ("forward", "reverse", "bidirectional"):
            raise ValueError(direction)
        self.hidden_size = hidden_size
        self.direction = direction
        self.w = nn.Parameter(w.clone())
        self.r = nn.Parameter(r.clone())
        self.b = nn.Parameter(b.clone())
        self.wq = WeightFakeQuant()
        self.rq = WeightFakeQuant()
        self.out_q = ActFakeQuant()

    def forward(self, x):
        h_size = self.hidden_size
        w = self.wq(self.w.reshape(-1, self.w.shape[-1])).reshape(self.w.shape)
        r = self.rq(self.r.reshape(-1, self.r.shape[-1])).reshape(self.r.shape)
        steps = x.shape[0]
        ys, hs, cs = [], [], []
        for d in range(w.shape[0]):
            xw = x @ w[d].T + self.b[d, :4 * h_size] + self.b[d, 4 * h_size:]  # [T, B, 4H]
            h = x.new_zeros(x.shape[1], h_size)
            c = x.new_zeros(x.shape[1], h_size)
            out = [None] * steps
            reverse = self.direction == "reverse" or d == 1
            for t in (reversed(range(steps)) if reverse else range(steps)):
                i, o, f, g = (xw[t] + h @ r[d].T).split(h_size, -1)
                c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
                h = self.out_q(torch.sigmoid(o) * torch.tanh(c))
                out[t] = h
            ys.append(torch.stack(out))
            hs.append(h)
            cs.append(c)
        return torch.stack(ys, 1), torch.stack(hs), torch.stack(cs)


# ----------------------------------------------------------------------------------------------------------------------
# state
# ----------------------------------------------------------------------------------------------------------------------
def set_state(model, weights, acts):
    for m in model.modules():
        if isinstance(m, ActFakeQuant):
            m.enabled = acts
        elif isinstance(m, WeightFakeQuant):
            m.enabled = weights


def describe(model):
    qs = [m for m in model.modules() if isinstance(m, ActFakeQuant)]
    groups = {id(q.params) for q in qs}
    nconv = sum(isinstance(m, WeightFakeQuant) for m in model.modules())
    return f"{nconv} weight quantizers, {len(qs)} activation quantizers in {len(groups)} range groups"


# ----------------------------------------------------------------------------------------------------------------------
# calibration: nncase's PTQ range selection (Quantizer.cs, Quantizer.Algorithms.cs), ported as is
# ----------------------------------------------------------------------------------------------------------------------
SRC_BINS, DST_BINS = 8192, 256  # nncase Quantizer.RunAsync


def _fix_up_range(lo, hi):
    """Quantizer.FixUpRange (asymmetric): the range contains 0 and is at least 1e-6 wide (else 0.1)."""
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    r = hi - lo
    return lo, lo + (0.1 if r < 1e-6 else r)


def _box_smooth(p, box=512):
    """Quantizer.Smooth: moving sum of `box` bins / box, on the zero-padded histogram."""
    pad = np.zeros(box - 1, np.float64)
    c = np.concatenate(([0.0], np.cumsum(np.concatenate((pad, p, pad)))))
    k = np.arange(len(p)) + box // 2
    return ((c[k + box] - c[k]) / box).astype(np.float32)


def _smooth_distribution(p, eps=1e-4):
    """Quantizer.SmoothDistribution, in place like nncase: eps on the empty bins, taken from the others."""
    zeros = p == 0
    n_zeros = int(zeros.sum())
    n_nonzeros = len(p) - n_zeros
    if n_nonzeros == 0 or eps * n_zeros / n_nonzeros >= 1.0:
        return None
    p += np.where(zeros, np.float32(eps), np.float32(-eps * n_zeros / n_nonzeros))
    return p


def _compute_kld(p, q):
    """Quantizer.ComputeKld; normalizes p and q in place like nncase."""
    if p is None or q is None:
        return float("inf")
    p /= p.sum()
    q /= q.sum()
    nz = p != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(q[nz] == 0, np.float32(1.0), p[nz] * np.log(p[nz] / q[nz]))
    return float(terms.sum())


def _kld_of_thresholds(lower, upper, src):
    """Quantizer.GetKldOptRanges for one (lower, upper) bin threshold pair. src is changed in place (smoothed and
    normalized), as in nncase, so the order of the calls matters."""
    n = len(src)
    src_range = upper - lower
    per = src_range // DST_BINS
    ref = src[lower:upper].copy()
    ref[0] += src[:lower].sum()
    ref[-1] += src[upper:].sum()
    q = ref.reshape(DST_BINS, per).sum(1)
    nonzero = ref != 0
    count = nonzero.reshape(DST_BINS, per).sum(1)
    ups = np.repeat(np.where(count > 0, q / np.maximum(count, 1), 0).astype(np.float32), per) * nonzero
    ups2 = np.zeros(n, np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        left = lower + per  # left outliers: their mean on the non-empty bins
        value = src[:left].sum() / np.count_nonzero(src[:left])
        ups2[:left] += np.where(src[:left] != 0, value, 0)
        ups2[left:upper - per] = ups[per:src_range - per]
        right = upper - per  # right outliers; nncase does not reset the sum of the left ones
        value = (value + src[right:].sum()) / np.count_nonzero(src[right:])
        ups2[right:] += np.where(src[right:] != 0, value, 0)
    return _compute_kld(_smooth_distribution(src), _smooth_distribution(ups2))


def kld_range(hist, lo, hi):
    """Quantizer.GetOptRanges (CalibMethod.Kld): the [min, max] with the lowest KL divergence between the histogram
    (SRC_BINS bins over [lo, hi]) and its DST_BINS-level quantization; first the upper threshold, then the lower."""
    src = _box_smooth(np.asarray(hist, np.float64))
    interval = (hi - lo) / SRC_BINS
    zero = int(np.clip((0 - lo) / interval, 0, SRC_BINS - 1))
    best, min_kld = (0, SRC_BINS), float("inf")
    upper = SRC_BINS
    while upper >= DST_BINS and upper >= zero:
        kld = _kld_of_thresholds(0, upper, src)
        if kld < min_kld:
            best, min_kld = (0, upper), kld
        upper -= DST_BINS
    min_kld, upper = float("inf"), best[1]
    lower = 0
    while lower <= zero and lower <= upper - DST_BINS:
        kld = _kld_of_thresholds(lower, upper, src)
        if kld < min_kld:
            best, min_kld = (lower, upper), kld
        lower += DST_BINS
    return (best[0] - 0.5) * interval + lo, (best[1] + 0.5) * interval + lo


@torch.no_grad()
def calibrate(model, loader, method="kld"):
    """Set the range of every activation quantizer group from the images of `loader`, like nncase's PTQ:
    "noclip" = min / max over all images (CalibMethod.NoClip), "kld" = a second pass that collects SRC_BINS-bin
    histograms inside those ranges and keeps the range with the lowest KL divergence (CalibMethod.Kld, the default of
    nncase compile). Quantizers that share a range get one histogram of all their tensors. Like nncase,
    it runs the float network (no fake quantization; quantized weights shift the histograms, and the KLD choice moves
    with them); afterwards weights and activations are fake-quantized. The loader must give the network the input
    the K230 sees (for a model compiled with --pre-process, the dequantized uint8 input)."""
    if method not in ("kld", "noclip"):
        raise ValueError(f"unknown calibration method {method!r}")
    qs = [m for m in model.modules() if isinstance(m, ActFakeQuant)]
    groups = {}  # id(ActRange) -> ActRange
    for q in qs:
        groups.setdefault(id(q.params), q.params)
    stats = {}
    device = next(model.parameters()).device

    def run(fn):
        hooks = [q.register_forward_hook(lambda m, inp, out: fn(id(m.params), inp[0].detach().float())) for q in qs]
        try:
            n = 0
            for batch in loader:
                model(batch[0].float().to(device))
                n += 1
            return n
        finally:
            for h in hooks:
                h.remove()

    def minmax(gid, x):
        lo, hi = float(x.amin()), float(x.amax())
        old = stats.get(gid)
        stats[gid] = (lo, hi) if old is None else (min(old[0], lo), max(old[1], hi))

    set_state(model, weights=False, acts=False)
    model.eval()
    batches = run(minmax)
    ranges = {gid: _fix_up_range(*r) for gid, r in stats.items()}

    if method == "kld":
        hists = {gid: torch.zeros(SRC_BINS, dtype=torch.float64, device=device) for gid in ranges}

        def histogram(gid, x):
            lo, hi = ranges[gid]
            idx = ((x.reshape(-1) - lo) / ((hi - lo) / SRC_BINS)).clamp_(0, SRC_BINS - 1).long()
            hists[gid] += torch.bincount(idx, minlength=SRC_BINS).double()

        run(histogram)
        ranges = {gid: kld_range(hists[gid].cpu().numpy(), *r) for gid, r in ranges.items()}

    for gid, (lo, hi) in ranges.items():
        groups[gid].min_val.fill_(lo)
        groups[gid].max_val.fill_(hi)
        groups[gid].seen.fill_(batches)
    set_state(model, weights=True, acts=True)
    print(f"calibrated {len(ranges)} range groups on {len(loader.dataset)} images ({method})")
    unseen = len(groups) - len(ranges)
    if unseen:
        print(f"WARNING: {unseen} range groups saw no data")


# ----------------------------------------------------------------------------------------------------------------------
# export
# ----------------------------------------------------------------------------------------------------------------------
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
