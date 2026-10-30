#!/usr/bin/env python3
"""Emulation of the K230 (nncase GNNE) arithmetic between the quantization points, for the fake-quant QAT model.

The QDQ (fake-quant) model fixes where values are rounded to u8 and with which scale, but not the arithmetic in
between. After nncase lowering (7_TargetDependentAfterQuantPass) every quantized layer runs as
  * GNNEConv2D: exact integer MAC, per-channel dequant with f16 parameters, f16 output
        z = f16(acc * s_in * s_w + f16(bias))
  * GNNEActivation: SiLU as a 16 segment piecewise-linear function (SwishToFakeActivation.cs) with the output
    quantization folded into the f16 segment coefficients, u8 output
        code = f16(f16(k_i / s_out) * z + f16(b_i / s_out + zp)),   q = round(code)
  * residual add: GNNEActivation with two inputs, the same folded output quantization
  * head: conv with f16 output; Sigmoid as a 16 segment piecewise-linear function (SigmoidToFakeActivation.cs), f16
On one isolated 1x1 conv + SiLU this model reproduces the board's u8 output to 99.93% (exact SiLU in f32: 99.72%).
Not modelled: the f16 rounding of the per-channel dequant slope s_in * s_w (the conv does not know s_in) and the
fixed-point shift details.

Everything keeps gradients (straight-through for the roundings, the segment slope for the PWL functions), so the same
functions can be used for K230-aware fine-tuning. The model specific wiring is qat_yolox.set_k230().
"""
import torch
import torch.nn.functional as F

from util.qat_generic import QMAX, QMIN, act_qparams

# SwishToFakeActivation.cs: segment i covers [xs[i-1], xs[i]), y = ks[i] * x + bs[i]
SILU_XS = [-6.8, -4.0, -2.2, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.2, 4.0, 6.8]
SILU_KS = [0.0, -0.021684465098961, -0.083146935094056, -0.083239373463281, -0.02318165804185, 0.073618083374296,
           0.218122950353851, 0.401072247994053, 0.598927752005947, 0.781877049646149, 0.926381916625704,
           1.023181658041854, 1.083239373463282, 1.083146935094056, 1.021684465098961, 1.0]
SILU_BS = [0.0, -0.146472330255445, -0.395702832349094, -0.404303627166028, -0.308112929911211, -0.193207045153763,
           -0.078813366995298, -0.006394948413948, -0.006394948413948, -0.078813366995297, -0.193207045153763,
           -0.308112929911212, -0.404303627166028, -0.395702832349094, -0.146472330255446, 0.0]

# SigmoidToFakeActivation.cs (the C# table interleaves slope / bias)
SIGMOID_XS = [-7.0, -4.5, -3.5, -2.7, -2.1, -1.6, -1.0, 0.0, 1.0, 1.6, 2.1, 2.7, 3.5, 4.5, 7.0]
SIGMOID_KS = [0.0005523135475095087, 0.003582545941984816, 0.017628076952972527, 0.04080084671519202,
              0.07504241042393778, 0.11550039574401527, 0.16589656476120918, 0.23123362875892262,
              0.23398496370676491, 0.17066333504274322, 0.1197647477013204, 0.07822321089963047,
              0.04270052410967129, 0.01849619693381177, 0.0037644096557241102, 0.000567715948649905]
SIGMOID_BS = [0.004717946782565874, 0.024643784893781717, 0.08920121475552378, 0.17057512199171598,
              0.2641958959068108, 0.35039917518496155, 0.4312276071962273, 0.4955178069668943, 0.5031840614393268,
              0.5625762717242175, 0.6417049229131112, 0.7281582013897308, 0.8235195920244173, 0.9073131477622514,
              0.9742926548134362, 0.9951637114353361]

_tables = {}


def _table(xs, ks, bs, device):
    key = (id(xs), device)
    if key not in _tables:
        txs = torch.tensor(xs, dtype=torch.float32, device=device).half().float()  # the breakpoints are f16 on the KPU
        _tables[key] = (txs,) + tuple(torch.tensor(v, dtype=torch.float32, device=device) for v in (ks, bs))
    return _tables[key]


def f16(x):
    """Round to float16 and back (straight-through gradient)."""
    return x + (x.half().float() - x).detach()


def _segments(x, xs, ks, bs):
    """Slope and offset of the segment of every element of x."""
    txs, tks, tbs = _table(xs, ks, bs, x.device)
    i = torch.bucketize(x.detach(), txs, right=True)  # number of breakpoints <= x
    return tks[i], tbs[i]


def silu(x):
    k, b = _segments(x, SILU_XS, SILU_KS, SILU_BS)
    return k * x + b


def sigmoid(x):
    k, b = _segments(x, SIGMOID_XS, SIGMOID_KS, SIGMOID_BS)
    return (f16(k) * x + f16(b)).clamp(0.0, 1.0)


def conv_f16(qconv, x):
    """qat_generic.QConv2d without its out_q, the GNNEConv2D way: f16 bias, f16 output."""
    c = qconv.conv
    y = F.conv2d(x, qconv.wq(c.weight), None, c.stride, c.padding, c.dilation, c.groups)
    if c.bias is not None:
        y = y + f16(c.bias).view(1, -1, 1, 1)
    return f16(y)


def _quant_code(code, q):
    """Round / clamp an output code computed in f16 and dequantize it with the parameters of ActFakeQuant q."""
    scale, zp = act_qparams(q.params.min_val, q.params.max_val)
    code = code + (torch.round(code) - code).detach()
    return (code.clamp(QMIN, QMAX) - zp) * scale


def silu_quant(z, q):
    """PWL SiLU of the f16 conv output z with the output quantization of ActFakeQuant q folded in (GNNEActivation)."""
    if q.observe:
        q.params.update(silu(z))
    if not q.enabled:
        return f16(silu(z))
    scale, zp = act_qparams(q.params.min_val, q.params.max_val)
    k, b = _segments(z, SILU_XS, SILU_KS, SILU_BS)
    code = f16(f16(f16(k / scale) * z) + f16(b / scale + zp))
    return _quant_code(code, q)


def add_quant(a, b, q):
    """Residual add of two dequantized tensors with the output quantization of ActFakeQuant q (GNNEActivation)."""
    s = f16(f16(a) + f16(b))
    if q.observe:
        q.params.update(s)
    if not q.enabled:
        return s
    scale, zp = act_qparams(q.params.min_val, q.params.max_val)
    code = f16(f16(s * f16(1.0 / scale)) + zp)
    return _quant_code(code, q)


def sigmoid_logit(z):
    """Logit whose exact sigmoid is the K230 PWL sigmoid of z stored as f16, so a model that applies torch.sigmoid
    afterwards (the YOLOX head) gets the K230 probabilities."""
    p = f16(sigmoid(z)).clamp(1e-6, 1.0 - 1e-6)
    return torch.log(p) - torch.log1p(-p)
