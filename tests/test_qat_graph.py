"""util/qat_graph.py: where make_quantizable() puts the fake quantizers.

Each case is a small model in ONNX text syntax (onnx.parser; the inputs after the first are weights and get random
values). It goes the way PP-OCR does: qat_graph.load() (onnx2torch), make_quantizable(), then the export with the
activation quantizers on and the weights baked (qat_export.py --act-qdq). The exported graph is printed in a compact
form and compared with the expected text:

    v3 = Q0(v2)          a QuantizeLinear -> DequantizeLinear pair; Q<k> is its range group (same k = shared range)
    v4 = Conv(v3, #, #)  every other node: op type and inputs; # is a constant (baked weight, scalar, shape)

x is the model input, y the output, v1, v2, ... the other tensors in node order. Every range group gets its own range
before the export, so equal scales mean a shared group.
"""
import numpy as np
import onnx
import onnx.parser
import pytest

from util import qat_generic, qat_graph

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning"),  # onnx2torch's notes on Resize modes
              pytest.mark.filterwarnings("ignore::DeprecationWarning")]  # the legacy TorchScript exporter

HEAD = '<ir_version: 8, opset_import: ["" : 13]>\n'


def build(text, path, seed=0):
    """Save the model of `text`, its inputs after the first turned into random initializers; returns the input
    shape."""
    m = onnx.parser.parse_model(HEAD + text)
    rng = np.random.default_rng(seed)
    g = m.graph
    for inp in list(g.input)[1:]:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        g.initializer.append(onnx.numpy_helper.from_array((rng.standard_normal(shape) * 0.3).astype(np.float32),
                                                          inp.name))
    del g.input[1:]
    onnx.checker.check_model(m)
    onnx.save(m, path)
    return [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]


def compact(path):
    """The exported graph as text (see the module docstring)."""
    g = onnx.load(path).graph
    values = {i.name: onnx.numpy_helper.to_array(i) for i in g.initializer}
    values.update({n.output[0]: onnx.numpy_helper.to_array(n.attribute[0].t) for n in g.node
                   if n.op_type == "Constant"})
    names = {g.input[0].name: "x", g.output[0].name: "y"}

    def name(t):
        return names.setdefault(t, f"v{len(names) - 1}")

    groups, quantized, lines = {}, {}, []
    for n in g.node:
        if n.op_type == "Constant":
            continue
        if n.op_type == "QuantizeLinear":
            gid = groups.setdefault(round(float(values[n.input[1]]), 9), len(groups))
            quantized[n.output[0]] = (name(n.input[0]), gid)
        elif n.op_type == "DequantizeLinear" and n.input[0] in quantized:
            src, gid = quantized[n.input[0]]
            lines.append(f"{name(n.output[0])} = Q{gid}({src})")
        else:
            args = ", ".join("#" if a in values else name(a) for a in n.input if a)
            lines.append(f"{', '.join(name(o) for o in n.output if o)} = {n.op_type}({args})")
    return "\n".join(lines)


def quantize(text, tmp_path):
    src, out = tmp_path / "in.onnx", tmp_path / "out.onnx"
    shape = build(text, src)
    gm = qat_graph.make_quantizable(qat_graph.load(str(src)))
    groups = {id(m.params): m.params for m in gm.modules() if isinstance(m, qat_generic.ActFakeQuant)}
    for k, r in enumerate(groups.values()):  # a different range per group: its scale identifies it
        r.min_val.fill_(-1.0)
        r.max_val.fill_(float(k + 1))
    qat_generic.set_state(gm, weights=True, acts=True)
    with qat_graph.static_batch(gm):
        qat_generic.export_onnx(gm, shape, str(out), quiet=True, bake_conv_weights=True, input_names=("x",),
                                output_names=("y",))
    return compact(out)


# name: (model, expected compact export)
CASES = {
    # the model input and output are not quantized; the activation output read by a conv is
    "conv_relu_conv": ("""
g (float[1,3,8,8] x, float[4,3,3,3] w1, float[4] b1, float[4,4,3,3] w2, float[4] b2) => (float[1,4,8,8] y) {
  c1 = Conv <pads=[1,1,1,1]> (x, w1, b1)
  r1 = Relu(c1)
  y = Conv <pads=[1,1,1,1]> (r1, w2, b2)
}""", """
v1 = Conv(x, #, #)
v2 = Relu(v1)
v3 = Q0(v2)
y = Conv(v3, #, #)"""),

    # SiLU = Mul(x, Sigmoid(x)) fuses into the conv; MaxPool keeps the grid: the quantizer goes in front of it
    "silu_maxpool": ("""
g (float[1,3,8,8] x, float[4,3,3,3] w1, float[4] b1, float[4,4,3,3] w2, float[4] b2) => (float[1,4,4,4] y) {
  c1 = Conv <pads=[1,1,1,1]> (x, w1, b1)
  s1 = Sigmoid(c1)
  a1 = Mul(c1, s1)
  p1 = MaxPool <kernel_shape=[2,2], strides=[2,2]> (a1)
  y = Conv <pads=[1,1,1,1]> (p1, w2, b2)
}""", """
v1 = Conv(x, #, #)
v2 = Sigmoid(v1)
v3 = Mul(v1, v2)
v4 = Q0(v3)
v5 = MaxPool(v4)
y = Conv(v5, #, #)"""),

    # the inputs of a Concat share one range
    "concat_ties_ranges": ("""
g (float[1,3,8,8] x, float[4,3,3,3] w1, float[4] b1, float[4,3,1,1] w2, float[4] b2, float[4,8,1,1] w3, float[4] b3)
  => (float[1,4,8,8] y) {
  c1 = Conv <pads=[1,1,1,1]> (x, w1, b1)
  r1 = Relu(c1)
  c2 = Conv (x, w2, b2)
  r2 = Relu(c2)
  k = Concat <axis=1> (r1, r2)
  y = Conv (k, w3, b3)
}""", """
v1 = Conv(x, #, #)
v2 = Relu(v1)
v3 = Q0(v2)
v4 = Conv(x, #, #)
v5 = Relu(v4)
v6 = Q0(v5)
v7 = Concat(v3, v6)
y = Conv(v7, #, #)"""),

    # a nearest Resize keeps the grid: its input joins the Concat group too
    "resize_nearest_concat": ("""
g (float[1,3,8,8] x, float[4,3,3,3] w1, float[4] b1, float[4,4,3,3] w2, float[4] b2, float[4,8,1,1] w3, float[4] b3)
  => (float[1,4,8,8] y) {
  c1 = Conv <pads=[1,1,1,1]> (x, w1, b1)
  r1 = Relu(c1)
  c2 = Conv <pads=[1,1,1,1], strides=[2,2]> (r1, w2, b2)
  r2 = Relu(c2)
  e = Constant <value = float[0] {}> ()
  sc = Constant <value = float[4] {1.0, 1.0, 2.0, 2.0}> ()
  u = Resize <mode="nearest"> (r2, e, sc)
  k = Concat <axis=1> (r1, u)
  y = Conv (k, w3, b3)
}""", """
v1 = Conv(x, #, #)
v2 = Relu(v1)
v3 = Q0(v2)
v4 = Conv(v3, #, #)
v5 = Relu(v4)
v6 = Q0(v5)
v7 = Resize(v6, #)
v8 = Concat(v3, v7)
y = Conv(v8, #, #)"""),

    # Add of two tensors is a quantized op: both inputs quantized (the conv output without an activation too), and
    # its fused Relu output read by the next conv
    "residual_add": ("""
g (float[1,4,8,8] x, float[4,4,3,3] w1, float[4] b1, float[4,4,3,3] w2, float[4] b2, float[4,4,1,1] w3, float[4] b3)
  => (float[1,4,8,8] y) {
  c1 = Conv <pads=[1,1,1,1]> (x, w1, b1)
  r1 = Relu(c1)
  c2 = Conv <pads=[1,1,1,1]> (r1, w2, b2)
  a = Add(r1, c2)
  r2 = Relu(a)
  y = Conv (r2, w3, b3)
}""", """
v1 = Conv(x, #, #)
v2 = Relu(v1)
v3 = Q0(v2)
v4 = Conv(v3, #, #)
v5 = Q1(v4)
v6 = Add(v3, v5)
v7 = Relu(v6)
v8 = Q2(v7)
y = Conv(v8, #, #)"""),

    # HardSwish spelled out as (x * Clip(x + 3, 0, 6)) / 6 is one activation: no quantizer inside it
    "hardswish_mul_div": ("""
g (float[1,3,8,8] x, float[4,3,3,3] w1, float[4] b1, float[4,4,1,1] w2, float[4] b2) => (float[1,4,8,8] y) {
  c1 = Conv <pads=[1,1,1,1]> (x, w1, b1)
  three = Constant <value = float {3.0}> ()
  zero = Constant <value = float {0.0}> ()
  six = Constant <value = float {6.0}> ()
  t = Add(c1, three)
  cl = Clip(t, zero, six)
  m = Mul(c1, cl)
  h = Div(m, six)
  y = Conv (h, w2, b2)
}""", """
v1 = Conv(x, #, #)
v2 = Add(v1, #)
v3 = Clip(v2, #, #)
v4 = Mul(v1, v3)
v5 = Div(v4, #)
v6 = Q0(v5)
y = Conv(v6, #, #)"""),

    # ... and as x * (Clip(x + 3, 0, 6) / 6)
    "hardswish_div_mul": ("""
g (float[1,3,8,8] x, float[4,3,3,3] w1, float[4] b1, float[4,4,1,1] w2, float[4] b2) => (float[1,4,8,8] y) {
  c1 = Conv <pads=[1,1,1,1]> (x, w1, b1)
  three = Constant <value = float {3.0}> ()
  zero = Constant <value = float {0.0}> ()
  six = Constant <value = float {6.0}> ()
  t = Add(c1, three)
  cl = Clip(t, zero, six)
  d = Div(cl, six)
  h = Mul(c1, d)
  y = Conv (h, w2, b2)
}""", """
v1 = Conv(x, #, #)
v2 = Add(v1, #)
v3 = Clip(v2, #, #)
v4 = Div(v3, #)
v5 = Mul(v1, v4)
v6 = Q0(v5)
y = Conv(v6, #, #)"""),

    # PP-OCR's squeeze-excitation: GlobalAveragePool and Mul of two tensors are quantized ops, HardSigmoid fuses
    # into its conv
    "squeeze_excitation": ("""
g (float[1,4,8,8] x, float[4,4,3,3] w1, float[4] b1, float[2,4,1,1] w2, float[2] b2, float[4,2,1,1] w3, float[4] b3,
   float[4,4,1,1] w4, float[4] b4) => (float[1,4,8,8] y) {
  c1 = Conv <pads=[1,1,1,1]> (x, w1, b1)
  r1 = Relu(c1)
  p = GlobalAveragePool(r1)
  c2 = Conv (p, w2, b2)
  r2 = Relu(c2)
  c3 = Conv (r2, w3, b3)
  hs = HardSigmoid <alpha=0.2, beta=0.5> (c3)
  m = Mul(r1, hs)
  y = Conv (m, w4, b4)
}""", """
v1 = Conv(x, #, #)
v2 = Relu(v1)
v3 = Q0(v2)
v4 = GlobalAveragePool(v3)
v5 = Q1(v4)
v6 = Conv(v5, #, #)
v7 = Relu(v6)
v8 = Q2(v7)
v9 = Conv(v8, #, #)
v10 = HardSigmoid(v9)
v11 = Q3(v10)
v12 = Mul(v3, v11)
v13 = Q4(v12)
y = Conv(v13, #, #)"""),

    # Flatten keeps the grid: the pooled tensor is quantized in front of it, for the Gemm
    "pool_flatten_gemm": ("""
g (float[1,3,8,8] x, float[4,3,3,3] w1, float[4] b1, float[5,4] w2, float[5] b2) => (float[1,5] y) {
  c1 = Conv <pads=[1,1,1,1]> (x, w1, b1)
  r1 = Relu(c1)
  p = GlobalAveragePool(r1)
  f = Flatten(p)
  y = Gemm <transB=1> (f, w2, b2)
}""", """
v1 = Conv(x, #, #)
v2 = Relu(v1)
v3 = Q0(v2)
v4 = GlobalAveragePool(v3)
v5 = Q1(v4)
v6 = Flatten(v5)
y = Gemm(v6, #, #)"""),

    # an LSTM exports as one LSTM node; its output (the hidden states) is quantized, its input is the model input
    "lstm_gemm": ("""
g (float[8,1,6] x, float[1,16,6] w, float[1,16,4] r, float[1,32] b, float[3,4] w2, float[3] b2) => (float[8,3] y) {
  l1, h1, c1 = LSTM <hidden_size=4> (x, w, r, b)
  s = Constant <value = int64[2] {-1, 4}> ()
  f = Reshape(l1, s)
  y = Gemm <transB=1> (f, w2, b2)
}""", """
v1, v2, v3 = LSTM(x, #, #, #)
v4 = Q0(v1)
v5 = Reshape(v4, #)
y = Gemm(v5, #, #)"""),
}


@pytest.mark.parametrize("case", CASES)
def test_placement(case, tmp_path):
    model, expected = CASES[case]
    assert quantize(model, tmp_path) == expected.strip()


def test_weights_are_quantized_per_channel(tmp_path):
    """Convs and Gemms get weight quantizers with one grid per output channel: every fake-quantized weight is
    (q - zero point) * scale of its row, q an integer in 0..255, and the rows have different scales."""
    src = tmp_path / "in.onnx"
    build(CASES["pool_flatten_gemm"][0], src)
    gm = qat_graph.make_quantizable(qat_graph.load(str(src)))
    layers = [m for m in gm.modules() if isinstance(m, (qat_generic.QConv2d, qat_generic.QLinear))]
    assert [type(m).__name__ for m in layers] == ["QConv2d", "QLinear"]
    qat_generic.set_state(gm, weights=True, acts=False)
    for m in layers:
        w = (m.conv if isinstance(m, qat_generic.QConv2d) else m.linear).weight.detach()
        scale, zp = (t.numpy().astype(np.float64).reshape(-1, 1) for t in qat_generic.weight_qparams(w))
        fq = m.wq(w).detach().reshape(len(w), -1).numpy().astype(np.float64)
        q = fq / scale + zp
        assert np.allclose(q, np.round(q), atol=1e-3) and q.min() > -0.5 and q.max() < 255.5
        assert len(np.unique(scale)) == len(w)
