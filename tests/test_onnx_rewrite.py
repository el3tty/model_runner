"""util/onnx_rewrite.py on small models: each rule on the pattern it was written for, compared with onnxruntime by
rewrite() itself, and close variants of those patterns, which must stay as they are. Needs onnx, onnxruntime and onnxscript."""
import collections

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper, parser

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")  # requirements-qat.txt

from util import onnx_rewrite as rw  # noqa: E402


def model(text, **consts):
    m = parser.parse_model('<ir_version: 8, opset_import: ["" : 13]>\n' + text)
    m.graph.initializer.extend(numpy_helper.from_array(np.asarray(v), k) for k, v in consts.items())
    return m


def rewrite(m):
    out, fired = rw.rewrite(m, log=lambda s: None)
    onnx.checker.check_model(out)
    return collections.Counter(n.op_type for n in out.graph.node), fired


def rand(*shape):
    return np.random.default_rng(sum(shape)).normal(size=shape).astype(np.float32)


# The Linears of a Lite-Mono DilatedConv block: everything but the Convs goes away.
LINEAR = """
g (float[1,4,6,8] x) => (float[1,4,6,8] y) {
    t = Transpose <perm = [0,2,3,1]> (x)
    a = MatMul(t, w1)
    b = Add(b1, a)
    e = Div(b, s2)
    f = Erf(e)
    h = Add(f, one)
    i = Mul(b, h)
    j = Mul(i, half)
    k = MatMul(j, w2)
    l = Add(b2, k)
    m = Mul(gamma, l)
    n = Transpose <perm = [0,3,1,2]> (m)
    y = Add(x, n)
}
"""
LINEAR_CONSTS = dict(w1=rand(4, 16), b1=rand(16), s2=np.float32(2 ** 0.5), one=np.float32(1), half=np.float32(0.5),
                     w2=rand(16, 4), b2=rand(4), gamma=rand(4))


def test_nhwc_linears_become_convs():
    ops, fired = rewrite(model(LINEAR, **LINEAR_CONSTS))
    assert ops["Conv"] == 2 and ops["Transpose"] == 0 and ops["MatMul"] == 0
    assert ops["Erf"] == 1 and ops["Mul"] == 2  # the GELU stays whole, gamma is folded
    assert fired["matmul_to_conv"] == 2 and fired["fold_conv_scale"] == 1


def test_batch_2_is_left_alone():
    ops, fired = rewrite(model(LINEAR.replace("float[1,4,6,8]", "float[2,4,6,8]"), **LINEAR_CONSTS))
    assert not fired and ops["MatMul"] == 2 and ops["Transpose"] == 2


def test_other_permutation_is_left_alone():
    text = """
    g (float[1,4,6,8] x) => (float[1,4,8,6] y) {
        y = Transpose <perm = [0,1,3,2]> (x)
    }
    """
    _, fired = rewrite(model(text))
    assert not fired


def test_sub_with_a_constant_does_not_sink():
    text = """
    g (float[1,4,6,8] x) => (float[1,6,8,4] y) {
        t = Transpose <perm = [0,2,3,1]> (x)
        y = Sub(t, c)
    }
    """
    ops, fired = rewrite(model(text, c=rand(4)))
    assert not fired and ops["Sub"] == 1


def test_expand_read_by_div_goes_away():
    text = """
    g (float[1,2,3,8] x) => (float[1,2,3,8] y) {
        n = ReduceL2 <axes = [-1], keepdims = 1> (x)
        e = Expand(n, shape)
        y = Div(x, e)
    }
    """
    ops, fired = rewrite(model(text, shape=np.array([1, 2, 3, 8], np.int64)))
    assert fired == {"drop_expand": 1} and ops["Expand"] == 0


def test_expand_read_by_mul_stays():
    text = """
    g (float[1,2,3,8] x) => (float[1,2,3,8] y) {
        n = ReduceL2 <axes = [-1], keepdims = 1> (x)
        e = Expand(n, shape)
        y = Mul(x, e)
    }
    """
    ops, fired = rewrite(model(text, shape=np.array([1, 2, 3, 8], np.int64)))
    assert not fired and ops["Expand"] == 1


SILU_SPLIT = """
g (float[1,3,8,8] x) => (float[1,4,8,8] p, float[1,4,8,8] q) {
    c = Conv <kernel_shape = [1,1]> (x, w, b)
    s = Sigmoid(c)
    m = Mul(c, s)
    %s
}
"""
SILU_CONSTS = dict(w=rand(8, 3, 1, 1), b=rand(8))


def test_split_silu_conv_by_slice():
    cut = """p = Slice(m, s0, s4, ax)
    q = Slice(m, s4, s8, ax)"""
    consts = dict(s0=np.array([0]), s4=np.array([4]), s8=np.array([8]), ax=np.array([1]), **SILU_CONSTS)
    ops, fired = rewrite(model(SILU_SPLIT % cut, **consts))
    assert fired == {"split_silu_conv": 1} and ops["Conv"] == 2 and ops["Slice"] == 0


def test_split_silu_conv_by_split():
    cut = "p, q = Split <axis = 1> (m, sizes)"
    ops, fired = rewrite(model(SILU_SPLIT % cut, sizes=np.array([4, 4]), **SILU_CONSTS))
    assert fired == {"split_silu_conv": 1} and ops["Conv"] == 2 and ops["Split"] == 0


def test_slice_of_part_of_the_channels_stays():
    cut = """p = Slice(m, s0, s4, ax)
    q = Slice(m, s2, s6, ax)"""
    consts = dict(s0=np.array([0]), s2=np.array([2]), s4=np.array([4]), s6=np.array([6]), ax=np.array([1]), **SILU_CONSTS)
    ops, fired = rewrite(model(SILU_SPLIT % cut, **consts))
    assert not fired and ops["Slice"] == 2


def test_split_qkv_heads_and_scale():
    text = """
    g (float[1,3,4,4] x) => (float[1,2,2,16] p, float[1,2,2,16] q, float[1,2,4,16] v) {
        c = Conv <kernel_shape = [1,1]> (x, w, b)
        r = Reshape(c, shape)
        a, q, v = Split <axis = 2> (r, sizes)
        p = Mul(a, scale)
    }
    """
    ops, fired = rewrite(model(text, w=rand(16, 3, 1, 1), b=rand(16), shape=np.array([1, 2, 8, 16]),
                               sizes=np.array([2, 2, 4]), scale=np.float32(0.5)))
    assert fired["split_qkv_heads"] == 1 and fired["fold_conv_reshape_scale"] == 1
    assert ops["Conv"] == 3 and ops["Split"] == 0 and ops["Mul"] == 0


def test_empty_clip_max_is_dropped():
    text = """
    g (float[1,4] x) => (float[1,4] y) {
        y = Clip(x, low)
    }
    """
    m = model(text, low=np.float32(0.25))
    m.graph.node[0].input.append("")  # the text parser drops an empty last input
    out, fired = rw.rewrite(m, log=lambda s: None)
    assert fired == {"drop_empty_clip_max": 1} and len(out.graph.node[0].input) == 2


def test_clip_with_a_max_stays():
    text = """
    g (float[1,4] x) => (float[1,4] y) {
        y = Clip(x, low, high)
    }
    """
    _, fired = rewrite(model(text, low=np.float32(0), high=np.float32(1)))
    assert not fired


def test_a_quantized_model_is_refused():
    text = """
    g (float[1,4] x) => (float[1,4] y) {
        q = QuantizeLinear(x, scale, zero)
        y = DequantizeLinear(q, scale, zero)
    }
    """
    m = model(text, scale=np.float32(0.1), zero=np.uint8(128))
    with pytest.raises(ValueError, match="2 Q/DQ nodes"):
        rw.rewrite(m, log=lambda s: None)


LAYERNORM = """
g (float[1,2,3,4] x) => (float[1,6,4] y) {
    r = Reshape(x, shape)
    m = ReduceMean <axes = [-1]> (r)
    d = Sub(r, m)
    p = Pow(d, two)
    v = ReduceMean <axes = [-1]> (p)
    e = Add(v, eps)
    t = Sqrt(e)
    n = Div(d, t)
    s = Mul(n, gamma)
    y = Add(s, beta)
}
"""
LAYERNORM_CONSTS = dict(shape=np.array([1, 6, 4]), two=np.float32(2), eps=np.float32(1e-6), gamma=rand(4), beta=rand(4))


def test_token_reshape_sinks_past_a_layernorm():
    _, fired = rewrite(model(LAYERNORM, **LAYERNORM_CONSTS))
    assert fired == {"sink_layernorm": 1}


@pytest.mark.parametrize("change", [
    {"two": np.float32(3)},  # not a variance
    {"eps": np.float32(0.5)},  # not an epsilon
    {"gamma": rand(6, 4)},  # not per channel
])
def test_not_quite_a_layernorm_constants(change):
    _, fired = rewrite(model(LAYERNORM, **{**LAYERNORM_CONSTS, **change}))
    assert not fired


@pytest.mark.parametrize("old, new", [
    ("n = Div(d, t)", "n = Div(r, t)"),  # divides the input, not the centered tensor
    ("v = ReduceMean <axes = [-1]> (p)", "v = ReduceMean <axes = [1]> (p)"),  # the variance over another axis
])
def test_not_quite_a_layernorm_wiring(old, new):
    _, fired = rewrite(model(LAYERNORM.replace(old, new), **LAYERNORM_CONSTS))
    assert not fired


NORMALIZED_MATMUL = """
g (float[1,2,3,8] q, float[1,2,3,8] k) => (float[1,2,3,3] y) {
    a = ReduceL2 <axes = [-1], keepdims = 1> (q)
    b = Clip(a, eps)
    c = Div(q, b)
    d = ReduceL2 <axes = [-1], keepdims = 1> (k)
    e = Clip(d, eps)
    f = Div(k, e)
    t = Transpose <perm = [0,1,3,2]> (f)
    y = MatMul(c, t)
}
"""


def test_normalize_after_matmul():
    ops, fired = rewrite(model(NORMALIZED_MATMUL, eps=np.float32(1e-12)))
    assert fired == {"normalize_after_matmul": 1} and ops["Div"] == 1


def test_normalized_matmul_with_a_large_eps_stays():
    _, fired = rewrite(model(NORMALIZED_MATMUL, eps=np.float32(0.5)))
    assert not fired


def test_lint_reports_what_costs_time():
    text = """
    g (float[1,4,6,8] x) => (float[1,4,8,10] y) {
        p = Pad <mode = "reflect"> (x, pads)
        y = Transpose <perm = [0,1,3,2]> (p)
    }
    """
    lines = "\n".join(rw.lint(model(text, pads=np.array([0, 0, 1, 0, 0, 0, 1, 0]))))
    assert "reflect Pad" in lines and "Transpose" in lines


def test_a_rule_that_breaks_the_model_is_caught():
    def match(g, node):
        return {"node": node} if node.op_type == "Sigmoid" else None

    def generate(g, m):
        n = m["node"]
        return rw.Replace([n], [helper.make_node("Relu", list(n.input), list(n.output))])

    text = "g (float[1,4] x) => (float[1,4] y) { y = Sigmoid(x) }"
    with pytest.raises(ValueError, match="changed the result"):
        rw.rewrite(model(text), rules=[rw.Rule("bad", match, generate)], log=lambda s: None)
