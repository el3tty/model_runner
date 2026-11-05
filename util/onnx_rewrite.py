#!/usr/bin/env python3
"""Model independent rewrites of an ONNX graph into the form the K230 compile path runs best, by small graph patterns.

A Rule is a pair of functions:
  * match(g, node): looks at a few nodes and shapes around `node`; returns what it found (a dict, so the generator needs
    no second walk) or None;
  * generate(g, m): the new nodes for that match, as Replace(remove, add, alias).
rewrite() applies the rules in the order of RULES, each until it no longer matches, and compares the graph with the
original on a random input after every rule that changed something. All rules keep the function, up to float rounding.

The rules are deliberately narrow: each matches only the ops, layouts and constant shapes seen in the models it was
written for (batch 1, Add / Mul with a per-channel or full-size constant, ...). Anything else is left as it is. A new
model that needs more should get its pattern added after looking at its graph, not by loosening a rule on a guess.

Why these rewrites: nncase cannot put a Reshape inside a layer group, so a tensor that goes through Reshape / Transpose
between two ops makes a round trip through DDR, and most of the KPU time is DDR traffic; Slice, Gather and Expand run
on the CPU. The rules move the layout changes out of the way:
  * a Linear on NHWC data (Transpose -> MatMul -> Add -> ... -> Transpose) becomes an NCHW 1x1 Conv: matmul_to_conv makes
    the Conv, the sink_* rules move the Transpose / token Reshape behind the ops that follow until an opposite one
    cancels it (cancel_*, merge_*), and the Add / Mul by a constant fold into the Conv (fold_conv_*);
  * Conv / MatMul + split into q, k, v becomes one Conv per part;
  * an Expand whose reader broadcasts anyway goes away.
"""
import collections
import copy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference

NHWC = (0, 2, 3, 1)
NCHW = (0, 3, 1, 2)


@dataclass
class Replace:
    remove: list  # nodes to drop
    add: list  # nodes to add
    alias: dict = field(default_factory=dict)  # tensor name -> the tensor its readers use from now on


@dataclass
class Rule:
    name: str
    match: Callable
    generate: Callable


class Graph:
    """An ONNX model with its readers, writers, constants and shapes at hand."""

    def __init__(self, model):
        self.model = model
        self.g = model.graph
        self.count = 0
        self.consts = {}
        for i in self.g.initializer:
            self.consts[i.name] = numpy_helper.to_array(i)
        keep = []
        for n in self.g.node:
            if n.op_type == "Constant" and len(n.attribute) == 1 and n.attribute[0].name == "value":
                self.consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
            else:
                keep.append(n)
        self.nodes = keep
        self.outputs = {o.name for o in self.g.output}
        self.refresh()

    def to_model(self):
        """The current graph as a model: only the constants still in use, in topological order."""
        m = copy.deepcopy(self.model)
        del m.graph.node[:]
        m.graph.node.extend(self.nodes)
        used = {i for n in self.nodes for i in n.input}
        del m.graph.initializer[:]
        m.graph.initializer.extend(numpy_helper.from_array(v, k) for k, v in self.consts.items() if k in used)
        del m.graph.value_info[:]
        return m

    def refresh(self):
        self._prune()
        self._sort()
        self.writer = {o: n for n in self.nodes for o in n.output}
        self.readers = collections.defaultdict(list)
        for n in self.nodes:
            for i in n.input:
                self.readers[i].append(n)
        m = shape_inference.infer_shapes(self.to_model())
        self.shapes = {}
        for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
            self.shapes[v.name] = tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
        self.shapes.update({k: v.shape for k, v in self.consts.items()})

    def _prune(self):
        """Drop nodes nothing reads, repeatedly."""
        while True:
            read = {i for n in self.nodes for i in n.input} | self.outputs
            keep = [n for n in self.nodes if any(o in read for o in n.output)]
            if len(keep) == len(self.nodes):
                return
            self.nodes = keep

    def _sort(self):
        done = set(self.consts) | {i.name for i in self.g.input} | {""}
        pending, order = list(self.nodes), []
        while pending:
            rest = []
            for n in pending:
                if all(i in done for i in n.input):
                    order.append(n)
                    done.update(n.output)
                else:
                    rest.append(n)
            if len(rest) == len(pending):
                raise ValueError(f"cycle or missing input at {rest[0].op_type} {list(rest[0].input)}")
            pending = rest
        self.nodes = order

    def fresh(self, hint="t"):
        self.count += 1
        return f"rw{self.count}_{hint.split('/')[-1]}"

    def const_name(self, array, hint="c"):
        name = self.fresh(hint)
        self.consts[name] = np.asarray(array)
        return name

    def shape(self, name):
        return self.shapes.get(name)

    def const(self, name):
        return self.consts.get(name)

    def one_const(self, node):
        """(tensor input, constant) of a binary node with exactly one constant input, else None."""
        a, b = node.input[0], node.input[1]
        if (a in self.consts) == (b in self.consts):
            return None
        return (b, self.consts[a]) if a in self.consts else (a, self.consts[b])

    def apply(self, r):
        gone = {id(n) for n in r.remove}
        self.nodes = [n for n in self.nodes if id(n) not in gone] + list(r.add)
        for old, new in r.alias.items():
            if old in self.outputs:
                self.nodes.append(helper.make_node("Identity", [new], [old]))
                continue
            for n in self.nodes:
                for k, i in enumerate(n.input):
                    if i == old:
                        n.input[k] = new
        self.refresh()

    def conv_hw(self, tokens):
        """(H, W) of a token count, from the Conv outputs of the graph (None unless exactly one size fits)."""
        sizes = {self.shape(n.output[0])[2:4] for n in self.nodes if n.op_type == "Conv"}
        fit = [s for s in sizes if s[0] * s[1] == tokens]
        return fit[0] if len(fit) == 1 else None


def perm_of(node):
    return next((tuple(a.ints) for a in node.attribute if a.name == "perm"), ())


def attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


def is_scalar_near(array, value, tol=1e-5):
    return array is not None and array.size == 1 and abs(float(array.reshape(())) - value) <= tol * max(1, abs(value))


def transpose(g, x, perm, out=None):
    return helper.make_node("Transpose", [x], [out or g.fresh("tr")], perm=list(perm))


def reshape(g, x, shape, out=None):
    return helper.make_node("Reshape", [x, g.const_name(np.array(shape, np.int64), "shape")], [out or g.fresh("rs")])


def conv1x1(g, x, weight, bias, out=None):
    """Conv 1x1 with weight [in, out] (the layout of a MatMul weight), bias [out] or None."""
    inputs = [x, g.const_name(np.ascontiguousarray(weight.T)[:, :, None, None], "w")]
    if bias is not None:
        inputs.append(g.const_name(bias, "b"))
    return helper.make_node("Conv", inputs, [out or g.fresh("conv")], kernel_shape=[1, 1])


# The wrappers that sink: Transpose NCHW -> NHWC, and a Reshape that merges H and W into tokens, [1,H,W,C] -> [1,N,C].

def wrapper_of(g, tensor):
    """The sinkable wrapper node that writes tensor, else None."""
    n = g.writer.get(tensor)
    if n is None:
        return None
    if n.op_type == "Transpose" and perm_of(n) == NHWC and g.shape(n.input[0])[0] == 1:
        return n
    if n.op_type == "Reshape":
        src, dst = g.shape(n.input[0]), g.shape(n.output[0])
        if src and dst and len(src) == 4 and dst == (1, src[1] * src[2], src[3]) and src[0] == 1:
            return n
    return None


def rewrap(g, wrapper, tensor, out):
    """The same wrapper around another tensor, written to `out`."""
    if wrapper.op_type == "Transpose":
        return transpose(g, tensor, perm_of(wrapper), out)
    return reshape(g, tensor, g.shape(out), out)


def sink(g, wrapper, block, inputs=None):
    """Replace for moving wrapper from the input of `block` (nodes, last one writes the result) to its output. The
    wrapper stays where it is if something else reads it (the pruning drops it otherwise). inputs: other inputs to
    swap, {old name: new name}."""
    swap = {wrapper.output[0]: wrapper.input[0], **(inputs or {})}
    result = block[-1].output[0]
    inner = g.fresh(result)
    add = []
    for n in block:
        c = copy.deepcopy(n)
        c.input[:] = [swap.get(i, i) for i in c.input]
        if n is block[-1]:
            c.output[0] = inner
        add.append(c)
    add.append(rewrap(g, wrapper, inner, result))
    return Replace(block, add)


def gelu_block(g, div):
    """The five nodes of x * (1 + Erf(x / sqrt 2)) * 0.5 starting at their Div, else None."""
    if div.op_type != "Div" or not is_scalar_near(g.const(div.input[1]), 2 ** 0.5):
        return None
    x = div.input[0]
    chain = [div]
    for op in ("Erf", "Add", "Mul", "Mul"):
        users = g.readers[chain[-1].output[0]]
        if len(users) != 1 or users[0].op_type != op:
            return None
        chain.append(users[0])
    if x not in chain[3].input or not is_scalar_near(g.one_const(chain[4])[1] if g.one_const(chain[4]) else None, 0.5):
        return None
    return chain


def layernorm_block(g, mean):
    """The nine nodes of a LayerNorm over the last axis starting at its first ReduceMean, else None."""
    if mean.op_type != "ReduceMean" or list(attr(mean, "axes", [])) != [-1]:
        return None
    x = mean.input[0]
    sub = next((u for u in g.readers[mean.output[0]] if u.op_type == "Sub" and u.input[0] == x), None)
    if sub is None:
        return None
    chain = [mean, sub]
    pow_ = next((u for u in g.readers[sub.output[0]] if u.op_type == "Pow"), None)
    if pow_ is None:
        return None
    chain.append(pow_)
    for op in ("ReduceMean", "Add", "Sqrt", "Div", "Mul", "Add"):
        users = [u for u in g.readers[chain[-1].output[0]] if u.op_type == op]
        if len(users) != 1:
            return None
        chain.append(users[0])
    return chain


# matmul_to_conv: a MatMul with a constant weight on [1, H, W, C] data is a 1x1 Conv between two Transposes.

def match_matmul_to_conv(g, node):
    shape = g.shape(node.input[0]) or ()
    if node.op_type != "MatMul" or len(shape) != 4 or shape[0] != 1:
        return None
    w = g.const(node.input[1])
    return {"node": node, "w": w} if w is not None and w.ndim == 2 else None


def generate_matmul_to_conv(g, m):
    node = m["node"]
    t = transpose(g, node.input[0], NCHW)
    c = conv1x1(g, t.output[0], m["w"], None)
    return Replace([node], [t, c, transpose(g, c.output[0], NHWC, node.output[0])])


# sink_elementwise: wrapper -> Add / Mul with a constant becomes that op, on the constant in the wrapper's layout, ->
# wrapper. The constants seen: per channel [C] (bias, layer scale) and full size (position embedding).

def match_sink_elementwise(g, node):
    if node.op_type not in ("Add", "Mul"):
        return None
    pair = g.one_const(node)
    if pair is None:
        return None
    x, c = pair
    w = wrapper_of(g, x)
    if w is None:
        return None
    shape, src = g.shape(x), g.shape(w.input[0])
    if c.shape == (shape[-1],):
        new = c if w.op_type == "Reshape" else c.reshape(1, -1, 1, 1)
    elif c.shape == shape:
        new = c.reshape(src) if w.op_type == "Reshape" else np.transpose(c, np.argsort(perm_of(w)))
    else:
        return None
    return {"node": node, "wrapper": w, "new": new}


def generate_sink_elementwise(g, m):
    node = m["node"]
    other = next(i for i in node.input if i in g.consts)
    return sink(g, m["wrapper"], [node], {other: g.const_name(m["new"], "c")})


# sink_gelu: wrapper -> GELU becomes GELU -> wrapper, the five nodes staying together.

def match_sink_gelu(g, node):
    block = gelu_block(g, node)
    w = wrapper_of(g, node.input[0]) if block else None
    return {"block": block, "wrapper": w} if w is not None else None


def generate_sink_block(g, m):
    return sink(g, m["wrapper"], m["block"])


# sink_layernorm: a token Reshape moves behind a LayerNorm over the last axis; a Transpose cannot (it changes that axis).

def match_sink_layernorm(g, node):
    block = layernorm_block(g, node)
    w = wrapper_of(g, node.input[0]) if block else None
    return {"block": block, "wrapper": w} if w is not None and w.op_type == "Reshape" else None


# sink_matmul: a token Reshape moves behind a MatMul with a constant weight, which acts on the last axis only.

def match_sink_matmul(g, node):
    if node.op_type != "MatMul" or g.const(node.input[1]) is None:
        return None
    w = wrapper_of(g, node.input[0])
    return {"block": [node], "wrapper": w} if w is not None and w.op_type == "Reshape" else None


# merge_add: Add(wrapper(a), wrapper(b)) with equal wrappers becomes wrapper(Add(a, b)): a residual connection.

def match_merge_add(g, node):
    if node.op_type != "Add":
        return None
    a, b = (wrapper_of(g, i) for i in node.input[:2])
    if a is None or b is None or a.op_type != b.op_type or g.shape(a.input[0]) != g.shape(b.input[0]):
        return None
    if a.op_type == "Transpose" and perm_of(a) != perm_of(b):
        return None
    return {"node": node, "a": a, "b": b}


def generate_merge_add(g, m):
    node, a, b = m["node"], m["a"], m["b"]
    inner = g.fresh(node.output[0])
    add = helper.make_node(node.op_type, [a.input[0], b.input[0]], [inner])
    return Replace([node], [add, rewrap(g, a, inner, node.output[0])])


# cancel_transposes: NCHW -> NHWC and back, in either order.

def match_cancel_transposes(g, node):
    if node.op_type != "Transpose":
        return None
    first = g.writer.get(node.input[0])
    if first is None or first.op_type != "Transpose":
        return None
    if {perm_of(first), perm_of(node)} != {NHWC, NCHW}:
        return None
    return {"node": node, "source": first.input[0]}


def generate_alias(g, m):
    return Replace([m["node"]], [], {m["node"].output[0]: m["source"]})


# fold_reshapes: Reshape of a Reshape is one Reshape, or nothing if the shape comes back.

def match_fold_reshapes(g, node):
    if node.op_type != "Reshape":
        return None
    first = g.writer.get(node.input[0])
    if first is None or first.op_type != "Reshape":
        return None
    return {"node": node, "source": first.input[0]}


def generate_fold_reshapes(g, m):
    node, source = m["node"], m["source"]
    if g.shape(source) == g.shape(node.output[0]):
        return Replace([node], [], {node.output[0]: source})
    return Replace([node], [helper.make_node("Reshape", [source, node.input[1]], [node.output[0]])])


# tokens_to_nhwc: Reshape [1,C,H,W] -> [1,C,N], Transpose (0,2,1) is an NHWC Transpose and a token Reshape.

def match_tokens_to_nhwc(g, node):
    if node.op_type != "Transpose" or perm_of(node) != (0, 2, 1):
        return None
    r = g.writer.get(node.input[0])
    if r is None or r.op_type != "Reshape":
        return None
    src = g.shape(r.input[0])
    if not src or len(src) != 4 or src[0] != 1 or g.shape(r.output[0]) != (1, src[1], src[2] * src[3]):
        return None
    return {"node": node, "source": r.input[0]}


def generate_tokens_to_nhwc(g, m):
    node = m["node"]
    t = transpose(g, m["source"], NHWC)
    return Replace([node], [t, reshape(g, t.output[0], g.shape(node.output[0]), node.output[0])])


# heads_to_tokens: [1,h,d,N] -> Transpose (0,3,1,2), Reshape [1,N,h*d] is [1,h*d,H,W] data under the same wrappers.

def match_heads_to_tokens(g, node):
    if node.op_type != "Reshape":
        return None
    t = g.writer.get(node.input[0])
    if t is None or t.op_type != "Transpose" or perm_of(t) != NCHW:
        return None
    src, dst = g.shape(t.input[0]), g.shape(node.output[0])
    if not src or len(src) != 4 or src[0] != 1 or dst != (1, src[3], src[1] * src[2]):
        return None
    hw = g.conv_hw(src[3])
    return {"node": node, "source": t.input[0], "hw": hw} if hw else None


def generate_heads_to_tokens(g, m):
    node, (h, w) = m["node"], m["hw"]
    b, n, c = g.shape(node.output[0])
    r = reshape(g, m["source"], (b, c, h, w))
    t = transpose(g, r.output[0], NHWC)
    return Replace([node], [r, t, reshape(g, t.output[0], (b, n, c), node.output[0])])


# split_qkv: tokens -> MatMul -> Add -> Reshape [1,N,3,h,d] -> Transpose (2,0,3,1,4) -> Gather 0, 1, 2 -> swap of the last
# two axes: three 1x1 Convs on NCHW data, each output reshaped to [1,h,d,N]. No Gather (CPU) and no Transposes left.

def match_split_qkv(g, node):
    if node.op_type != "MatMul" or g.const(node.input[1]) is None or len(g.shape(node.input[0]) or ()) != 3:
        return None
    users = g.readers[node.output[0]]
    if len(users) != 1 or users[0].op_type != "Add":
        return None
    add = users[0]
    pair = g.one_const(add)
    users = g.readers[add.output[0]]
    if pair is None or len(users) != 1 or users[0].op_type != "Reshape":
        return None
    r = users[0]
    shape = g.shape(r.output[0])
    users = g.readers[r.output[0]]
    if len(shape) != 5 or shape[0] != 1 or shape[2] != 3 or len(users) != 1 or perm_of(users[0]) != (2, 0, 3, 1, 4):
        return None
    t = users[0]
    gathers = g.readers[t.output[0]]
    if len(gathers) != 3 or any(u.op_type != "Gather" or attr(u, "axis", 0) != 0 for u in gathers):
        return None
    parts = {}
    for u in gathers:
        swap = g.readers[u.output[0]]
        if len(swap) != 1 or perm_of(swap[0]) != (0, 1, 3, 2):
            return None
        parts[int(g.const(u.input[1]))] = swap[0]
    hw = g.conv_hw(shape[1])
    if hw is None or sorted(parts) != [0, 1, 2] or pair[1].shape != (g.const(node.input[1]).shape[1],):
        return None
    return {"nodes": [node, add, r, t] + gathers + list(parts.values()), "x": node.input[0], "w": g.const(node.input[1]),
            "b": pair[1], "parts": parts, "shape": shape, "hw": hw}


def generate_split_qkv(g, m):
    b, n, _, heads, d = m["shape"]
    h, w = m["hw"]
    c = heads * d
    r = reshape(g, m["x"], (b, h, w, c))
    t = transpose(g, r.output[0], NCHW)
    add = [r, t]
    for i, swap in m["parts"].items():
        conv = conv1x1(g, t.output[0], m["w"][:, i * c:(i + 1) * c], m["b"][i * c:(i + 1) * c])
        add += [conv, reshape(g, conv.output[0], (b, heads, d, n), swap.output[0])]
    return Replace(m["nodes"], add)


def plain_conv1x1(g, conv):
    """conv is a 1x1 Conv, group 1, with constant weight and bias."""
    if conv.op_type != "Conv" or len(conv.input) != 3:
        return False
    w = g.const(conv.input[1])
    return w is not None and w.shape[2:] == (1, 1) and attr(conv, "group", 1) == 1 and g.const(conv.input[2]) is not None


def channel_parts(g, readers):
    """[(first channel, end, output tensor)] when `readers` cut a tensor into consecutive channel ranges: two Slices
    (starts, ends, axes 1) or one Split on axis 1; else None."""
    if len(readers) == 1 and readers[0].op_type == "Split" and attr(readers[0], "axis") == 1:
        sizes = g.const(readers[0].input[1])
        if sizes is None:
            return None
        ends = np.cumsum(sizes)
        return [(int(e - s), int(e), o) for s, e, o in zip(sizes, ends, readers[0].output)]
    if len(readers) != 2 or any(r.op_type != "Slice" or len(r.input) != 4 for r in readers):
        return None
    parts = []
    for r in readers:
        start, end, axes = (g.const(i) for i in r.input[1:4])
        if start is None or end is None or axes is None or axes.tolist() != [1]:
            return None
        parts.append((int(start[0]), int(end[0]), r.output[0]))
    parts.sort()
    return parts if parts[0][1] == parts[1][0] and parts[0][0] == 0 else None


# split_silu_conv: 1x1 Conv -> SiLU -> Slice / Split on channels: one Conv and SiLU per part. YOLO's cv1(x).chunk(2).

def match_split_silu_conv(g, node):
    if not plain_conv1x1(g, node) or len(g.readers[node.output[0]]) != 2:
        return None
    sig = next((u for u in g.readers[node.output[0]] if u.op_type == "Sigmoid"), None)
    mul = next((u for u in g.readers[node.output[0]] if u.op_type == "Mul"), None)
    if sig is None or mul is None or sig.output[0] not in mul.input or len(g.readers[sig.output[0]]) != 1:
        return None
    readers = g.readers[mul.output[0]]
    parts = channel_parts(g, readers)
    if parts is None or parts[-1][1] != g.const(node.input[1]).shape[0]:
        return None
    return {"nodes": [node, sig, mul] + readers, "conv": node, "parts": parts}


def generate_split_silu_conv(g, m):
    conv = m["conv"]
    w, b = g.const(conv.input[1]), g.const(conv.input[2])
    add = []
    for first, end, out in m["parts"]:
        c = conv_with(g, conv, g.fresh("conv"), w[first:end], b[first:end])
        s = helper.make_node("Sigmoid", [c.output[0]], [g.fresh("sig")])
        add += [c, s, helper.make_node("Mul", [c.output[0], s.output[0]], [out])]
    return Replace(m["nodes"], add)


# split_qkv_heads: 1x1 Conv -> Reshape [1,h,q+k+v,N] -> Split on axis 2: per head the channels are [q, k, v], so one Conv
# per part, with the rows of all heads, and a Reshape to [1,h,part,N]. YOLO's Attention.

def match_split_qkv_heads(g, node):
    if not plain_conv1x1(g, node) or len(g.readers[node.output[0]]) != 1:
        return None
    r = g.readers[node.output[0]][0]
    shape = g.shape(r.output[0]) if r.op_type == "Reshape" else None
    if not shape or len(shape) != 4 or shape[0] != 1 or len(g.readers[r.output[0]]) != 1:
        return None
    split = g.readers[r.output[0]][0]
    if split.op_type != "Split" or attr(split, "axis") != 2 or len(split.output) != 3:
        return None
    sizes = g.const(split.input[1])
    return {"nodes": [node, r, split], "conv": node, "split": split, "sizes": sizes, "shape": shape} \
        if sizes is not None else None


def generate_split_qkv_heads(g, m):
    conv, (b, heads, per_head, n) = m["conv"], m["shape"]
    w, bias = g.const(conv.input[1]), g.const(conv.input[2])
    add, first = [], 0
    for size, out in zip(m["sizes"].tolist(), m["split"].output):
        rows = [h * per_head + first + i for h in range(heads) for i in range(size)]
        c = conv_with(g, conv, g.fresh("conv"), w[rows], bias[rows])
        add += [c, reshape(g, c.output[0], (b, heads, size, n), out)]
        first += size
    return Replace(m["nodes"], add)


# fold_conv_reshape_scale: Conv -> Reshape -> Mul by a scalar: the attention scale of q, into the Conv.

def match_fold_conv_reshape_scale(g, node):
    if node.op_type != "Mul":
        return None
    pair = g.one_const(node)
    if pair is None or pair[1].size != 1:
        return None
    r = g.writer.get(pair[0])
    if r is None or r.op_type != "Reshape" or len(g.readers[r.output[0]]) != 1:
        return None
    conv = g.writer.get(r.input[0])
    if conv is None or not plain_conv1x1(g, conv) or len(g.readers[conv.output[0]]) != 1:
        return None
    return {"node": node, "reshape": r, "conv": conv, "s": float(pair[1].reshape(()))}


def generate_fold_conv_reshape_scale(g, m):
    conv, r, s = m["conv"], m["reshape"], m["s"]
    new = conv_with(g, conv, conv.output[0], g.const(conv.input[1]) * np.float32(s), g.const(conv.input[2]) * np.float32(s))
    return Replace([conv, r, m["node"]], [new, helper.make_node("Reshape", list(r.input), [m["node"].output[0]])])


# drop_expand: an Expand read only as the divisor of a Div on a tensor of the expanded shape, which broadcasts by itself
# (the last step of F.normalize).

def match_drop_expand(g, node):
    if node.op_type != "Expand":
        return None
    out = node.output[0]
    for u in g.readers[out]:
        others = [i for i in u.input if i != out]
        if u.op_type != "Div" or u.input[1] != out or len(others) != 1 or g.shape(others[0]) != g.shape(out):
            return None
    return {"node": node, "source": node.input[0]}


# fold_conv_bias / fold_conv_scale: Add / Mul by a per-channel constant [1,C,1,1], the only reader of a Conv.

def conv_channel_const(g, node, op):
    if node.op_type != op:
        return None
    pair = g.one_const(node)
    if pair is None:
        return None
    x, c = pair
    conv = g.writer.get(x)
    if conv is None or conv.op_type != "Conv" or len(g.readers[x]) != 1 or g.const(conv.input[1]) is None:
        return None
    if c.shape != (1, g.const(conv.input[1]).shape[0], 1, 1):
        return None
    return {"node": node, "conv": conv, "c": c.reshape(-1)}


def match_fold_conv_bias(g, node):
    return conv_channel_const(g, node, "Add")


def match_fold_conv_scale(g, node):
    return conv_channel_const(g, node, "Mul")


def conv_with(g, conv, out, weight, bias):
    c = copy.deepcopy(conv)
    del c.input[:]
    c.input.extend([conv.input[0], g.const_name(weight, "w"), g.const_name(bias, "b")])
    c.output[0] = out
    c.name = g.fresh(conv.name)
    return c


def conv_bias(g, conv):
    w = g.const(conv.input[1])
    return g.const(conv.input[2]) if len(conv.input) > 2 else np.zeros(w.shape[0], w.dtype)


def generate_fold_conv_bias(g, m):
    conv, node = m["conv"], m["node"]
    new = conv_with(g, conv, node.output[0], g.const(conv.input[1]), conv_bias(g, conv) + m["c"])
    return Replace([conv, node], [new])


def generate_fold_conv_scale(g, m):
    conv, node, s = m["conv"], m["node"], m["c"]
    w = g.const(conv.input[1]) * s[:, None, None, None]
    return Replace([conv, node], [conv_with(g, conv, node.output[0], w, conv_bias(g, conv) * s)])


RULES = [
    Rule("split_qkv", match_split_qkv, generate_split_qkv),
    Rule("split_silu_conv", match_split_silu_conv, generate_split_silu_conv),
    Rule("split_qkv_heads", match_split_qkv_heads, generate_split_qkv_heads),
    Rule("fold_conv_reshape_scale", match_fold_conv_reshape_scale, generate_fold_conv_reshape_scale),
    Rule("tokens_to_nhwc", match_tokens_to_nhwc, generate_tokens_to_nhwc),
    Rule("heads_to_tokens", match_heads_to_tokens, generate_heads_to_tokens),
    Rule("drop_expand", match_drop_expand, generate_alias),
    Rule("fold_reshapes", match_fold_reshapes, generate_fold_reshapes),
    Rule("cancel_transposes", match_cancel_transposes, generate_alias),
    Rule("merge_add", match_merge_add, generate_merge_add),
    Rule("sink_layernorm", match_sink_layernorm, generate_sink_block),
    Rule("sink_matmul", match_sink_matmul, generate_sink_block),
    Rule("matmul_to_conv", match_matmul_to_conv, generate_matmul_to_conv),
    Rule("sink_gelu", match_sink_gelu, generate_sink_block),
    Rule("sink_elementwise", match_sink_elementwise, generate_sink_elementwise),
    Rule("fold_conv_bias", match_fold_conv_bias, generate_fold_conv_bias),
    Rule("fold_conv_scale", match_fold_conv_scale, generate_fold_conv_scale),
]


def lint(model):
    """Lines describing what will cost time on the K230, from what was measured so far."""
    g = Graph(model)
    ops = collections.Counter(n.op_type for n in g.nodes)
    lines = []
    between = sum(1 for n in g.nodes if n.op_type == "Reshape" and g.writer.get(n.input[0]) is not None
                  and any(u.op_type in ("Conv", "MatMul", "Mul", "Add") for u in g.readers[n.output[0]]))
    if between:
        lines.append(f"{between} Reshape between compute ops: nncase cannot fuse across them, a DDR round trip each")
    if ops["Transpose"]:
        lines.append(f"{ops['Transpose']} Transpose: each a separate KPU pass or a fusion boundary")
    for op, why in (("Slice", "runs on the CPU"), ("Split", "its outputs are fetched on the CPU"),
                    ("Gather", "runs on the CPU"), ("Expand", "runs on the CPU, and copies")):
        if ops[op]:
            lines.append(f"{ops[op]} {op}: {why}")
    reflect = sum(1 for n in g.nodes if n.op_type == "Pad" and attr(n, "mode", b"constant") == b"reflect")
    if reflect:
        lines.append(f"{reflect} reflect Pad: runs on the CPU, AI2D gave zero borders in mirror mode")
    erf = sum(1 for n in g.nodes if n.op_type == "Erf")
    whole = sum(1 for n in g.nodes if gelu_block(g, n))
    if erf != whole:
        lines.append(f"{erf - whole} of {erf} Erf not in a whole GELU: nncase fails on them")
    return lines


def run_model(model, feeds):
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    s = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return s.run(None, feeds)


def random_feeds(model, seed=0):
    rng = np.random.default_rng(seed)
    feeds = {}
    for i in model.graph.input:
        shape = [d.dim_value for d in i.type.tensor_type.shape.dim]
        feeds[i.name] = rng.uniform(0, 1, shape).astype(np.float32)
    return feeds


def rewrite(model, rules=None, tolerance=1e-3, log=print, limit=5000):
    """model rewritten by rules (default RULES), each applied until it no longer matches; repeated while any rule
    still fires. Compared with the original on a random input after every rule that fired. Returns the model and
    {rule name: times applied}."""
    rules = RULES if rules is None else rules
    feeds = random_feeds(model)
    want = run_model(model, feeds)
    scale = max(float(np.abs(w).max()) for w in want) or 1.0
    g = Graph(model)
    fired = collections.Counter()
    steps = 0
    changed = True
    while changed:
        changed = False
        for rule in rules:
            n = 0
            while True:
                m = next((m for m in (rule.match(g, x) for x in list(g.nodes)) if m), None)
                if m is None:
                    break
                g.apply(rule.generate(g, m))
                n += 1
                steps += 1
                if steps > limit:
                    raise RuntimeError(f"no end after {limit} rewrites, last rule {rule.name}")
            if n:
                fired[rule.name] += n
                changed = True
                got = run_model(g.to_model(), feeds)
                err = max(float(np.abs(a - b).max()) for a, b in zip(want, got))
                log(f"{rule.name}: {n} x, {len(g.nodes)} nodes, max difference {err:.2e}")
                if err > tolerance * scale:
                    raise ValueError(f"rule {rule.name} changed the result: {err}")
    return (g.to_model() if fired else model), dict(fired)
