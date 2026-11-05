#!/usr/bin/env python3
"""Model independent placement of the fake quantizers, on a torch.fx graph: an ONNX model converted by onnx2torch
(load()), or a torch model traced with torch.fx.symbolic_trace (qat_yolox.py).

The quantizers go where the K230 compile path quantizes, by graph rules, so a new model needs no placement code:
  * quantized ops (the KPU runs them on uint8 inputs): Conv (weights fake-quantized per output channel), Gemm (per
    output feature), LSTM (QLSTM: W and R per row, the hidden state quantized at every step), Add and Mul of two
    tensors, AveragePool / GlobalAveragePool;
  * an activation that is the only consumer of a quantized op's result fuses into it (Relu, LeakyRelu, Clip, Sigmoid,
    HardSigmoid, HardSwish, SiLU = Mul(x, Sigmoid(x)) and HardSwish spelled out as x * Clip(x + 3, 0, 6) / 6): the
    tensor the hardware stores is the activation output. An activation that shares the result with other readers
    fuses too: nncase gives it its own copy of the op, so its output is a stored tensor as well;
  * a stored tensor gets an activation fake-quant if a quantized op reads it, directly or through ops that keep
    the quantization grid (MaxPool, nearest Resize, Concat, Slice, Reshape, Transpose, Flatten, Squeeze, Unsqueeze);
    except a Conv / Gemm output without an activation that only Add / Mul read: nncase keeps it in float16 (checked
    in its quantized IR: e.g. a MobileNetV3 project conv into the residual Add). That tensor gets an ExportQuant:
    no rounding here, but a Q/DQ pair in the export, without which nncase leaves the Add / Mul outside the KPU;
  * the inputs of a Concat share one range (nncase unites concat ranges), so do the tensors a grid keeping op
    connects; the model input is not quantized (it is uint8 already).
load() also turns Constant weights into initializers and makes a constant batch size of 1 dynamic; load_scheme_ranges()
takes the activation ranges from an nncase quant scheme. The fake-quant modules, the calibration and the export are
those of qat_generic.py.
"""
import contextlib
import functools
import json

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper
from onnx2torch import convert
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.utils.common import OnnxMapping, OperationConverterResult

from util.qat_generic import ActFakeQuant, ExportQuant, QConv2d, QLinear, QLSTM

# node kinds: module class names (onnx2torch's or torch.nn's), function / method names of a traced torch model
ACTIVATIONS = ("Relu", "LeakyRelu", "Clip", "Sigmoid", "HardSigmoid", "Hardsigmoid", "Hardswish", "OnnxHardSigmoid",
               "OnnxHardSwish", "OnnxClip", "ReLU", "ReLU6", "SiLU", "Hardtanh",
               "relu", "relu6", "leaky_relu", "sigmoid", "silu", "hardswish", "hardsigmoid", "clamp", "hardtanh")
SIGMOIDS = ("Sigmoid", "sigmoid")
CLIPS = ("OnnxClip", "Clip", "ReLU6", "Hardtanh", "relu6", "hardtanh", "clamp")  # Clip(x + 3, 0, 6) of a HardSwish
CONCATS = ("OnnxConcat", "cat", "concat")
GRID_KEEPING = ("MaxPool2d", "OnnxSlice", "OnnxReshape", "OnnxTranspose", "OnnxFlatten", "OnnxSqueeze",
                "OnnxSqueezeStaticAxes", "OnnxSqueezeDynamicAxes", "OnnxUnsqueezeStaticAxes",
                "OnnxUnsqueezeDynamicAxes", "Flatten", "Identity", "max_pool2d", "getitem", "view", "reshape", "permute",
                "flatten", "contiguous", "transpose", "squeeze", "unsqueeze") + CONCATS
AVG_POOLS = ("AvgPool2d", "OnnxGlobalAveragePool", "OnnxGlobalAveragePoolWithKnownInputShape", "AdaptiveAvgPool2d",
             "avg_pool2d", "adaptive_avg_pool2d")
BINARY = {"add": "add", "iadd": "add", "sub": "sub", "isub": "sub", "mul": "mul", "imul": "mul", "truediv": "div",
          "itruediv": "div", "div": "div"}


@add_converter(operation_type="LSTM", version=7)
@add_converter(operation_type="LSTM", version=14)
def _convert_lstm(node, graph):
    """ONNX LSTM -> QLSTM. Only the plain form: constant W, R, B, no sequence lengths, initial states or peepholes,
    default activations."""
    ins = list(node.input_values) + [""] * (8 - len(node.input_values))
    if any(ins[4:]) or any(i not in graph.initializers for i in ins[1:4] if i):
        raise NotImplementedError(f"LSTM {node.name}: only constant W, R, B and no optional inputs are supported")
    a = node.attributes
    if a.get("activations") or a.get("clip") or a.get("input_forget", 0) or a.get("layout", 0):
        raise NotImplementedError(f"LSTM {node.name}: only the default activations and layout are supported")
    w, r = (graph.initializers[ins[i]].to_torch() for i in (1, 2))
    b = graph.initializers[ins[3]].to_torch() if ins[3] else torch.zeros(w.shape[0], 2 * w.shape[1])
    direction = a.get("direction", "forward")
    direction = direction.decode() if isinstance(direction, bytes) else direction
    module = QLSTM(w, r, b, int(a["hidden_size"]), direction)
    return OperationConverterResult(torch_module=module,
                                    onnx_mapping=OnnxMapping(inputs=(ins[0],), outputs=tuple(node.output_values)))


class _HardSigmoidFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, beta):
        return torch.clip(x * alpha + beta, 0.0, 1.0)

    @staticmethod
    def symbolic(g, x, alpha, beta):
        return g.op("HardSigmoid", x, alpha_f=alpha, beta_f=beta)


class HardSigmoid(nn.Module):
    """ONNX HardSigmoid, clip(alpha * x + beta, 0, 1), exported as one HardSigmoid node (onnx2torch's module traces to
    Mul / Add / Clip, which nncase does not fuse: they would run on the CPU)."""

    def __init__(self, alpha=0.2, beta=0.5):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, x):
        if torch.onnx.is_in_onnx_export():
            return _HardSigmoidFn.apply(x, self.alpha, self.beta)
        return torch.clip(x * self.alpha + self.beta, 0.0, 1.0)


def _replace_modules(gm):
    """Modules of onnx2torch that would not export back to the same ONNX op, replaced by ones that do."""
    for name, m in list(gm.named_modules()):
        if type(m).__name__ == "OnnxHardSigmoid":
            _set_module(gm, name, HardSigmoid(m.alpha, m.beta))


def constants_to_initializers(model):
    """Constant nodes -> initializers (onnx2torch takes Conv / Gemm / LSTM weights from initializers only)."""
    g = model.graph
    keep = []
    for n in g.node:
        if n.op_type == "Constant" and len(n.attribute) == 1 and n.attribute[0].name == "value":
            g.initializer.append(numpy_helper.from_array(numpy_helper.to_array(n.attribute[0].t), n.output[0]))
        else:
            keep.append(n)
    del g.node[:]
    g.node.extend(keep)
    return model


def _dynamic_batch(model):
    """Reshapes to a constant shape that starts with the model's batch size 1 (and has no -1) get -1 there, so the
    converted model takes any batch size. Returns the model and the changed shapes."""
    g = model.graph
    changed = []
    inits = {i.name: i for i in g.initializer}
    for n in g.node:
        if n.op_type != "Reshape" or n.input[1] not in inits:
            continue
        shape = numpy_helper.to_array(inits[n.input[1]])
        if shape.ndim == 1 and len(shape) > 1 and shape[0] == 1 and -1 not in shape:
            shape = shape.copy()
            shape[0] = -1
            name = n.input[1] + "_dynamic_batch"
            g.initializer.append(numpy_helper.from_array(shape, name))
            n.input[1] = name
            changed.append(shape)
    return model, changed


def load(path):
    """onnx2torch GraphModule of the float ONNX model at path (BatchNorm must already be folded into the convs), for
    any batch size (static_batch() restores the model's own reshapes for an export)."""
    model, changed = _dynamic_batch(constants_to_initializers(onnx.load(path)))
    gm = convert(model)
    _replace_modules(gm)
    gm.dynamic_batch_buffers = [n for n, b in gm.named_buffers() if b.dtype in (torch.int64, torch.int32)
                                and any(b.numel() == len(c) and b.tolist() == c.tolist() for c in changed)]
    return gm


@contextlib.contextmanager
def static_batch(model):
    """The reshape shapes load() made dynamic (-1) back to the model's batch size 1, for an ONNX export (nncase wants
    the static shapes of the original model). Any model works: one without such reshapes is left alone."""
    buffers = []
    for m in model.modules():
        for name in getattr(m, "dynamic_batch_buffers", []):
            buffers.append(m.get_buffer(name))
    for b in buffers:
        b[0] = 1
    try:
        yield model
    finally:
        for b in buffers:
            b[0] = -1


class _Graph:
    """Node classification of a GraphModule: converted by onnx2torch, or a torch model traced by torch.fx."""

    def __init__(self, gm):
        self.gm = gm
        self.mods = dict(gm.named_modules())
        self.nodes = list(gm.graph.nodes)

    def mod(self, n):
        return self.mods.get(n.target) if n.op == "call_module" else None

    def kind(self, n):
        """Module class name, function name or method name of node n."""
        if n.op == "call_module":
            return type(self.mod(n)).__name__
        if n.op == "call_function":
            return getattr(n.target, "__name__", str(n.target))
        if n.op == "call_method":
            return n.target
        return n.op

    def binary(self, n):
        """'add' / 'sub' / 'mul' / 'div' for a binary op (onnx2torch module, operator / torch function, method)."""
        f = getattr(self.mod(n), "math_op_function", None)
        if f is not None:
            return getattr(f, "__name__", "").removeprefix("_onnx_") or None
        if n.op in ("call_function", "call_method"):
            return BINARY.get(self.kind(n))
        return None

    def is_resize_nearest(self, n):
        m = self.mod(n)
        if type(m).__name__ == "OnnxResize":
            return getattr(m, "onnx_mode", "") == "nearest"
        if type(m).__name__ == "Upsample":
            return m.mode == "nearest"
        return self.kind(n) == "interpolate" and n.kwargs.get("mode", "nearest") == "nearest"

    def is_const(self, n):
        return self.kind(n) == "OnnxConstant" or n.op == "get_attr"

    def data_inputs(self, n):
        """The tensor inputs of n that are not constants (the tensors of a list argument, e.g. torch.cat's, too)."""
        out = []
        for a in n.args:
            for b in (a if isinstance(a, (list, tuple)) else [a]):
                if hasattr(b, "op") and not self.is_const(b):
                    out.append(b)
        return out

    def _const_value(self, n):
        """Python float of a scalar constant (node or number), else None."""
        if isinstance(n, (int, float)) and not isinstance(n, bool):
            return float(n)
        if not hasattr(n, "op"):
            return None
        if n.op == "get_attr":
            t = functools.reduce(getattr, n.target.split("."), self.gm)
        elif self.kind(n) == "OnnxConstant":
            t = self.mod(n).value
        else:
            return None
        return float(t) if t.numel() == 1 else None

    def _is_op(self, n, op, x=None, const=None):
        """n is binary op `op` with data input x (any, if None) and, if given, a scalar constant `const`."""
        if not hasattr(n, "op") or self.binary(n) != op:
            return False
        if x is not None and x not in n.args[:2]:
            return False
        if const is not None:
            return any(self._const_value(a) == const for a in n.args[:2])
        return True

    def act_pattern(self, n):
        """(x, internal nodes) when n is the output of a fusable activation of x, else None. Single op activations,
        SiLU = Mul(x, Sigmoid(x)) and HardSwish spelled out as x * Clip(x + 3, 0, 6) / 6 (either association)."""
        if self.binary(n) == "mul":  # SiLU
            for x, s in (n.args[:2], n.args[1::-1]):
                if hasattr(s, "op") and self.kind(s) in SIGMOIDS and s.args[0] is x:
                    return x, {s}
        if self.binary(n) == "div" and self._const_value(n.args[1]) == 6.0:  # (x * Clip(x + 3)) / 6
            m = n.args[0]
            if self._is_op(m, "mul"):
                for x, c in (m.args[:2], m.args[1::-1]):
                    if hasattr(c, "op") and self.kind(c) in CLIPS and self._is_op(c.args[0], "add", x, 3.0):
                        return x, {m, c, c.args[0]}
        if self.binary(n) == "mul":  # x * (Clip(x + 3) / 6)
            for x, d in (n.args[:2], n.args[1::-1]):
                if self._is_op(d, "div", const=6.0) and hasattr(d.args[0], "op") and self.kind(d.args[0]) in CLIPS:
                    c = d.args[0]
                    if self._is_op(c.args[0], "add", x, 3.0):
                        return x, {d, c, c.args[0]}
        if self.kind(n) in ACTIVATIONS:
            return self.data_inputs(n)[0], set()
        return None

    def internal_nodes(self):
        """Nodes that are inner parts of a multi-node activation (they are not quantized ops themselves)."""
        out = set()
        for n in self.nodes:
            p = self.act_pattern(n)
            if p is not None:
                out |= p[1]
        return out

    def is_quantized_op(self, n, internal):
        k = self.kind(n)
        if k in ("Conv2d", "Linear", "QLSTM") or k in AVG_POOLS:
            return True
        return (n not in internal and self.act_pattern(n) is None and self.binary(n) in ("add", "mul")
                and len(self.data_inputs(n)) == 2)

    def stores_own_output(self, n):
        """Quantized ops whose output quantization is part of the op (the LSTM quantizes h at every step)."""
        return self.kind(n) == "QLSTM"

    def keeps_grid(self, n):
        return self.kind(n) in GRID_KEEPING or self.is_resize_nearest(n)

    def fused_output(self, n):
        """Follow n through activations that are its only consumers (the inner nodes of a SiLU / HardSwish count as
        part of it); the longest pattern wins (SiLU over its Sigmoid)."""
        while True:
            users = set(n.users)
            cands = []
            for u in self.nodes:
                p = self.act_pattern(u)
                if p is not None and p[0] is n and users <= {u} | p[1]:
                    cands.append((len(p[1]), u))
            if not cands:
                return n
            n = max(cands, key=lambda c: c[0])[1]


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] is not x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra is not rb:
            self.parent[rb] = ra


def plan(gm, float16_rule=True):
    """The quantization points of gm: (list of stored tensor nodes to fake-quantize, {node: range group root},
    list of conv nodes). Nothing is changed. float16_rule=False gives the earlier placement, without the float16 rule
    and the side activations (checkpoints made with it load through upgrade_state_dict)."""
    g = _Graph(gm)
    uf = _UnionFind()
    internal = g.internal_nodes()
    anchors = [n for n in g.nodes if g.is_quantized_op(n, internal)]
    stored = set()
    for a in anchors:
        if g.stores_own_output(a):
            continue
        out = g.fused_output(a)
        if not float16_rule:
            stored.add(out)
            continue
        # activations sharing the result with other readers: nncase fuses each into its own copy of the op
        side = [u for u in g.nodes if u not in internal and (p := g.act_pattern(u)) is not None and p[0] is out]
        stored |= {g.fused_output(u) for u in side}
        side_nodes = set(side).union(*(g.act_pattern(u)[1] for u in side))
        readers = [u for u in out.users if u not in side_nodes]
        if (out is a and g.kind(a) in ("Conv2d", "Linear") and readers
                and all(g.binary(u) in ("add", "mul") and g.is_quantized_op(u, internal) for u in readers)):
            continue  # float16 on the K230: a plain conv / gemm output that only Add / Mul read
        stored.add(out)
    for n in g.nodes:  # grid keeping ops: their output is the same quantized tensor (Concat also ties its inputs)
        if g.keeps_grid(n):
            ins = g.data_inputs(n) if g.kind(n) in CONCATS else g.data_inputs(n)[:1]
            for a in ins:
                uf.union(a, n)
    read = set()  # groups some quantized op reads
    for a in anchors:
        for x in g.data_inputs(a):
            read.add(uf.find(x))
    points = [n for n in g.nodes if n in stored and uf.find(n) in read]
    weighted = [n for n in anchors if g.kind(n) in ("Conv2d", "Linear")]
    return points, {n: uf.find(n) for n in points}, weighted


def float16_points(gm):
    """[(plain conv / gemm output of plan()'s float16 rule, the Add / Mul nodes reading it)], in graph order."""
    g = _Graph(gm)
    kept = set(plan(gm)[0])
    return [(n, [u for u in n.users if g.binary(u) in ("add", "mul")])
            for n in plan(gm, float16_rule=False)[0] if n not in kept]


def _set_module(gm, target, module):
    parent, _, name = target.rpartition(".")
    setattr(gm.get_submodule(parent) if parent else gm, name, module)


def make_quantizable(gm):
    """Insert the fake quantizers into the onnx2torch GraphModule gm (in place) and return it: QConv2d for every
    Conv, an ActFakeQuant after every quantization point of plan(), one shared ActRange per range group, an ExportQuant
    in front of the Add / Mul readers of float16_points(). The quantizers start disabled."""
    points, groups, weighted = plan(gm)
    float16 = float16_points(gm)
    index = {n: i for i, n in enumerate(points)}
    gm.legacy_actq = [index.get(n) for n in plan(gm, float16_rule=False)[0]]  # see upgrade_state_dict
    for n in weighted:
        m = gm.get_submodule(n.target)
        _set_module(gm, n.target, QConv2d(m) if isinstance(m, nn.Conv2d) else QLinear(m))
    ranges = {}
    graph = gm.graph
    for i, n in enumerate(points):
        q = ActFakeQuant()
        root = groups[n]
        if root not in ranges:
            ranges[root] = q.params
        else:
            q.params = ranges[root]
        name = f"actq_{i}"
        gm.add_module(name, q)
        users = list(n.users)
        with graph.inserting_after(n):
            qn = graph.call_module(name, (n,))
        for u in users:
            u.replace_input_with(n, qn)
    for i, (n, readers) in enumerate(float16):
        name = f"exportq_{i}"
        gm.add_module(name, ExportQuant())
        with graph.inserting_after(n):
            qn = graph.call_module(name, (n,))
        for u in readers:
            u.replace_input_with(n, qn)
    graph.lint()
    gm.recompile()
    return gm


def upgrade_state_dict(gm, state_dict):
    """A checkpoint of gm made with the earlier placement (plan(float16_rule=False): its actq_<i> numbered over more
    points) -> the same ranges under the current actq_<i> and exportq_<i> names. A checkpoint from before the
    ExportQuants has no ranges for them: they stay unset (seen = 0), and Task.export refuses to use them. Other state
    dicts are returned as they are."""
    legacy = getattr(gm, "legacy_actq", None)
    saved = {int(k.split(".")[0][5:]) for k in state_dict if k.startswith("actq_")}
    current = sum(1 for name, _ in gm.named_children() if name.startswith("actq_"))
    exported = sum(1 for name, _ in gm.named_children() if name.startswith("exportq_"))
    if legacy is None or not exported or any(k.startswith("exportq_") for k in state_dict):
        return state_dict
    if len(saved) == current:
        print(f"checkpoint without the ranges of the {exported} float16 tensors: it evaluates, but cannot be exported "
              "with --act-qdq")
        return {**state_dict, **{k: v for k, v in gm.state_dict().items() if k.startswith("exportq_")}}
    if len(saved) != len(legacy):
        return state_dict
    out, float16 = {}, {i: n for n, i in enumerate(i for i, j in enumerate(legacy) if j is None)}
    for k, v in state_dict.items():
        if not k.startswith("actq_"):
            out[k] = v
            continue
        module, rest = k.split(".", 1)
        i = int(module[5:])
        out[f"actq_{legacy[i]}.{rest}" if legacy[i] is not None else f"exportq_{float16[i]}.{rest}"] = v
    print(f"checkpoint of the earlier quantizer placement: {len(saved)} activation quantizers -> {current} "
          f"(+ {exported} on tensors the K230 keeps in float16, only in the export)")
    return out


def describe_plan(gm):
    """Text listing of the quantization points (for checking): the stored tensor, its group, and its readers."""
    points, groups, weighted = plan(gm)
    ids = {}
    lines = [f"{len(weighted)} convs / linears, {len(points)} activation quantizers in {len(set(groups.values()))} range groups"]
    for n in points:
        gid = ids.setdefault(groups[n], len(ids))
        lines.append(f"  g{gid:3d}  {n.name}  -> {', '.join(u.name for u in n.users)}")
    lines += [f"  export  {n.name}  -> {', '.join(u.name for u in readers)}" for n, readers in float16_points(gm)]
    return "\n".join(lines)



def _onnx_tensors(onnx_path, names, x):
    """Values of the named intermediate tensors of the ONNX model on input x (onnxruntime)."""
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    m = onnx.load(onnx_path)
    produced = {o for n in m.graph.node for o in n.output}
    names = [n for n in names if n in produced]
    del m.graph.output[:]
    m.graph.output.extend(onnx.helper.make_empty_tensor_value_info(n) for n in names)
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    return dict(zip(names, sess.run(None, {sess.get_inputs()[0].name: x})))


def match_scheme(gm, onnx_path, scheme_path, seed=0):
    """[(quantizer module name, ONNX tensor name)] for the activation quantizers of the fake-quant GraphModule gm (from
    make_quantizable, untrained: its weights must still be those of onnx_path): each quantizer is matched to the
    tensor of the nncase quant scheme with the same values on a random input (a QLSTM's out_q to the LSTM's Y
    output). The names hold for any model built the same way from onnx_path (e.g. a trained one). Raises if a
    quantizer has no tensor."""
    names = [o["Name"] for o in json.load(open(scheme_path))["Outputs"]
             if o.get("DataRangeMode", "by_tensor") == "by_tensor"]
    shape = [d.dim_value for d in onnx.load(onnx_path).graph.input[0].type.tensor_type.shape.dim]
    x = np.random.default_rng(seed).uniform(-1, 1, shape).astype(np.float32)
    ref = _onnx_tensors(onnx_path, names, x)
    device = next(gm.parameters()).device
    seen = []
    mods = dict(gm.named_modules())
    lstm_qs = {n + ".out_q" for n, m in mods.items() if isinstance(m, QLSTM)}
    hooks = [m.register_forward_hook(lambda m, i, o, n=n: seen.append((n, i[0])))
             for n, m in mods.items() if isinstance(m, ActFakeQuant) and n not in lstm_qs]
    hooks += [m.register_forward_hook(lambda m, i, o, n=n: seen.append((n + ".out_q", o[0])))
              for n, m in mods.items() if isinstance(m, QLSTM)]
    try:
        with torch.no_grad():
            gm(torch.from_numpy(x).to(device))
    finally:
        for h in hooks:
            h.remove()
    pairs = []
    for qname, t in seen:
        t = t.detach().cpu().numpy()
        tol = 1e-3 * (np.abs(t).max() + 1e-6)
        found = [n for n, v in ref.items() if v.size == t.size and np.abs(v.reshape(-1) - t.reshape(-1)).max() < tol]
        if not found:
            raise ValueError(f"no tensor of {scheme_path} matches a quantizer input of shape {tuple(t.shape)}")
        pairs.append((qname, found[0]))
    groups = {id(m.params) for m in gm.modules() if isinstance(m, ActFakeQuant)}
    missing = groups - {id(mods[q].params) for q, _ in pairs}
    if missing:
        raise ValueError(f"{len(missing)} range groups have no tensor in {scheme_path}")
    return pairs


def load_scheme_ranges(gm, onnx_path, scheme_path):
    """Set the activation ranges of the fake-quant GraphModule gm (from make_quantizable) from an nncase quant scheme
    (QuantScheme.json of --export-quant-scheme, or a .qs.json); a range group takes the union of its members' ranges.
    Returns the number of range groups set."""
    scheme = {o["Name"]: o["DataRange"][0] for o in json.load(open(scheme_path))["Outputs"]}
    ranges = {}
    for qname, name in match_scheme(gm, onnx_path, scheme_path):
        params = gm.get_submodule(qname).params
        r = scheme[name]
        _, lo, hi = ranges.get(id(params), (params, r["Min"], r["Max"]))
        ranges[id(params)] = (params, min(lo, r["Min"]), max(hi, r["Max"]))
    for params, lo, hi in ranges.values():
        params.min_val.fill_(lo)
        params.max_val.fill_(hi)
        params.seen.fill_(1)
    return len(ranges)
