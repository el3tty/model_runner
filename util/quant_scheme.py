"""Write an nncase QuantScheme.json for a QAT-exported ONNX model (the one with QuantizeLinear/DequantizeLinear).

The ranges are read back from the Q/DQ nodes of the exported graph, so the file always matches the model:

  * activation  Q/DQ pair on tensor x          -> entry "x", by_tensor, [(qmin - zp) * scale, (qmax - zp) * scale]
  * weight      Q/DQ pair on an initializer w  -> entry "w", by_channel, one range per output channel
  * Conv weight without Q/DQ (baked weights)   -> entry "w", by_channel, per-channel min/max of the (already fake
                                                 quantized) weights, like Quantizer.GetWeightsRangesByChannel
  * graph input and the tensors that only move/copy values of a tensor with a range (Reshape, Transpose, Slice,
    MaxPool, Resize, Concat of equal ranges ...) -> the same range as their source

The names are the ONNX tensor names. They stay valid when the Q/DQ nodes are removed, because the quantized tensor keeps
the name of x. Tensors without a range (e.g. the Conv output in front of SiLU, which QAT does not quantize) are not in
the file.

No `onnx` package is needed: the model is read at protobuf wire level (numpy only).
"""
import json
import struct

import numpy as np

_DTYPES = {1: np.float32, 2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16, 6: np.int32, 7: np.int64,
           10: np.float16, 11: np.float64}
# range-preserving ops: output values are copies/rearrangements of input values
_PRESERVE = {"Reshape", "Transpose", "Slice", "Squeeze", "Unsqueeze", "Flatten", "Identity", "MaxPool", "Resize",
             "Upsample", "Split", "Pad", "Gather", "Expand", "Tile"}


def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def _fields(b):
    i, n = 0, len(b)
    while i < n:
        k, i = _varint(b, i)
        f, w = k >> 3, k & 7
        if w == 0:
            v, i = _varint(b, i)
        elif w == 2:
            ln, i = _varint(b, i)
            v = b[i:i + ln]
            i += ln
        elif w == 1:
            v = b[i:i + 8]
            i += 8
        elif w == 5:
            v = b[i:i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {w}")
        yield f, w, v


def _tensor(b):
    """TensorProto -> (name, numpy array)."""
    dims, dt, name, raw, fl, i32, i64 = [], 1, "", None, b"", b"", b""
    for f, w, v in _fields(b):
        if f == 1:
            if w == 2:
                j = 0
                while j < len(v):
                    x, j = _varint(v, j)
                    dims.append(x)
            else:
                dims.append(v)
        elif f == 2:
            dt = v
        elif f == 4:
            fl = v
        elif f == 5:
            i32 = v
        elif f == 7:
            i64 = v
        elif f == 8:
            name = v.decode()
        elif f == 9:
            raw = v
    dtype = _DTYPES.get(dt)
    if dtype is None:
        raise ValueError(f"tensor {name}: unsupported data type {dt}")
    if raw is not None:
        a = np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder("<"))
    elif fl:
        a = np.frombuffer(fl, dtype="<f4")
    elif i32:
        j, vals = 0, []
        while j < len(i32):
            x, j = _varint(i32, j)
            vals.append(x)
        a = np.array(vals, dtype=np.int64).astype(dtype)
    elif i64:
        j, vals = 0, []
        while j < len(i64):
            x, j = _varint(i64, j)
            vals.append(x - (1 << 64) if x >= 1 << 63 else x)
        a = np.array(vals, dtype=np.int64)
    else:
        a = np.zeros(0, dtype=dtype)
    return name, a.reshape(dims).copy()


def load_graph(path):
    """-> nodes [{op, in, out}], initializers {name: array}, graph input names."""
    with open(path, "rb") as f:
        model = f.read()
    graph = next(v for f, _, v in _fields(model) if f == 7)
    nodes, inits, inputs = [], {}, []
    for f, _, v in _fields(graph):
        if f == 1:
            node = {"op": "", "in": [], "out": [], "value": None}
            for f2, _, v2 in _fields(v):
                if f2 == 1:
                    node["in"].append(v2.decode())
                elif f2 == 2:
                    node["out"].append(v2.decode())
                elif f2 == 4:
                    node["op"] = v2.decode()
                elif f2 == 5:  # AttributeProto
                    name, t = "", None
                    for f3, _, v3 in _fields(v2):
                        if f3 == 1:
                            name = v3.decode()
                        elif f3 == 5:
                            t = v3
                    if name == "value" and t is not None:
                        node["value"] = _tensor(t)[1]
            nodes.append(node)
        elif f == 5:
            name, a = _tensor(v)
            inits[name] = a
        elif f == 11:
            inputs.append(next(v2 for f2, _, v2 in _fields(v) if f2 == 1).decode())
    for n in nodes:  # Constant nodes act as initializers
        if n["op"] == "Constant" and n["value"] is not None and n["out"]:
            inits.setdefault(n["out"][0], n["value"])
    return nodes, inits, inputs


def _fix_up(lo, hi):
    """Quantizer.FixUpRange (asymmetric): the range contains 0 and is at least 0.1 wide when degenerate."""
    lo, hi = float(lo), float(hi)
    hi = max(hi, 0.0)
    lo = min(lo, 0.0)
    r = hi - lo
    if r < 1e-6:
        r = 0.1
    return lo, lo + r


def _entry(name, dtype, mode, ranges):
    return {"Name": name, "DataType": dtype, "DataRange":
            [{"Min": float(lo), "Max": float(hi), "IsFull": False} for lo, hi in ranges], "DataRangeMode": mode}


def build_scheme(onnx_path, input_range=(0.0, 255.0)):
    nodes, inits, inputs = load_graph(onnx_path)
    entries = {}  # tensor name -> entry, in discovery order
    alias = {}  # DequantizeLinear output -> the tensor that was quantized
    quantized = {}  # QuantizeLinear output -> x

    def qrange(scale, zp):
        info = np.iinfo(zp.dtype)
        scale = np.asarray(scale, dtype=np.float64).reshape(-1)
        zpv = np.asarray(zp, dtype=np.int64).reshape(-1)
        return [((int(info.min) - int(z)) * float(s), (int(info.max) - int(z)) * float(s))
                for s, z in zip(scale, np.broadcast_to(zpv, scale.shape))], ("u8" if info.min == 0 else "i8")

    for n in nodes:
        if n["op"] == "QuantizeLinear":
            quantized[n["out"][0]] = n["in"][0]
        elif n["op"] == "DequantizeLinear" and n["in"][0] in quantized:
            x = quantized[n["in"][0]]
            alias[n["out"][0]] = x
            scale, zp = inits.get(n["in"][1]), inits.get(n["in"][2]) if len(n["in"]) > 2 else None
            if scale is None or zp is None:
                raise SystemExit(f"{x}: scale / zero point of the Q/DQ pair are not constants")
            ranges, dtype = qrange(scale, zp)
            is_weight = x in inits
            if not is_weight and len(ranges) != 1:
                raise SystemExit(f"{x}: per-channel activation quantization is not supported by nncase")
            entries[x] = _entry(x, dtype, "by_channel" if is_weight else "by_tensor", ranges)

    # Conv weights that are plain initializers (baked fake-quantized weights)
    for n in nodes:
        if n["op"] in ("Conv", "ConvTranspose") and len(n["in"]) > 1:
            w = alias.get(n["in"][1], n["in"][1])
            if w in inits and w not in entries and inits[w].dtype == np.float32:
                a = inits[w]
                a = a.transpose(1, 0, *range(2, a.ndim)) if n["op"] == "ConvTranspose" else a
                flat = a.reshape(a.shape[0], -1)
                entries[w] = _entry(w, "u8", "by_channel", [_fix_up(r.min(), r.max()) for r in flat])

    # graph input and range-preserving ops
    def rng(t):
        e = entries.get(alias.get(t, t))
        return None if e is None else (e["DataRange"][0]["Min"], e["DataRange"][0]["Max"])

    for name in inputs:
        if name not in inits and name not in entries:
            entries[name] = _entry(name, "u8", "by_tensor", [input_range])
    for n in nodes:
        if n["op"] in ("QuantizeLinear", "DequantizeLinear", "Constant") or not n["in"]:
            continue
        out = n["out"][0]
        if alias.get(out, out) in entries:
            continue
        src = None
        if n["op"] in _PRESERVE:
            src = rng(n["in"][0])
        elif n["op"] == "Concat":
            rs = [rng(t) for t in n["in"]]
            if rs and rs[0] is not None and all(r == rs[0] for r in rs):
                src = rs[0]
        if src is not None:
            for o in n["out"]:
                entries[o] = _entry(o, "u8", "by_tensor", [src])
    return {"Version": "1.0", "Model": None, "Outputs": list(entries.values())}


def write_scheme(onnx_path, out_path, input_range=(0.0, 255.0)):
    scheme = build_scheme(onnx_path, input_range)
    with open(out_path, "w") as f:
        json.dump(scheme, f, indent=2)
    outs = scheme["Outputs"]
    n_w = sum(e["DataRangeMode"] == "by_channel" for e in outs)
    print(f"wrote {out_path}: {len(outs)} tensors ({len(outs) - n_w} by_tensor, {n_w} by_channel)")
    return scheme
