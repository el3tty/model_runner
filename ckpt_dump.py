#!/usr/bin/env python3
"""Print nncase K230 simulator / runtime dump files in human readable form.

    ckpt_dump.py <file> [--values N] [--record SEQ] [--kind P|L|S|A] [--lines N] [--sum K]

Dumps come from the Debug nncase runtime (built with ENABLE_DUMP_MANAGER, `kpu.set_dump_root(dir)` before
load_model) and, with NNCASE_SIM_CHECKPOINTS=1, from the K230 simulator. Supported files (detected by name):

    <n>$<op>, <n>$<op>$<arg>   runtime dump of a stackvm op output / input ("type:<code>", "shape:<dims>", one value
                               per line; several such blocks for a tuple). A K230 call is an op named after its
                               function ("1_7"): its $arg_<i> files are the KPU inputs, the file itself the output.
    <n>$<op>.ckpt/act_dm_write.chk
                               the activation units of one KPU call, one record per activation:
                               P  ACT0 input: int32 PSUM, PSUM scale 2^-shift, 7 fp16 line-fit parameters per channel
                               L  ACT0 output kept in PSUM_L1 for the next engine (uint8 or fp16)
                               S  ACT0 output stored to GLB (bank, byte offset, buffer and tile shapes)
                               A  MFU ACT1: fp16 input -> piece-wise linear fit -> stored output, with the fit table
    pdp0_pe_if_in.chk, pdp0_pe_we_in.chk, pdp0_pe_out.chk
                               PDP0 (depthwise conv / pooling): one line per kernel tap of one output pixel, 32 lanes
                               (= channels), lane 31 printed first; pixels and weights are 12-bit, products / pooled
                               values 20-bit two's complement (zero points already subtracted)
    other *.chk                printed as they are (hardware check point formats that are not decoded here)

Options:
    --values N   values shown per tensor (default 12; 0 = all)
    --record S   act_dm_write.chk: only the record(s) with this sequence number
    --kind K     act_dm_write.chk: only records of this kind (P, L, S or A)
    --lines N    pdp0 / raw files: lines shown (default 20; 0 = all)
    --sum K      pdp0_pe_out.chk: also sum every K consecutive lines per lane (K = kernel taps, e.g. 9 for 3x3):
                 the integer accumulator of each output pixel
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

TYPECODES = {0: "bool", 1: "utf8", 2: "int8", 3: "int16", 4: "int32", 5: "int64", 6: "uint8", 7: "uint16",
             8: "uint32", 9: "uint64", 10: "float16", 11: "float32", 12: "float64", 13: "bfloat16"}
ACT_TYPES = {0: "uint8", 1: "int8", 2: "fp16", 3: "int16"}
ACT1_DST_TYPES = {0: "fp16", 1: "uint8", 2: "int8", 3: "int16"}
ROUTES = {0: "PSUM_L1 (next engine)", 1: "GLB", 2: "GLB and PSUM_L1"}
PARAM_NAMES = ["slope_lo", "slope_hi", "offset_lo", "offset_hi", "clamp_lo", "clamp_hi", "knee"]


def fmt_values(v, n):
    v = np.asarray(v).ravel()
    shown = v if n == 0 or v.size <= n else v[:n]
    if np.issubdtype(shown.dtype, np.integer) or np.all(np.mod(shown, 1) == 0):
        text = " ".join(str(int(x)) for x in shown)
    else:
        text = " ".join(f"{x:.6g}" for x in shown)
    return text + (" ..." if shown.size < v.size else "")


def stats(v):
    v = np.asarray(v, np.float64).ravel()
    if v.size == 0:
        return "empty"
    return f"min {v.min():.6g}  max {v.max():.6g}  mean {v.mean():.6g}  std {v.std():.6g}"


def print_tensor(label, v, shape, n):
    print(f"  {label}: shape {list(shape)}, {np.asarray(v).size} values, {stats(v)}")
    if n != 0 and np.asarray(v).ndim == 3 and 1 < v.shape[0] <= 32:
        # first values of every channel
        for c in range(v.shape[0]):
            print(f"    ch{c:<3d} {fmt_values(v[c], n)}")
    else:
        print(f"    {fmt_values(v, n)}")


def hex_fp16(s):
    return np.frombuffer(bytes.fromhex(s), ">u2").view(">f2").astype(np.float64)


# ---- runtime dumps -----------------------------------------------------------------------------------------------
def dump_runtime(path, args):
    m = re.fullmatch(r"(\d+)\$([^$]+)(?:\$(.+))?", path.name)
    if m:
        what = "output" if m.group(3) is None else f"input '{m.group(3)}'"
        op = m.group(2)
        kind = "K230 (KPU) function" if re.fullmatch(r"\d+_\d+", op) else "stackvm op"
        print(f"runtime dump #{m.group(1)}: {what} of {kind} '{op}'")
    lines = path.read_text().split("\n")
    if len(lines) == 2 or not lines[0].startswith("type:"):
        print("  (op name only)" if lines[0].strip() else "  (empty)")
        return
    blocks, cur = [], None
    for line in lines:
        if line.startswith("type:"):
            cur = {"type": int(line[5:]), "shape": None, "values": []}
            blocks.append(cur)
        elif line.startswith("shape:"):
            cur["shape"] = line[6:].split()
        elif line.strip() and cur is not None:
            cur["values"].append(line.strip())
    for i, b in enumerate(blocks):
        tname = TYPECODES.get(b["type"], f"type {b['type']}")
        label = f"tensor {i}" if len(blocks) > 1 else "tensor"
        if b["shape"] == ["scalar"]:
            print(f"  {label}: {tname} scalar = {b['values'][0] if b['values'] else '?'}")
            continue
        shape = [int(d) for d in b["shape"]]
        try:
            v = np.array([float(x) for x in b["values"]])
        except ValueError:
            print(f"  {label}: {tname} {shape}: {' '.join(b['values'][:args.values])}")
            continue
        v = v.reshape(shape) if v.size == int(np.prod(shape)) else v
        print(f"  {label}: {tname}")
        print_tensor("values", v.reshape(v.shape[-3:]) if v.ndim >= 3 and np.prod(v.shape[:-3]) == 1 else v,
                     shape, args.values)


# ---- act_dm_write.chk ----------------------------------------------------------------------------------------------
def parse_act_records(path):
    lines = path.read_text().split("\n")
    i = 0
    while i < len(lines) and lines[i]:
        h = lines[i].split()
        kind = h[0]
        if kind == "P":
            c, hh, w = map(int, h[9:12])
            rec = dict(kind="P", seq=int(h[1]), shift=int(h[3]), route=int(h[5]), type=int(h[7]), shape=(c, hh, w),
                       params=hex_fp16(lines[i + 1]).reshape(-1, 7)[:c],
                       psum=np.frombuffer(bytes.fromhex(lines[i + 2]), ">i4").astype(np.int64).reshape(c, hh, w))
            i += 3
        elif kind == "L":
            c, hh, w = map(int, h[5:8])
            t = int(h[9])
            raw = bytes.fromhex(lines[i + 1])
            data = hex_fp16(lines[i + 1]) if t == 2 else np.frombuffer(raw, np.uint8).astype(np.int64)
            rec = dict(kind="L", seq=int(h[1]), base=int(h[3], 16), type=t, shape=(c, hh, w), data=data.reshape(c, hh, w))
            i += 2
        elif kind == "S":
            c, hh, w = map(int, h[17:20])
            t = int(h[21])
            raw = bytes.fromhex(lines[i + 1])
            data = (np.frombuffer(raw, ">i2").astype(np.int64) if t == 3 else hex_fp16(lines[i + 1])) if t >= 2 \
                else np.frombuffer(raw, np.int8 if t == 1 else np.uint8).astype(np.int64)
            rec = dict(kind="S", seq=int(h[1]), bank=int(h[3]), off=int(h[5], 16), mmu=int(h[7], 16),
                       buf=tuple(map(int, h[13:16])), type=t, shape=(c, hh, w), data=data.reshape(c, hh, w))
            i += 2
        elif kind == "A":
            c, hh, w = int(h[3]), int(h[4]), int(h[5])
            rest = dict(zip(h[6::2], h[7::2]))
            dst = int(rest["dst_type"])
            out_raw = bytes.fromhex(lines[i + 3])
            out = np.frombuffer(out_raw, np.uint8 if dst == 1 else np.int8).astype(np.int64) if dst in (1, 2) \
                else (np.frombuffer(out_raw, ">i2").astype(np.int64) if dst == 3 else hex_fp16(lines[i + 3]))
            rec = dict(kind="A", seq=int(h[1]), shape=(c, hh, w), fit16=int(rest["fit16"]), fit_shift=int(rest["fit_shift"]),
                       dst=dst, per_channel=int(rest["per_channel"]), src1_psum=int(rest["src1_psum"]),
                       op_mul=int(rest["op_mul"]), src2=int(rest["src2"]), table=hex_fp16(lines[i + 1]),
                       inp=hex_fp16(lines[i + 2]).reshape(c, hh, w), out=out.reshape(c, hh, w))
            i += 4
        else:
            raise ValueError(f"line {i + 1}: unknown record '{kind}'")
        yield rec


def print_act_record(r, n):
    c, h, w = r["shape"]
    if r["kind"] == "P":
        print(f"P {r['seq']}: ACT0 input, PSUM int32 {c}x{h}x{w}; x = PSUM * 2^-{r['shift']} (fp16, mantissa truncated); "
              f"y = slope*x + offset per segment, clamped; output {ACT_TYPES.get(r['type'], r['type'])} "
              f"-> {ROUTES.get(r['route'], r['route'])}")
        p = r["params"]
        same = np.all(p == p[0])
        print("  line-fit parameters" + (" (same for all channels)" if same else " per channel") + ":  "
              + "  ".join(f"{name:>9s}" for name in PARAM_NAMES))
        for ch in range(1 if same else c):
            print(f"    {'all' if same else 'ch' + str(ch):<6s}" + "".join(f"  {v:9.5g}" for v in p[ch]))
        print_tensor("PSUM", r["psum"], r["shape"], n)
        print(f"  x = PSUM * 2^-{r['shift']}: {stats(r['psum'] * 2.0 ** -r['shift'])}")
    elif r["kind"] == "L":
        print(f"L {r['seq']}: ACT0 output kept in PSUM_L1 at byte {r['base']:#x}: {ACT_TYPES.get(r['type'], r['type'])} {c}x{h}x{w}")
        print_tensor("values", r["data"], r["shape"], n)
    elif r["kind"] == "S":
        bc, bh, bw = r["buf"]
        print(f"S {r['seq']}: ACT0 output stored to GLB bank {r['bank']} offset {r['off']:#x} (mmu {r['mmu']:#x}): "
              f"{ACT_TYPES.get(r['type'], r['type'])} {c}x{h}x{w} tile of a {bc}x{bh}x{bw} (C x H x W) buffer; "
              f"element (c, y, x) of the buffer is at its start + (c*{bh} + y)*{bw} + x")
        print_tensor("values", r["data"], r["shape"], n)
    else:
        src = ("PSUM_L1 (fp16)" if r["src1_psum"] else "GLB") + (" * src2" if r["src2"] and r["op_mul"] else
                                                                 " + src2" if r["src2"] else "")
        print(f"A {r['seq']}: MFU ACT1 {c}x{h}x{w}: input {src} -> "
              f"{'16-segment' if r['fit16'] else '2-segment'} fit (shift {r['fit_shift']}) -> "
              f"{ACT1_DST_TYPES.get(r['dst'], r['dst'])}, parameters {'per channel' if r['per_channel'] else 'shared'}")
        t = r["table"]
        per = 49 if r["fit16"] else 7
        for s in range(min(len(t) // per, c if r["per_channel"] else 1)):
            tab = t[s * per:(s + 1) * per]
            label = f"ch{s}" if r["per_channel"] else "table"
            if r["fit16"]:
                print(f"  {label}: knees      {fmt_values(np.round(tab[:15], 4), 0)}")
                print(f"  {' ' * len(label)}  slopes     {fmt_values(np.round(tab[15:31], 4), 0)}")
                print(f"  {' ' * len(label)}  intercepts {fmt_values(np.round(tab[31:47], 4), 0)}")
                print(f"  {' ' * len(label)}  clamp      {tab[47]:.5g} .. {tab[48]:.5g}")
            else:
                print(f"  {label}: " + "  ".join(f"{k}={v:.5g}" for k, v in
                                                 zip(["knee", "slope_lo", "slope_hi", "offset_lo", "offset_hi", "clamp_lo", "clamp_hi"], tab)))
        print_tensor("input (fp16)", r["inp"], r["shape"], n)
        print_tensor("output", r["out"], r["shape"], n)


def dump_act(path, args):
    print(f"{path}: activation records (P = ACT0 input, L = ACT0 -> PSUM_L1, S = ACT0 -> GLB, A = MFU ACT1)")
    counts = {}
    for r in parse_act_records(path):
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
        if args.record is not None and r["seq"] != args.record:
            continue
        if args.kind and r["kind"] != args.kind:
            continue
        print()
        print_act_record(r, args.values)
    print("\nrecords: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


# ---- pdp0 ----------------------------------------------------------------------------------------------------------
def dump_pdp0(path, args):
    width = 5 if path.name == "pdp0_pe_out.chk" else 3
    bits = 4 * width
    meaning = {"pdp0_pe_if_in.chk": "input pixel (zero point subtracted)", "pdp0_pe_we_in.chk": "weight (zero point subtracted)",
               "pdp0_pe_out.chk": "pixel * weight product, or the running min / max when pooling"}[path.name]
    print(f"{path}: PDP0, one line per kernel tap of one output pixel (order: output row, output column, kernel row, "
          f"kernel column); 32 lanes = channels; value = {meaning}; {bits}-bit two's complement")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if len(line) == 32 * width:
                v = np.array([int(line[k:k + width], 16) for k in range(0, 32 * width, width)][::-1], np.int64)
                rows.append(np.where(v >= 1 << (bits - 1), v - (1 << bits), v))
    a = np.array(rows) if rows else np.zeros((0, 32), np.int64)
    used = int(np.max(np.nonzero(np.any(a != 0, axis=0))[0]) + 1) if a.size and np.any(a) else 0
    print(f"  {len(a)} lines, lanes in use: 0..{used - 1}" if used else f"  {len(a)} lines, all zero")
    shown = a if args.lines == 0 else a[:args.lines]
    for k, v in enumerate(shown):
        print(f"  {k:7d}: {' '.join(f'{x:7d}' for x in v[:max(used, 1)])}")
    if len(shown) < len(a):
        print(f"  ... ({len(a) - len(shown)} more lines; --lines 0 shows all)")
    if args.sum and path.name == "pdp0_pe_out.chk":
        k = args.sum
        acc = a[:len(a) // k * k].reshape(-1, k, 32).sum(1)
        print(f"  sums of {k} consecutive lines (integer accumulator per output pixel), {len(acc)} pixels:")
        for j, v in enumerate(acc if args.lines == 0 else acc[:args.lines]):
            print(f"  {j:7d}: {' '.join(f'{x:8d}' for x in v[:max(used, 1)])}")


def dump_raw(path, args):
    print(f"{path}: hardware check point file, not decoded here; raw lines:")
    with open(path) as f:
        for k, line in enumerate(f):
            if args.lines and k >= args.lines:
                print("  ... (--lines 0 shows all)")
                break
            print(f"  {k:7d}: {line.rstrip()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file")
    ap.add_argument("--values", type=int, default=12)
    ap.add_argument("--record", type=int)
    ap.add_argument("--kind", choices=["P", "L", "S", "A"])
    ap.add_argument("--lines", type=int, default=20)
    ap.add_argument("--sum", type=int, default=0)
    args = ap.parse_args()
    path = Path(args.file)
    if not path.is_file():
        sys.exit(f"{path}: not a file")
    if path.name == "act_dm_write.chk":
        dump_act(path, args)
    elif path.name in ("pdp0_pe_if_in.chk", "pdp0_pe_we_in.chk", "pdp0_pe_out.chk"):
        dump_pdp0(path, args)
    elif path.suffix == ".chk":
        dump_raw(path, args)
    else:
        dump_runtime(path, args)


if __name__ == "__main__":
    main()
