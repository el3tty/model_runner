#!/usr/bin/env python3
"""
Run a detector kmodel on the images written by coco_prepare.py and keep the candidate cells of its raw outputs, for
coco_eval.py on the PC. Needs only numpy and nncaseruntime (copy this script and util/ to the board); works the same
in the nncase simulator on the PC (slow: use --limit).

Every <id>.u8 input is converted to the kmodel's input type (0..255 for yolox, / 255 for the 0..1 models). Of each
output only the cells (a YOLOX row, or one anchor of an anchor-head grid cell) whose objectness * best class score
>= --conf are kept, as float16; the file layout is described in util/coco.py.

Usage:
    python coco_run.py yolox-qat.kmodel --input coco-yolox --output dets-yolox.npz [--detect yolox] [--conf 0.01]
                       [--limit N]
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # util/ and nncaseruntime of the repository

import numpy as np
import nncaseruntime as nn

from util import coco as cu
from util import yolo_post
from util.common import temp_dump_root


def read_uncached(path):
    """Read a file as uint8 and drop it from the page cache. The page cache may use the CMA pool the KPU runtime
    allocates its MMZ buffers from; reading thousands of input files fills it, and the runtime then crashes in
    mmz_allocator::allocate_segment (CmaFree in /proc/meminfo goes to ~0)."""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kmodel")
    ap.add_argument("--input", required=True, help="folder written by coco_prepare.py (index.json, <id>.u8)")
    ap.add_argument("--output", required=True, help="candidate file to write (.npz)")
    ap.add_argument("--detect", choices=sorted(cu.PREP), default=None,
                    help="detector head (default: the model the input folder was prepared for)")
    ap.add_argument("--conf", type=float, default=None,
                    help="keep cells with objectness * best class score >= CONF (default: the model's evaluation "
                         "threshold, util/coco.py PREP; coco_eval.py cannot use a lower one)")
    ap.add_argument("--limit", type=int, default=0, help="first N images only (default: 0 = all)")
    ap.add_argument("--dump-root", default=None, help="folder for runtime dumps, if the runtime writes any "
                                                      "(default: a new folder under <system temp>/nncase-rt-dumps)")
    args = ap.parse_args()

    meta = cu.load_index(args.input)
    model = args.detect or meta["model"]
    if model != meta["model"]:
        print(f"warning: {args.input} was prepared for {meta['model']}, running it as {model}")
    conf = cu.PREP[model]["conf"] if args.conf is None else args.conf
    k = cu.kind(model)
    white = yolo_post.MODELS[model]["white"]

    kpu = nn.Interpreter()
    dump_root = args.dump_root or temp_dump_root(args.kmodel)
    os.makedirs(dump_root, exist_ok=True)
    kpu.set_dump_root(dump_root)
    with open(args.kmodel, "rb") as f:
        kpu.load_model(f.read())
    shape = list(kpu.get_input_shape(0))
    dtype = np.dtype(kpu.get_input_desc(0).dtype)
    images = meta["images"][:args.limit] if args.limit > 0 else meta["images"]
    print(f"{args.kmodel}: input {shape} {dtype}, {kpu.outputs_size} output(s), {model}, conf {conf}, "
          f"{len(images)} images")

    n_out = kpu.outputs_size // 2 if k == "dense" else kpu.outputs_size  # yolo26: box + class map per stride
    ids, shapes = [], [None] * n_out
    rows, index, values = ([[] for _ in range(n_out)] for _ in range(3))
    t0 = time.time()
    t_run = 0.0
    for n, e in enumerate(images):
        x = read_uncached(os.path.join(args.input, e["file"]))
        if x.size != int(np.prod(shape)):
            print(f"error: {e['file']} has {x.size} values, the model input needs {shape}")
            return 1
        x = cu.model_input(x.reshape(shape), dtype, white)
        t = time.time()
        kpu.set_input_tensor(0, nn.RuntimeTensor.from_numpy(x))
        kpu.run()
        outs = cu.group([kpu.get_output_tensor(o).to_numpy() for o in range(kpu.outputs_size)], model)
        t_run += time.time() - t
        ids.append(e["id"])
        for o, out in enumerate(outs):
            shapes[o] = out.shape
            c = cu.cells(out, k)
            keep = cu.candidates(c, k, conf)
            rows[o].append(len(keep))
            index[o].append(keep.astype(np.int32))
            values[o].append(c[keep].astype(np.float16))
        if (n + 1) % 250 == 0 or n + 1 == len(images):
            print(f"  {n + 1}/{len(images)}  {time.time() - t0:.0f} s  (run {1000 * t_run / (n + 1):.1f} ms/image)",
                  flush=True)

    z = {"model": np.array(model), "conf": np.array(conf), "ids": np.array(ids, np.int64)}
    for o in range(n_out):
        z[f"shape{o}"] = np.array(shapes[o], np.int64)
        z[f"rows{o}"] = np.array(rows[o], np.int64)
        z[f"index{o}"] = np.concatenate(index[o])
        z[f"values{o}"] = np.concatenate(values[o])
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(args.output, **z)
    kept = sum(int(np.sum(r)) for r in rows)
    print(f"saved {os.path.abspath(args.output)}: {kept} candidate cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
