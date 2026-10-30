#!/usr/bin/env python3
"""
YOLOX-nano (416x416) kmodel test for the nncaseruntime K230 extension.

Loads a kmodel, feeds it a raw uint8 tensor read from a file, runs one
inference and writes each output tensor to <prefix><index>.txt, one value per
line (row-major order). Runtime dumps (text/rdata/in/out etc., if the runtime was built
with DUMP_K230_INVOCATIONS / ENABLE_DUMP_MANAGER) go to a separate folder
instead of the current directory.

The input file is raw bytes, with no header, laid out exactly as the model
expects (e.g. 1x3x416x416 NCHW or 1x416x416x3 NHWC): its size must equal the
model's input size. The expected shape and size are printed on error.

With --yolox the first output is treated as the raw YOLOX head output
[1, N, 85] = (dx, dy, log w, log h per grid cell, objectness, 80 class
scores), as exported by pth_to_onnx.py, and decoded to boxes;
score = objectness * best class score. Boxes above --score-thr (after NMS)
are drawn on the input image, colored by score, with the score (3 decimals)
above each box. The result is written to <prefix>yolox.rgb as a raw tensor
with exactly the layout, element type and size of the input (e.g. 1x3x416x416).
Input may be u8, u16, f16 or f32 (as the model expects); pixel values are
assumed to be in 0..255 for every type.

Usage:
    python nncaseruntime_test.py model.kmodel input.u8 [--output-prefix output]
                                 [--dump-root rt-dumps] [--yolox]

Example:
    python nncaseruntime_test.py yolox_nano_416.kmodel image_416.u8 --output-prefix out_
    # -> out_0.txt, out_1.txt, ...
"""

import argparse
import os
import sys
import time

import numpy as np
import nncaseruntime as nn

from util import yolox_post
from util.common import save_tensor_text


class ScopedTiming:
    """Simple timer context manager."""

    def __init__(self, info=""):
        self.info = info

    def __enter__(self):
        self.start_time = time.time_ns()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.time_ns() - self.start_time
        print(f"{self.info} : {elapsed / 1e6} ms")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOX-nano kmodel runner")
    parser.add_argument("kmodel", help="path to the .kmodel file")
    parser.add_argument("input", help="raw input tensor file (element type and layout as the model expects, e.g. u8 or f32)")
    parser.add_argument("--output-prefix", default="output",
                        help="output tensors are saved as <prefix>0.txt, "
                             "<prefix>1.txt, ... (default: output)")
    parser.add_argument("--dump-root", default="rt-dumps",
                        help="folder for runtime dumps (default: rt-dumps)")
    yolox_post.add_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"nncaseruntime version: {nn.__version__}")
    print(f"module: {nn.__file__}\n")

    kpu = nn.Interpreter()

    # Keep runtime dumps out of the current directory.
    os.makedirs(args.dump_root, exist_ok=True)
    kpu.set_dump_root(args.dump_root)
    print(f"dump root: {os.path.abspath(args.dump_root)}")

    with open(args.kmodel, "rb") as f:
        kmodel = f.read()
    with ScopedTiming("load_model"):
        kpu.load_model(kmodel)
    print(f"model: {args.kmodel}, {kpu.inputs_size} input(s), "
          f"{kpu.outputs_size} output(s)")

    if kpu.inputs_size != 1:
        print(f"error: expected a single input, model has {kpu.inputs_size}")
        return 1

    # Input: reshape the raw bytes to the shape the model expects.
    # (kpu.get_input_tensor(0) is an empty tensor until one is set, so the
    # shape and dtype come from the model description instead.)
    in_shape = list(kpu.get_input_shape(0))
    in_dtype = np.dtype(kpu.get_input_desc(0).dtype)
    print(f"input[0]: shape={in_shape}, dtype={in_dtype}")

    raw = np.fromfile(args.input, dtype=np.uint8)
    expected = int(np.prod(in_shape)) * in_dtype.itemsize
    if raw.nbytes != expected:
        print(f"error: {args.input} has {raw.nbytes} bytes, "
              f"model input needs {expected} ({in_shape} {in_dtype})")
        return 1
    data = raw.view(in_dtype).reshape(in_shape)

    # Create the tensor from a numpy array and set it (kept in a function
    # scope upstream to release it early; here a single run, so fine inline).
    kpu.set_input_tensor(0, nn.RuntimeTensor.from_numpy(np.ascontiguousarray(data)))

    with ScopedTiming("run"):
        kpu.run()

    # Outputs: print a summary and save each tensor as <prefix><i>.txt.
    out_dir = os.path.dirname(args.output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out0 = None
    for i in range(kpu.outputs_size):
        out = kpu.get_output_tensor(i).to_numpy()
        if i == 0:
            out0 = out
        print(f"output[{i}]: shape={list(out.shape)}, dtype={out.dtype}, "
              f"min={out.min():.6f}, max={out.max():.6f}, "
              f"mean={out.mean():.6f}")
        path = f"{args.output_prefix}{i}.txt"
        save_tensor_text(path, out)
        print(f"  saved {os.path.abspath(path)} ({out.size} values)")

    if args.yolox:
        return yolox_post.draw_yolox(args, data, out0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
