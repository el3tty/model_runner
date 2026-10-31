#!/usr/bin/env python3
"""
Run an ONNX model (onnxruntime, CPU) on a raw input tensor file - the float reference for run-model.py.

The input file is raw little-endian bytes without a header, in the element type and layout of the model input
(e.g. 1x3x416x416 f32, as written by tensor_from_image.py --type f32). Every output tensor is written to
<prefix><index>.txt, one value per line (same format as run-model.py).

With --yolox the first output is treated as the raw YOLOX head output [1, N, 85] and the boxes are drawn on the input
exactly like in run-model.py (same options); the result is saved as <prefix>yolox.rgb, a raw tensor with the layout
and element type of the input, which tensor_to_image.py turns into an image.

Usage:
    python run-onnx.py model.onnx input.in [--output-prefix output]
                       [--yolox [--score-thr 0.25] [--nms-thr 0.45]]

Example:
    python tensor_from_image.py test.jpg 416x416 --bgr --type f32 --output yolox-f32-bgr.in
    python run-onnx.py yolox_nano.onnx yolox-f32-bgr.in --yolox
    python tensor_to_image.py outputyolox.rgb 416x416 f32 --bgr --output result.png
"""
import argparse
import os
import sys

import numpy as np

from util import yolox_post
from util.common import save_tensor_text

ORT_TYPES = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(uint8)": np.uint8,
             "tensor(uint16)": np.uint16, "tensor(int8)": np.int8, "tensor(int32)": np.int32,
             "tensor(int64)": np.int64, "tensor(double)": np.float64}


def parse_args():
    ap = argparse.ArgumentParser(description="ONNX model runner (float reference for run-model.py)")
    ap.add_argument("model", help="path to the .onnx file")
    ap.add_argument("input", help="raw input tensor file (element type and layout as the model expects)")
    ap.add_argument("--output-prefix", default="output",
                    help="output tensors are saved as <prefix>0.txt, ...; --yolox writes <prefix>yolox.rgb "
                         "(default: output)")
    yolox_post.add_arguments(ap)
    return ap.parse_args()


def main():
    args = parse_args()
    try:
        import onnxruntime as ort
    except ImportError:
        print("error: onnxruntime is not installed (pip install onnxruntime)")
        return 1

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    if len(inputs) != 1:
        print(f"error: expected a single model input, {args.model} has {len(inputs)}")
        return 1
    inp = inputs[0]
    dtype = np.dtype(ORT_TYPES.get(inp.type, np.float32))
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    print(f"model: {args.model}\ninput[0]: {inp.name}, shape={shape}, dtype={dtype}")

    expected = int(np.prod(shape)) * dtype.itemsize
    size = os.path.getsize(args.input)
    if size != expected:
        print(f"error: {args.input} has {size} bytes, model input needs {expected} ({shape} {dtype})")
        return 1
    data = np.fromfile(args.input, dtype=dtype.newbyteorder("<")).astype(dtype).reshape(shape)

    outs = sess.run(None, {inp.name: data})
    out_dir = os.path.dirname(args.output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for i, out in enumerate(outs):
        print(f"output[{i}]: shape={list(out.shape)}, dtype={out.dtype}, "
              f"min={out.min():.6f}, max={out.max():.6f}, mean={out.mean():.6f}")
        path = f"{args.output_prefix}{i}.txt"
        save_tensor_text(path, out)
        print(f"  saved {os.path.abspath(path)} ({out.size} values)")

    if args.yolox:
        return yolox_post.draw_yolox(args, data, outs[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
