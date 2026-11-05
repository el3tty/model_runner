#!/usr/bin/env python3
"""
Run an ONNX model (onnxruntime, CPU) on a raw input tensor file - the float reference for run-model.py.

The input file is raw little-endian bytes without a header, in the element type and layout of the model input
(e.g. 1x3x416x416 f32, as written by tensor_from_image.py --type f32). A summary of every output tensor is printed;
with --dump-outputs each one is also written to <prefix><index>.txt, one value per line (same format as run-model.py).

--postprocess <model> post-processes the outputs exactly like run-model.py (util/postprocess.py): for a detector
(yolox, yolov3, yolo-fastest, yolo26) the boxes are drawn on the input and saved as <prefix><model>.rgb, a raw tensor
with the layout and element type of the input, which tensor_to_image.py turns into an image; for ppocr the text is
printed.

--input-range LO HI takes a uint8 input file and maps it to LO..HI the way nncase's --pre-process does
(x - zero_point) * scale with scale = (HI - LO) / 255 and an integer zero point, so the ONNX model runs on the same
file as the kmodel compiled with --pre-process true --input-type Uint8 --input-range LO HI.

onnxruntime's graph optimizations are off: they replace QuantizeLinear/DequantizeLinear + Conv with integer kernels
that round differently, so a QAT export would not give its reference result.

Usage:
    python run-onnx.py model.onnx input.in [--output-prefix output] [--dump-outputs] [--input-range LO HI]
                       [--postprocess yolox|yolov3|yolo-fastest|yolo26|ppocr [--score-thr 0.25] [--nms-thr 0.45]]

Examples:
    python tensor_from_image.py test.jpg 416x416 --bgr --type f32 --output yolox-f32-bgr.in
    python run-onnx.py yolox_nano.onnx yolox-f32-bgr.in --postprocess yolox
    python tensor_to_image.py outputyolox.rgb 416x416 f32 --bgr --output result.png

    python tensor_from_image.py test.jpg 416x416 --letterbox --pad 128 --range 1 --type f32 --output yolov3.in
    python run-onnx.py yolov3-12-int8-heads.onnx yolov3.in --postprocess yolov3
    python tensor_to_image.py outputyolov3.rgb 416x416 f32 --range 1 --output result.png

    python tensor_from_image.py test.jpg 320x320 --range 1 --type f32 --output fastest.in
    python run-onnx.py yolo-fastest-1.1-320_int8_qop.onnx fastest.in --postprocess yolo-fastest

    python run-onnx.py ppocr-en-rec-qat.onnx build/data/ocr-synth/test/0000.u8 --input-range -1 1 --postprocess ppocr
"""
import argparse
import os
import sys

import numpy as np

from util import postprocess
from util.common import save_tensor_text

ORT_TYPES = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(uint8)": np.uint8,
             "tensor(uint16)": np.uint16, "tensor(int8)": np.int8, "tensor(int32)": np.int32,
             "tensor(int64)": np.int64, "tensor(double)": np.float64}


def parse_args():
    ap = argparse.ArgumentParser(description="ONNX model runner (float reference for run-model.py)")
    ap.add_argument("model", help="path to the .onnx file")
    ap.add_argument("input", help="raw input tensor file (element type and layout as the model expects)")
    ap.add_argument("--output-prefix", default="output",
                    help="prefix of the files written: <prefix>0.txt, ... with --dump-outputs, <prefix><model>.rgb "
                         "with --postprocess for a detector (default: output)")
    ap.add_argument("--dump-outputs", action="store_true",
                    help="save each output tensor as text, <prefix><index>.txt, one value per line")
    ap.add_argument("--input-range", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                    help="the input file is uint8; map it to LO..HI as nncase's --pre-process does (e.g. -1 1 for "
                         "PP-OCR), for the float model input")
    postprocess.add_arguments(ap)
    return ap.parse_args()


def map_input_range(u8, lo, hi):
    """uint8 -> float32 in lo..hi like nncase's pre-process Dequantize (integer zero point: -1..1 is
    (x - 128) / 127.5)."""
    scale = (hi - lo) / 255.0
    zero_point = np.rint(-lo / scale)
    return ((u8.astype(np.float32) - zero_point) * scale).astype(np.float32)


def main():
    args = parse_args()
    try:
        import onnxruntime as ort
    except ImportError:
        print("error: onnxruntime is not installed (pip install onnxruntime)")
        return 1

    # no graph optimizations: they turn Q/DQ + Conv into integer kernels that round differently from the Q/DQ
    # reference (and from nncase); a float model gives the same result either way
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(args.model, opts, providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    if len(inputs) != 1:
        print(f"error: expected a single model input, {args.model} has {len(inputs)}")
        return 1
    inp = inputs[0]
    dtype = np.dtype(ORT_TYPES.get(inp.type, np.float32))
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    print(f"model: {args.model}\ninput[0]: {inp.name}, shape={shape}, dtype={dtype}")

    file_dtype = np.dtype(np.uint8) if args.input_range else dtype
    expected = int(np.prod(shape)) * file_dtype.itemsize
    size = os.path.getsize(args.input)
    if size != expected:
        print(f"error: {args.input} has {size} bytes, model input needs {expected} ({shape} {file_dtype})")
        return 1
    data = np.fromfile(args.input, dtype=file_dtype.newbyteorder("<")).astype(file_dtype).reshape(shape)
    if args.input_range:
        data = map_input_range(data, *args.input_range).astype(dtype)

    outs = sess.run(None, {inp.name: data})
    out_dir = os.path.dirname(args.output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for i, out in enumerate(outs):
        print(f"output[{i}]: shape={list(out.shape)}, dtype={out.dtype}, "
              f"min={out.min():.6f}, max={out.max():.6f}, mean={out.mean():.6f}")
        if args.dump_outputs:
            path = f"{args.output_prefix}{i}.txt"
            save_tensor_text(path, out)
            print(f"  saved {os.path.abspath(path)} ({out.size} values)")

    return postprocess.run(args, data, outs)


if __name__ == "__main__":
    sys.exit(main())
