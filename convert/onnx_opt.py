#!/usr/bin/env python3
"""
Rewrite an ONNX model into the form the K230 compile path runs best (util/onnx_rewrite.py), and report what is left
that will cost time on the K230:

    python convert/onnx_opt.py build/onnx/lite-mono-tiny-640x192.onnx build/onnx/lite-mono-tiny-640x192-opt.onnx
    python convert/onnx_opt.py build/onnx/some-model.onnx --lint

The model is constant folded first (onnxsim, without its optimizers: they fold the 0.5 of a GELU into the next Conv,
and nncase then fails on the Erf left behind). The rules match only the patterns seen so far; the output says which ones
fired. The result is compared with the input on a random tensor after every rule.

A model from elsewhere may need preparing first. The PP-OCR recognizer as RapidOCR converted it (PP-OCRv1
en_number_mobile_v2.0: CRNN = MobileNetV3 with squeeze-excitation, two bidirectional LSTMs, CTC; 97 classes) has a dynamic
input and ends in a Softmax. Cut before it, made static, with its BatchNorm folded by the optimizers (it has no GELU):

    curl -L -o build/weights/ppocr/en_number_mobile_v2.0_rec_infer.onnx \
        https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv1/en_number_mobile_v2.0_rec_infer.onnx
    python convert/onnx_opt.py build/weights/ppocr/en_number_mobile_v2.0_rec_infer.onnx build/onnx/ppocr-en-rec-32x320.onnx \
        --input-shape 1,3,32,320 --cut ctc_fc.tmp_1 --output-shape 1,80,97 --optimize

--output-shape also gives the Reshape that writes the output a constant shape (PP-OCR's takes it from Shape / Slice /
Concat, which folding alone leaves in the graph).
"""
import argparse
import collections
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, utils

sys.path.append(str(Path(__file__).resolve().parent.parent))  # util/ of the repository

from util import onnx_rewrite  # noqa: E402


def shape_arg(text):
    return [int(v) for v in text.split(",")]


def set_dims(value, shape):
    for d, v in zip(value.type.tensor_type.shape.dim, shape):
        d.ClearField("dim_param")
        d.dim_value = v


def prepare(args):
    """The model of args.source cut, made static and folded."""
    import onnxsim

    if args.cut:
        name = onnx.load(args.source).graph.input[0].name
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cut.onnx")
            utils.extract_model(args.source, path, [name], [args.cut])
            model = onnx.load(path)
    else:
        model = onnx.load(args.source)
    name = model.graph.input[0].name
    options = dict(perform_optimization=args.optimize)
    if args.input_shape:
        options["overwrite_input_shapes"] = {name: args.input_shape}

    model, ok = onnxsim.simplify(model, **options)
    if ok and args.output_shape:
        out = model.graph.output[0].name
        final = next(n for n in model.graph.node if n.op_type == "Reshape" and n.output[0] == out)
        shape = out.replace(".", "_") + "_shape"
        model.graph.initializer.append(numpy_helper.from_array(np.array(args.output_shape, np.int64), shape))
        final.input[1] = shape
        model, ok = onnxsim.simplify(model, **options)
    if not ok:
        raise RuntimeError("onnxsim could not validate the folded model")
    if args.input_shape:
        set_dims(model.graph.input[0], args.input_shape)
    if args.output_shape:
        set_dims(model.graph.output[0], args.output_shape)
    if args.input_shape or args.output_shape:
        del model.graph.value_info[:]
        model = onnx.shape_inference.infer_shapes(model)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("output", nargs="?", help="rewritten model; not needed with --lint")
    ap.add_argument("--lint", action="store_true", help="only report, change nothing")
    ap.add_argument("--input-shape", type=shape_arg, help="make the input static, e.g. 1,3,32,320")
    ap.add_argument("--cut", help="end the model at this tensor")
    ap.add_argument("--output-shape", type=shape_arg, help="constant shape of the output, see above")
    ap.add_argument("--optimize", action="store_true",
                    help="run onnxsim's optimizers too (BatchNorm folding, ...); not for a model with GELU")
    args = ap.parse_args()
    if not args.lint and not args.output:
        ap.error("give an output file, or --lint")

    model = prepare(args)
    if not args.lint:
        model, fired = onnx_rewrite.rewrite(model)
        print("rules fired:", fired or "none")
        onnx.checker.check_model(model)
        onnx.save(model, args.output)
    ops = collections.Counter(n.op_type for n in model.graph.node if n.op_type != "Constant")
    print(f"{sum(ops.values())} nodes:", dict(ops.most_common()))
    for line in onnx_rewrite.lint(model):
        print("lint:", line)


if __name__ == "__main__":
    main()
