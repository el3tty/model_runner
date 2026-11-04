#!/usr/bin/env python3
"""
Make the PP-OCR recognizer ONNX the K230 scripts use from the PaddleOCR English/number recognizer (PP-OCRv1
en_number_mobile_v2.0: CRNN = MobileNetV3 with squeeze-excitation, two bidirectional LSTMs, CTC; 97 classes), as
converted to ONNX by RapidOCR:

    curl -L -o weights/ppocr/en_number_mobile_v2.0_rec_infer.onnx \
        https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv1/en_number_mobile_v2.0_rec_infer.onnx
    python ppocr_to_onnx.py weights/ppocr/en_number_mobile_v2.0_rec_infer.onnx onnx/ppocr-en-rec-32x320.onnx

The result: input x [1, 3, 32, 320] (BGR, normalized to -1..1), output ctc_fc.tmp_1 [1, 80, 97] = the CTC logits per
time step (the final Softmax is cut off: argmax and the distillation work on logits, and nncase does not need it).
The input size is static (nncase needs it; onnxsim's simplification would bring the dynamic batch back, so the input
is fixed again after it). The character set is the model's 'character' metadata (util/ocr_post.py has it built in).
"""
import argparse
import collections

import numpy as np
import onnx
from onnx import utils

SHAPE = (1, 3, 32, 320)
OUT_SHAPE = (1, 80, 97)  # 80 time steps (320 / 4), 97 classes


def set_static_input(m):
    dims = m.graph.input[0].type.tensor_type.shape.dim
    for d, v in zip(dims, SHAPE):
        d.ClearField("dim_param")
        d.dim_value = v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="en_number_mobile_v2.0_rec_infer.onnx (RapidOCR's ONNX of the PaddleOCR model)")
    ap.add_argument("output", help="output .onnx, e.g. onnx/ppocr-en-rec-32x320.onnx")
    args = ap.parse_args()

    import onnxsim
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmp:
        cut = os.path.join(tmp, "cut.onnx")
        utils.extract_model(args.source, cut, ["x"], ["ctc_fc.tmp_1"])  # without the final Softmax
        m = onnx.load(cut)
    m, ok = onnxsim.simplify(m, overwrite_input_shapes={"x": list(SHAPE)})  # folds the shape computations
    if not ok:
        raise SystemExit("onnxsim could not validate the simplified model")
    # the last Reshape ([80, 97] of the Gemm -> [1, 80, 97]) still takes its shape from Shape / Slice / Concat:
    # give it the constant shape, the second simplification drops the unused shape ops
    final = next(n for n in m.graph.node if n.op_type == "Reshape" and n.output[0] == "ctc_fc.tmp_1")
    m.graph.initializer.append(onnx.numpy_helper.from_array(np.array(OUT_SHAPE, np.int64), "ctc_fc_out_shape"))
    final.input[1] = "ctc_fc_out_shape"
    m, ok = onnxsim.simplify(m, overwrite_input_shapes={"x": list(SHAPE)})
    if not ok:
        raise SystemExit("onnxsim could not validate the simplified model")
    set_static_input(m)
    out_dims = m.graph.output[0].type.tensor_type.shape.dim
    for d, v in zip(out_dims, OUT_SHAPE):
        d.ClearField("dim_param")
        d.dim_value = v
    del m.graph.value_info[:]
    m = onnx.shape_inference.infer_shapes(m)
    onnx.checker.check_model(m)
    onnx.save(m, args.output)

    out = m.graph.output[0]
    print(f"{args.output}: input {m.graph.input[0].name} {list(SHAPE)}, output {out.name} "
          f"{[d.dim_value for d in out.type.tensor_type.shape.dim]}")
    print("ops:", dict(collections.Counter(n.op_type for n in m.graph.node)))


if __name__ == "__main__":
    main()
