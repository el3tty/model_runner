"""
Text recognizer post-processing shared by run-model.py (K230 board / simulator) and run-onnx.py (onnxruntime), for
the model selected with --postprocess (util/postprocess.py):

  ppocr   PP-OCR en_number_mobile_v2.0 recognizer (onnx/ppocr-en-rec-32x320.onnx and its QAT export): input
          1x3x32x320 BGR, output [1, 80, 97] CTC logits per time step. The kmodel (--pre-process, --input-range -1 1)
          takes the uint8 image; the ONNX model takes it mapped to -1..1 as (x - 128) / 127.5 (run-onnx.py
          --input-range -1 1 does that from the same uint8 file).

Decoding is greedy CTC: argmax per time step, repeats merged, the blank (class 0) dropped. The confidence is the mean
softmax probability of the kept characters. numpy only, so it runs on the board.
"""
import numpy as np

# blank, the 'character' metadata of en_number_mobile_v2.0_rec_infer.onnx, then the space PaddleOCR appends
PPOCR_CHARS = (["<blank>"] + [chr(c) for c in range(ord("0"), ord("~") + 1)] + list("!\"#$%&'()*+,-./") + [" "] +
               [" "])
MODELS = {"ppocr": PPOCR_CHARS}


def decode(logits, chars=PPOCR_CHARS):
    """Greedy CTC of logits [T, C]: (text, mean probability of the kept characters)."""
    logits = np.asarray(logits, np.float32)
    e = np.exp(logits - logits.max(-1, keepdims=True))
    prob = e / e.sum(-1, keepdims=True)
    best = logits.argmax(-1)
    text, conf, prev = [], [], 0
    for t, c in enumerate(best):
        if c != prev and c != 0:
            text.append(chars[c])
            conf.append(prob[t, c])
        prev = c
    return "".join(text), float(np.mean(conf)) if conf else 0.0


def recognize(args, outs):
    """Print the text the --postprocess recognizer read; returns the exit code."""
    name = args.postprocess
    chars = MODELS[name]
    out = np.asarray(outs[0])
    if out.ndim != 3 or out.shape[0] != 1 or out.shape[2] != len(chars):
        print(f"error: --postprocess {name} needs an output [1, T, {len(chars)}], got {list(out.shape)}")
        return 1
    text, conf = decode(out[0], chars)
    print(f"text: {text!r}  (confidence {conf:.3f})")
    return 0
