#!/usr/bin/env python3
"""The PP-OCR recognizer for QAT (qat_train.py ppocr, qat_export.py ppocr): edit distance, the labeled test set and
the calibration images (a folder written by python -m util.ocr_synth: labels.json, test/<i>.u8 1x3x32x320 BGR uint8,
calib/*.png) and PpocrTask. The character set and the greedy CTC decoding are util/ocr_post.py's."""
import json
import os

import numpy as np
import torch

from util import qat_graph
from util.ocr_post import PPOCR_CHARS, decode
from util.ocr_synth import normalize
from util.qat_train import Task


def edit(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def load_test(synth_dir, limit=0):
    """(normalized float32 [N, 3, 32, 320], texts) of the labeled test set."""
    labels = json.load(open(os.path.join(synth_dir, "labels.json")))
    labels = labels[:limit] if limit else labels
    x = np.stack([normalize(np.fromfile(os.path.join(synth_dir, "test", l["file"]), np.uint8).reshape(3, 32, 320))
                  for l in labels])
    return x, [l["text"] for l in labels]


@torch.no_grad()
def evaluate(model, x, texts, chars=PPOCR_CHARS, batch=100):
    """(exact word accuracy, character error rate) of model on the test set."""
    device = next(model.parameters()).device
    preds = []
    for i in range(0, len(x), batch):
        out = model(torch.from_numpy(x[i:i + batch]).to(device)).float().cpu().numpy()
        preds += [decode(o, chars)[0] for o in out]
    words = sum(p == t for p, t in zip(preds, texts)) / len(texts)
    cer = sum(edit(p, t) for p, t in zip(preds, texts)) / sum(len(t) for t in texts)
    return words, cer


def load_calib(calib_dir, batch=20):
    """DataLoader of the calibration PNGs (320x32, as nncase reads them: BGR, --pre-process dequantized)."""
    from PIL import Image

    xs = [normalize(np.asarray(Image.open(os.path.join(calib_dir, f)).convert("RGB"))[:, :, ::-1].transpose(2, 0, 1))
          for f in sorted(os.listdir(calib_dir)) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(np.stack(xs))),
                                       batch_size=batch)


def distill_loss(student_logits, teacher_logits, temperature=1.0):
    """KL(teacher || student) per time step, averaged over steps and lines."""
    import torch.nn.functional as F

    c = student_logits.shape[-1]
    s = F.log_softmax(student_logits.reshape(-1, c) / temperature, -1)
    t = F.softmax(teacher_logits.reshape(-1, c) / temperature, -1)
    return F.kl_div(s, t, reduction="batchmean") * temperature ** 2


class PpocrTask(Task):
    """The PP-OCR recognizer, distilled from its float model on rendered text lines."""

    name = "ppocr"
    description = ("PP-OCR recognizer (ONNX -> torch, any graph qat_graph.py handles): distillation from the float "
                   "model (KL of the per-time-step class distributions, no labels) on text lines rendered on the fly "
                   "(util/ocr_synth.py), Adam, words / CER on a labeled test set")
    default_lr = 1e-4
    default_batch_size = 128
    default_epochs = 20
    default_iters_per_epoch = 1000
    default_clip_grad = 5.0
    supports_scheme = True

    def add_args(self, ap):
        # the float recognizer, made by convert/onnx_opt.py (input x 1x3x32x320, output the CTC logits [1, 80, 97]):
        # the model the fake quantizers go into, and the teacher of the distillation
        ap.add_argument("--onnx", required=True, help="float recognizer, e.g. build/onnx/ppocr-en-rec-32x320.onnx")
        # written by python -m util.ocr_synth <dir>: test/*.u8 + labels.json (the words / CER evaluation), calib/*.png
        # (the activation range calibration, the images nncase's --dataset would get)
        ap.add_argument("--data", required=True, help="text line folder of util.ocr_synth, e.g. build/data/ocr-synth")
        ap.add_argument("--fonts", default=None, help="directory with .ttf/.otf fonts for the training lines")
        ap.add_argument("--temperature", type=float, default=1.0, help="distillation temperature")
        ap.add_argument("--seed", type=int, default=0, help="seed of the training lines")

    def build(self, args, device, load_float_weights):
        return qat_graph.make_quantizable(qat_graph.load(args.onnx)).to(device)

    def calib_loader(self, args):
        return load_calib(os.path.join(args.data, "calib"))

    def load_scheme(self, model, args):
        return qat_graph.load_scheme_ranges(model, args.onnx, args.scheme)

    def optimizer(self, params, lr, args):
        return torch.optim.Adam(params, lr=lr)

    def iters_per_epoch(self, args):
        return 1000

    def _teacher(self, args, device):
        if not hasattr(self, "_teacher_model"):
            self._teacher_model = qat_graph.load(args.onnx).to(device).eval()
            for p in self._teacher_model.parameters():
                p.requires_grad_(False)
        return self._teacher_model

    def batches(self, model, args, epoch, iters):
        from util.ocr_synth import OcrLines, find_fonts

        if not hasattr(self, "_lines"):
            self._lines = OcrLines(iters * args.batch_size, find_fonts(args.fonts), args.seed)
        self._lines.epoch = epoch
        loader = torch.utils.data.DataLoader(self._lines, batch_size=args.batch_size, num_workers=args.workers,
                                             pin_memory=True, drop_last=True)
        device = next(model.parameters()).device
        for x, _ in loader:
            yield x.to(device, non_blocking=True)

    def loss(self, model, batch, args):
        with torch.no_grad():
            t_logits = self._teacher(args, batch.device)(batch)
        loss = distill_loss(model(batch), t_logits, args.temperature)
        return loss, {"loss": loss.item()}

    def evaluate(self, model, args):
        if not hasattr(self, "_test"):
            self._test = load_test(args.data)
        x, texts = self._test
        words, cer = evaluate(model, x, texts)
        return {"words": words, "CER": cer}

    def score(self, metrics):
        return metrics["words"], -metrics["CER"]

    export_input_names = ("x",)
    export_output_names = ("ctc_fc.tmp_1",)  # the names of the original model

    def export_input_shape(self, args):
        return (1, 3, 32, 320)

    def add_export_args(self, ap):
        ap.description = (
            "ONNX of the fake-quant recognizer: input x 1x3x32x320 float32 (as the K230 pre-process maps the uint8 "
            "image: (x - 128) / 127.5), output [1, 80, 97]. Each LSTM is one ONNX LSTM node with its float weights "
            "(nncase quantizes them per row, as the training did) and a Q/DQ pair on its output with the trained "
            "hidden-state range.")
        super().add_export_args(ap)
