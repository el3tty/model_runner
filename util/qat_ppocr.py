#!/usr/bin/env python3
"""The PP-OCR recognizer for QAT (qat_train.py ppocr, qat_export.py ppocr): the character set, greedy CTC decoding,
edit distance, the labeled test set (data/ocr-synth: test/<i>.u8 1x3x32x320 BGR uint8 + labels.json), the
calibration images, the export (float ONNX with the trained weights + nncase quant scheme) and PpocrTask."""
import json
import os

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from util import qat_graph
from util.ocr_synth import normalize
from util.qat_train import Task


def charset(dict_onnx):
    """Classes of the recognizer: blank, the characters of the original model's 'character' metadata, space."""
    import onnx

    meta = {p.key: p.value for p in onnx.load(dict_onnx).metadata_props}
    return ["<blank>"] + [c for c in meta["character"].split("\n") if c] + [" "]


def decode(logits, chars):
    """Greedy CTC: argmax per time step, merge repeats, drop the blank (0)."""
    out, prev = [], 0
    for c in logits.argmax(-1):
        if c != prev and c != 0:
            out.append(chars[c])
        prev = c
    return "".join(out)


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
def evaluate(model, x, texts, chars, batch=100):
    """(exact word accuracy, character error rate) of model on the test set."""
    device = next(model.parameters()).device
    preds = []
    for i in range(0, len(x), batch):
        out = model(torch.from_numpy(x[i:i + batch]).to(device)).float().cpu().numpy()
        preds += [decode(o, chars) for o in out]
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


def trained_initializers(model, onnx_path):
    """{initializer name: trained float32 array} for the ONNX initializers that are parameters of model (matched by
    the values of a freshly converted, untrained model)."""
    fresh = qat_graph.load(onnx_path)
    fresh_params = {k: v.detach().numpy() for k, v in fresh.state_dict().items()}
    trained = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    m = qat_graph.constants_to_initializers(onnx.load(onnx_path))
    out = {}
    for init in m.graph.initializer:
        w = numpy_helper.to_array(init)
        if w.dtype != np.float32 or w.size < 2:
            continue
        for key, v in fresh_params.items():
            transposed = v.ndim == 2 and v.T.shape == w.shape and np.array_equal(v.T, w)
            if not (v.shape == w.shape and np.array_equal(v, w)) and not transposed:
                continue
            head, _, leaf = key.rpartition(".")
            for k in (key, f"{head}.conv.{leaf}", f"{head}.linear.{leaf}"):
                if k in trained:
                    out[init.name] = trained[k].T if transposed else trained[k]
                    break
            break
    return m, out


def channel_ranges(w, count):
    """[(min, max)] of w per channel along the axis of size count (axis 0 first), or of the whole tensor."""
    if count == 1:
        return [(float(w.min()), float(w.max()))]
    axis = next(a for a in range(w.ndim) if w.shape[a] == count)
    flat = np.moveaxis(w, axis, 0).reshape(count, -1)
    return list(zip(flat.min(1).tolist(), flat.max(1).tolist()))


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
        ap.add_argument("--onnx", default="onnx/ppocr-en-rec-32x320.onnx", help="float recognizer (1x3x32x320)")
        ap.add_argument("--dict", default="weights/ppocr/en_number_mobile_v2.0_rec_infer.onnx",
                        help="the original PaddleOCR ONNX (its 'character' metadata is the character set)")
        ap.add_argument("--test", default="data/ocr-synth", help="labeled test set (labels.json, test/*.u8)")
        ap.add_argument("--calib-dir", default="data/ocr-synth/calib", help="calibration images (nncase's --dataset)")
        ap.add_argument("--fonts", default=None, help="directory with .ttf/.otf fonts for the training lines")
        ap.add_argument("--temperature", type=float, default=1.0, help="distillation temperature")
        ap.add_argument("--seed", type=int, default=0, help="seed of the training lines")

    def build(self, args, device, load_float_weights):
        return qat_graph.make_quantizable(qat_graph.load(args.onnx)).to(device)

    def calib_loader(self, args):
        return load_calib(args.calib_dir)

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
            self._test = load_test(args.test), charset(args.dict)
        (x, texts), chars = self._test
        words, cer = evaluate(model, x, texts, chars)
        return {"words": words, "CER": cer}

    def score(self, metrics):
        return metrics["words"], -metrics["CER"]

    def add_export_args(self, ap):
        ap.description = (
            "The float recognizer with the trained weights (same graph, input and output as the training's --onnx; "
            "the weights as float values: nncase quantizes them per output channel between min and max, the "
            "training's fake quantization) and <output>.qs.json, the nncase quant scheme to compile it with "
            "(--quant-scheme): --scheme with the checkpoint's activation ranges and the trained weights' per-channel "
            "ranges. A Q/DQ export does not work for this model: an unrolled LSTM would not match nncase's LSTM.")
        ap.add_argument("--scheme", default=None,
                        help="nncase QuantScheme.json of a PTQ compile of the float model (--export-quant-scheme "
                             "--export-weight-range-by-channel; it names every tensor nncase quantizes); default: "
                             "the training's --scheme")

    def export(self, model, ckpt, args):
        onnx_path = ckpt["args"]["onnx"]
        scheme_path = args.scheme or ckpt["args"].get("scheme")
        if not scheme_path:
            raise SystemExit("--scheme is needed: the checkpoint was not started from a quant scheme")
        m, weights = trained_initializers(model, onnx_path)
        for init in m.graph.initializer:
            if init.name in weights:
                init.CopyFrom(numpy_helper.from_array(weights[init.name].astype(np.float32), init.name))
        onnx.checker.check_model(m)
        onnx.save(m, args.output)
        print(f"wrote {args.output} ({len(weights)} trained initializers)")

        scheme = json.load(open(scheme_path))
        names = qat_graph.match_scheme(qat_graph.make_quantizable(qat_graph.load(onnx_path)), onnx_path, scheme_path)
        acts = {}
        for qname, name in names:
            p = model.get_submodule(qname).params
            acts[name] = (float(p.min_val), float(p.max_val))
        n_w = n_a = 0
        for entry in scheme["Outputs"]:
            if entry["Name"] in weights:
                ranges = channel_ranges(weights[entry["Name"]], len(entry["DataRange"]))
                entry["DataRange"] = [{"Min": lo, "Max": hi, "IsFull": False} for lo, hi in ranges]
                n_w += 1
            elif entry["Name"] in acts:
                lo, hi = acts[entry["Name"]]
                entry["DataRange"] = [{"Min": lo, "Max": hi, "IsFull": False}]
                n_a += 1
        with open(args.output + ".qs.json", "w") as f:
            json.dump(scheme, f, indent=2)
        print(f"wrote {args.output}.qs.json ({n_w} weight and {n_a} activation entries updated, "
              f"{len(scheme['Outputs']) - n_w - n_a} kept)")
