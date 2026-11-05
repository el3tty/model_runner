#!/usr/bin/env python3
"""A real-text test set for the PP-OCR recognizer: the text lines of the FUNSD test forms, in the folder format of
util.ocr_synth, so the same evaluation reads it.

    python -m util.funsd data/funsd/dataset.zip data/funsd/lines

dataset.zip comes from https://guillaumejaume.github.io/FUNSD/ (non-commercial research and education only, the images
stay under copyright: keep it and the crops out of the repository). It writes <out>/test/<i>.u8 (1x3x32x320 BGR uint8)
and <out>/labels.json, whose items also have the form, the FUNSD label (question, answer, header, other) and the box.
A FUNSD entity can span several lines: the lines are rebuilt from its word boxes. No test line trains or selects
anything. FUNSD does not mark handwriting; most of it is in the answers, hence the label.
"""
import argparse
import collections
import io
import json
import os
import zipfile

from PIL import Image

from util.ocr_post import PPOCR_CHARS
from util.ocr_synth import H, W, to_tensor

TEST = "dataset/testing_data/"
MARGIN = 2  # pixels around the word boxes, the annotations are tight


def same_line(a, b):
    """Is word box b the continuation of word box a: to its right, the heights overlapping by half the smaller one."""
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    return b[0] >= a[0] and overlap > 0.5 * min(a[3] - a[1], b[3] - b[1])


def split_lines(entity):
    """[(text, box)] of a FUNSD entity, one item per visual line."""
    lines = []
    for w in entity["words"]:
        if not w["text"].strip():
            continue
        if lines and same_line(lines[-1][-1]["box"], w["box"]):
            lines[-1].append(w)
        else:
            lines.append([w])
    return [(" ".join(w["text"].strip() for w in line),
             [min(w["box"][0] for w in line), min(w["box"][1] for w in line),
              max(w["box"][2] for w in line), max(w["box"][3] for w in line)]) for line in lines]


def reject(text, box):
    """Why a line is not usable, or None."""
    width, height = box[2] - box[0], box[3] - box[1]
    if width <= 0 or height <= 0:
        return "empty box"
    if len(text) < 2 or not any(c.isalnum() for c in text):
        return "no word"
    if any(c not in PPOCR_CHARS for c in text):
        return "character outside the model's set"
    if height > width:
        return "vertical"
    if (width + 2 * MARGIN) * H / (height + 2 * MARGIN) > W:
        return f"wider than {W} at height {H}"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip")
    ap.add_argument("out")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "test"), exist_ok=True)
    labels, dropped = [], collections.Counter()
    with zipfile.ZipFile(args.zip) as z:
        forms = sorted(n for n in z.namelist() if n.startswith(TEST + "annotations/") and n.endswith(".json"))
        for name in forms:
            form = os.path.basename(name)[:-5]
            page = Image.open(io.BytesIO(z.read(f"{TEST}images/{form}.png"))).convert("RGB")
            for entity in json.loads(z.read(name))["form"]:
                for text, box in split_lines(entity):
                    why = reject(text, box)
                    if why:
                        dropped[why] += 1
                        continue
                    crop = page.crop((max(box[0] - MARGIN, 0), max(box[1] - MARGIN, 0),
                                      min(box[2] + MARGIN, page.width), min(box[3] + MARGIN, page.height)))
                    file = f"{len(labels):05d}.u8"
                    to_tensor(crop).transpose(2, 0, 1).tofile(os.path.join(args.out, "test", file))
                    labels.append({"file": file, "text": text, "form": form, "label": entity["label"], "box": box})
    json.dump(labels, open(os.path.join(args.out, "labels.json"), "w"), indent=0)
    print(f"{len(labels)} lines of {len(forms)} forms in {args.out}; dropped {sum(dropped.values())}: {dict(dropped)}")
    print(dict(collections.Counter(l["label"] for l in labels)))


if __name__ == "__main__":
    main()
