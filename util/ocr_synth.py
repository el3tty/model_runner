#!/usr/bin/env python3
"""Synthetic text lines for the PP-OCR English recognizer (32x320), rendered with PIL.

    python -m util.ocr_synth <out_dir> [--test 1000] [--calib 100] [--fonts DIR]

writes <out>/calib/*.png (already 320x32, so nncase's resize is the identity), <out>/test/<i>.u8 (1x3x32x320 BGR uint8)
and <out>/labels.json. build/data/ocr-synth was made this way with the Windows fonts. OcrLines renders lines on the fly
for training (ppocr_qat_train.py), so no data set has to be stored or uploaded.

A line: a random word, number, date, price or symbol-joined pair (word()), one random font and size, a grey background
and darker text, sometimes a blur, Gaussian noise; resized like PaddleOCR (height 32, width by aspect ratio up to 320,
right-padded with 128) and stored as BGR.

Fonts: --fonts DIR (every .ttf / .otf in it), else the Windows fonts listed in WINDOWS_FONTS if they are there, else
the DejaVu / Liberation fonts of a Linux system (apt install fonts-dejavu-core fonts-liberation).
"""
import argparse
import functools
import glob
import json
import os
import random
import string

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

WINDOWS_FONTS = ["arial.ttf", "arialbd.ttf", "ariali.ttf", "calibri.ttf", "calibrib.ttf", "consola.ttf", "cour.ttf",
                 "courbd.ttf", "times.ttf", "timesbd.ttf", "verdana.ttf", "georgia.ttf", "tahoma.ttf", "segoeui.ttf"]
LINUX_FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts")]
H, W = 32, 320


def find_fonts(font_dir=None):
    if font_dir:
        fonts = sorted(glob.glob(os.path.join(font_dir, "**", "*.[to]tf"), recursive=True))
    else:
        win = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")
        fonts = [os.path.join(win, f) for f in WINDOWS_FONTS if os.path.exists(os.path.join(win, f))]
        if not fonts:
            fonts = sorted(f for d in LINUX_FONT_DIRS for f in glob.glob(os.path.join(d, "**", "*.ttf"), recursive=True)
                           if os.path.basename(f).startswith(("DejaVu", "Liberation")))
    if not fonts:
        raise SystemExit("no fonts found: pass --fonts DIR (or apt install fonts-dejavu-core fonts-liberation)")
    return fonts


def word(rng):
    kind = rng.random()
    if kind < 0.35:
        return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 10)))
    if kind < 0.55:
        w = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 9)))
        return w.capitalize() if rng.random() < 0.7 else w.upper()
    if kind < 0.7:
        return str(rng.randint(0, 10 ** rng.randint(1, 7)))
    if kind < 0.8:
        return f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(1990, 2030)}"
    if kind < 0.9:
        return f"${rng.randint(1, 9999)}.{rng.randint(0, 99):02d}"
    parts = ["".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(2, 6))) for _ in range(2)]
    return rng.choice(["-", "_", "/", ":", "@", "#"]).join(parts)


@functools.lru_cache(maxsize=512)
def _font(path, size):
    return ImageFont.truetype(path, size)


def render(text, rng, fonts):
    font = _font(rng.choice(fonts), rng.randint(28, 48))
    x0, y0, x1, y1 = font.getbbox(text)
    pad = rng.randint(2, 8)
    bg = rng.randint(150, 255)
    fg = rng.randint(0, max(0, bg - 100))
    img = Image.new("L", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), bg)
    ImageDraw.Draw(img).text((pad - x0, pad - y0), text, font=font, fill=fg)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    noise = np.random.default_rng(rng.randint(0, 2 ** 31)).normal(0, rng.uniform(0, 8), img.size[::-1])
    return Image.fromarray(np.clip(np.asarray(img, np.float32) + noise, 0, 255).astype(np.uint8)).convert("RGB")


def to_tensor(img):
    """HxWx3 BGR uint8, PaddleOCR's resize"""
    w = min(W, max(1, int(round(img.width * H / img.height))))
    a = np.full((H, W, 3), 128, np.uint8)
    a[:, :w] = np.asarray(img.resize((w, H), Image.BILINEAR))[:, :, ::-1]  # RGB -> BGR, like cv2
    return a


def normalize(u8):
    """uint8 BGR CHW -> the recognizer's float input as the K230 computes it: a kmodel compiled with --pre-process
    --input-type Uint8 --input-range -1 1 dequantizes with the integer zero point 128, (x - 128) / 127.5, half a step
    below PaddleOCR's (x / 255 - 0.5) / 0.5 (0.001 words on the float model; but nncase's KLD ranges move with it)."""
    return (u8.astype(np.float32) - 128.0) / 127.5


class OcrLines:
    """torch-style Dataset of n random lines: item i = (normalized 3x32x320 float32 array, text). Line i of epoch e is
    always the same (seeded by seed, e, i), whichever DataLoader worker renders it."""

    def __init__(self, n, fonts, seed=0):
        self.n = n
        self.fonts = list(fonts)
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random((self.seed * 1_000_003 + self.epoch) * 1_000_000_007 + i)
        text = word(rng)
        return normalize(to_tensor(render(text, rng, self.fonts)).transpose(2, 0, 1)), text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--calib", type=int, default=100)
    ap.add_argument("--fonts", default=None, help="directory with the .ttf / .otf fonts to use")
    args = ap.parse_args()
    fonts = find_fonts(args.fonts)
    rng = random.Random(0)
    os.makedirs(os.path.join(args.out, "calib"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "test"), exist_ok=True)
    for i in range(args.calib):
        Image.fromarray(to_tensor(render(word(rng), rng, fonts))[:, :, ::-1]).save(
            os.path.join(args.out, "calib", f"{i:04d}.png"))
    labels = []
    for i in range(args.test):
        text = word(rng)
        to_tensor(render(text, rng, fonts)).transpose(2, 0, 1).tofile(os.path.join(args.out, "test", f"{i:04d}.u8"))
        labels.append({"file": f"{i:04d}.u8", "text": text})
    with open(os.path.join(args.out, "labels.json"), "w") as f:
        json.dump(labels, f, indent=0)
    print(f"{args.calib} calibration PNGs, {args.test} test tensors in {args.out} ({len(fonts)} fonts)")


if __name__ == "__main__":
    main()
