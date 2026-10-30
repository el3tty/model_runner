#!/usr/bin/env python3
"""Convert an image to a raw NCHW (1x3xHxW, RGB) tensor stored in a .bin file next to the image.

    tensor_from_image.py C:\\myimage.jpg 320x320 [--letterbox] [--bgr] [--type u8|u16|f16|f32] [--output FILE]

The output is written to --output if given, otherwise to the image path with the extension replaced by .bin
(C:\\myimage.bin).
Pixel values keep their 0..255 range in every output type (no normalisation); they are only converted to the
requested element type. Data is little-endian and has no header.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

DTYPES = {"u8": np.uint8, "u16": np.uint16, "f16": np.float16, "f32": np.float32}
LETTERBOX_PAD = (114, 114, 114)  # gray padding used by letterbox (YOLO convention)


def parse_resolution(text):
    m = re.fullmatch(r"(\d+)[xX](\d+)", text)
    if not m or int(m.group(1)) == 0 or int(m.group(2)) == 0:
        raise argparse.ArgumentTypeError(f"invalid resolution '{text}', expected WIDTHxHEIGHT (e.g. 320x320)")
    return int(m.group(1)), int(m.group(2))


def load_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # honour camera orientation
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))  # flatten transparency onto white
        img = Image.alpha_composite(background, rgba)
    return img.convert("RGB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path, help="input image (bmp, jpeg, png, ...)")
    ap.add_argument("resolution", type=parse_resolution, help="output resolution WIDTHxHEIGHT, e.g. 320x320")
    ap.add_argument("--letterbox", action="store_true",
                    help="keep the aspect ratio and pad (default: stretch to the resolution)")
    ap.add_argument("--bgr", action="store_true",
                    help="store the channels in BGR order instead of RGB (YOLOX/OpenCV-trained models expect BGR)")
    ap.add_argument("--type", dest="dtype", choices=sorted(DTYPES), default="u8",
                    help="output element type (default: u8)")
    ap.add_argument("--output", type=Path, default=None,
                    help="output file (default: the image path with the extension replaced by .bin)")
    args = ap.parse_args()

    size = args.resolution  # (width, height)
    img = load_image(args.image)
    if args.letterbox:
        img = ImageOps.pad(img, size, method=Image.Resampling.BILINEAR, color=LETTERBOX_PAD, centering=(0.5, 0.5))
    else:
        img = img.resize(size, Image.Resampling.BILINEAR)

    hwc = np.asarray(img, dtype=np.uint8)                 # H x W x C
    if args.bgr:
        hwc = hwc[:, :, ::-1]
    nchw = np.ascontiguousarray(hwc.transpose(2, 0, 1)[None]).astype(DTYPES[args.dtype])  # 1 x C x H x W

    out = args.output if args.output is not None else args.image.with_suffix(".bin")
    nchw.astype(nchw.dtype.newbyteorder("<")).tofile(out)
    print(f"{out}: shape {tuple(nchw.shape)} (NCHW, {'BGR' if args.bgr else 'RGB'}), type {args.dtype}, {nchw.nbytes} bytes")


if __name__ == "__main__":
    sys.exit(main())
