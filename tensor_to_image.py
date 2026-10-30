#!/usr/bin/env python3
"""Convert a raw NCHW (1x3xHxW, RGB) tensor file back to an image (png, jpg or bmp).

    tensor_to_image.py test.rgb 416x416 f32 --format bmp --output test.bmp

This is the reverse of tensor_from_image.py: the file has no header, is little-endian and holds
1 x 3 x H x W values (RGB) of the given type. Pixel values are expected in the 0..255 range for every type
(no normalisation); they are rounded and clipped to u8.

Arguments:
    tensor       raw tensor file
    resolution   WIDTHxHEIGHT, e.g. 416x416
    type         element type: u8, u16, f16 or f32 (default: u8)

Options:
    --bgr        the tensor holds BGR channels (as made by tensor_from_image.py --bgr)
    --layout     nchw (default, 1x3xHxW) or nhwc (1xHxWx3)
    --format     png, jpg or bmp. Default: taken from the --output extension, otherwise png
    --output     output file. Default: the tensor path with the extension replaced by the format's extension
    --quality    JPEG quality 1..100 (default 95)
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

DTYPES = {"u8": np.uint8, "u16": np.uint16, "f16": np.float16, "f32": np.float32}
FORMATS = {"png": "PNG", "jpg": "JPEG", "jpeg": "JPEG", "bmp": "BMP"}
EXT = {"png": ".png", "jpg": ".jpg", "jpeg": ".jpg", "bmp": ".bmp"}


def parse_resolution(text):
    m = re.fullmatch(r"(\d+)[xX](\d+)", text)
    if not m or int(m.group(1)) == 0 or int(m.group(2)) == 0:
        raise argparse.ArgumentTypeError(f"invalid resolution '{text}', expected WIDTHxHEIGHT (e.g. 416x416)")
    return int(m.group(1)), int(m.group(2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tensor", type=Path, help="raw tensor file")
    ap.add_argument("resolution", type=parse_resolution, help="WIDTHxHEIGHT, e.g. 416x416")
    ap.add_argument("dtype", nargs="?", default="u8", choices=sorted(DTYPES), help="element type (default: u8)")
    ap.add_argument("--layout", choices=("nchw", "nhwc"), default="nchw", help="tensor layout (default: nchw)")
    ap.add_argument("--bgr", action="store_true", help="the tensor channels are in BGR order (swap to RGB for the image)")
    ap.add_argument("--format", dest="fmt", choices=sorted(FORMATS), default=None,
                    help="image format (default: from the --output extension, else png)")
    ap.add_argument("--output", type=Path, default=None, help="output file")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")
    args = ap.parse_args()

    fmt = args.fmt
    if fmt is None:
        ext = args.output.suffix.lower().lstrip(".") if args.output else ""
        fmt = ext if ext in FORMATS else "png"
    out = args.output or args.tensor.with_suffix(EXT[fmt])

    w, h = args.resolution
    dtype = DTYPES[args.dtype]
    expected = 3 * h * w * np.dtype(dtype).itemsize
    size = args.tensor.stat().st_size
    if size != expected:
        sys.exit(f"error: {args.tensor} has {size} bytes, but 1x3x{h}x{w} {args.dtype} needs {expected}"
                 f" (is the resolution or type wrong?)")

    data = np.fromfile(args.tensor, dtype=np.dtype(dtype).newbyteorder("<"))
    hwc = data.reshape(h, w, 3) if args.layout == "nhwc" else data.reshape(3, h, w).transpose(1, 2, 0)
    if hwc.dtype.kind == "f" and not np.isfinite(hwc).all():
        print("warning: tensor contains NaN/Inf, replaced by 0")
        hwc = np.nan_to_num(hwc, nan=0.0, posinf=255.0, neginf=0.0)
    if args.bgr:
        hwc = hwc[:, :, ::-1]
    lo, hi = float(hwc.min()), float(hwc.max())
    img = Image.fromarray(np.clip(np.rint(hwc.astype(np.float32)), 0, 255).astype(np.uint8), "RGB")

    kw = {"quality": args.quality, "subsampling": 0} if FORMATS[fmt] == "JPEG" else {}
    img.save(out, FORMATS[fmt], **kw)
    print(f"{args.tensor} ({args.dtype}, value range {lo:g}..{hi:g}) -> {out} ({fmt}, {w}x{h})")
    if hi <= 1.0 and args.dtype in ("f16", "f32"):
        print("warning: all values are <= 1.0; if the tensor is normalised to 0..1 the image will look black")


if __name__ == "__main__":
    main()
