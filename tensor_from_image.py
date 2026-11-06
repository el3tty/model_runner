#!/usr/bin/env python3
"""Convert an image to a raw NCHW (1x3xHxW, RGB) tensor stored in a .bin file next to the image.

    tensor_from_image.py C:\\myimage.jpg 320x320 [--letterbox [--pad 114] | --crop [--crop-y 0.5]] [--bgr]
                         [--range 255|1] [--type u8|u16|f16|f32] [--output FILE]

The output is written to --output if given, otherwise to the image path with the extension replaced by .bin
(C:\\myimage.bin).
Pixel values keep their 0..255 range in every output type (no normalisation) and are only converted to the requested
element type; --range 1 divides them by 255 (f16/f32 only), for models that take 0..1 input. Data is little-endian and
has no header.

Typical inputs: YOLOX --letterbox --bgr (pad 114, 0..255); YOLOv3 (ONNX model zoo) --letterbox --pad 128 --range 1;
Yolo-Fastest (darknet) --range 1 (plain resize); Lite-Mono --crop.

--crop keeps the shapes in the image and uses every output pixel: the image is scaled up or down as little as it takes
to cover the resolution, and the part that sticks out is cut off. Sideways the crop is centered; --crop-y places it
vertically, from 0 (the top of the image) to 1 (the bottom).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

DTYPES = {"u8": np.uint8, "u16": np.uint16, "f16": np.float16, "f32": np.float32}


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


def crop_box(width, height, out_width, out_height, crop_y):
    """The part of a width x height image with the shape of the output: as large as fits, centered sideways."""
    if out_width * height >= out_height * width:  # the image is too tall: all of its width
        w, h = width, width * out_height / out_width
    else:
        w, h = height * out_width / out_height, height
    x, y = (width - w) / 2, (height - h) * crop_y
    return x, y, x + w, y + h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path, help="input image (bmp, jpeg, png, ...)")
    ap.add_argument("resolution", type=parse_resolution, help="output resolution WIDTHxHEIGHT, e.g. 320x320")
    ap.add_argument("--letterbox", action="store_true",
                    help="keep the aspect ratio and pad (default: stretch to the resolution)")
    ap.add_argument("--pad", type=int, default=114,
                    help="gray value 0..255 of the letterbox padding (default: 114, YOLOX; keras-yolo3 uses 128)")
    ap.add_argument("--crop", action="store_true",
                    help="keep the aspect ratio: scale to cover the resolution and cut off the rest")
    ap.add_argument("--crop-y", type=float, default=None,
                    help="vertical place of the --crop: 0 the top of the image, 1 the bottom (default: 0.5)")
    ap.add_argument("--range", dest="vrange", type=int, choices=(255, 1), default=255,
                    help="pixel value range 0..255 (default) or 0..1 (divided by 255; needs --type f16 or f32)")
    ap.add_argument("--bgr", action="store_true",
                    help="store the channels in BGR order instead of RGB (YOLOX/OpenCV-trained models expect BGR)")
    ap.add_argument("--type", dest="dtype", choices=sorted(DTYPES), default="u8",
                    help="output element type (default: u8)")
    ap.add_argument("--output", type=Path, default=None,
                    help="output file (default: the image path with the extension replaced by .bin)")
    args = ap.parse_args()
    if args.vrange == 1 and args.dtype not in ("f16", "f32"):
        ap.error("--range 1 needs --type f16 or f32")
    if not 0 <= args.pad <= 255:
        ap.error("--pad must be 0..255")
    if args.crop and args.letterbox:
        ap.error("--crop and --letterbox exclude each other")
    if args.crop_y is not None and not args.crop:
        ap.error("--crop-y needs --crop")
    if args.crop_y is not None and not 0 <= args.crop_y <= 1:
        ap.error("--crop-y must be 0..1")

    size = args.resolution  # (width, height)
    img = load_image(args.image)
    if args.letterbox:
        img = ImageOps.pad(img, size, method=Image.Resampling.BILINEAR, color=(args.pad,) * 3, centering=(0.5, 0.5))
    elif args.crop:
        crop_y = 0.5 if args.crop_y is None else args.crop_y
        img = img.resize(size, Image.Resampling.BILINEAR, box=crop_box(*img.size, *size, crop_y))
    else:
        img = img.resize(size, Image.Resampling.BILINEAR)

    hwc = np.asarray(img, dtype=np.uint8)                 # H x W x C
    if args.bgr:
        hwc = hwc[:, :, ::-1]
    nchw = np.ascontiguousarray(hwc.transpose(2, 0, 1)[None])  # 1 x C x H x W
    if args.vrange == 1:
        nchw = nchw.astype(np.float32) / 255.0
    nchw = nchw.astype(DTYPES[args.dtype])

    out = args.output if args.output is not None else args.image.with_suffix(".bin")
    nchw.astype(nchw.dtype.newbyteorder("<")).tofile(out)
    print(f"{out}: shape {tuple(nchw.shape)} (NCHW, {'BGR' if args.bgr else 'RGB'}, 0..{args.vrange}), "
          f"type {args.dtype}, {nchw.nbytes} bytes")


if __name__ == "__main__":
    sys.exit(main())
