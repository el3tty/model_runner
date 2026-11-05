#!/usr/bin/env python3
"""
Write COCO val2017 images preprocessed for one detector as raw uint8 tensors, for coco_run.py (board or simulator) and
coco_eval.py (PC).

The preprocessing is the one of the model's own evaluation (util/coco.py PREP):
    yolox         416x416, letterbox in the top-left corner, pad 114, BGR, cv2 INTER_LINEAR (YOLOX ValTransform,
                  bit-exact)
    yolov3        416x416, centered letterbox, pad 128, RGB, Pillow bicubic (keras-yolo3 / ONNX model zoo)
    yolo-fastest  320x320, plain resize, RGB, cv2 INTER_LINEAR (darknet, letter_box=0)
Every image becomes <output>/<image id>.u8 (1x3xHxW, 0..255; the 0..1 models get / 255 in coco_run.py and
coco_eval.py), plus <output>/index.json with the original sizes and the box mapping back to the original image.

Usage:
    python coco_prepare.py --detect yolox --output coco-yolox --coco <COCO> (or $COCO_DIR) [--images 5000]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # util/ and nncaseruntime of the repository

import numpy as np

from util import coco as cu



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detect", required=True, choices=sorted(cu.PREP), help="detector whose preprocessing to use")
    ap.add_argument("--output", required=True, help="folder for the <id>.u8 tensors and index.json")
    # the COCO 2017 dataset root (cocodataset.org: val2017/ images, annotations/instances_val2017.json); the
    # ground truth and the images of the prepared inputs
    ap.add_argument("--coco", default=os.environ.get("COCO_DIR"), required="COCO_DIR" not in os.environ,
                    help="COCO root with val2017/ and annotations/ (default: $COCO_DIR)")
    ap.add_argument("--images", type=int, default=0, help="first N val2017 images (default: 0 = all 5000)")
    args = ap.parse_args()

    ann = os.path.join(args.coco, "annotations", "instances_val2017.json")
    with open(ann) as f:
        images = json.load(f)["images"]
    # file order = pycocotools COCO.getImgIds(), the order YOLOX evaluates in: --images N picks the same N images
    if args.images > 0:
        images = images[:args.images]

    os.makedirs(args.output, exist_ok=True)
    index = []
    for n, im in enumerate(images):
        x, geo = cu.preprocess(os.path.join(args.coco, "val2017", im["file_name"]), args.detect)
        name = f"{im['id']}.u8"
        x.tofile(os.path.join(args.output, name))
        index.append({"id": im["id"], "file": name, "width": im["width"], "height": im["height"], **geo})
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(images)}", flush=True)
    w, h = cu.PREP[args.detect]["size"]
    with open(os.path.join(args.output, "index.json"), "w") as f:
        json.dump({"model": args.detect, "size": [w, h], "images": index}, f)
    print(f"wrote {len(index)} {args.detect} inputs (1x3x{h}x{w} u8, {len(index) * 3 * h * w / 2**20:.0f} MB) "
          f"to {os.path.abspath(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
