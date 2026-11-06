"""
--postprocess for run-model.py (K230 board / simulator) and run-onnx.py (onnxruntime): what to do with the outputs of
one inference, by model.

  detectors   yolox, yolov3, yolo-fastest, yolo26 (util/yolo_post.py): decode the raw heads to boxes, keep those
              above --score-thr after NMS (--nms-thr), print them, draw them on the input and save
              <prefix><model>.rgb (raw, same layout, channel order, value range and element type as the input)
  recognizers ppocr (util/ocr_post.py): greedy CTC decoding of the output, print the text
  depth       litemono (util/litemono_post.py): save the disparity in the magma color map as <prefix><model>.rgb
              (raw uint8 RGB 1x3xHxW), pale yellow close and black far

numpy only, so it runs on the board.
"""
from util import litemono_post, ocr_post, yolo_post

HANDLERS = {**{name: yolo_post.draw for name in yolo_post.MODELS},
            **{name: lambda args, data, outs: ocr_post.recognize(args, outs) for name in ocr_post.MODELS},
            **{name: lambda args, data, outs: litemono_post.save(args, outs) for name in litemono_post.MODELS}}


def add_arguments(parser):
    """Add --postprocess and its options to an argparse parser."""
    parser.add_argument("--postprocess", choices=sorted(HANDLERS), default=None,
                        help="post-process the outputs for this model: detectors (yolox: BGR 0..255 input, yolov3 / "
                             "yolo-fastest: RGB 0..1, yolo26: RGB 0..255) draw their boxes on the input and save "
                             "<prefix><model>.rgb; ppocr (BGR 1x3x32x320) prints the text it read; litemono saves its depth "
                             "map in colors as <prefix><model>.rgb (uint8), pale yellow close, black far")
    parser.add_argument("--score-thr", type=float, default=0.25,
                        help="detectors: minimum objectness*class score (default 0.25)")
    parser.add_argument("--nms-thr", type=float, default=0.45,
                        help="detectors: NMS IoU threshold (default 0.45)")


def run(args, data, outs):
    """Post-process the outputs `outs` of the input `data` for args.postprocess; returns the exit code."""
    return HANDLERS[args.postprocess](args, data, outs) if args.postprocess else 0
