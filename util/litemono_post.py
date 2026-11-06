"""
--postprocess litemono: the disparity map of Lite-Mono as a picture in the magma color map, pale yellow close, through
orange and purple, to black far.

Disparity is 0..1 and works like inverse distance, so it needs no conversion. The scale is the same for every image
and every model, so equal colors mean equal outputs: the float model and a kmodel can be compared, and the near limit
of a kmodel shows. The distances are relative, without a unit (see "Lite-Mono" in README.md).

numpy only, so it runs on the board.
"""
import os

import numpy as np

MODELS = ("litemono",)

# matplotlib's magma at 17 even steps; in between, linear interpolation stays within 3 of 255 of the full table
MAGMA = np.array([[0, 0, 4], [10, 8, 34], [29, 17, 71], [54, 16, 107], [81, 18, 124], [106, 28, 129], [131, 38, 129],
                  [156, 46, 127], [183, 55, 121], [207, 64, 112], [229, 80, 100], [244, 105, 92], [251, 135, 97],
                  [254, 165, 113], [254, 194, 135], [253, 224, 161], [252, 253, 191]], np.float32)


def to_magma(disp):
    """disp: HxW, 0..1 -> uint8 RGB 1x3xHxW."""
    steps = np.linspace(0, 1, len(MAGMA))
    x = np.clip(disp.astype(np.float32), 0, 1)
    return np.ascontiguousarray(np.rint([[np.interp(x, steps, MAGMA[:, c]) for c in range(3)]]).astype(np.uint8))


def save(args, outs):
    """Save the --postprocess output as <prefix>litemono.rgb; returns the exit code."""
    name = args.postprocess
    out = np.asarray(outs[0])
    if len(outs) != 1 or out.ndim != 4 or out.shape[:2] != (1, 1):
        print(f"error: --postprocess {name} needs one output [1, 1, H, W], got {[list(o.shape) for o in outs]}")
        return 1
    res = to_magma(out[0, 0])
    path = f"{args.output_prefix}{name}.rgb"
    res.tofile(path)
    print(f"disparity {float(out.min()):.3f}..{float(out.max()):.3f}, pale yellow is close, black is far")
    print(f"saved {os.path.abspath(path)}: raw {res.dtype} {list(res.shape)} ({res.nbytes} bytes)")
    return 0
