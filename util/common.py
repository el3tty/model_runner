"""Helpers shared by run-model.py, run-onnx.py and coco_run.py."""
import os
import tempfile

import numpy as np


def temp_dump_root(kmodel):
    """A new folder for the runtime dumps of one run of kmodel: <system temp>/nncase-rt-dumps/<kmodel name>-XXXXXXXX.
    An interpreter without a dump root writes its dumps to the current directory."""
    parent = os.path.join(tempfile.gettempdir(), "nncase-rt-dumps")
    os.makedirs(parent, exist_ok=True)
    return tempfile.mkdtemp(prefix=os.path.splitext(os.path.basename(kmodel))[0] + "-", dir=parent)


def save_tensor_text(path, array):
    """Write a tensor as text, one value per line, in row-major order."""
    flat = np.ascontiguousarray(array).reshape(-1)
    if flat.dtype.kind == "f":
        # %.9g round-trips float32; float16 values print exactly with it too.
        fmt = "%.9g"
    elif flat.dtype.kind == "b":
        flat = flat.astype(np.uint8)
        fmt = "%d"
    else:
        fmt = "%d"
    np.savetxt(path, flat, fmt=fmt)
