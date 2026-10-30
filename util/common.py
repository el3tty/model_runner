"""Helpers shared by run-model.py and run-onnx.py."""
import numpy as np


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
