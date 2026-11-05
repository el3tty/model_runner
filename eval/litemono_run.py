#!/usr/bin/env python3
"""
Run a Lite-Mono kmodel on the *.u8 tensors of kitti_prepare.py and save its disparity maps, for litemono_eval.py on the
PC. Needs only numpy and nncaseruntime (copy this script and util/ to the board, side by side); the same in the nncase
simulator on the PC, which gives the board's results bit for bit.

Time per image is `run`: set the input, run, read the output, as in coco_run.py. --warmup runs are left out of it.

Usage:
    python eval/litemono_run.py lite-mono-tiny-640x192-opt-u8-zoom.kmodel --input test --output disp-tiny.npz [--limit N] [--warmup 2]
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # util/ and nncaseruntime of the repository

import numpy as np
import nncaseruntime as nn

from util.common import temp_dump_root


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kmodel")
    ap.add_argument("--input", required=True, help="folder with <frame>.u8 (kitti_prepare.py's <out>/test)")
    ap.add_argument("--output", required=True, help="file to write (.npz: frame name -> disparity, float16)")
    ap.add_argument("--limit", type=int, default=0, help="first N frames only (default: 0 = all)")
    ap.add_argument("--warmup", type=int, default=2, help="untimed runs before the first frame (default: 2)")
    ap.add_argument("--dump-root", default=None, help="folder for runtime dumps, if the runtime writes any "
                                                      "(default: a new folder under <system temp>/nncase-rt-dumps)")
    args = ap.parse_args()

    kpu = nn.Interpreter()
    if hasattr(kpu, "set_dump_root"):  # a runtime built without dumps may not have it
        dump_root = args.dump_root or temp_dump_root(args.kmodel)
        os.makedirs(dump_root, exist_ok=True)
        kpu.set_dump_root(dump_root)
    with open(args.kmodel, "rb") as f:
        kpu.load_model(f.read())
    shape = list(kpu.get_input_shape(0))
    dtype = np.dtype(kpu.get_input_desc(0).dtype)
    files = sorted(glob.glob(os.path.join(args.input, "*.u8")))
    if args.limit > 0:
        files = files[:args.limit]
    print(f"{args.kmodel}: input {shape} {dtype}, {kpu.outputs_size} output(s), {len(files)} frames", flush=True)
    if dtype != np.uint8:
        print("error: expected a uint8 input (a kmodel built with --pre-process true --input-type Uint8)")
        return 1

    def infer(path):
        x = np.fromfile(path, dtype=np.uint8).reshape(shape)
        t = time.time()
        kpu.set_input_tensor(0, nn.RuntimeTensor.from_numpy(x))
        kpu.run()
        out = kpu.get_output_tensor(0).to_numpy()
        return out, time.time() - t

    for _ in range(args.warmup):
        infer(files[0])
    times, disp = [], {}
    for n, path in enumerate(files):
        out, t = infer(path)
        times.append(t)
        disp[os.path.splitext(os.path.basename(path))[0]] = out.astype(np.float16)
        print(f"  {n + 1}/{len(files)}  {1000 * t:.1f} ms", flush=True)
    t = 1000 * np.array(times)
    print(f"run: mean {t.mean():.2f} ms, median {np.median(t):.2f}, min {t.min():.2f}, max {t.max():.2f} "
          f"({len(t)} frames)")
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(args.output, **disp)
    print("saved", os.path.abspath(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
