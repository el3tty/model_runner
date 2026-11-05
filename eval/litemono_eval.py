#!/usr/bin/env python3
"""
Compare Lite-Mono disparity maps (litemono_run.py) with the float ONNX model on the same frames, and both with the LiDAR
ground truth of kitti_prepare.py.

    python eval/litemono_eval.py --input build/data/kitti-lm/test --onnx build/onnx/lite-mono-tiny-640x192.onnx --disp build/data/kitti-lm/disp-tiny.npz

float vs kmodel: depth = 1 / (1/100 + (1/0.1 - 1/100) * disp) (Lite-Mono's disp_to_depth), abs-rel over all pixels.
Against the LiDAR: the evaluation of monodepth2 / Lite-Mono on the raw points (the Eigen split's numbers use the same
code on different frames): disparity resized to the image size, Garg crop, depth 1e-3..80 m, per image median scaling.
Frames are few, so each number comes with the standard deviation over the frames. The result is not comparable with
the paper's table: one drive, and the points are not the improved ground truth.
"""
import argparse
import glob
import os

import cv2
import numpy as np

MIN_DEPTH, MAX_DEPTH = 1e-3, 80.0


def to_depth(disp):
    lo, hi = 1 / 100.0, 1 / 0.1
    return 1.0 / (lo + (hi - lo) * disp.astype(np.float64))


def lidar_metrics(disp, gt):
    """(abs_rel, rmse, delta1, median scale) of a disparity map against a sparse depth map."""
    h, w = gt.shape
    pred = to_depth(cv2.resize(disp.astype(np.float32), (w, h)))  # as evaluate_depth.py: disp_to_depth, then resized
    crop = np.zeros(gt.shape, bool)
    crop[int(0.40810811 * h):int(0.99189189 * h), int(0.03594771 * w):int(0.96405229 * w)] = True
    ok = crop & (gt > MIN_DEPTH) & (gt < MAX_DEPTH)
    g, p = gt[ok].astype(np.float64), pred[ok].astype(np.float64)
    scale = np.median(g) / np.median(p)
    p = np.clip(p * scale, MIN_DEPTH, MAX_DEPTH)
    thresh = np.maximum(g / p, p / g)
    return np.mean(np.abs(g - p) / g), np.sqrt(np.mean((g - p) ** 2)), np.mean(thresh < 1.25), scale


def summary(name, rows):
    a = np.array(rows)
    print(f"{name:24s} abs_rel {a[:, 0].mean():.4f} (sd {a[:, 0].std():.4f})  rmse {a[:, 1].mean():.3f} m  "
          f"delta1 {a[:, 2].mean():.4f}  median scale {a[:, 3].mean():.2f} (sd {a[:, 3].std():.2f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="kitti_prepare.py's <out>/test")
    ap.add_argument("--onnx", required=True, help="the float model")
    ap.add_argument("--disp", nargs="*", default=[], help=".npz files of litemono_run.py")
    args = ap.parse_args()

    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    names = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(args.input, "*.u8")))
    ref = {}
    for n in names:
        x = np.fromfile(os.path.join(args.input, n + ".u8"), dtype=np.uint8).reshape(1, 3, 192, 640)
        ref[n] = sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float32) / 255})[0][0, 0]
    gts = {n: np.load(os.path.join(args.input, n + ".npy")) for n in names}

    print(f"{len(names)} frames")
    summary("float vs LiDAR", [lidar_metrics(ref[n], gts[n]) for n in names])
    for path in args.disp:
        z = np.load(path)
        have = [n for n in names if n in z]
        q = {n: z[n].reshape(ref[n].shape).astype(np.float32) for n in have}
        print(f"\n{path} ({len(have)} frames): disp {min(v.min() for v in q.values()):.4f}.."
              f"{max(v.max() for v in q.values()):.4f}, float {min(ref[n].min() for n in have):.4f}.."
              f"{max(ref[n].max() for n in have):.4f}")
        rel = [np.mean(np.abs(to_depth(q[n]) - to_depth(ref[n])) / to_depth(ref[n])) for n in have]
        corr = [np.corrcoef(q[n].ravel(), ref[n].ravel())[0, 1] for n in have]
        dd = [np.abs(q[n] - ref[n]).mean() for n in have]
        print(f"vs float: depth abs-rel {np.mean(rel):.4f} (sd {np.std(rel):.4f}, max {np.max(rel):.4f}), "
              f"|disp diff| {np.mean(dd):.5f}, correlation {np.mean(corr):.4f} (min {np.min(corr):.4f})")
        summary("float, same frames", [lidar_metrics(ref[n], gts[n]) for n in have])
        summary("kmodel vs LiDAR", [lidar_metrics(q[n], gts[n]) for n in have])


if __name__ == "__main__":
    main()
