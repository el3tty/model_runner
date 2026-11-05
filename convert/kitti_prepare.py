#!/usr/bin/env python3
"""
Lite-Mono inputs from KITTI raw drives (image_02, the left colour camera), resized the way monodepth2 / Lite-Mono do:
Pillow LANCZOS to 640x192, plain resize.

    python convert/kitti_prepare.py build/data/kitti build/data/kitti-lm --calib 2011_09_26_drive_0002 --test 2011_09_26_drive_0048

<out>/calib/*.png: the calibration images (nncase's --dataset; already 640x192, so its resize leaves them alone).
<out>/calib-zoom/*.png: the same and bottom crops of the calibration drive zoomed 2x and 3x. Use this set: the kmodel
cannot output anything nearer than the calibration images show, and a drive without close objects cuts off the near range.
<out>/test/*.u8: raw uint8 RGB 1x3x192x640 tensors, as the kmodel takes them.
<out>/test/*.npy: the LiDAR ground truth of each test frame, float32 metres, [375-ish, 1242-ish] at the original image
size, 0 where there is no point (monodepth2's generate_depth_map: velodyne points projected into camera 2, nearest
point per pixel).
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image

W, H = 640, 192


def read_calib(path):
    """{key: float array} of a KITTI calibration text file."""
    out = {}
    for line in open(path):
        key, _, value = line.partition(":")
        try:
            out[key] = np.array([float(v) for v in value.split()], dtype=np.float64)
        except ValueError:
            pass
    return out


def lidar_depth(root, velo_path, shape):
    """Depth map of one velodyne scan in camera 2 (monodepth2's generate_depth_map, cam 2, without vel_depth)."""
    cam = read_calib(os.path.join(root, "calib_cam_to_cam.txt"))
    velo = read_calib(os.path.join(root, "calib_velo_to_cam.txt"))
    v2c = np.eye(4)
    v2c[:3, :3] = velo["R"].reshape(3, 3)
    v2c[:3, 3] = velo["T"]
    r_rect = np.eye(4)
    r_rect[:3, :3] = cam["R_rect_00"].reshape(3, 3)
    proj = cam["P_rect_02"].reshape(3, 4) @ r_rect @ v2c

    pts = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4).astype(np.float64)
    pts = pts[pts[:, 0] >= 0]
    pts[:, 3] = 1.0
    uvz = (proj @ pts.T).T
    uvz[:, :2] /= uvz[:, 2:3]
    u = np.round(uvz[:, 0]).astype(int) - 1
    v = np.round(uvz[:, 1]).astype(int) - 1
    ok = (u >= 0) & (v >= 0) & (u < shape[1]) & (v < shape[0]) & (uvz[:, 2] > 0)
    u, v, z = u[ok], v[ok], uvz[ok, 2]
    order = np.argsort(-z)  # far points first, so the nearest point of a pixel is written last
    depth = np.zeros(shape, dtype=np.float32)
    depth[v[order], u[order]] = z[order]
    return depth


def frames(root, drive):
    return sorted(glob.glob(os.path.join(root, drive + "_sync", "image_02", "data", "*.png")))


def resized(path):
    return Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kitti", help="folder with the unpacked archives (<date>/<drive>_sync, <date>/calib_*.txt)")
    ap.add_argument("out")
    ap.add_argument("--date", default="2011_09_26")
    ap.add_argument("--calib", required=True, help="drive of the calibration images")
    ap.add_argument("--test", required=True, help="drive of the test frames")
    ap.add_argument("--calib-images", type=int, default=20, help="calibration images, evenly spaced (default: 20)")
    args = ap.parse_args()

    root = os.path.join(args.kitti, args.date)
    calib_dir, test_dir = os.path.join(args.out, "calib"), os.path.join(args.out, "test")
    os.makedirs(calib_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    files = frames(root, args.calib)
    pick = np.linspace(0, len(files) - 1, min(args.calib_images, len(files))).round().astype(int)
    for i in pick:
        resized(files[i]).save(os.path.join(calib_dir, os.path.basename(files[i])))

    zoom_dir = os.path.join(args.out, "calib-zoom")
    os.makedirs(zoom_dir, exist_ok=True)
    for i in pick:
        resized(files[i]).save(os.path.join(zoom_dir, os.path.basename(files[i])))
    rng = np.random.default_rng(0)
    sources = files[::4][:args.calib_images]
    for zoom in (1.5, 2, 3):  # 1.5 only keeps the random sequence of the published set
        for i, f in enumerate(sources):
            image = Image.open(f).convert("RGB")
            w, h = int(image.width / zoom), int(image.height / zoom)
            x = int(rng.uniform(0, image.width - w))
            if zoom > 1.5 and i % 2 == 0:
                crop = image.crop((x, image.height - h, x + w, image.height)).resize((W, H), Image.LANCZOS)
                crop.save(os.path.join(zoom_dir, f"z{zoom}-{i:02d}.png"))

    files = frames(root, args.test)
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        np.asarray(resized(f)).transpose(2, 0, 1).tofile(os.path.join(test_dir, name + ".u8"))
        velo = os.path.join(root, args.test + "_sync", "velodyne_points", "data", name + ".bin")
        size = Image.open(f).size
        np.save(os.path.join(test_dir, name + ".npy"), lidar_depth(root, velo, (size[1], size[0])))
    print(f"{len(pick)} calibration images, {len(files)} test frames in {args.out}")


if __name__ == "__main__":
    main()
