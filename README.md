# model_runner

Scripts to run YOLO detectors as ONNX (onnxruntime, PC) and as K230 kmodels (board or nncase simulator), and to
measure their COCO val2017 AP the same way for both.

Placeholders used below:

| Placeholder | Meaning |
|---|---|
| `<COCO>` | COCO root with `train2017/`, `val2017/`, `annotations/`. The scripts default to `../YOLOX/datasets/COCO` (relative to this folder), or `$COCO_DIR` |
| `<nncase>` | nncase checkout (K230 branch) |
| `<board>` | the board's address; `<bdir>` a working folder on it |

`data/`, `onnx/`, `kmodel/`, `weights/` are git-ignored: everything in them is produced by the commands below.

## Requirements

* PC: Python 3, numpy, opencv-python, Pillow, onnxruntime, onnx, pycocotools. For the exports: torch,
  ultralytics (YOLO26), the `yolox` package (YOLOX QAT).
* nncase compiler: build the CLI in Release (calibration runs the model in nncase's own evaluator, about 5 s per
  image and pass even in Release):

  ```
  cd <nncase>
  dotnet build src/Nncase.Cli/Nncase.Cli.csproj -c Release
  ```

  It is `src/Nncase.Cli/bin/Release/net9.0/Nncase.Cli.exe`. Arguments can be given in a response file, one
  argument per line: `Nncase.Cli.exe @args.txt`.
* Board: Python 3 with numpy and nncaseruntime. Copy `coco_run.py`, `run-model.py` and `util/`. The board clock may
  be wrong, so copy with `tar xmf` (does not restore modification times):

  ```
  tar cf - coco_run.py run-model.py util | ssh root@<board> "tar xmf - -C <bdir>"
  ```

## Models

| `--detect` | Input | ONNX | kmodel |
|---|---|---|---|
| `yolox` | 416×416, BGR 0..255, letterbox top-left, pad 114 | YOLOX-nano QAT, `onnx/yolox-qat-qdq-wfuse.onnx` | `kmodel/yolox-qat-qdq.kmodel` |
| `yolov3` | 416×416, RGB 0..1, letterbox centered, pad 128 | ONNX model zoo YOLOv3 int8, `onnx/yolov3-12-int8-heads.onnx` | `kmodel/yolov3-12-int8.kmodel` |
| `yolo-fastest` | 320×320, RGB 0..1, plain resize | Yolo-Fastest 1.1 int8, `onnx/yolo-fastest-1.1-320_int8_qop.onnx` | `kmodel/yolo-fastest-1.1-320-qop.kmodel` |
| `yolo26` | 640×640, RGB 0..255, letterbox centered, pad 114 | Ultralytics YOLO26n, `onnx/yolo26n-640-nosplit.onnx` | `kmodel/yolo26n-640-nosplit-u8-ptq.kmodel` |

All ONNX models end at the raw head outputs (no box decoding, no NMS); `util/yolo_post.py` decodes them.

### YOLO26n

1. Export (downloads `yolo26n.pt` from the Ultralytics release on first use):

   ```
   python -c "from ultralytics import YOLO; YOLO('weights/yolo26n.pt')"
   python yolo26_to_onnx.py weights/yolo26n.pt onnx/yolo26n-640-nosplit.onnx
   ```

   The graph ends at the one-to-many head (`end2end=False`): per stride 8/16/32 a box map `[1, 4, H/s, W/s]`
   (l, t, r, b in strides) and a class-logit map `[1, 80, H/s, W/s]`, kept apart so each gets its own quantization
   scale. BatchNorm and the 1/255 input scaling are folded, and the `cv1(x).chunk(2)` of every C3k2 / C2PSA block is
   exported as two convs, because nncase runs Slice on the CPU. `--keep-split` gives the original graph (same AP,
   about 20 ms slower on the K230).

2. Calibration set: 300 random train2017 images (seed 0):

   ```
   python -c "import os, random, shutil, sys; s, d = sys.argv[1:]; os.makedirs(d, exist_ok=True); [shutil.copy(os.path.join(s, f), d) for f in random.Random(0).sample(sorted(os.listdir(s)), 300)]" <COCO>/train2017 calib300
   ```

3. Compile (PTQ, KLD calibration, uint8 input). Takes about 1 hour, nearly all of it calibration:

   ```
   compile
   --dataset calib300
   --dataset-format Image
   --calib-image-resize=Letterbox
   --calib-bgr=false
   --pre-process true
   --input-type Uint8
   --input-shape 1 3 640 640
   --input-layout NCHW
   --model-layout NCHW
   --input-range 0 255
   --model-quant-mode UsePTQ
   onnx/yolo26n-640-nosplit.onnx
   kmodel/yolo26n-640-nosplit-u8-ptq.kmodel
   k230
   ```

   Without the five pre-process lines the kmodel takes float32 0..255 input (about 4 ms slower per image).

### YOLOX-nano QAT

Needs the `yolox` package and the COCO folder YOLOX expects:

```
python yolox_qat.py -c yolox_nano.pth --out qat_out
python yolox_qat_train.py --init qat_out/qat_init.pth -b 64 --epochs 15 --no-aug-epochs 3 --out qat_run
python pth_to_onnx.py qat_run/best.pth onnx/yolox-qat-qdq-wfuse.onnx --act-qdq
```

`yolox_qat.py` sets the activation ranges like nncase's PTQ (`--calib-method kld`, the default, or `noclip`) on 512
train2017 images and prints the AP with weights-only and with weights + activations fake quantization. The ranges stay
fixed during `yolox_qat_train.py`; only the weights train. The board result in the table below comes from an earlier
version that calibrated with the mean of per-batch min / max and let the ranges follow the training data.

`--act-qdq` puts Q/DQ pairs with the trained ranges on the activations; the weight quantization is baked into the
conv weights. nncase reads the ranges from the Q/DQ nodes, so no calibration dataset is needed:

```
compile
--input-range 0 255
--model-quant-mode UsePTQ
onnx/yolox-qat-qdq-wfuse.onnx
kmodel/yolox-qat-qdq.kmodel
k230
```

### YOLOv3 and Yolo-Fastest

Both are already int8-quantized ONNX models (QLinearConv), so nncase uses their quantization and needs no dataset.
How they were made is not scripted here:

* `yolov3-12-int8-heads.onnx`: `yolov3-12-int8.onnx` from the ONNX model zoo, with the decode / NMS part cut off,
  leaving the three raw heads `[1, 255, H/s, W/s]` (strides 8, 16, 32).
* `yolo-fastest-1.1-320_int8_qop.onnx`: Yolo-Fastest 1.1 (darknet) quantized with onnxruntime's static quantization
  (QOperator format); two raw heads, strides 16 and 32.

```
compile
--input-range 0 1
--model-quant-mode UsePTQ
onnx/yolov3-12-int8-heads.onnx
kmodel/yolov3-12-int8.kmodel
k230
```

```
compile
--model-quant-mode UsePTQ
onnx/yolo-fastest-1.1-320_int8_qop.onnx
kmodel/yolo-fastest-1.1-320-qop.kmodel
k230
```

## One image

`tensor_from_image.py` writes the raw input tensor, `run-onnx.py` / `run-model.py` run the model, draw the
detections into a tensor of the same layout, and `tensor_to_image.py` turns it into an image:

| `--detect` | `tensor_from_image.py` options | `tensor_to_image.py` options |
|---|---|---|
| `yolox` | `416x416 --letterbox --bgr --type f32` | `416x416 f32 --bgr` |
| `yolov3` | `416x416 --letterbox --pad 128 --range 1 --type f32` | `416x416 f32 --range 1` |
| `yolo-fastest` | `320x320 --range 1 --type f32` | `320x320 f32 --range 1` |
| `yolo26` | `640x640 --letterbox --pad 114 --type f32` (`--type u8` for the u8 kmodel) | `640x640 f32` (`u8`) |

Example (YOLO26):

```
python tensor_from_image.py data/test.jpg 640x640 --letterbox --pad 114 --type f32 --output data/yolo26/yolo26-f32.in
python run-onnx.py onnx/yolo26n-640-nosplit.onnx data/yolo26/yolo26-f32.in --detect yolo26 --output-prefix=data/yolo26/out-onnx-
python tensor_to_image.py data/yolo26/out-onnx-yolo26.rgb 640x640 f32 --format bmp --output data/yolo26/out-onnx-yolo26.bmp
```

On the board (or in the simulator) with the u8 kmodel:

```
python tensor_from_image.py data/test.jpg 640x640 --letterbox --pad 114 --type u8 --output data/yolo26/yolo26-u8.in
python3 run-model.py yolo26n-640-nosplit-u8-ptq.kmodel yolo26-u8.in --detect yolo26 --output-prefix out-
```

These single-image inputs use Pillow's resize, so they differ slightly from the COCO inputs below.

## COCO val2017 AP

The same 5000 prepared input tensors go to the kmodel and to the ONNX model; the decoding, thresholds and NMS are the
model's own evaluation settings (`PREP` in `util/coco.py`).

1. Prepare the inputs with the model's own evaluation preprocessing (5000 images; about 6 GB for 640×640):

   ```
   python coco_prepare.py --detect yolo26 --output data/coco-yolo26
   ```

2. Run the kmodel on the board and fetch the candidates. Only output cells whose score can reach the evaluation
   threshold are kept, as float16:

   ```
   tar cf - -C data coco-yolo26 | ssh root@<board> "tar xmf - -C <bdir>/data"
   scp kmodel/yolo26n-640-nosplit-u8-ptq.kmodel root@<board>:<bdir>/
   ssh root@<board> "cd <bdir> && python3 coco_run.py yolo26n-640-nosplit-u8-ptq.kmodel --input data/coco-yolo26 --output dets/yolo26.npz"
   scp root@<board>:<bdir>/dets/yolo26.npz data/dets/
   ```

   `coco_run.py` also works with the simulator (slow: use `--limit N`).

3. Score the kmodel candidates, and optionally ONNX models on the same tensors:

   ```
   python coco_eval.py --input data/coco-yolo26 data/dets/yolo26.npz --onnx onnx/yolo26n-640-nosplit.onnx
   ```

Evaluation settings per model:

| `--detect` | conf | NMS IoU | labels per box |
|---|---|---|---|
| `yolox` | 0.01 | 0.65 | best class |
| `yolov3`, `yolo-fastest` | 0.005 | 0.45 | every class above conf |
| `yolo26` | 0.001 | 0.7 | every class above conf (Ultralytics val; square 640 input instead of rectangular batches) |

## Results

COCO val2017, 5000 images, measured with the steps above:

| Model | Runs on | AP | AP50 | Time per image (board) |
|---|---|---|---|---|
| YOLO26n 640, float ONNX | onnxruntime | 0.409 | 0.569 | |
| YOLO26n 640, PTQ, original graph, float32 input | board | 0.401 | 0.560 | 77 ms |
| YOLO26n 640, PTQ, no-split, uint8 input | board | 0.401 | 0.560 | 53.5 ms |
| YOLOX-nano 416, QAT | board | 0.199 | 0.330 | |
| YOLOv3 416, int8 | board | 0.373 | 0.669 | |
| Yolo-Fastest 1.1 320, int8 | board | 0.075 | 0.183 | |

Time per image is `coco_run.py`'s `run`: set the input, run, read all outputs. The YOLOv3 AP is likely inflated: the
ONNX model zoo model may have been trained on images that are in val2017.
