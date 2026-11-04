# model_runner: Python scripts for preparing and running models on the K230

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
  ultralytics (YOLO26), the `yolox` package (YOLOX QAT). For QAT: torch (CUDA), onnx2torch.
* nncase compiler: build the CLI in Release (calibration runs the model in nncase's own evaluator, about 5 s per
  image and pass even in Release):

  ```
  cd <nncase>
  dotnet build src/Nncase.Cli/Nncase.Cli.csproj -c Release
  ```

  It is `src/Nncase.Cli/bin/Release/net9.0/Nncase.Cli.exe`. Arguments can be given in a response file (one or more
  per line, as in the compile blocks below): `Nncase.Cli.exe @args.txt`. Relative paths in it are relative to the
  current directory (the blocks below assume this folder); `${here}` stands for the response file's folder, e.g.
  `${here}/../kmodel/model.kmodel`.
* Board: Python 3 with numpy and nncaseruntime. Copy `coco_run.py`, `run-model.py` and `util/`. The board clock may
  be wrong, so copy with `tar xmf` (does not restore modification times):

  ```
  tar cf - coco_run.py run-model.py util | ssh root@<board> "tar xmf - -C <bdir>"
  ```

## Models

| `--detect` | Model | Input |
|---|---|---|
| `yolox` | [YOLOX-nano](#yolox-nano-qat) | 416×416, BGR 0..255, letterbox top-left, pad 114 |
| `yolov3` | [YOLOv3](#yolov3-and-yolo-fastest) | 416×416, RGB 0..1, letterbox centered, pad 128 |
| `yolo-fastest` | [Yolo-Fastest 1.1](#yolov3-and-yolo-fastest) | 320×320, RGB 0..1, plain resize |
| `yolo26` | [YOLO26n](#yolo26n) | 640×640, RGB 0..255, letterbox centered, pad 114 |

All ONNX models end at the raw head outputs (no box decoding, no NMS); `util/yolo_post.py` decodes them.

### YOLO26n

1. Export (downloads `yolo26n.pt` from the Ultralytics release on first use):

   ```
   python -c "from ultralytics import YOLO; YOLO('weights/yolo26n.pt')"
   python yolo_to_onnx.py weights/yolo26n.pt onnx/yolo26n-640-nosplit.onnx
   ```

   The graph ends at the one-to-many head (`end2end=False`): per stride 8/16/32 a box map `[1, 4, H/s, W/s]`
   (l, t, r, b in strides) and a class-logit map `[1, 80, H/s, W/s]`, kept apart so each gets its own quantization
   scale. BatchNorm and the 1/255 input scaling are folded, the `cv1(x).chunk(2)` of every C3k2 / C2PSA block is
   exported as two convs and the qkv conv + split of both attention blocks as three (q, k, v), because nncase runs
   Slice / Split on the CPU. `--keep-split` gives the original graph (same AP, about 20 ms slower on the K230 with
   the same input type; see the results).

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

   The original graph (the results' "original graph, float32 input" row):

   ```
   python yolo_to_onnx.py weights/yolo26n.pt onnx/yolo26n-640.onnx --keep-split
   ```

   ```
   compile
   --dataset calib300
   --dataset-format Image
   --calib-image-resize=Letterbox
   --calib-bgr=false
   --input-range 0 255
   --model-quant-mode UsePTQ
   onnx/yolo26n-640.onnx
   kmodel/yolo26n-640-ptq.kmodel
   k230
   ```

### QAT: YOLOX-nano and the PP-OCR recognizer

`qat_train.py <task>` fine-tunes a model for the K230's uint8 quantization; `yolox` and `ppocr` are the tasks. The
fake-quant model is the network as the K230 runs it: `util/qat_graph.py` puts the quantizers where the compile path
quantizes, by graph rules, on a `torch.fx` graph (YOLOX: the float model traced; PP-OCR: the ONNX file converted with
`onnx2torch`):

* quantized ops: Conv (weights per output channel), Gemm (weights per output feature), LSTM, AveragePool /
  GlobalAveragePool, Add / Mul of two tensors (residual adds, squeeze-excitation scaling);
* an activation that is the only consumer of a quantized op fuses into it (ReLU, Clip, Sigmoid, HardSigmoid, SiLU,
  HardSwish, also when spelled out as several ops); the activation output is the stored tensor;
* a stored tensor is fake-quantized (uint8, per tensor) when a quantized op reads it, directly or through ops that
  keep the quantization grid (MaxPool, nearest Resize / Upsample, Concat, slicing, Reshape, Transpose, ...); the
  inputs of a Concat share one range (nncase unites concat ranges); the model input is not quantized;
* the LSTM (`QLSTM`): the KPU runs it with uint8 x, W, R (per row) and hidden state, and an fp16 cell state; the
  fake-quant version quantizes the hidden state at every time step (the tensor stored as Y and fed back).

For YOLOX these rules give exactly the placement of the former hand-written code (111 activation quantizers in 99
range groups, each at the same tensor). Then:

1. the activation ranges: calibrated like nncase's PTQ (`--calib kld`, nncase's default KL divergence of 8192-bin
   histograms, or `noclip` = min / max), on the float network and the input as the K230 sees it; that gives nncase's
   own ranges (PP-OCR: all 80 equal to nncase's QuantScheme.json). `--init <checkpoint>` takes them (and the weights)
   from a checkpoint, `--scheme <QuantScheme.json>` (ppocr) from an nncase quant scheme. The calibrated model is
   `<out>/init.pth` (+ `ranges.json`); `--epochs 0` stops after evaluating it (the PTQ accuracy);
2. training: the ranges stay fixed, the weights train; warmup + cosine lr down to 5% of `--lr`; `--clip-grad`; after
   every epoch the task's metric, `latest.pth` (with the optimizer, for `--resume`), `best.pth` and `ranges_best.json`
   in `--out`. `--eval-first` evaluates before training, `--iters-per-epoch` shortens epochs for a pilot run.

| task | data | training | metric | export |
|---|---|---|---|---|
| `yolox` | COCO, YOLOX's pipeline (mosaic / mixup, off for the last `--no-aug-epochs` epochs, with the L1 loss) | YOLOX's loss (labels), SGD, `--lr` per 64 images, stem conv frozen | AP on val2017 | ONNX with Q/DQ pairs |
| `ppocr` | text lines rendered on the fly (`util/ocr_synth.py`) | distillation: KL of the float and the fake-quant model's per-time-step class distributions (no labels), Adam | words / CER on `data/ocr-synth` | ONNX with Q/DQ pairs |

`qat_export.py <task> <checkpoint> <output.onnx> --act-qdq` exports a checkpoint for nncase: the fake-quant model
traced to ONNX, every quantized activation with a QuantizeLinear / DequantizeLinear pair holding its trained range,
the weights baked onto their per-channel grid; `--float` exports the trained weights without any quantization. The
ONNX file is all nncase needs: it reads the ranges from the Q/DQ pairs, so the compile takes no calibration dataset
and no quant scheme file. `--weight-qdq` keeps per-channel Q/DQ nodes on the conv / gemm weights instead of baking
them (the same values; for onnxruntime or inspection, nncase's ONNX import rejects per-channel Q/DQ). An LSTM is
exported as one ONNX LSTM node (float weights, which nncase quantizes per row as the training did, and a Q/DQ pair on
its output), not as the step loop the training runs. Runs on the CPU.
To measure a Q/DQ model with onnxruntime, switch its graph optimizations off (`ORT_DISABLE_ALL`): they replace
Q/DQ + Conv with integer kernels that round differently (PP-OCR: 0.932 words instead of 0.941).
`python qat_train.py <task> -h` and `python qat_export.py <task> -h` list the options. The code: `util/qat_train.py`
(training and export drivers, schedule, checkpoints), `util/qat_graph.py` (placement, ONNX import, quant scheme
matching), `util/qat_generic.py` (fake-quant modules, nncase's range rules and calibration, ONNX export),
`util/qat_yolox.py` and `util/qat_ppocr.py` (the tasks, with their exports), `util/ocr_synth.py` (text lines).

#### YOLOX-nano

Needs the `yolox` package, the COCO folder YOLOX expects and `yolox_nano.pth` (the pretrained float model from the
YOLOX release page):

```
python qat_train.py yolox -c yolox_nano.pth --epochs 0 --out qat_yolox
python qat_train.py yolox --init qat_yolox/init.pth --epochs 15 --eval-first --out qat_yolox
python qat_export.py yolox qat_yolox/best.pth onnx/yolox-qat-kld.onnx --act-qdq
```

The first line calibrates on `--calib-images` (512) random train2017 images and reports the PTQ AP (0.182, AP50 0.305);
`-c yolox_nano.pth --epochs 15` does both in one run. The stem conv stays frozen (it sees raw 0..255 pixels, and
with BatchNorm folded its gradients are far too large for the shared lr).

`qat_export.py yolox` exports `init.pth`, `best.pth` or `latest.pth` (checkpoints of the former `yolox_qat.py` /
`yolox_qat_train.py` load too): input `1x3x416x416` float32 BGR 0..255, output `[1, 3549, 85]` without box
decoding. The compile:

```
compile
--input-range 0 255
--model-quant-mode UsePTQ
onnx/yolox-qat-kld.onnx
kmodel/yolox-qat-kld.kmodel
k230
```

#### PP-OCR recognizer

`onnx/ppocr-en-rec-32x320.onnx` (MobileNetV3 with squeeze-excitation blocks, two bidirectional LSTMs, CTC; input
`1x3x32x320`, output `[1, 80, 97]`). Needs the original PaddleOCR ONNX
`weights/ppocr/en_number_mobile_v2.0_rec_infer.onnx` (its metadata holds the character set), the calibration images
`data/ocr-synth/calib` (nncase's `--dataset`) and the labeled test set `data/ocr-synth` (1000 lines):

```
python qat_train.py ppocr --epochs 0 --out qat_ppocr
python qat_train.py ppocr --init qat_ppocr/init.pth -b 128 --epochs 20 --out qat_ppocr
python qat_export.py ppocr qat_ppocr/best.pth onnx/ppocr-en-rec-qat.onnx --act-qdq
```

The calibrated model scores what the board does (PTQ: 0.853 words / CER 0.029, the board 0.855 / 0.029; float
0.931 / 0.014). The default run reached its best after 3 epochs: 0.942 / 0.0105 in PyTorch, and its kmodel 0.942 /
0.0115 on the K230 simulator (bit-exact with the board), against 0.855 / 0.029 for the PTQ kmodel. The input is
mapped like the K230's pre-process, `(x - 128) / 127.5` (nncase's integer zero point; PaddleOCR uses 127.5).

Training speed: the LSTMs run step by step in Python, about 125 lines/s on a GTX 1060 at `-b 128` (`-b 256` does not
fit in 6 GB); the default 20 x 1000 x 128 lines took 5.6 hours there (the best epoch was the third: `--epochs 5`
would have done). Fonts for the training lines:
`--fonts <dir>` (every `.ttf` / `.otf` in it), else the Windows fonts `data/ocr-synth` was made with, else the DejaVu
/ Liberation fonts of a Linux system (`apt-get install fonts-dejavu-core fonts-liberation`).
`python -m util.ocr_synth <dir>` writes a new test set.

The PTQ build (the float model, calibrated by nncase on `data/ocr-synth/calib`; uint8 BGR input, which the
pre-process maps to -1..1):

```
compile
--dataset data/ocr-synth/calib
--dataset-format Image
--calib-image-resize=Letterbox
--calib-bgr=true
--pre-process true
--input-type Uint8
--input-shape 1 3 32 320
--input-layout NCHW
--model-layout NCHW
--input-range -1 1
--model-quant-mode UsePTQ
onnx/ppocr-en-rec-32x320.onnx
kmodel/ppocr-en-rec-32x320.kmodel
k230
```

The export (`--act-qdq`) is the original graph operator for operator plus 80 Q/DQ pairs; it compiles like the PTQ
build but without the four calibration lines:

```
compile
--pre-process true
--input-type Uint8
--input-shape 1 3 32 320
--input-layout NCHW
--model-layout NCHW
--input-range -1 1
--model-quant-mode UsePTQ
onnx/ppocr-en-rec-qat.onnx
kmodel/ppocr-en-rec-qat.kmodel
k230
```

That needs an nncase whose Q/DQ import (`QdqQuantSchemePass`) handles LSTM and MatMul weights and the pre-processed
input (the k230 branch of `<nncase>`); an older one fails on the LSTM with "Invalid w_quant_type".

#### QAT training on RunPod

YOLOX training is CPU-bound (data augmentation): about 5 hours on an RTX 4090 pod with 10 vCPUs at `-b 16`. PP-OCR
renders its training lines on the pod (no data set to upload) and fits a local GPU as well. Use a PyTorch template
with a network volume (`/workspace` survives pod restarts; everything else does not) and "SSH over exposed TCP": the
`ssh.runpod.io` proxy gives only an interactive shell, so `scp` and `rsync` fail through it. Placeholders:
`<pod-ip>` and `<port>` from the pod's Connect page, `<key>` the SSH key added to RunPod.

On the PC, pack this repository (git bundle: no remote needed) and copy it with what the task needs (YOLOX: the float
checkpoint; PP-OCR: the two ONNX files and `data/ocr-synth`, about 40 MB):

```
git bundle create model_runner.bundle master
scp -i <key> -P <port> model_runner.bundle yolox_nano.pth root@<pod-ip>:/workspace/
scp -i <key> -P <port> onnx/ppocr-en-rec-32x320.onnx weights/ppocr/en_number_mobile_v2.0_rec_infer.onnx root@<pod-ip>:/workspace/
scp -i <key> -P <port> -r data/ocr-synth root@<pod-ip>:/workspace/
ssh -i <key> -p <port> root@<pod-ip>
```

On the pod: system packages, this repository and its PP-OCR files; for YOLOX also the `yolox` package (its
`onnx-simplifier` pin does not build on recent Python and is not needed; OpenCV needs libGL) and COCO 2017 (about
20 GB):

```
apt-get update && apt-get install -y libgl1 libglib2.0-0 tmux unzip rsync fonts-dejavu-core fonts-liberation
pip install onnx onnxruntime onnx2torch pillow
cd /workspace && git clone model_runner.bundle model_runner && cd model_runner
mkdir -p onnx weights/ppocr data && mv ../ppocr-en-rec-32x320.onnx onnx/ && mv ../en_number_mobile_v2.0_rec_infer.onnx weights/ppocr/ && mv ../ocr-synth data/ && mv ../yolox_nano.pth .
cd /workspace && git clone https://github.com/Megvii-BaseDetection/YOLOX && cd YOLOX
sed -i '/onnx-simplifier/d' requirements.txt && pip install -r requirements.txt && pip install -v -e . --no-build-isolation
mkdir -p /workspace/datasets/COCO && cd /workspace/datasets/COCO
for f in train2017 val2017; do wget -q http://images.cocodataset.org/zips/$f.zip && unzip -q $f.zip && rm $f.zip; done
wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -q annotations_trainval2017.zip && rm annotations_trainval2017.zip
```

Count the CPUs the pod may use: `nproc` reports the host's (96 on a 4090 pod), and that many data-loader workers
exhaust memory and stall the loader. With cgroup v1 the limit is quota / period:

```
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us
```

(`cat /sys/fs/cgroup/cpu.max` with cgroup v2.) Run the training inside tmux, so it survives a dropped SSH connection
(`Ctrl-b d` detaches, `tmux attach -t qat` returns), with `--workers` a bit below that CPU count:

```
tmux new -s qat
export YOLOX_DATADIR=/workspace/datasets
cd /workspace/model_runner
python qat_train.py yolox -c yolox_nano.pth --epochs 15 --workers 8 --eval-first --out qat_yolox 2>&1 | tee train-yolox.log
python qat_train.py ppocr -b 128 --epochs 20 --workers 8 --eval-first --out qat_ppocr 2>&1 | tee train-ppocr.log
```

If the pod restarts, continue with `--resume <out>/latest.pth` and the same other arguments. `nvidia-smi -l 1` shows
the GPU load (YOLOX: expect well under 100%, the CPUs are the limit). Back on the PC, fetch the run and export it:

```
scp -i <key> -P <port> -r root@<pod-ip>:/workspace/model_runner/qat_yolox .
python qat_export.py yolox qat_yolox/best.pth onnx/yolox-qat-kld.onnx --act-qdq
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
| YOLO26n 640, PTQ, no-split, uint8 input | board | 0.400 | 0.559 | 51.1 ms |
| YOLOX-nano 416, QAT | board | 0.243 | 0.396 | 25.2 ms |
| YOLOv3 416, int8 | board | 0.373 | 0.669 | |
| Yolo-Fastest 1.1 320, int8 | board | 0.075 | 0.183 | |

Time per image is `coco_run.py`'s `run`: set the input, run, read all outputs. The YOLOv3 AP is likely inflated: the
ONNX model zoo model may have been trained on images that are in val2017.

## Runtime and simulator dumps

The K230 simulator gives the board's results bit for bit, so a kmodel can be debugged on the PC. nncase's Debug
runtime dumps the tensors of every CPU op, including the inputs and outputs of each KPU call, and with
`NNCASE_SIM_CHECKPOINTS=1` the simulator also dumps what the KPU's activation units compute inside a call (one record
per tile: the convolution's integer sums, the dequantisation / activation parameters and the outputs). How to make
the dumps and how to read them: "Debug dumps and check points" in `<nncase>/src/Native/src/simulator/README.md`;
`<nncase>/tools/ckpt_dump.py` prints them in readable form:

```
python <nncase>/tools/ckpt_dump.py 'dumps/Runtime/1$1_7.ckpt/act_dm_write.chk' --kind P --record 0
```
