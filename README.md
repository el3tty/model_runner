# model_runner: Python scripts for preparing and running models on the K230

Scripts to run YOLO detectors as ONNX (onnxruntime, PC) and as K230 kmodels (board or nncase simulator), and to
measure their COCO val2017 AP the same way for both. `convert/` holds the exports and the ONNX rewriter, `eval/` the
measurements; run everything from this folder.

## Contents

* [Test PC and placeholders](#test-pc-and-placeholders)
* [Requirements](#requirements)
* [Supported models](#supported-models)
  * [YOLO26n](#yolo26n)
  * [Quantization-aware training (QAT): YOLOX-nano and PP-OCR](#quantization-aware-training-qat-yolox-nano-and-pp-ocr)
    * [YOLOX-nano](#yolox-nano)
    * [PP-OCR recognizer](#pp-ocr-recognizer)
    * [Running QAT on RunPod](#running-qat-on-runpod)
  * [YOLOv3 and Yolo-Fastest](#yolov3-and-yolo-fastest)
  * [Lite-Mono](#lite-mono)
* [Optimizing a model with graph rewrites](#optimizing-a-model-with-graph-rewrites)
  * [Why a correct model can be slow](#why-a-correct-model-can-be-slow)
  * [What stops boxes from merging](#what-stops-boxes-from-merging)
  * [What a rewrite rule is](#what-a-rewrite-rule-is)
  * [The rewrites, one by one](#the-rewrites-one-by-one)
  * [All the rewrite rules](#all-the-rewrite-rules)
  * [Where the rewrites fit in the workflow](#where-the-rewrites-fit-in-the-workflow)
* [Running a model on one input](#running-a-model-on-one-input)
* [Measuring detection accuracy on COCO val2017](#measuring-detection-accuracy-on-coco-val2017)
* [COCO val2017 results](#coco-val2017-results)
* [Debugging with runtime and simulator dumps](#debugging-with-runtime-and-simulator-dumps)
* [Running the tests](#running-the-tests)

## Test PC and placeholders

PC times and memory limits below were measured on this PC (the "test PC"); times on the board are K230 times:

| Part | Test PC |
|---|---|
| CPU | AMD Ryzen 9 3950X, 16 cores / 32 threads |
| GPU | NVIDIA GeForce GTX 1060 6 GB |
| RAM | 64 GB |
| Storage | 2 TB NVMe SSD |

Placeholders used below:

| Placeholder | Meaning |
|---|---|
| `<COCO>` | COCO 2017 root with `train2017/`, `val2017/`, `annotations/` (from [cocodataset.org](https://cocodataset.org/#download): the train / val 2017 images and the 2017 train/val annotations). The COCO scripts take it as `--coco`, or `$COCO_DIR` when that is set |
| `<nncase>` | nncase checkout (K230 branch) |
| `<board>` | the board's address; `<bdir>` a working folder on it |

`build/` is git-ignored: everything in it (`build/data`, `build/onnx`, `build/kmodel`, `build/weights`) is produced by
the commands below.

## Requirements

* PC: Python 3. For running models, the COCO evaluation and the tests:

  ```
  pip install -r requirements.txt
  ```

  For the exports and QAT, torch with CUDA first, then the rest (without torch, its tests are skipped):

  ```
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  pip install -r requirements-qat.txt
  ```

  YOLOX QAT also needs the `yolox` package, from its repository (see [the RunPod commands](#running-qat-on-runpod)).
  `run-model.py` and `eval/coco_run.py` need `nncaseruntime` from the nncase build: a link to its `nncaseruntime` folder
  in this folder is enough, e.g. to `<nncase>/build-rel/install/nncaseruntime`.
* nncase compiler: build the CLI in Release (calibration runs the model in nncase's own evaluator, about 5 s per
  image and pass on the test PC, even in Release):

  ```
  cd <nncase>
  dotnet build src/Nncase.Cli/Nncase.Cli.csproj -c Release
  ```

  It is `src/Nncase.Cli/bin/Release/net9.0/Nncase.Cli.exe`. Arguments can be given in a response file (one or more
  per line, as in the compile blocks below): `Nncase.Cli.exe @args.txt`. Relative paths in it are relative to the
  current directory (the blocks below assume this folder); `${here}` stands for the response file's folder, e.g.
  `${here}/../kmodel/model.kmodel`.
* Board: Python 3 with numpy and nncaseruntime. Copy `eval/coco_run.py`, `eval/litemono_run.py`, `run-model.py` and
  `util/` into one folder. The board clock may be wrong, so copy with `tar xmf` (does not restore modification times):

  ```
  tar cf - --exclude=__pycache__ --exclude=.pytest_cache -C eval coco_run.py litemono_run.py -C .. run-model.py util | ssh root@<board> "tar xmf - -C <bdir>"
  ```

## Supported models

The name is the `--postprocess` of `run-onnx.py` / `run-model.py` and, for the detectors, the `--detect` of the
COCO scripts.

| Name | Model | Input |
|---|---|---|
| `yolox` | [YOLOX-nano](#yolox-nano) | 416×416, BGR 0..255, letterbox top-left, pad 114 |
| `yolov3` | [YOLOv3](#yolov3-and-yolo-fastest) | 416×416, RGB 0..1, letterbox centered, pad 128 |
| `yolo-fastest` | [Yolo-Fastest 1.1](#yolov3-and-yolo-fastest) | 320×320, RGB 0..1, plain resize |
| `yolo26` | [YOLO26n](#yolo26n) | 640×640, RGB 0..255, letterbox centered, pad 114 |
| `ppocr` | [PP-OCR recognizer](#pp-ocr-recognizer) | 320×32, one text line, BGR; uint8 (kmodel) or -1..1 (ONNX) |
| | [Lite-Mono](#lite-mono) (depth, no `--postprocess`) | 640×192, RGB; uint8 (kmodel) or 0..1 (ONNX) |

All ONNX models end at the raw head outputs (no box decoding, no NMS); `util/yolo_post.py` decodes them.

### YOLO26n

1. Export (downloads `yolo26n.pt` from the Ultralytics release on first use):

   ```
   python -c "from ultralytics import YOLO; YOLO('build/weights/yolo26n.pt')"
   python convert/yolo_to_onnx.py build/weights/yolo26n.pt build/onnx/yolo26n-640.onnx
   python convert/onnx_opt.py build/onnx/yolo26n-640.onnx build/onnx/yolo26n-640-nosplit.onnx
   ```

   The graph ends at the one-to-many head (`end2end=False`): per stride 8/16/32 a box map `[1, 4, H/s, W/s]`
   (l, t, r, b in strides) and a class-logit map `[1, 80, H/s, W/s]`, kept apart so each gets its own quantization
   scale. BatchNorm and the 1/255 input scaling are folded. [`onnx_opt.py`](#optimizing-a-model-with-graph-rewrites) then makes the
   `cv1(x).chunk(2)` of every C3k2 / C2PSA block two convs and the qkv conv + split of both attention blocks three
   (q, k, v), because nncase runs Slice / Split on the CPU. `build/onnx/yolo26n-640.onnx` is the original graph (same AP, about
   20 ms slower on the K230 with the same input type; see the results).

2. Calibration set: 300 random train2017 images (seed 0):

   ```
   python -c "import os, random, shutil, sys; s, d = sys.argv[1:]; os.makedirs(d, exist_ok=True); [shutil.copy(os.path.join(s, f), d) for f in random.Random(0).sample(sorted(os.listdir(s)), 300)]" <COCO>/train2017 calib300
   ```

3. Compile (PTQ, KLD calibration, uint8 input). Takes about 1 hour on the test PC, nearly all of it calibration:

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
   build/onnx/yolo26n-640-nosplit.onnx
   build/kmodel/yolo26n-640-nosplit-u8-ptq.kmodel
   k230
   ```

   Without the five pre-process lines the kmodel takes float32 0..255 input (about 4 ms slower per image).

   The original graph (the results' "original graph, float32 input" row):

   ```
   compile
   --dataset calib300
   --dataset-format Image
   --calib-image-resize=Letterbox
   --calib-bgr=false
   --input-range 0 255
   --model-quant-mode UsePTQ
   build/onnx/yolo26n-640.onnx
   build/kmodel/yolo26n-640-ptq.kmodel
   k230
   ```

### Quantization-aware training (QAT): YOLOX-nano and PP-OCR

`qat_train.py <task>` fine-tunes a model for the K230's uint8 quantization; `yolox` and `ppocr` are the tasks. The
fake-quant model is the network as the K230 runs it: `util/qat_graph.py` puts the quantizers where the compile path
quantizes, by graph rules, on a `torch.fx` graph (YOLOX: the float model traced; PP-OCR: the ONNX file converted with
`onnx2torch`):

* quantized ops: Conv (weights per output channel), Gemm (weights per output feature), LSTM, AveragePool /
  GlobalAveragePool, Add / Mul of two tensors (residual adds, squeeze-excitation scaling);
* an activation that is the only consumer of a quantized op fuses into it (ReLU, Clip, Sigmoid, HardSigmoid, SiLU,
  HardSwish, also when spelled out as several ops); the activation output is the stored tensor. An activation that
  shares the op's result with other readers fuses too: nncase gives it its own copy of the op;
* a stored tensor is fake-quantized (uint8, per tensor) when a quantized op reads it, directly or through ops that
  keep the quantization grid (MaxPool, nearest Resize / Upsample, Concat, slicing, Reshape, Transpose, ...); the
  inputs of a Concat share one range (nncase unites concat ranges); the model input is not quantized. Except a
  Conv / Gemm output without an activation that only Add / Mul read: the K230 keeps it in float16 (nncase's
  quantized IR stores it f32 and loads it f16), so training does not round it (PP-OCR: the 7 MobileNetV3 project convs
  into the residual Adds; YOLOX: none). The export still gives it a Q/DQ pair (`ExportQuant`): nncase needs a range on
  both inputs of an Add / Mul to run it on the KPU, without one it stays a float32 `Binary` outside the KPU;
* the LSTM (`QLSTM`): the KPU runs it with uint8 x, W, R (per row) and hidden state, and an fp16 cell state; the
  fake-quant version quantizes the hidden state at every time step (the tensor stored as Y and fed back).

**Where the Q/DQ pairs go, in any graph.** `util/qat_graph.py` sorts the ops of a model into four kinds:

| Kind | Ops |
|---|---|
| quantized op | Conv (depthwise and grouped too), Gemm, LSTM, AveragePool, GlobalAveragePool, Add / Mul of two tensors |
| fused activation | Relu, LeakyRelu, Clip, Sigmoid, HardSigmoid, HardSwish, SiLU, also spelled out as several ops |
| grid keeping op | MaxPool, nearest Resize, Concat, Slice, Reshape, Transpose, Flatten, Squeeze, Unsqueeze |
| anything else | e.g. Softmax, Sub, Div, Add / Mul with a constant, a Resize that interpolates |

Then, for the output of every quantized op:

1. Follow it through the activations that are its only readers: that tensor is what the K230 stores.
2. A quantized op reads it, directly or through grid keeping ops: it gets a pair, right after the op or its activation.
   Nothing reads it that way: no pair.
3. Except when it is a Conv / Gemm output without an activation and every reader is an Add / Mul of two tensors: the K230
   keeps it in float16. The pair goes on the edges into the Add / Mul, only in the export (`ExportQuant`): training does
   not round it and neither does the K230, but without it the Add / Mul leaves the KPU.
4. An activation beside other readers gets its own copy of the op: its output follows rule 2, and the raw output rules 2
   and 3 for the other readers.
5. An LSTM stores its hidden state itself: one pair on its output, rounded at every time step in training.

The pairs of a grid keeping op's input and output are one tensor with one range, so the pair goes in front of the op, and
all the inputs of a Concat share a range. By the reader of a tensor:

| Producer | Reader | Result |
|---|---|---|
| Conv -> Relu | Conv, depthwise Conv, Gemm | `Conv -> Relu -> QDQ -> Conv` |
| Conv, no activation | Conv, depthwise Conv, Gemm | `Conv -> QDQ -> Conv` |
| Conv, no activation | Add / Mul only | `Conv -> QDQ (export only) -> Add` |
| Conv, no activation | Add / Mul and a Conv | `Conv -> QDQ`, both read the quantized tensor |
| Conv -> Relu | Add / Mul | `Conv -> Relu -> QDQ -> Add` |
| Add -> Relu, or Add | Conv, Add / Mul | `Add -> Relu -> QDQ -> Conv` |
| GlobalAveragePool | Conv, Gemm | `Pool -> QDQ -> Conv` |
| Conv -> Relu | MaxPool, Reshape, ... -> Conv | `Conv -> Relu -> QDQ -> MaxPool -> Conv` |
| any quantized op | nothing quantized (e.g. the model output) | no pair |

There is never a pair on the model input (uint8 already), between an op and its fused activation, or on Conv / Gemm
weights (baked onto their grid, unless `--weight-qdq`). The output of an op of the last kind gets no pair either, even if
a Conv reads it: the rules only cover the outputs of quantized ops. For PP-OCR these rules give 71 + 2 LSTM + 7 export
only = 80 pairs.

For YOLOX these rules give exactly the placement of the former hand-written code (111 activation quantizers in 99
range groups, each at the same tensor). Checkpoints made before the float16 rule (PP-OCR: 80 activation quantizers,
now 73 and 7 only in the export) load with each range on the same tensor. Then:

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
| `ppocr` | text lines rendered on the fly (`util/ocr_synth.py`) | distillation: KL of the float and the fake-quant model's per-time-step class distributions (no labels), Adam | words / CER on `build/data/ocr-synth` | ONNX with Q/DQ pairs |

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
python qat_export.py yolox qat_yolox/best.pth build/onnx/yolox-qat-kld.onnx --act-qdq
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
build/onnx/yolox-qat-kld.onnx
build/kmodel/yolox-qat-kld.kmodel
k230
```

#### PP-OCR recognizer

The PaddleOCR English / number recognizer en_number_mobile_v2.0 (PP-OCRv1: MobileNetV3 with squeeze-excitation
blocks, two bidirectional LSTMs, CTC over 97 classes). The inputs:

* `build/onnx/ppocr-en-rec-32x320.onnx`, the float model (`--onnx`): RapidOCR's ONNX of it, cut before the final Softmax
  (output: the CTC logits `[1, 80, 97]`) and made static `1x3x32x320` by `convert/onnx_opt.py` (needs onnxsim):

  ```
  curl -L -o build/weights/ppocr/en_number_mobile_v2.0_rec_infer.onnx https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv1/en_number_mobile_v2.0_rec_infer.onnx
  python convert/onnx_opt.py build/weights/ppocr/en_number_mobile_v2.0_rec_infer.onnx build/onnx/ppocr-en-rec-32x320.onnx --input-shape 1,3,32,320 --cut ctc_fc.tmp_1 --output-shape 1,80,97 --optimize
  ```

  The character set (the model's `character` metadata) is built into `util/ocr_post.py`.
* `build/data/ocr-synth`, rendered text lines (`--data`): `test/*.u8` with `labels.json` (1000 lines, the words / CER
  evaluation) and `calib/*.png` (100 lines, the range calibration and nncase's `--dataset`). The fonts are those of
  the training lines (below); the set used for the results here was rendered with the Windows fonts:

  ```
  python -m util.ocr_synth build/data/ocr-synth
  ```

```
python qat_train.py ppocr --onnx build/onnx/ppocr-en-rec-32x320.onnx --data build/data/ocr-synth --epochs 0 --out qat_ppocr
python qat_train.py ppocr --onnx build/onnx/ppocr-en-rec-32x320.onnx --data build/data/ocr-synth --init qat_ppocr/init.pth -b 128 --epochs 20 --out qat_ppocr
python qat_export.py ppocr qat_ppocr/best.pth build/onnx/ppocr-en-rec-qat.onnx --act-qdq
```

The calibrated model scores what the board does (PTQ: 0.853 words / CER 0.029, the board 0.855 / 0.029; float
0.931 / 0.014). The default run reached its best after 3 epochs: 0.942 / 0.0105 in PyTorch, and its kmodel 0.942 /
0.0115 on the K230 simulator (bit-exact with the board), against 0.855 / 0.029 for the PTQ kmodel. The input is
mapped like the K230's pre-process, `(x - 128) / 127.5` (nncase's integer zero point; PaddleOCR uses 127.5).

Training speed: the LSTMs run step by step in Python, about 125 lines/s on the test PC at `-b 128` (`-b 256` does not
fit in its 6 GB); the default 20 x 1000 x 128 lines took 5.6 hours there (the best epoch was the third: `--epochs 5`
would have done). Fonts for the training lines:
`--fonts <dir>` (every `.ttf` / `.otf` in it), else the Windows fonts `build/data/ocr-synth` was made with, else the DejaVu
/ Liberation fonts of a Linux system (`apt-get install fonts-dejavu-core fonts-liberation`).

The PTQ build (the float model, calibrated by nncase on `build/data/ocr-synth/calib`; uint8 BGR input, which the
pre-process maps to -1..1):

```
compile
--dataset build/data/ocr-synth/calib
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
build/onnx/ppocr-en-rec-32x320.onnx
build/kmodel/ppocr-en-rec-32x320.kmodel
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
build/onnx/ppocr-en-rec-qat.onnx
build/kmodel/ppocr-en-rec-qat.kmodel
k230
```

That needs an nncase whose Q/DQ import (`QdqQuantSchemePass`) handles LSTM and MatMul weights and the pre-processed
input (the k230 branch of `<nncase>`); an older one fails on the LSTM with "Invalid w_quant_type".

#### Running QAT on RunPod

YOLOX training is CPU-bound (data augmentation): about 5 hours on a rented RTX 4090 pod with 10 vCPUs at `-b 16`, not
on the test PC. PP-OCR renders its training lines on the pod (no data set to upload) and fits the test PC's GPU as well.
Use a PyTorch template
with a network volume (`/workspace` survives pod restarts; everything else does not) and "SSH over exposed TCP": the
`ssh.runpod.io` proxy gives only an interactive shell, so `scp` and `rsync` fail through it. Placeholders:
`<pod-ip>` and `<port>` from the pod's Connect page, `<key>` the SSH key added to RunPod.

On the PC, pack this repository (git bundle: no remote needed) and copy it with what the task needs (YOLOX: the float
checkpoint; PP-OCR: the float model and `build/data/ocr-synth`, about 40 MB):

```
git bundle create model_runner.bundle master
scp -i <key> -P <port> model_runner.bundle yolox_nano.pth root@<pod-ip>:/workspace/
scp -i <key> -P <port> build/onnx/ppocr-en-rec-32x320.onnx root@<pod-ip>:/workspace/
scp -i <key> -P <port> -r build/data/ocr-synth root@<pod-ip>:/workspace/
ssh -i <key> -p <port> root@<pod-ip>
```

On the pod: system packages, this repository and its PP-OCR files; for YOLOX also the `yolox` package (its
`onnx-simplifier` pin does not build on recent Python and is not needed; OpenCV needs libGL) and COCO 2017 (about
20 GB):

```
apt-get update && apt-get install -y libgl1 libglib2.0-0 tmux unzip rsync fonts-dejavu-core fonts-liberation
pip install onnx onnxruntime onnx2torch pillow
cd /workspace && git clone model_runner.bundle model_runner && cd model_runner
mkdir -p build/onnx build/data && mv ../ppocr-en-rec-32x320.onnx build/onnx/ && mv ../ocr-synth build/data/ && mv ../yolox_nano.pth .
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
python qat_train.py ppocr --onnx build/onnx/ppocr-en-rec-32x320.onnx --data build/data/ocr-synth -b 128 --epochs 20 --workers 8 --eval-first --out qat_ppocr 2>&1 | tee train-ppocr.log
```

If the pod restarts, continue with `--resume <out>/latest.pth` and the same other arguments. `nvidia-smi -l 1` shows
the GPU load (YOLOX: expect well under 100%, the CPUs are the limit). Back on the PC, fetch the run and export it:

```
scp -i <key> -P <port> -r root@<pod-ip>:/workspace/model_runner/qat_yolox .
python qat_export.py yolox qat_yolox/best.pth build/onnx/yolox-qat-kld.onnx --act-qdq
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
build/onnx/yolov3-12-int8-heads.onnx
build/kmodel/yolov3-12-int8.kmodel
k230
```

```
compile
--model-quant-mode UsePTQ
build/onnx/yolo-fastest-1.1-320_int8_qop.onnx
build/kmodel/yolo-fastest-1.1-320-qop.kmodel
k230
```

### Lite-Mono

Monocular depth: [Lite-Mono](https://github.com/noahzn/Lite-Mono) (MIT), trained on KITTI, encoder and depth decoder as
one model with the full resolution disparity as output, `disp [1, 1, 192, 640]`. Needs the repository's
`networks/depth_encoder.py`, `networks/depth_decoder.py` and `layers.py` in `build/weights/lite-mono/upstream` and `encoder.pth`
and `depth.pth` of a variant in `build/weights/lite-mono/<variant>` (the table in its README links them), and two KITTI raw
drives with the calibration files, unpacked in `build/data/kitti` (they are in the Eigen test split, so the model did not
train on them; drive 0002 calibrates, drive 0048 measures):

```
python convert/litemono_to_onnx.py build/weights/lite-mono/lite-mono-tiny build/onnx/lite-mono-tiny-640x192.onnx --model lite-mono-tiny
python convert/onnx_opt.py build/onnx/lite-mono-tiny-640x192.onnx build/onnx/lite-mono-tiny-640x192-opt.onnx
python convert/kitti_prepare.py build/data/kitti build/data/kitti-lm --calib 2011_09_26_drive_0002 --test 2011_09_26_drive_0048
```

```
compile
--dataset build/data/kitti-lm/calib-zoom
--dataset-format Image
--calib-image-resize=Letterbox
--calib-bgr=false
--calib-method NoClip
--pre-process true
--input-type Uint8
--input-shape 1 3 192 640
--input-layout NCHW
--model-layout NCHW
--input-range 0 1
--model-quant-mode UsePTQ
build/onnx/lite-mono-tiny-640x192-opt.onnx
build/kmodel/lite-mono-tiny-640x192-opt-u8-zoom.kmodel
k230
```

`calib-zoom` matters: the kmodel cannot output anything nearer than the calibration images show, and the 20 plain
images cap the disparity at 0.685, about 5 m at KITTI scale. Zoomed crops raise that to 0.88.

On the board or in the simulator, then on the PC:

```
python eval/litemono_run.py build/kmodel/lite-mono-tiny-640x192-opt-u8-zoom.kmodel --input build/data/kitti-lm/test --output build/data/kitti-lm/disp-tiny.npz
python eval/litemono_eval.py --input build/data/kitti-lm/test --onnx build/onnx/lite-mono-tiny-640x192-opt.onnx --disp build/data/kitti-lm/disp-tiny.npz
```

22 frames of drive 0048 against the LiDAR, per image median scaling (not comparable with the paper's table). The board
times need the k230 branch of `<nncase>`: its runtime has the fast reflect Pad, Broadcast and GetItem, its compiler
`CombineQuantizePad`:

| Model | Runs on | abs-rel | abs-rel under 5 m | Time per image |
|---|---|---|---|---|
| lite-mono-tiny, float | onnxruntime | 0.107 | 0.256 | |
| lite-mono-tiny, PTQ, rewritten | board | 0.112 | 0.289 | 61 ms |
| lite-mono, float | onnxruntime | 0.117 | 0.262 | |
| lite-mono, PTQ, rewritten, without `normalize_after_matmul` and `CombineQuantizePad` | board | 0.120 | 0.307 | 111 ms |

The frame to frame sd of abs-rel is about 0.03. Abs-rel is the mean error of the distance relative to the true distance:
0.112 is about 11%.

How lite-mono-tiny got from 527 ms to 61 ms, on the same board, without changing what the model computes:

| Change | Where | Time per image |
|---|---|---|
| Starting point | | 527 ms |
| Reflect Pad copies whole rows | runtime | 270 ms |
| Broadcast fills whole rows | runtime | 152 ms |
| GetItem copies one block | runtime | 126 ms |
| [Graph rewrites](#optimizing-a-model-with-graph-rewrites): Linears as 1x1 Convs, q, k, v as three Convs | `onnx_opt.py` | 78 ms |
| `normalize_after_matmul` | `onnx_opt.py` | 73 ms |
| `CombineQuantizePad`: uint8 instead of float32 through the Pads | nncase | 61 ms |

## Optimizing a model with graph rewrites

`convert/onnx_opt.py` rewrites a model into a form the K230 runs faster, without changing what it computes:

```
python convert/onnx_opt.py build/onnx/lite-mono-tiny-640x192.onnx build/onnx/lite-mono-tiny-640x192-opt.onnx
python convert/onnx_opt.py build/onnx/some-model.onnx --lint
```

The first line writes the new model and reports which rules fired. The second changes nothing and only reports what is
in the model that will cost time. This section explains the idea without assuming a machine learning background.

### Why a correct model can be slow

An ONNX file describes a network as a graph: boxes are operations, arrows are tensors (arrays of numbers) flowing
between them.

```
image --> [Conv] --> [GELU] --> [Conv] --> depth map
```

The K230's NPU, the KPU, is very fast at arithmetic and has a small fast memory of its own. Everything else lives in
DDR, the main memory, which is much slower. nncase groups boxes into calls. Inside a call the data stays on the chip,
between calls it goes out to DDR and comes back:

```
three calls:   DDR -> [Conv] -> DDR -> [GELU] -> DDR -> [Conv] -> DDR       6 trips through DDR

one call:      DDR -> [Conv -> GELU -> Conv] -> DDR                         2 trips through DDR
```

That matters more than the arithmetic. Before these rewrites Lite-Mono spent about 95 ms per image waiting for the KPU,
and of that about 77 ms was moving data and only about 3.5 ms was arithmetic. So the aim is: **fewer trips to DDR**.

### What stops boxes from merging

A picture can be stored in two ways. NCHW keeps each colour in its own plane, NHWC keeps the colours of a pixel
together:

```
NCHW                     NHWC

R R R R                  RGB RGB RGB RGB
R R R R                  RGB RGB RGB RGB
G G G G
G G G G
B B B B
B B B B
```

* A **Transpose** converts between such layouts. It moves every value, so it is a full pass over the tensor.
* A **Reshape** keeps the values in place and gives them a new shape, as `[48, 160]` to `[7680]`.

The KPU's Conv works on NCHW. nncase splits a group of boxes into tiles that fit on the chip, and follows each tile
backwards through the boxes. It cannot do that through a Reshape, so **a Reshape between two boxes ends the call**, and
the data goes to DDR. Many networks written in PyTorch are full of these, because it is natural to write them that way.

### What a rewrite rule is

The rewriter does what a person would do with the drawing: find a small known shape in the graph, and replace it with
a better one. Every rule in `util/onnx_rewrite.py` is two small functions:

```
match(graph, node)     do the nodes around this one look like my pattern?
generate(graph, m)     what should replace them?
```

For example, a Transpose to NHWC followed by one back to NCHW does nothing:

```
before:   x -> [to NHWC] -> [to NCHW] -> y

after:    x -> y
```

`rewrite()` tries the rules in order until none fires. After each one it runs the graph on a random input with
onnxruntime and compares the output with the original model's, and stops with an error if it changed. The longest
patterns, GELU and LayerNorm, are written with `onnxscript`, which looks like the formula itself:

```python
def layernorm_pattern(op, x, two, eps, gamma, beta):
    d = op.Sub(x, op.ReduceMean(x, axes=[-1], _allow_other_attributes=True))
    var = op.ReduceMean(op.Pow(d, two), axes=[-1], _allow_other_attributes=True)
    return op.Add(op.Mul(op.Div(d, op.Sqrt(op.Add(var, eps))), gamma), beta)
```

### The rewrites, one by one

**A Linear is a 1x1 Conv.** A Linear layer mixes the channels of each pixel separately. A 1x1 Conv does the same thing:
it looks at one pixel at a time, with the same weights. They only differ in the layout they expect. PyTorch's Linear
wants NHWC, so Lite-Mono's blocks are wrapped in Transposes, and nncase then adds Reshapes around every Linear:

```
before   x -> [to NHWC] -> [Linear] -> [GELU] -> [Linear] -> [x gamma] -> [to NCHW] -> y

1.       `matmul_to_conv`: each Linear becomes a Conv with a Transpose on each side

         x -> [to NHWC] -> [to NCHW] -> [Conv] -> [to NHWC] -> [GELU] -> [to NCHW] -> [Conv] -> [to NHWC] -> ...

2.       `cancel_transposes`, `sink_gelu`, `sink_elementwise`: Transposes move along and cancel in pairs

3.       `fold_conv_scale`: multiplying by a constant is done by changing the Conv's weights

after    x -> [Conv] -> [GELU] -> [Conv] -> y
```

There is nothing left to stop those boxes merging: Lite-Mono's small blocks went from about five KPU calls to two.

**Constants fold into a Conv.** A Conv already multiplies and adds, so a following Add or Mul by a constant can be
built into its weights ahead of time (`fold_conv_bias`, `fold_conv_scale`, `fold_conv_reshape_scale`):

```
before   x -> [Conv] -> [+ b] -> [x gamma] -> y

after    x -> [Conv with new weights] -> y
```

**One Conv and a split become several Convs.** nncase runs Slice, Split and Gather on the CPU, which cuts the graph in
two. But each part of a Conv's output comes from its own part of the weights, so the Conv can be cut instead
(`split_silu_conv`, `split_qkv_heads`, `split_qkv`):

```
before                                     after

x -> [Conv] -> [Split] --> a               x --> [Conv, first half]  --> a
                       \-> b                 \-> [Conv, second half] --> b
```

**An Expand that only copies goes away.** Dividing a table by one number per row does not need that number written out
for every column first, Div does that by itself (`drop_expand`):

```
before   n -> [Expand to the size of x] -> [x / n]

after    n -> [x / n]
```

**Divide the small result, not the large inputs.** Lite-Mono's attention compares channels with each other. It scales
every row of `q` and `k` to length 1 and then multiplies them. Scaling a row before a multiplication gives the same
answer as scaling the result afterwards, and the result is tiny (`normalize_after_matmul`):

```
before   q (245,760 numbers) -> [/ |q|] --\
                                           [MatMul] -> attention (128 numbers)
         k (245,760 numbers) -> [/ |k|] --/

after    q ----------\
                      [MatMul] -> [/ (|q| x |k|)] -> attention
         k ----------/
```

The rule only fires when the result really is much smaller: at least 16 tokens per channel of a head.

### All the rewrite rules

| Rule | Pattern | Result |
|---|---|---|
| `split_silu_conv` | 1x1 Conv, SiLU, Slice / Split on channels | one Conv + SiLU per part |
| `split_qkv_heads` | 1x1 Conv, Reshape, Split on the head axis | one Conv per part |
| `split_qkv` | MatMul, Add, Reshape, Transpose, three Gather | three 1x1 Convs |
| `matmul_to_conv` | MatMul with a constant on NHWC data | Transpose, 1x1 Conv, Transpose |
| `tokens_to_nhwc`, `heads_to_tokens` | token Reshape / Transpose pairs | NHWC Transpose + token Reshape |
| `sink_elementwise`, `sink_gelu`, `sink_layernorm`, `sink_matmul` | Transpose / token Reshape before an op | the same after it |
| `merge_add` | Add of two equal wrappers | one wrapper after the Add |
| `cancel_transposes`, `fold_reshapes` | NCHW / NHWC pairs, Reshape of a Reshape | gone |
| `fold_conv_bias`, `fold_conv_scale`, `fold_conv_reshape_scale` | Add / Mul with a constant after a Conv | in the Conv |
| `drop_expand` | Expand read as a divisor | gone, Div broadcasts |
| `drop_empty_clip_max` | `Clip(x, min, "")` | `Clip(x, min)`, which `onnx2torch` can load |
| `normalize_after_matmul` | `normalize(q) @ normalize(k)^T` | `(q @ k^T) / (\|q\| * \|k\|^T)`: the Div on the small matrix |

The rules are narrow on purpose: they match only the ops, layouts and shapes seen in YOLO26 and Lite-Mono, and do
nothing to PP-OCR and YOLOX. A close look-alike is left as it is. Add a pattern for a new model after looking at its graph,
rather than loosening a rule on a guess. `--lint` is the place to start. For plain Lite-Mono tiny it prints:

```
lint: 3 Reshape between compute ops: nncase cannot fuse across them, a DDR round trip each
lint: 48 Transpose: each a separate KPU pass or a fusion boundary
lint: 9 Gather: runs on the CPU
lint: 6 Expand: runs on the CPU, and copies
lint: 7 reflect Pad: runs on the CPU, AI2D gave zero borders in mirror mode
```

Its lines are hints from what was measured so far, not measurements of the new model. It counts ops and does not know
how large they are, and some Reshapes are really needed: attention needs its own layout.

### Where the rewrites fit in the workflow

The rules keep the float function, so the trained weights carry over. They run on the float model, before quantization:

```
.pth --> export --> onnx_opt.py --+--> nncase with calibration images --> kmodel       (PTQ)
                                  |
                                  +--> qat_train.py --> qat_export.py --> nncase --> kmodel    (QAT)
```

`qat_graph.load()` takes the rewritten ONNX, so QAT trains the graph the K230 runs. `onnx_opt.py` refuses a model that
already has Q/DQ nodes: the patterns do not match across a pair, and a fold would remove a tensor that has a range.

Changes that alter the function, like zero padding instead of reflect padding, do not belong here. They need the
accuracy checked, and usually training: Lite-Mono's abs-rel went from 0.107 to 0.112 in float with zero padding.

## Running a model on one input

`tensor_from_image.py` writes the raw input tensor, `run-onnx.py` / `run-model.py` run the model and, with
`--postprocess <detector>`, draw the detections into a tensor of the same layout, and `tensor_to_image.py` turns it
into an image:

| `--postprocess` | `tensor_from_image.py` options | `tensor_to_image.py` options |
|---|---|---|
| `yolox` | `416x416 --letterbox --bgr --type f32` | `416x416 f32 --bgr` |
| `yolov3` | `416x416 --letterbox --pad 128 --range 1 --type f32` | `416x416 f32 --range 1` |
| `yolo-fastest` | `320x320 --range 1 --type f32` | `320x320 f32 --range 1` |
| `yolo26` | `640x640 --letterbox --pad 114 --type f32` (`--type u8` for the u8 kmodel) | `640x640 f32` (`u8`) |

Example (YOLO26):

```
python tensor_from_image.py build/data/test.jpg 640x640 --letterbox --pad 114 --type f32 --output build/data/yolo26/yolo26-f32.in
python run-onnx.py build/onnx/yolo26n-640-nosplit.onnx build/data/yolo26/yolo26-f32.in --postprocess yolo26 --output-prefix=build/data/yolo26/out-onnx-
python tensor_to_image.py build/data/yolo26/out-onnx-yolo26.rgb 640x640 f32 --format bmp --output build/data/yolo26/out-onnx-yolo26.bmp
```

On the board (or in the simulator) with the u8 kmodel:

```
python tensor_from_image.py build/data/test.jpg 640x640 --letterbox --pad 114 --type u8 --output build/data/yolo26/yolo26-u8.in
python3 run-model.py yolo26n-640-nosplit-u8-ptq.kmodel yolo26-u8.in --postprocess yolo26 --output-prefix out-
```

These single-image inputs use Pillow's resize, so they differ slightly from the COCO inputs below.

Text recognition (PP-OCR): `--postprocess ppocr` decodes the CTC output and prints the text (`util/ocr_post.py`). The
input is a uint8 BGR `1x3x32x320` text line, e.g. a `build/data/ocr-synth/test/*.u8` file (labels in
`build/data/ocr-synth/labels.json`) or `tensor_from_image.py <line image> 320x32 --bgr` (plain resize: crop the image
to one line of text first). The kmodel (built with `--pre-process`) takes the file as it is; `run-onnx.py
--input-range -1 1` maps it like the kmodel's pre-process, `(x - 128) / 127.5`, for the float or QAT ONNX:

```
python run-onnx.py build/onnx/ppocr-en-rec-qat.onnx build/data/ocr-synth/test/0001.u8 --input-range -1 1 --postprocess ppocr
python run-model.py build/kmodel/ppocr-en-rec-qat.kmodel build/data/ocr-synth/test/0001.u8 --postprocess ppocr
```

`run-onnx.py` runs onnxruntime without graph optimizations, which would turn the Q/DQ pairs of a QAT export into
integer kernels that round differently.

## Measuring detection accuracy on COCO val2017

The same 5000 prepared input tensors go to the kmodel and to the ONNX model; the decoding, thresholds and NMS are the
model's own evaluation settings (`PREP` in `util/coco.py`).

1. Prepare the inputs with the model's own evaluation preprocessing (5000 images; about 6 GB for 640×640):

   ```
   python eval/coco_prepare.py --detect yolo26 --output build/data/coco-yolo26
   ```

2. Run the kmodel on the board and fetch the candidates. Only output cells whose score can reach the evaluation
   threshold are kept, as float16:

   ```
   tar cf - -C build/data coco-yolo26 | ssh root@<board> "tar xmf - -C <bdir>/data"
   scp build/kmodel/yolo26n-640-nosplit-u8-ptq.kmodel root@<board>:<bdir>/
   ssh root@<board> "cd <bdir> && python3 coco_run.py yolo26n-640-nosplit-u8-ptq.kmodel --input data/coco-yolo26 --output dets/yolo26.npz"
   scp root@<board>:<bdir>/dets/yolo26.npz build/data/dets/
   ```

   `eval/coco_run.py` also works with the simulator (slow: use `--limit N`).

3. Score the kmodel candidates, and optionally ONNX models on the same tensors:

   ```
   python eval/coco_eval.py --input build/data/coco-yolo26 build/data/dets/yolo26.npz --onnx build/onnx/yolo26n-640-nosplit.onnx
   ```

Evaluation settings per model:

| `--detect` | conf | NMS IoU | labels per box |
|---|---|---|---|
| `yolox` | 0.01 | 0.65 | best class |
| `yolov3`, `yolo-fastest` | 0.005 | 0.45 | every class above conf |
| `yolo26` | 0.001 | 0.7 | every class above conf (Ultralytics val; square 640 input instead of rectangular batches) |

## COCO val2017 results

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

## Debugging with runtime and simulator dumps

The K230 simulator gives the board's results bit for bit, so a kmodel can be debugged on the PC. nncase's Debug
runtime dumps the tensors of every CPU op, including the inputs and outputs of each KPU call, and with
`NNCASE_SIM_CHECKPOINTS=1` the simulator also dumps what the KPU's activation units compute inside a call (one record
per tile: the convolution's integer sums, the dequantisation / activation parameters and the outputs). How to make
the dumps and how to read them: "Debug dumps and check points" in `<nncase>/src/Native/src/simulator/README.md`;
`<nncase>/tools/ckpt_dump.py` prints them in readable form:

```
python <nncase>/tools/ckpt_dump.py 'dumps/Runtime/1$1_7.ckpt/act_dm_write.chk' --kind P --record 0
```

## Running the tests

```
python -m pytest tests
```

needs pytest; no data files or models. `tests/test_tensor_image.py` runs `tensor_from_image.py` and
`tensor_to_image.py` on small generated images: layout, channel order, element types and value range, letterbox
padding, transparency and EXIF orientation, the round trip back to the same bytes, rounding / clipping and the
errors. `tests/test_yolo_post.py` encodes known boxes into the raw head outputs of every `--postprocess` detector
(YOLOX, YOLOv3, Yolo-Fastest, YOLO26 split / joined / DFL; square and rectangular inputs) and compares the printed
detections with them: every stride, per-class NMS, the score threshold, outputs in any order, and the errors.
`tests/test_ocr_post.py` does the same for `--postprocess ppocr`: text encoded as CTC logits (repeats, blanks, every
character, the longest line), the printed text and confidence, and the errors; one test compares the built-in
character set with the PaddleOCR model's metadata and is skipped without `build/weights/ppocr`.
`tests/test_qat_graph.py` writes small models in ONNX text syntax (conv / activation chains, SiLU and HardSwish
patterns, MaxPool, Concat, nearest Resize, residual Adds, an activation beside another reader, squeeze-excitation,
pooling + Gemm, LSTM), runs `qat_graph.load` and `make_quantizable`, exports them with `--act-qdq` and compares the
exported graph, printed in a compact text form, with where the quantizers and shared ranges belong (the float16
cases as nncase's quantized IR stores them, with the Q/DQ pairs of the export); also the per-channel weight grid, the
loading of checkpoints of earlier placements and that a float16 tensor is not rounded outside the export. Needs torch,
onnx and onnx2torch.
`tests/test_qat_calibrate.py` runs `qat_generic.calibrate` on the same models and compares every range group, for
noclip and KLD, with the float tensors from onnxruntime; also the state after calibration and degenerate tensors.
`kld_range` is compared with thresholds taken from nncase's own `Nncase.Quantization.dll`.
`tests/test_onnx_rewrite.py` runs the rules of `util/onnx_rewrite.py` on small models in ONNX text syntax: the patterns
they were written for and close variants that must stay as they are, the lint report and a wrong rule being caught.
`tests/test_coco.py` runs `util/coco.py` on generated images, raw outputs and annotations: the input geometry of
every detector against sizes worked out by hand, cells and candidates, NMS, the mapping back to the image, AP with
known values, and a white rectangle followed from an image through to COCO results. Needs Pillow, OpenCV and
pycocotools.
