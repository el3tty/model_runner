#!/usr/bin/env python3
"""Export a QAT checkpoint of qat_train.py (best.pth, latest.pth or init.pth) for nncase:

    python qat_export.py yolox qat_yolox/best.pth onnx/yolox-qat-kld.onnx --act-qdq
    python qat_export.py ppocr qat_ppocr/best.pth onnx/ppocr-en-rec-qat.onnx --act-qdq
    python qat_export.py <task> -h                    # the task's export options

The ONNX of the fake-quant model: --act-qdq puts QuantizeLinear/DequantizeLinear pairs with the trained ranges on the
activations, so nncase compiles it without calibration data or a quant scheme; the weights are baked onto their
per-channel grid (--weight-qdq keeps Q/DQ nodes on them). Checkpoints of the former yolox_qat.py / yolox_qat_train.py
load too. The exports are the tasks' (util/qat_yolox.py, util/qat_ppocr.py).
"""
from util.qat_ppocr import PpocrTask
from util.qat_train import export, parse_export_args
from util.qat_yolox import YoloxTask

TASKS = [YoloxTask(), PpocrTask()]


def main():
    export(TASKS, parse_export_args(TASKS, description=__doc__))


if __name__ == "__main__":
    main()
