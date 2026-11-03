#!/usr/bin/env python3
"""Export a QAT checkpoint of qat_train.py (best.pth, latest.pth or init.pth) for nncase:

    python qat_export.py yolox qat_yolox/best.pth onnx/yolox-qat-kld.onnx --act-qdq
    python qat_export.py ppocr qat_ppocr/best.pth onnx/ppocr-en-rec-qat.onnx --scheme QuantScheme.json
    python qat_export.py <task> -h                    # the task's export options

yolox: ONNX with QuantizeLinear/DequantizeLinear pairs that nncase reads (+ <output>.qs.json); checkpoints of the
former yolox_qat.py / yolox_qat_train.py load too. ppocr: the float ONNX with the trained weights + <output>.qs.json
to compile it with (--quant-scheme). The exports are the tasks' (util/qat_yolox.py, util/qat_ppocr.py).
"""
from util.qat_ppocr import PpocrTask
from util.qat_train import export, parse_export_args
from util.qat_yolox import YoloxTask

TASKS = [YoloxTask(), PpocrTask()]


def main():
    export(TASKS, parse_export_args(TASKS, description=__doc__))


if __name__ == "__main__":
    main()
