#!/usr/bin/env python3
"""Quantization-aware training for the K230 (nncase uint8): one script for every supported model.

    python qat_train.py yolox -c yolox_nano.pth --epochs 0 --out qat_yolox          # calibrate + PTQ AP only
    python qat_train.py yolox -c yolox_nano.pth --epochs 15 --eval-first --out qat_yolox
    python qat_train.py ppocr --epochs 0 --out qat_ppocr                             # calibrate + PTQ words/CER
    python qat_train.py ppocr --epochs 20 --eval-first --out qat_ppocr
    python qat_train.py <task> -h                                                    # the task's options

The fake quantizers are placed by graph rules (util/qat_graph.py: where the KPU quantizes), the activation ranges
calibrated like nncase's PTQ (--calib kld, its default; init.pth = the calibrated model) or loaded (--init, --scheme),
then the weights train with fixed ranges; latest.pth / best.pth / ranges_best.json per epoch in --out. Export with
qat_export.py. The common code is util/qat_train.py, the model specific parts util/qat_yolox.py and
util/qat_ppocr.py.
"""
from util.qat_ppocr import PpocrTask
from util.qat_train import parse_args, run
from util.qat_yolox import YoloxTask

TASKS = [YoloxTask(), PpocrTask()]


def main():
    args = parse_args(TASKS)
    run({t.name: t for t in TASKS}[args.task], args)


if __name__ == "__main__":
    main()
