#!/usr/bin/env python3
"""YOLOX for QAT (qat_train.py yolox, qat_export.py yolox): the fake-quant model, data, loss, COCO evaluation, export.

The network is the float YOLOX with BatchNorm folded; YoloxBody (backbone + head convs, raw per-level outputs) is
traced with torch.fx and qat_graph.py puts the fake quantizers in it by the graph rules (the same 111 activation
quantizers in 99 range groups as the former hand-written placement: every conv output after SiLU, the residual adds,
the concat inputs sharing one range; the head's prediction convs keep float outputs). QatYolox wraps the traced body
with YOLOX's own head code for the loss (training) and the box decoding (evaluation). Needs the yolox package.
"""
import copy
import os
import random
import shutil

import torch
import torch.fx
import torch.nn as nn

from util import qat_generic, qat_graph
from util.qat_train import Task


class YoloxBody(nn.Module):
    """Backbone + head convs of a BatchNorm folded YOLOX model: per level the raw [reg (4), obj (1), cls] output, before
    any sigmoid. Traceable by torch.fx (YOLOXHead.forward is not: its loss is data dependent)."""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, x):
        fpn = self.backbone(x)
        h = self.head
        outs = []
        for k in range(len(h.stems)):
            f = h.stems[k](fpn[k])
            cls_feat = h.cls_convs[k](f)
            reg_feat = h.reg_convs[k](f)
            outs.append(torch.cat([h.reg_preds[k](reg_feat), h.obj_preds[k](reg_feat), h.cls_preds[k](cls_feat)], 1))
        return outs


class QatYolox(nn.Module):
    """The traced, fake-quantized YoloxBody (`body`) with YOLOXHead.forward's code around it: model(x, targets) in
    training mode returns YOLOX's loss dict, model(x) in eval mode the [B, N, 85] outputs, decoded unless
    head.decode_in_inference is False (the ONNX export)."""

    def __init__(self, body, head):
        super().__init__()
        self.body = body
        object.__setattr__(self, "head", head)  # its methods and settings only; its convs are in body

    def forward(self, x, targets=None):
        outs = self.body(x)
        head = self.head
        dtype = outs[0].type()
        if self.training:
            outputs, origin_preds, x_shifts, y_shifts, strides = [], [], [], [], []
            for k, (output, stride) in enumerate(zip(outs, head.strides)):
                if head.use_l1:
                    b, _, hsize, wsize = output.shape
                    reg = output[:, :4].view(b, 1, 4, hsize, wsize).permute(0, 1, 3, 4, 2).reshape(b, -1, 4)
                    origin_preds.append(reg.clone())
                output, grid = head.get_output_and_grid(output, k, stride, dtype)
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                strides.append(torch.zeros(1, grid.shape[1]).fill_(stride).type_as(outs[0]))
                outputs.append(output)
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = head.get_losses(
                x, x_shifts, y_shifts, strides, targets, torch.cat(outputs, 1), origin_preds, dtype=outs[0].dtype)
            return {"total_loss": loss, "iou_loss": iou_loss, "l1_loss": l1_loss, "conf_loss": conf_loss,
                    "cls_loss": cls_loss, "num_fg": num_fg}
        outs = [torch.cat([o[:, :4], o[:, 4:5].sigmoid(), o[:, 5:].sigmoid()], 1) for o in outs]
        head.hw = [o.shape[-2:] for o in outs]
        outputs = torch.cat([o.flatten(start_dim=2) for o in outs], dim=2).permute(0, 2, 1)
        return head.decode_outputs(outputs, dtype=dtype) if head.decode_in_inference else outputs


def fold(model):
    """BatchNorm folding with YOLOX's own function, on a copy; every parameter a trainable leaf again (fuse_model
    freezes the fused convs and leaves their weights attached to the BN graph)."""
    from yolox.utils import fuse_model

    with torch.no_grad():
        folded = fuse_model(copy.deepcopy(model))
    for m in folded.modules():
        for name, prm in list(m._parameters.items()):
            if prm is not None and not prm.is_leaf:
                m._parameters[name] = nn.Parameter(prm.detach().clone())
    for prm in folded.parameters():
        prm.requires_grad_(True)
    return folded


def make_quantizable(folded):
    """QatYolox of a folded YOLOX model, the quantizers disabled."""
    body = qat_graph.make_quantizable(torch.fx.symbolic_trace(YoloxBody(folded)))
    return QatYolox(body, folded.head)


def legacy_state_dict(folded, old):
    """The state_dict of a checkpoint of the former hand-written placement (QBaseConv / QBottleneck: <conv>.out_q,
    <bottleneck>.add_q) in the layout of make_quantizable(folded). The weights keep their names (prefixed with
    "body."), each range goes to the quantizer of the same tensor."""
    gm = torch.fx.symbolic_trace(YoloxBody(folded))
    points, _, _ = qat_graph.plan(gm)
    new = {"body." + k: v for k, v in old.items() if ".out_q." not in k and ".add_q." not in k}
    for i, n in enumerate(points):
        if n.op == "call_module" and n.target.endswith(".act"):
            src = n.target[:-len(".act")] + ".out_q"
        else:
            src = list(n.meta["nn_module_stack"])[-1] + ".add_q"
        for field in ("min_val", "max_val", "seen"):
            new[f"body.actq_{i}.params.{field}"] = old[f"{src}.params.{field}"]
    return new


def make_calib_loader(exp, n_images, batch_size, workers, seed=0):
    from yolox.data import COCODataset, ValTransform

    ds = COCODataset(
        data_dir=exp.data_dir,
        json_file=exp.train_ann,
        name="train2017",
        img_size=exp.test_size,
        preproc=ValTransform(legacy=False),  # same preprocessing as evaluation: raw 0..255 BGR, no normalization
    )
    idx = random.Random(seed).sample(range(len(ds)), min(n_images, len(ds)))
    sub = torch.utils.data.Subset(ds, idx)
    return torch.utils.data.DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=workers)


def _use_standard_cocoeval_without_compiler():
    """YOLOX's evaluator JIT-compiles a C++ COCOeval and fails without a compiler (e.g. no MSVC on Windows); its own
    fallback only catches ImportError. Use pycocotools' COCOeval then (same AP, slower)."""
    import yolox.layers
    from pycocotools.cocoeval import COCOeval

    if shutil.which("cl" if os.name == "nt" else "c++") is None:
        yolox.layers.COCOeval_opt = COCOeval


def evaluate(exp, model, batch_size):
    """(AP 0.50:0.95, AP50) on val2017."""
    _use_standard_cocoeval_without_compiler()
    evaluator = exp.get_evaluator(batch_size, False)
    with torch.no_grad():
        ap50_95, ap50, _ = evaluator.evaluate(model, False, False, None, None, exp.test_size)
    return ap50_95, ap50


class YoloxTask(Task):
    """YOLOX on COCO with YOLOX's own training recipe and loss (labels)."""

    name = "yolox"
    description = ("YOLOX (default yolox-nano) on COCO: YOLOX's loss and data pipeline (mosaic / mixup, switched off "
                   "for the last --no-aug-epochs epochs, with the L1 loss), SGD, AP on val2017; the stem conv frozen")
    default_lr = 1e-3
    default_batch_size = 16
    default_epochs = 15
    default_clip_grad = 0.0

    def add_args(self, ap):
        ap.add_argument("-n", "--name", default="yolox-nano", help="YOLOX experiment name")
        ap.add_argument("-f", "--exp-file", default=None, help="YOLOX experiment file (instead of --name)")
        ap.add_argument("-c", "--ckpt", default=None, help="float YOLOX checkpoint (e.g. yolox_nano.pth)")
        ap.add_argument("--calib-images", type=int, default=512, help="random train2017 images to calibrate on")
        ap.add_argument("--no-aug-epochs", type=int, default=3, help="last epochs without mosaic/mixup, with L1 loss")
        ap.add_argument("--eval-batch-size", type=int, default=64)

    def exp(self, args):
        from yolox.exp import get_exp

        if not hasattr(self, "_exp"):
            self._exp = get_exp(args.exp_file, args.name)
            self._exp.data_num_workers = getattr(args, "workers", 4)
        return self._exp

    def build(self, args, device, load_float_weights):
        model = self.exp(args).get_model()
        if load_float_weights:
            if not args.ckpt:
                raise SystemExit("yolox: -c / --ckpt (the float checkpoint) is needed without --init / --resume")
            model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"])
        model.eval()
        self._folded = fold(model)
        return make_quantizable(self._folded).to(device)

    def load_state_dict(self, model, state_dict):
        if not any(k.startswith("body.") for k in state_dict):
            state_dict = legacy_state_dict(self._folded, state_dict)
        model.load_state_dict(state_dict)

    def calib_loader(self, args):
        return make_calib_loader(self.exp(args), args.calib_images, 32, args.workers)

    def frozen_parameters(self, model, args):
        # The stem conv sees raw 0..255 pixels: with BatchNorm folded its weights are tiny and their gradients ~255x
        # larger than elsewhere, so one shared lr wrecks it within a few dozen steps.
        return list(model.body.backbone.backbone.stem.parameters())

    def base_lr(self, args):
        return args.lr * args.batch_size / 64.0

    def optimizer(self, params, lr, args):
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)

    def _loader(self, args, no_aug):
        from yolox.data import DataPrefetcher

        self._train_loader = self.exp(args).get_data_loader(batch_size=args.batch_size, is_distributed=False,
                                                            no_aug=no_aug, cache_img=None)
        self._prefetcher = DataPrefetcher(self._train_loader)
        self._no_aug = no_aug

    def iters_per_epoch(self, args):
        if not hasattr(self, "_train_loader"):
            self._loader(args, False)
        return len(self._train_loader)

    def schedule(self, args, iters_per_epoch):
        return int(0.3 * iters_per_epoch), 2, args.no_aug_epochs * iters_per_epoch  # YOLOX's yoloxwarmcos

    def batches(self, model, args, epoch, iters):
        no_aug = epoch >= args.epochs - args.no_aug_epochs
        if not hasattr(self, "_train_loader"):
            self._loader(args, no_aug)
        elif no_aug and not self._no_aug:
            print("---> no mosaic/mixup from now on, L1 loss added")
            self._train_loader.close_mosaic()
            self._no_aug = True
        model.head.use_l1 = no_aug
        for _ in range(iters):
            yield self._prefetcher.next()

    def loss(self, model, batch, args):
        exp = self.exp(args)
        inps, targets = batch
        targets = targets.float()
        targets.requires_grad = False
        inps, targets = exp.preprocess(inps.float(), targets, exp.input_size)
        out = model(inps, targets)
        return out["total_loss"], {k: float(out[k]) for k in ("total_loss", "iou_loss", "conf_loss", "cls_loss",
                                                               "l1_loss")}

    def evaluate(self, model, args):
        ap50_95, ap50 = evaluate(self.exp(args), model, args.eval_batch_size)
        return {"AP": ap50_95, "AP50": ap50}

    export_input_names = ("images",)
    export_output_names = ("output",)

    def export_input_shape(self, args):
        return (1, 3, *self.exp(args).test_size)

    def add_export_args(self, ap):
        ap.description = (
            "ONNX of the fake-quant model: input 1x3xHxW float32 (raw 0..255 BGR), output [1, N, 85] without box "
            "decoding (raw dx, dy, log w, log h, sigmoid objectness, sigmoid class scores).")
        ap.add_argument("-n", "--name", default="yolox-nano", help="YOLOX experiment name (older checkpoints)")
        ap.add_argument("-f", "--exp-file", default=None, help="YOLOX experiment file (older checkpoints)")
        super().add_export_args(ap)

    def export_onnx(self, model, path, args, example):
        head = model.head
        old_decode = head.decode_in_inference
        head.decode_in_inference = False
        try:
            super().export_onnx(model, path, args, example)
        finally:
            head.decode_in_inference = old_decode
