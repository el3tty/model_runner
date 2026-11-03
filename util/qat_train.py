#!/usr/bin/env python3
"""Model independent QAT for the K230: the drivers of qat_train.py and qat_export.py.

A Task (util/qat_yolox.py, util/qat_ppocr.py) supplies what differs between models: the fake-quant model, calibration
images, training batches, loss, metric, optimizer, export. Everything else is here:
  * the start: the float model with the fake quantizers of qat_graph.py (weights per channel, activations where the
    KPU stores uint8), the activation ranges calibrated like nncase's PTQ (qat_generic.calibrate: --calib kld|noclip),
    or taken from an nncase quant scheme (--scheme, if the task supports it), or from a checkpoint (--init);
    the calibrated model is saved as <out>/init.pth;
  * training: the ranges stay fixed, the weights train; warmup (linear or quadratic) + cosine lr down to 5% of the
    base lr, optionally the last iterations at that minimum; gradient clipping; --resume;
  * after every epoch: the task's metric, <out>/latest.pth (with the optimizer), best.pth and ranges_best.json.
A checkpoint stores the task name and its arguments, so load_checkpoint() rebuilds the model without them; export()
loads one and hands it to the task's export.
"""
import argparse
import math
import os
import time

import torch

from util import qat_generic
from util.qat_generic import describe, dump_ranges, set_state

MIN_LR_RATIO = 0.05
PRINT_INTERVAL = 50


class Task:
    """What a model needs for QAT. Subclasses set name / default_lr and implement the methods marked as required."""

    name = None
    description = ""
    default_lr = 1e-3
    default_batch_size = 16
    default_epochs = 10
    default_iters_per_epoch = 0  # 0 = the task's own epoch length
    default_clip_grad = 0.0
    supports_scheme = False

    def add_args(self, ap):
        """Task specific command line arguments."""

    def build(self, args, device, load_float_weights):
        """Required. The model with fake quantizers (disabled), on device; float weights loaded if asked."""
        raise NotImplementedError

    def load_state_dict(self, model, state_dict):
        """Load a checkpoint's model state (a task may convert older formats here)."""
        model.load_state_dict(state_dict)

    def calib_loader(self, args):
        """Required for calibration. Batches whose [0] is the model input, as the K230 sees it."""
        raise NotImplementedError

    def load_scheme(self, model, args):
        """Set the activation ranges from args.scheme (if supports_scheme). Returns the number of range groups set."""
        raise NotImplementedError

    def frozen_parameters(self, model, args):
        return []

    def base_lr(self, args):
        return args.lr

    def optimizer(self, params, lr, args):
        """Required."""
        raise NotImplementedError

    def iters_per_epoch(self, args):
        """Required."""
        raise NotImplementedError

    def schedule(self, args, iters_per_epoch):
        """(warmup iterations, warmup power, iterations at the minimum lr at the end)."""
        return min(300, iters_per_epoch), 1, 0

    def batches(self, model, args, epoch, iters):
        """Required. Iterator of `iters` training batches of `epoch` (may switch the data pipeline / model mode)."""
        raise NotImplementedError

    def loss(self, model, batch, args):
        """Required. (loss tensor, {name: float} to log)."""
        raise NotImplementedError

    def evaluate(self, model, args):
        """Required. {metric: value}; the model is in eval mode with its quantizers on."""
        raise NotImplementedError

    def score(self, metrics):
        """Sort key of metrics (larger is better) for best.pth."""
        return tuple(metrics.values())

    def add_export_args(self, ap):
        """The task's qat_export.py arguments."""

    def export(self, model, ckpt, args):
        """Required for qat_export.py. Write args.output (and whatever nncase needs with it) from the loaded model
        (eval mode, quantizers on) and its checkpoint."""
        raise NotImplementedError


def format_metrics(metrics):
    return "  ".join(f"{k} {v:.4f}" for k, v in metrics.items())


def lr_at(step, total, base, warmup, power=1, tail=0):
    """warmup ((step / warmup) ** power), cosine down to MIN_LR_RATIO * base, the last `tail` steps at that minimum
    (YOLOX's yoloxwarmcos with power 2)."""
    low = base * MIN_LR_RATIO
    if step < warmup:
        return base * ((step + 1) / warmup) ** power
    if step >= total - tail:
        return low
    t = (step - warmup) / max(1, total - warmup - tail)
    return low + (base - low) * 0.5 * (1 + math.cos(math.pi * t))


def add_common_args(ap, task):
    ap.add_argument("--init", default=None, help="start from this checkpoint (weights and ranges), e.g. init.pth")
    ap.add_argument("--resume", default=None, help="latest.pth of an interrupted run (same other arguments)")
    ap.add_argument("--calib", choices=("kld", "noclip"), default="kld",
                    help="activation range selection when there is no --init / --scheme, as nncase's --calib-method")
    if task.supports_scheme:
        ap.add_argument("--scheme", default=None,
                        help="take the activation ranges from this nncase quant scheme instead of calibrating")
    ap.add_argument("-b", "--batch-size", type=int, default=task.default_batch_size)
    ap.add_argument("--epochs", type=int, default=task.default_epochs,
                    help="0 = calibrate (or load) and evaluate only")
    ap.add_argument("--iters-per-epoch", type=int, default=task.default_iters_per_epoch,
                    help="0 = the task's own epoch length; smaller for a pilot run")
    ap.add_argument("--lr", type=float, default=task.default_lr)
    ap.add_argument("--clip-grad", type=float, default=task.default_clip_grad, help="0 = no gradient clipping")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-first", action="store_true", help="evaluate before training (the PTQ starting point)")
    ap.add_argument("--out", default=f"qat_{task.name}")


def parse_args(tasks, argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="task", required=True)
    for task in tasks:
        p = sub.add_parser(task.name, help=task.description, description=task.description,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        task.add_args(p)
        add_common_args(p, task)
    return ap.parse_args(argv)


def checkpoint(model, epoch, metrics, best, args, opt=None):
    state = {"task": args.task, "model": model.state_dict(), "epoch": epoch, "metrics": metrics, "best": best,
             "args": vars(args)}
    if opt is not None:
        state["optimizer"] = opt.state_dict()
    return state


def load_checkpoint(path, tasks, device="cpu", task_name=None, **task_args):
    """(task, model, checkpoint) of a checkpoint written by qat_train.py; the model in eval mode, quantizers on.
    task_name and task_args stand in for what a checkpoint of an older script does not store."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if task_name and ckpt.get("task", task_name) != task_name:
        raise SystemExit(f"{path} is a {ckpt['task']} checkpoint, not {task_name}")
    task = {t.name: t for t in tasks}[ckpt.get("task", task_name)]
    args = argparse.Namespace(**{**task_args, **ckpt.get("args", {})})
    model = task.build(args, device, load_float_weights=False)
    task.load_state_dict(model, ckpt["model"])
    set_state(model, weights=True, acts=True)
    return task, model.eval(), ckpt


def run(task, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA, running on the CPU")
    os.makedirs(args.out, exist_ok=True)
    start = args.resume or args.init
    model = task.build(args, device, load_float_weights=start is None)
    if start:
        ckpt = torch.load(start, map_location=device, weights_only=False)
        task.load_state_dict(model, ckpt["model"])
        print(f"weights and ranges from {start}")
    elif getattr(args, "scheme", None):
        print(f"{task.load_scheme(model, args)} activation ranges from {args.scheme}")
    else:
        t = time.time()
        qat_generic.calibrate(model, task.calib_loader(args), args.calib)
        print(f"calibration took {time.time() - t:.0f} s")
        torch.save(checkpoint(model, -1, {}, None, args), os.path.join(args.out, "init.pth"))
        dump_ranges(model, os.path.join(args.out, "ranges.json"))
    set_state(model, weights=True, acts=True)
    print(describe(model))

    if args.eval_first or args.epochs == 0:
        model.eval()
        print(f"[before training] {format_metrics(task.evaluate(model, args))}", flush=True)
    if args.epochs == 0:
        return

    for p in task.frozen_parameters(model, args):
        p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"{sum(p.numel() for p in params) / 1e6:.2f}M trainable parameters")
    base_lr = task.base_lr(args)
    opt = task.optimizer(params, base_lr, args)
    start_epoch, best = 0, None
    if args.resume:
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch, best = ckpt["epoch"] + 1, ckpt["best"]
        print(f"resumed after epoch {ckpt['epoch'] + 1}")

    iters = args.iters_per_epoch or task.iters_per_epoch(args)
    total = args.epochs * iters
    warmup, power, tail = task.schedule(args, iters)
    print(f"{iters} iterations per epoch, {args.epochs} epochs, base lr {base_lr:.2e}")
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t_epoch, running = time.time(), {}
        for it, batch in enumerate(task.batches(model, args, epoch, iters)):
            step = epoch * iters + it
            lr = lr_at(step, total, base_lr, warmup, power, tail)
            for g in opt.param_groups:
                g["lr"] = lr
            loss, parts = task.loss(model, batch, args)
            if not torch.isfinite(loss):
                raise SystemExit(f"loss is {loss.item()} at epoch {epoch + 1} iter {it + 1}; lower --lr")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
            opt.step()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            if (it + 1) % PRINT_INTERVAL == 0:
                done = time.time() - t_epoch
                eta = done / (it + 1) * (iters - it - 1)
                print(f"epoch {epoch + 1}/{args.epochs} iter {it + 1}/{iters} lr {lr:.2e} "
                      + " ".join(f"{k} {v / PRINT_INTERVAL:.4f}" for k, v in running.items())
                      + f" | {(it + 1) * args.batch_size / done:.1f} samples/s, epoch ETA {eta / 60:.1f} min",
                      flush=True)
                running = {}

        model.eval()
        metrics = task.evaluate(model, args)
        print(f"[epoch {epoch + 1}] {format_metrics(metrics)}", flush=True)
        if best is None or task.score(metrics) > task.score(best):
            best = metrics
            torch.save(checkpoint(model, epoch, metrics, best, args), os.path.join(args.out, "best.pth"))
            dump_ranges(model, os.path.join(args.out, "ranges_best.json"))
            print(f"---> new best: {format_metrics(best)}")
        torch.save(checkpoint(model, epoch, metrics, best, args, opt), os.path.join(args.out, "latest.pth"))
        print(f"epoch {epoch + 1} took {(time.time() - t_epoch) / 60:.1f} min, total "
              f"{(time.time() - t_start) / 60:.1f} min", flush=True)
    print(f"done. best {format_metrics(best)}  ->  {os.path.join(args.out, 'best.pth')}")


def parse_export_args(tasks, argv=None, description=None):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="task", required=True)
    for task in tasks:
        p = sub.add_parser(task.name, help=f"export a {task.name} checkpoint",
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        p.add_argument("ckpt", help="best.pth / latest.pth / init.pth of qat_train.py")
        p.add_argument("output", help="output .onnx")
        task.add_export_args(p)
    return ap.parse_args(argv)


def export(tasks, args):
    """Load args.ckpt (the arguments of args stand in for those an older checkpoint does not store) and export it."""
    task, model, ckpt = load_checkpoint(args.ckpt, tasks, task_name=args.task, **vars(args))
    metrics = ckpt.get("metrics") or {k: ckpt[k] for k in ("ap50_95", "ap50") if k in ckpt}  # older checkpoints
    print(f"{args.ckpt}: epoch {ckpt.get('epoch', -1) + 1}  {format_metrics(metrics)}")
    task.export(model, ckpt, args)
