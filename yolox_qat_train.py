#!/usr/bin/env python3
"""QAT fine-tuning of YOLOX (nano). Needs util\qat_generic.py and util\qat_yolox.py next to this script (they provide the fake-quant model).

Start from the calibrated model written by yolox_qat.py (qat_out/qat_init.pth).

Short pilot (check that the pipeline works and the AP goes up):

    python yolox_qat_train.py --init qat_out/qat_init.pth -b 16 --epochs 2 --iters-per-epoch 500 ^
        --no-aug-epochs 1 --eval-first --out qat_pilot

Real run (one epoch = the whole train2017, 15 epochs, mosaic/mixup off for the last 3):

    python yolox_qat_train.py --init qat_out/qat_init.pth -b 16 --epochs 15 --no-aug-epochs 3 --out qat_run

What it does (the YOLOX training recipe, adapted for fine-tuning):
  * the model is the BN-folded network with fake quantizers (weights: per-channel, recomputed from the weights at
    every step; activations: per-tensor, ranges tracked by a moving average of the per-batch min/max);
  * SGD (momentum 0.9, nesterov), the same learning rate for every parameter, YOLOX's warmup + cosine schedule,
    lr = --lr * batch_size / 64; conv weights get weight decay --wd, biases none;
  * mosaic/mixup are switched off for the last --no-aug-epochs epochs and the L1 loss is added (as in YOLOX);
  * the activation ranges stop moving for the last --freeze-obs-epochs epochs;
  * after every epoch: evaluation on val2017 with fake quantization, latest.pth, best.pth (+ ranges.json).
"""
import argparse
import json
import os
import time
from dataclasses import asdict

import torch

from yolox.data import DataPrefetcher
from yolox.exp import get_exp
from yolox.utils import LRScheduler

from util.qat_generic import QuantConfig, describe, dump_ranges, set_observer_momentum, set_state
from util.qat_yolox import evaluate, fold, make_quantizable


def build_model(exp, cfg):
    model = exp.get_model()  # the weights come from the checkpoint below; only the structure matters here
    model.eval()
    qat = make_quantizable(fold(model), cfg)
    return qat.cuda()


def make_optimizer(model, lr, wd):
    weights = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
    biases = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    opt = torch.optim.SGD([{"params": weights, "weight_decay": wd}, {"params": biases, "weight_decay": 0.0}],
                          lr=lr, momentum=0.9, nesterov=True)
    print(f"optimizer: {len(weights)} weight tensors (wd={wd}), {len(biases)} bias tensors, base lr={lr:.2e}")
    return opt


def make_loader(exp, batch_size, no_aug):
    loader = exp.get_data_loader(batch_size=batch_size, is_distributed=False, no_aug=no_aug, cache_img=None)
    return loader, DataPrefetcher(loader)


def run_eval(exp, qat, batch_size, label, observing):
    set_state(qat, weights=True, acts=True, observe=False)  # evaluation must not move the ranges
    with torch.no_grad():
        ap = evaluate(exp, qat, batch_size, label)
    set_state(qat, weights=True, acts=True, observe=observing)
    qat.train()
    return ap


def save(path, qat, cfg, epoch, ap, args, opt=None):
    state = {"model": qat.state_dict(), "cfg": asdict(cfg), "epoch": epoch, "ap50_95": ap[0], "ap50": ap[1],
             "args": vars(args)}
    if opt is not None:
        state["optimizer"] = opt.state_dict()
    torch.save(state, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--name", default="yolox-nano")
    ap.add_argument("-f", "--exp-file", default=None)
    ap.add_argument("--init", default="qat_out/qat_init.pth", help="calibrated fake-quant model from yolox_qat.py")
    ap.add_argument("--resume", default=None, help="latest.pth of an interrupted run")
    ap.add_argument("-b", "--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--iters-per-epoch", type=int, default=0, help="0 = the whole training set; smaller = pilot")
    ap.add_argument("--lr", type=float, default=1e-3, help="base lr at batch size 64 (scaled by batch_size/64)")
    ap.add_argument("--wd", type=float, default=0.0, help="weight decay of the conv weights (YOLOX: 5e-4)")
    ap.add_argument("--warmup-epochs", type=float, default=0.3)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--no-aug-epochs", type=int, default=3, help="last epochs without mosaic/mixup, with L1 loss")
    ap.add_argument("--obs-momentum", type=float, default=0.01, help="EMA factor of the activation range observers")
    ap.add_argument("--freeze-obs-epochs", type=int, default=2, help="last epochs with frozen activation ranges")
    ap.add_argument("--clip-grad", type=float, default=0.0, help="max gradient norm (0 = off)")
    ap.add_argument("--print-interval", type=int, default=50)
    ap.add_argument("--eval-first", action="store_true", help="evaluate before training (must reproduce yolox_qat.py)")
    ap.add_argument("--out", default="qat_train")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    os.makedirs(args.out, exist_ok=True)

    exp = get_exp(args.exp_file, args.name)
    exp.data_num_workers = args.workers
    ckpt = torch.load(args.resume or args.init, map_location="cpu", weights_only=False)
    cfg = QuantConfig.from_dict(ckpt["cfg"])
    print("config:", asdict(cfg))
    qat = build_model(exp, cfg)
    qat.load_state_dict(ckpt["model"])
    print(describe(qat))
    n_params = sum(p.numel() for p in qat.parameters())
    print(f"{n_params / 1e6:.2f}M trainable parameters, input size {exp.input_size}")

    lr = args.lr * args.batch_size / 64.0
    opt = make_optimizer(qat, lr, args.wd)
    start_epoch = 0
    best = (-1.0, -1.0)
    if args.resume:
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best = (ckpt["ap50_95"], ckpt["ap50"])
        print(f"resumed after epoch {ckpt['epoch'] + 1}")

    no_aug = start_epoch >= args.epochs - args.no_aug_epochs
    loader, prefetcher = make_loader(exp, args.batch_size, no_aug)
    qat.head.use_l1 = no_aug
    iters = args.iters_per_epoch or len(loader)
    sched = LRScheduler("yoloxwarmcos", lr, iters, args.epochs, warmup_epochs=args.warmup_epochs, warmup_lr_start=0,
                        no_aug_epochs=args.no_aug_epochs, min_lr_ratio=args.min_lr_ratio)
    print(f"{iters} iterations per epoch, {args.epochs} epochs")

    set_observer_momentum(qat, args.obs_momentum)
    observing = True
    set_state(qat, weights=True, acts=True, observe=True)
    if args.eval_first:
        run_eval(exp, qat, args.eval_batch_size, "before training", observing)
    qat.train()

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        if epoch == args.epochs - args.no_aug_epochs and not no_aug:
            print("---> no mosaic/mixup from now on, L1 loss added")
            loader.close_mosaic()
            qat.head.use_l1 = True
            no_aug = True
        if observing and epoch >= args.epochs - args.freeze_obs_epochs:
            print("---> activation ranges frozen")
            observing = False
            set_state(qat, weights=True, acts=True, observe=False)
        qat.train()

        t_epoch = time.time()
        running = {}
        for it in range(iters):
            step = epoch * iters + it
            cur_lr = sched.update_lr(step + 1)
            for g in opt.param_groups:
                g["lr"] = cur_lr

            inps, targets = prefetcher.next()
            inps = inps.float()
            targets = targets.float()
            targets.requires_grad = False
            inps, targets = exp.preprocess(inps, targets, exp.input_size)

            outputs = qat(inps, targets)
            loss = outputs["total_loss"]
            if not torch.isfinite(loss):
                raise SystemExit(f"loss is {loss.item()} at epoch {epoch + 1} iter {it + 1}; lower --lr")
            opt.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(qat.parameters(), args.clip_grad)
            opt.step()

            for k in ("total_loss", "iou_loss", "conf_loss", "cls_loss", "l1_loss"):
                v = outputs[k]
                running[k] = running.get(k, 0.0) + float(v)
            if (it + 1) % args.print_interval == 0:
                n = args.print_interval
                done = time.time() - t_epoch
                eta = done / (it + 1) * (iters - it - 1)
                print(f"epoch {epoch + 1}/{args.epochs} iter {it + 1}/{iters} lr {cur_lr:.2e} "
                      + " ".join(f"{k.replace('_loss', '')} {v / n:.2f}" for k, v in running.items())
                      + f" | {(it + 1) * args.batch_size / done:.1f} img/s, epoch ETA {eta / 60:.1f} min")
                running = {}

        ap = run_eval(exp, qat, args.eval_batch_size, f"epoch {epoch + 1}", observing)
        save(os.path.join(args.out, "latest.pth"), qat, cfg, epoch, ap, args, opt)
        if ap[0] > best[0]:
            best = ap
            save(os.path.join(args.out, "best.pth"), qat, cfg, epoch, ap, args)
            dump_ranges(qat, os.path.join(args.out, "ranges_best.json"))
            print(f"---> new best AP {ap[0]:.4f}")
        print(f"epoch {epoch + 1} took {(time.time() - t_epoch) / 60:.1f} min, "
              f"total {(time.time() - t_start) / 60:.1f} min, best AP {best[0]:.4f}")

    print(f"done. best AP(0.50:0.95) = {best[0]:.4f}  AP50 = {best[1]:.4f}  ->  {os.path.join(args.out, 'best.pth')}")


if __name__ == "__main__":
    main()
