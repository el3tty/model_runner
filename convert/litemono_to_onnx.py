#!/usr/bin/env python3
"""
Export Lite-Mono (monocular depth, https://github.com/noahzn/Lite-Mono, MIT) to one static ONNX model: encoder and
depth decoder together, input x [1, 3, 192, 640] (RGB, 0..1), output disp [1, 1, 192, 640] (sigmoid disparity, 0..1,
the full resolution scale only: the other decoder outputs are training losses and are dropped).

    python convert/litemono_to_onnx.py build/weights/lite-mono/lite-mono-tiny build/onnx/lite-mono-tiny-640x192.onnx --model lite-mono-tiny

<upstream> (--upstream, default build/weights/lite-mono/upstream) holds networks/depth_encoder.py, networks/depth_decoder.py
and layers.py of the repository. The weights folder holds encoder.pth and depth.pth, loaded with weights_only=True.
The result is compared with the PyTorch model on onnxruntime. convert/onnx_opt.py then turns it into the form the K230 runs
best (about 40% less time on the board). timm is not needed: the model takes only DropPath
(inactive in eval) and an initializer from it.
"""
import argparse
import importlib.util
import inspect
import os
import sys
import types

import numpy as np
import onnx
import torch
import torch.nn as nn

H, W = 192, 640


def load_upstream(folder):
    """Lite-Mono's encoder and decoder classes from its source files, without timm and without its package __init__
    (that also imports the pose networks and torchvision)."""
    timm = types.ModuleType("timm")
    timm.models = types.ModuleType("timm.models")
    timm.models.layers = types.ModuleType("timm.models.layers")
    timm.models.layers.DropPath = nn.Identity
    timm.models.layers.trunc_normal_ = nn.init.trunc_normal_
    sys.modules.update({"timm": timm, "timm.models": timm.models, "timm.models.layers": timm.models.layers})

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, os.path.join(folder, path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    load("layers", "layers.py")
    return load("depth_encoder", "networks/depth_encoder.py").LiteMono, \
        load("depth_decoder", "networks/depth_decoder.py").DepthDecoder


class LiteMonoDepth(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))[("disp", 0)]


def build(upstream, weights, model):
    LiteMono, DepthDecoder = load_upstream(upstream)
    encoder = LiteMono(model=model, height=H, width=W)
    decoder = DepthDecoder(encoder.num_ch_enc, scales=range(3))
    for net, name in ((encoder, "encoder.pth"), (decoder, "depth.pth")):
        state = torch.load(os.path.join(weights, name), map_location="cpu", weights_only=True)
        own = net.state_dict()
        missing = set(own) - set(state)
        if missing:
            raise ValueError(f"{name} lacks {sorted(missing)[:5]}")
        net.load_state_dict({k: v for k, v in state.items() if k in own})
    return LiteMonoDepth(encoder, decoder).eval()


def check_gelu(m):
    """Every Erf must still be a whole GELU, x * (1 + Erf(x / sqrt 2)) * 0.5, for nncase to fold it."""
    readers = {}
    for n in m.graph.node:
        for i in n.input:
            readers.setdefault(i, []).append(n)
    for n in m.graph.node:
        if n.op_type == "Erf":
            add = readers[n.output[0]][0]
            mul = readers[add.output[0]][0]
            if [r.op_type for r in readers.get(mul.output[0], [])] != ["Mul"]:
                raise ValueError(f"{n.name}: not a whole GELU")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", help="folder with encoder.pth and depth.pth")
    ap.add_argument("output", help="output .onnx")
    ap.add_argument("--model", default="lite-mono-tiny", choices=("lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"))
    ap.add_argument("--upstream", default="build/weights/lite-mono/upstream", help="folder with the repository's source files")
    ap.add_argument("--opset", type=int, default=13, help="below 17 LayerNorm and GELU are spelled out, as nncase's "
                                                          "importer and its fold rules expect (default: 13)")
    args = ap.parse_args()

    net = build(args.upstream, args.weights, args.model)
    print("parameters:", sum(p.numel() for p in net.parameters()))
    x = torch.rand(1, 3, H, W, generator=torch.Generator().manual_seed(0))

    kwargs = dict(input_names=["x"], output_names=["disp"], opset_version=args.opset, do_constant_folding=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    with torch.no_grad():
        ref = net(x).numpy()
        torch.onnx.export(net, x, args.output, **kwargs)

    import onnxsim
    import onnxruntime as ort

    # constant folding only: onnxsim's optimizers fold the 0.5 of a GELU into the next Conv, and the Erf that leaves
    # behind is not matched by nncase's FoldGeneralGelu (the compile then fails on Erf)
    m, ok = onnxsim.simplify(onnx.load(args.output), perform_optimization=False)
    if not ok:
        raise RuntimeError("onnxsim could not validate the simplified model")
    check_gelu(m)
    onnx.save(m, args.output)
    onnx.checker.check_model(m)
    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"x": x.numpy()})[0]
    print(f"{args.output}: {len(m.graph.node)} nodes, output {out.shape}, disp {ref.min():.4f}..{ref.max():.4f}, "
          f"max |onnx - torch| {np.abs(out - ref).max():.3e}")


if __name__ == "__main__":
    main()
