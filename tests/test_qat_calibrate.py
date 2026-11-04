"""util/qat_generic.py calibrate(): the activation ranges, chosen like nncase's PTQ.

The models are the ONNX text cases of test_qat_graph.py. The expected ranges do not come from the code under test:
onnxruntime runs the float model with the tensors of each range group as extra outputs, and the min / max (noclip) or
the histogram (kld) of those tensors is computed here.

kld_range() itself is compared with nncase: GOLDEN holds the thresholds that Quantizer.Smooth and GetKldOptRanges of
Nncase.Quantization.dll chose for the histograms of HISTOGRAMS, with the loops of Quantizer.GetOptRanges around them.
"""
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn

from test_qat_graph import CASES, build
from util import qat_generic, qat_graph
from util.qat_generic import DST_BINS, SRC_BINS, ActFakeQuant, calibrate, kld_range

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning"),  # onnx2torch's notes on Resize modes
              pytest.mark.filterwarnings("ignore::DeprecationWarning")]

# case: the tensors of each range group, in quantizer order
GROUPS = {
    "conv_relu_conv": [["r1"]],
    "silu_maxpool": [["a1"]],
    "concat_ties_ranges": [["r1", "r2"]],
    "resize_nearest_concat": [["r1", "r2"]],
    "residual_add": [["r1"], ["r2"]],
    "add_chain": [["r0"], ["a1"], ["a2"]],
    "plain_conv_to_add_and_conv": [["r0"], ["c1"], ["a", "c2"]],
    "activation_beside_another_reader": [["r1"], ["a"]],
    "hardswish_mul_div": [["h"]],
    "squeeze_excitation": [["r1"], ["p"], ["r2"], ["hs"], ["m"]],
    "pool_flatten_gemm": [["r1"], ["p"]],
    "lstm_gemm": [["l1"]],
}


class Batches(list):
    """What calibrate() needs of a DataLoader: batches whose [0] is the input, and a dataset with a length."""

    @property
    def dataset(self):
        return self


def batches(shape, gains=(1.0, 3.0, 0.5), seed=1):
    """Random inputs. The middle batch is the widest: neither the first nor the last one has the overall range."""
    rng = np.random.default_rng(seed)
    return Batches((torch.from_numpy((rng.standard_normal(shape) * g).astype(np.float32)),) for g in gains)


def prepare(case, tmp_path):
    """(the fake-quant model of a case, the path of its float ONNX, its input shape)."""
    src = tmp_path / "in.onnx"
    shape = build(CASES[case][0], src)
    return qat_graph.make_quantizable(qat_graph.load(str(src))), src, shape


def range_groups(model):
    """The ActRange of each range group, in quantizer order."""
    groups = {}
    for m in model.modules():
        if isinstance(m, ActFakeQuant):
            groups.setdefault(id(m.params), m.params)
    return list(groups.values())


def ranges_of(model):
    return [(float(r.min_val), float(r.max_val)) for r in range_groups(model)]


def float_tensors(src, names, data):
    """{name: the values of that tensor of the float ONNX model on all the batches, flattened}."""
    m = onnx.load(src)
    m.graph.output.extend(onnx.ValueInfoProto(name=n) for n in names)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(m.SerializeToString(), options, providers=["CPUExecutionProvider"])
    runs = [sess.run(names, {sess.get_inputs()[0].name: b[0].numpy()}) for b in data]
    return {n: np.concatenate([r[i].ravel() for r in runs]) for i, n in enumerate(names)}


def expected_noclip(values):
    return min(float(values.min()), 0.0), max(float(values.max()), 0.0)


def expected_kld(values):
    lo, hi = expected_noclip(values)
    index = np.clip((values.astype(np.float64) - lo) / ((hi - lo) / SRC_BINS), 0, SRC_BINS - 1).astype(np.int64)
    return kld_range(np.bincount(index, minlength=SRC_BINS), lo, hi)


@pytest.mark.parametrize("method, expected", [("noclip", expected_noclip), ("kld", expected_kld)])
@pytest.mark.parametrize("case", GROUPS)
def test_ranges_of_the_float_tensors(case, method, expected, tmp_path):
    """Every range group gets the range of exactly its tensors over all the batches; a shared group gets one range
    of all its tensors."""
    model, src, shape = prepare(case, tmp_path)
    data = batches(shape)
    calibrate(model, data, method)
    names = [n for group in GROUPS[case] for n in group]
    tensors = float_tensors(src, names, data)
    want = [expected(np.concatenate([tensors[n] for n in group])) for group in GROUPS[case]]
    assert ranges_of(model) == [pytest.approx(w, rel=1e-4, abs=1e-5) for w in want]


def test_state_after_calibration(tmp_path, capsys):
    """Calibration turns the quantizers on, counts the batches and leaves no hooks behind."""
    model, _, shape = prepare("residual_add", tmp_path)
    data = batches(shape)
    calibrate(model, data, "noclip")
    fake = [m for m in model.modules() if isinstance(m, (ActFakeQuant, qat_generic.WeightFakeQuant))]
    assert all(m.enabled for m in fake) and not any(m._forward_hooks for m in fake)
    assert [int(r.seen) for r in range_groups(model)] == [3, 3]
    assert "calibrated 2 range groups on 3 images (noclip)" in capsys.readouterr().out


@pytest.mark.parametrize("method", ["noclip", "kld"])
def test_it_runs_the_float_network(method, tmp_path):
    """Calibrating an already fake-quantized model gives the same ranges as the first time."""
    model, _, shape = prepare("squeeze_excitation", tmp_path)
    data = batches(shape)
    calibrate(model, data, method)
    first = ranges_of(model)
    for r in range_groups(model):  # a useless range: fake quantization would clip everything to it
        r.min_val.fill_(-1e-3)
        r.max_val.fill_(1e-3)
    calibrate(model, data, method)
    assert ranges_of(model) == first


class Tap(nn.Module):
    """x * gain through one activation quantizer, plus a quantizer no data reaches."""

    def __init__(self, gain):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(float(gain)))
        self.q = ActFakeQuant()
        self.unused = ActFakeQuant()

    def forward(self, x):
        return self.q(x * self.gain)


POSITIVE = Batches([(torch.tensor([[1.0, 2.0, 4.0]]),), (torch.tensor([[3.0, 2.5, 3.5]]),)])


@pytest.mark.parametrize("gain, want", [(1, (0.0, 4.0)), (-1, (-4.0, 0.0)), (0, (0.0, 0.1))])
def test_the_range_contains_zero(gain, want):
    """All positive, all negative, all zero (a dead activation: the 0.1 of Quantizer.FixUpRange)."""
    model = Tap(gain)
    calibrate(model, POSITIVE, "noclip")
    assert (float(model.q.params.min_val), float(model.q.params.max_val)) == pytest.approx(want)


@pytest.mark.parametrize("gain", [1, -1, 0])
def test_kld_of_degenerate_tensors(gain):
    """Few values in a few bins: the range stays finite, contains zero and has a usable width."""
    model = Tap(gain)
    calibrate(model, POSITIVE, "kld")
    lo, hi = float(model.q.params.min_val), float(model.q.params.max_val)
    assert np.isfinite([lo, hi]).all() and lo <= 0 <= hi and hi - lo > 0


def test_kld_drops_an_outlier_that_noclip_keeps():
    x = torch.from_numpy(np.random.default_rng(2).standard_normal((1, 100000)).astype(np.float32))
    x[0, 0] = 40.0
    data = Batches([(x,)])
    model = Tap(1)
    calibrate(model, data, "noclip")
    assert float(model.q.params.max_val) == 40.0
    calibrate(model, data, "kld")
    assert 3.0 < float(model.q.params.max_val) < 10.0


def test_it_runs_in_eval_mode():
    """Batch normalization uses its stored statistics: in training mode it would use those of the batch and change
    them."""
    model = nn.Sequential(nn.BatchNorm1d(1), ActFakeQuant())
    model[0].running_mean.fill_(1.0)
    model.train()
    calibrate(model, Batches([(torch.tensor([[1.0], [3.0], [5.0]]),)]), "noclip")
    assert float(model[0].running_mean) == 1.0
    assert float(model[1].params.max_val) == pytest.approx(4.0, rel=1e-3)


def test_a_group_without_data_is_reported(capsys):
    model = Tap(1)
    calibrate(model, POSITIVE, "noclip")
    assert "WARNING: 1 range groups saw no data" in capsys.readouterr().out
    assert int(model.unused.params.seen) == 0 and float(model.unused.params.max_val) == 0.0


def test_unknown_method():
    with pytest.raises(ValueError, match="unknown calibration method"):
        calibrate(Tap(1), POSITIVE, "percentile")


def centers(lo, hi):
    return lo + (np.arange(SRC_BINS) + 0.5) * (hi - lo) / SRC_BINS


def bell(t, mean, sigma, height=1e5):
    return np.floor(height * np.exp(-0.5 * ((t - mean) / sigma) ** 2))


def spikes(bins, count=5):
    h = np.zeros(SRC_BINS)
    h[list(bins)] = count
    return h


def histograms():
    """name: (histogram, lo, hi), closed forms so that GOLDEN does not depend on a random generator."""
    t = centers(0, 12)
    relu = (bell(t, 0, 1.5) + spikes([8000, 8100, 8191]), 0.0, 12.0)
    t = centers(-20, 30)
    signed = (bell(t, 0.5, 2) + spikes([0, 40, 8100, 8191]), -20.0, 30.0)
    uniform = (np.full(SRC_BINS, 100.0), -1.0, 3.0)
    t = centers(-6, 10)
    bumps = (bell(t, -3, 0.5) + bell(t, 6, 1), -6.0, 10.0)
    t = centers(-0.375, 9)
    decay = (np.floor(1e5 * np.exp(-2 * np.abs(t))), -0.375, 9.0)
    t = centers(-4, 0)
    negative = (bell(t, -0.5, 0.4) + spikes([0, 100]), -4.0, 0.0)
    dead = (spikes([0], 1000), 0.0, 0.1)
    return {"relu": relu, "signed": signed, "uniform": uniform, "bumps": bumps, "decay": decay, "negative": negative,
            "dead": dead}


# name: the (lower, upper) thresholds of nncase
GOLDEN = {"relu": (0, 4352), "signed": (1536, 5120), "uniform": (0, 8192), "bumps": (256, 7936), "decay": (0, 8192),
          "negative": (3072, 8192), "dead": (0, 8192)}


@pytest.mark.parametrize("name", GOLDEN)
def test_kld_range_matches_nncase(name):
    hist, lo, hi = histograms()[name]
    lower, upper = GOLDEN[name]
    interval = (hi - lo) / SRC_BINS
    want = ((lower - 0.5) * interval + lo, (upper + 0.5) * interval + lo)
    assert kld_range(hist, lo, hi) == pytest.approx(want)


def test_kld_clips_outliers_and_keeps_a_full_distribution():
    h = histograms()
    assert kld_range(*h["relu"])[1] < 7.0
    lo, hi = kld_range(*h["signed"])
    assert -11.0 < lo and hi < 12.0
    lo, hi = kld_range(*h["uniform"])
    assert lo == pytest.approx(-1.0, abs=4.0 / DST_BINS) and hi == pytest.approx(3.0, abs=4.0 / DST_BINS)
