"""util/ocr_post.py through --postprocess ppocr (util/postprocess.py): known text is encoded into CTC logits
[1, 80, 97] (the inverse of greedy decoding: one or more time steps per character, blanks between repeats and as
padding), post-processed, and the printed text and confidence are compared with it."""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from util import ocr_post, postprocess

CHARS = ocr_post.PPOCR_CHARS
STEPS = 80
HIT = 12.0  # logit of the chosen class; the other 96 are 0


def index(c):
    return CHARS.index(c)  # the first of the two spaces


def encode(steps, n_steps=STEPS):
    """steps: one class per time step (a character, or None for the blank) -> logits [1, n_steps, 97], blank padded."""
    assert len(steps) <= n_steps
    out = np.zeros((1, n_steps, len(CHARS)), np.float32)
    for t in range(n_steps):
        c = steps[t] if t < len(steps) else None
        out[0, t, 0 if c is None else index(c)] = HIT
    return out


def spell(text, repeat=1):
    """CTC steps that greedy decoding turns into text: each character `repeat` times, a blank after each one."""
    steps = []
    for c in text:
        steps += [c] * repeat + [None]
    return steps


def hit_prob():
    return math.exp(HIT) / (math.exp(HIT) + len(CHARS) - 1)


def run(outs, capsys):
    args = SimpleNamespace(postprocess="ppocr", score_thr=0.25, nms_thr=0.45, output_prefix="unused-")
    code = postprocess.run(args, np.zeros((1, 3, 32, 320), np.uint8), outs)
    return code, capsys.readouterr().out


def line(text, conf):
    return f"text: {text!r}  (confidence {conf:.3f})"


@pytest.mark.parametrize("text", ["21.03.2020", "Uib#MmcA", "Hello, World!", "x", "a+b=c; [1]{2}(3) <4>"])
def test_text(text, capsys):
    code, out = run([encode(spell(text))], capsys)
    assert code == 0
    assert out.strip() == line(text, hit_prob())


def test_repeats_merge_and_blank_separates(capsys):
    """'hheelll' + blank + 'l' + 'oo' -> 'hello': runs of a class merge, a blank between two runs keeps both."""
    steps = list("hheelll") + [None] + list("loo")
    code, out = run([encode(steps)], capsys)
    assert code == 0
    assert out.strip() == line("hello", hit_prob())


def test_repeated_characters_need_a_blank(capsys):
    """Without blanks between them, 'aaaa' over four steps is one 'a'; with them, four."""
    assert run([encode(list("aaaa"))], capsys)[1].strip() == line("a", hit_prob())
    assert run([encode(spell("aaaa"))], capsys)[1].strip() == line("aaaa", hit_prob())


def test_longest_line(capsys):
    """40 identical characters fill the 80 steps (each needs a blank after it)."""
    code, out = run([encode(spell("7" * 40))], capsys)
    assert code == 0
    assert out.strip() == line("7" * 40, hit_prob())


def test_every_character(capsys):
    """Every class decodes to its character (two lines: the 96 characters do not fit in 80 steps)."""
    chars = "".join(CHARS[1:96])
    for part in (chars[:40], chars[40:80], chars[80:]):
        assert run([encode(spell(part))], capsys)[1].strip() == line(part, hit_prob())


def test_appended_space_class(capsys):
    """Class 96, the space PaddleOCR appends to the dictionary, also decodes to a space."""
    out = encode(spell("ab"))
    out[0, 1] = 0
    out[0, 1, 96] = HIT  # the blank after 'a' becomes class 96
    assert run([out], capsys)[1].strip() == line("a b", hit_prob())


def test_all_blank(capsys):
    code, out = run([encode([])], capsys)
    assert code == 0
    assert out.strip() == line("", 0.0)


def test_confidence_is_mean_of_kept_characters(capsys):
    """The kept steps' probabilities are averaged; blanks and merged repeats do not count."""
    out = encode(spell("ab"))
    out[0, 0, index("a")] = 2.0  # p(a) = e^2 / (e^2 + 96)
    out[0, 1, 0] = 1.0  # a low-confidence blank: not counted
    p_a = math.exp(2.0) / (math.exp(2.0) + 96)
    assert run([out], capsys)[1].strip() == line("ab", (p_a + hit_prob()) / 2)


def test_decode_returns_text_and_confidence():
    text, conf = ocr_post.decode(encode(spell("OK"))[0])
    assert text == "OK" and conf == pytest.approx(hit_prob(), rel=1e-6)


@pytest.mark.parametrize("shape", [(1, 80, 96), (80, 97), (2, 80, 97), (1, 97, 80)])
def test_wrong_output_shape(shape, capsys):
    code, out = run([np.zeros(shape, np.float32)], capsys)
    assert code == 1
    assert f"error: --postprocess ppocr needs an output [1, T, 97], got {list(shape)}" in out


def test_charset_matches_the_model():
    """The built-in character set is the 'character' metadata of the original PaddleOCR model."""
    onnx = pytest.importorskip("onnx")
    path = Path(__file__).resolve().parent.parent / "weights/ppocr/en_number_mobile_v2.0_rec_infer.onnx"
    if not path.exists():
        pytest.skip(f"{path} not found")
    meta = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
    assert CHARS == ["<blank>"] + [c for c in meta["character"].split("\n") if c] + [" "]
