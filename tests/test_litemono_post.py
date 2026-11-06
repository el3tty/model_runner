"""util/litemono_post.py through --postprocess litemono (util/postprocess.py): known disparities give known colors."""
from types import SimpleNamespace

import numpy as np
import pytest

from util import postprocess


def run(outs, tmp_path, capsys):
    args = SimpleNamespace(postprocess="litemono", score_thr=0.25, nms_thr=0.45, output_prefix=f"{tmp_path}/out-")
    code = postprocess.run(args, np.zeros((1, 3, 2, 3), np.uint8), outs)
    return code, tmp_path / "out-litemono.rgb", capsys.readouterr().out


def colors(path, h, w):
    return np.fromfile(path, np.uint8).reshape(3, h, w).transpose(1, 2, 0)


def test_close_is_bright_and_far_is_dark(tmp_path, capsys):
    disp = np.array([[[[0, 1]]]], np.float32)
    code, path, out = run([disp], tmp_path, capsys)
    assert code == 0
    assert colors(path, 1, 2).tolist() == [[[0, 0, 4], [252, 253, 191]]]
    assert "disparity 0.000..1.000" in out


def test_magma(tmp_path, capsys):
    cm = pytest.importorskip("matplotlib").colormaps["magma"]
    x = np.arange(256, dtype=np.float32) / 255
    _, path, _ = run([x.reshape(1, 1, 1, 256)], tmp_path, capsys)
    want = np.rint(cm(np.arange(256))[:, :3] * 255)
    assert np.abs(colors(path, 1, 256)[0].astype(int) - want).max() <= 3


def test_values_outside_the_range_are_clipped(tmp_path, capsys):
    disp = np.array([[[[-0.5, 1.5]]]], np.float32)
    _, path, _ = run([disp], tmp_path, capsys)
    assert colors(path, 1, 2).tolist() == [[[0, 0, 4], [252, 253, 191]]]


def test_float16_output(tmp_path, capsys):
    _, path, _ = run([np.full((1, 1, 2, 2), 0.5, np.float16)], tmp_path, capsys)
    assert (colors(path, 2, 2) == [183, 55, 121]).all()


def test_wrong_outputs(tmp_path, capsys):
    for outs in ([np.zeros((1, 3, 2, 2), np.float32)], [np.zeros((2, 2), np.float32)],
                 [np.zeros((1, 1, 2, 2), np.float32)] * 2):
        code, path, out = run(outs, tmp_path, capsys)
        assert code == 1 and out.startswith("error: --postprocess litemono needs") and not path.exists()
