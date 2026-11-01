"""CPU 上验证机制、数据身份和 CLI；无 GPU/权重/联网依赖。"""
import io

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from miniocr.nn.cnn import (output_size, receptive_field, conv_forward, conv_backward,
                            numerical_gradient, ManualBatchNorm2d, SymbolClassifier)
from book.part4_vlm.ch28_cnn.tour import downsample, damage, main


@pytest.mark.parametrize("stride,padding,kernel", [(1, 0, (2, 3)), (1, 1, (3, 3)), (2, 1, (2, 3)), (3, 0, (1, 1))])
def test_conv_forward_backward_library_and_finite_difference(stride, padding, kernel):
    rng = np.random.default_rng(7)
    values = [rng.normal(size=(2, 2, 4, 5)), rng.normal(size=(3, 2, *kernel)), rng.normal(size=3)]
    out, cache = conv_forward(*values, stride, padding)
    grad = rng.normal(size=out.shape)
    actual = conv_backward(grad, cache)
    ts = [torch.tensor(v, requires_grad=True) for v in values]
    ref = F.conv2d(*ts, stride=stride, padding=padding)
    (ref * torch.tensor(grad)).sum().backward()
    np.testing.assert_allclose(out, ref.detach().numpy(), atol=1e-12)
    for i, (got, t) in enumerate(zip(actual, ts)):
        np.testing.assert_allclose(got, t.grad.numpy(), atol=1e-12)
        def objective(v):
            args = values.copy()
            args[i] = v
            return (conv_forward(*args, stride, padding)[0] * grad).sum()
        num = numerical_gradient(objective, values[i])
        np.testing.assert_allclose(got, num, atol=2e-8, rtol=1e-6)


def test_geometry_floor_dilation_and_invalid():
    assert output_size(8, 3, 2, 1) == 4
    assert output_size(9, 3, 1, 0, 2) == 5
    rows = receptive_field([(3, 1, 1, 1), (2, 2, 0, 1), (3, 1, 1, 1), (2, 2, 0, 1)])
    assert rows[-1] == dict(field=10, jump=4, center=2.)
    assert receptive_field([(3, 1, 0, 2)])[-1]["field"] == 5
    for args in [(3, 5), (3, 2, 0), (3, 2, 1, -1)]:
        with pytest.raises(ValueError):
            output_size(*args)


def test_batchnorm_train_eval_and_gradients():
    torch.manual_seed(0)
    ours, ref = ManualBatchNorm2d(3).double(), torch.nn.BatchNorm2d(3).double()
    for _ in range(3):
        x = torch.randn(4, 3, 5, 6, dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        a, b = ours(x), ref(y)
        torch.testing.assert_close(a, b)
        upstream = torch.randn_like(a)
        (a*upstream).sum().backward()
        (b*upstream).sum().backward()
        for left, right in [(x.grad, y.grad), (ours.weight.grad, ref.weight.grad),
                            (ours.bias.grad, ref.bias.grad), (ours.running_mean, ref.running_mean),
                            (ours.running_var, ref.running_var)]:
            torch.testing.assert_close(left, right)
    ours.eval()
    ref.eval()
    snapshot = ours.running_var.clone()
    torch.testing.assert_close(ours(x), ref(x))
    torch.testing.assert_close(ours.running_var, snapshot)
    assert ours.num_batches_tracked.item() == 3
    ours.train()
    with pytest.raises(ValueError):
        ours(torch.zeros(1, 3, 1, 1, dtype=torch.float64))


def test_downsampling_thin_line_and_odd_size():
    ink = np.zeros((17, 19), np.float32)
    ink[8, 2:17] = 1
    np.testing.assert_array_equal(downsample(ink, 0), ink)
    restored = downsample(ink, 3)
    assert restored.shape == ink.shape
    assert damage(ink, restored)["foreground_recall"] == 0
    assert damage(np.zeros((3, 4)), np.zeros((3, 4)))["foreground_recall"] is None


@pytest.mark.parametrize("kind", ["mlp", "cnn"])
def test_classifier_learns_tiny_spatial_task(kind):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(1)
        x = torch.zeros(8, 1, 32, 32)
        x[:4, :, :8, :8] = 1
        x[4:, :, -8:, -8:] = 1
        y = torch.tensor([0]*4 + [1]*4)
        model = SymbolClassifier(kind, width=4, classes=2)
        opt = torch.optim.Adam(model.parameters(), lr=.02)
        initial = F.cross_entropy(model(x), y).item()
        for _ in range(25):
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert F.cross_entropy(model(x), y).item() < initial * .2
        assert torch.equal(model(x).argmax(1), y)
    finally:
        torch.set_num_threads(old)


def test_png_loader_split_dedup_and_identity(tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa
    import miniocr.data.formula_images as module
    buffer = io.BytesIO()
    Image.new("RGBA", (3, 2), (0, 0, 0, 0)).save(buffer, format="PNG")
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"latex_formula": f, "image": {"bytes": buffer.getvalue(), "path": None}}
        for f in ["train", "x^2", "x^{2}", "z"]]), path)
    monkeypatch.setattr(module, "_parquet_paths", lambda: [path])
    monkeypatch.setattr(module, "standard_split", lambda **kw: {"val": ["x^{2}", "z"], "train": ["train"]})
    rows = module.load_formula_images(2)
    assert rows[0]["id"].endswith(":1") and rows[1]["id"].endswith(":3")
    assert len({r["key_sha256"] for r in rows}) == 2
    assert np.max(rows[0]["ink"]) == 0  # 透明背景合成白色。
    with pytest.raises(ValueError):
        module.load_formula_images(1, "test")


def test_cli_parts_and_metadata(tmp_path):
    main(["2", "4", "--out", str(tmp_path), "--device", "cpu"])
    assert (tmp_path / "run.json").is_file()
    assert (tmp_path / "part2.json").is_file()
    assert (tmp_path / "normalization.csv").is_file()
    assert not (tmp_path / "final_test.json").exists()
