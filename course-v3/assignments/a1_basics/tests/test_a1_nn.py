"""A1 问题 linear、conv2d、max_pool、batchnorm、softmax：与 PyTorch 自带的实现逐值对拍（前向和梯度）。"""
import math

import pytest
import torch
import torch.nn.functional as F

from ._impl import adapters as A


def _grads(fn, *xs):
    xs = [x.detach().clone().requires_grad_(True) for x in xs]
    out = fn(*xs)
    out.backward(torch.randn(out.shape, generator=torch.Generator().manual_seed(9), dtype=out.dtype))
    return out.detach(), [x.grad for x in xs]


def test_linear():
    g = torch.Generator().manual_seed(0)
    w, b, x = torch.randn(5, 7, generator=g), torch.randn(5, generator=g), torch.randn(3, 4, 7, generator=g)
    torch.testing.assert_close(A.run_linear(7, 5, w, b, x), x @ w.T + b)


def test_init_scale():
    torch.manual_seed(0)
    lin = A.get_linear(1000, 500)
    assert lin.weight.shape == (500, 1000) and torch.all(lin.bias == 0)
    assert abs(lin.weight.std().item() / math.sqrt(2 / 1000) - 1) < 0.05
    conv = A.get_conv2d(64, 128, 3)
    assert conv.weight.shape == (128, 64, 3, 3)
    assert abs(conv.weight.std().item() / math.sqrt(2 / (64 * 9)) - 1) < 0.05


@pytest.mark.parametrize("stride,padding,k,bias", [(1, 0, 3, True), (1, 1, 3, False), (2, 1, 3, True), (2, 2, 5, True), (3, 0, 2, False)])
def test_conv2d(stride, padding, k, bias):
    g = torch.Generator().manual_seed(stride * 10 + k)
    x = torch.randn(2, 3, 11, 9, generator=g, dtype=torch.float64)
    w = torch.randn(4, 3, k, k, generator=g, dtype=torch.float64)
    b = torch.randn(4, generator=g, dtype=torch.float64)
    if bias:
        y1, g1 = _grads(lambda x, w, b: A.run_conv2d(x, w, b, stride, padding), x, w, b)
        y2, g2 = _grads(lambda x, w, b: F.conv2d(x, w, b, stride, padding), x, w, b)
    else:
        y1, g1 = _grads(lambda x, w: A.run_conv2d(x, w, None, stride, padding), x, w)
        y2, g2 = _grads(lambda x, w: F.conv2d(x, w, None, stride, padding), x, w)
    torch.testing.assert_close(y1, y2)
    for a, b_ in zip(g1, g2):
        torch.testing.assert_close(a, b_)


@pytest.mark.parametrize("shape,k", [((2, 3, 8, 8), 2), ((1, 2, 9, 7), 2), ((2, 1, 9, 9), 3)])
def test_max_pool2d(shape, k):
    x = torch.randn(*shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    y1, g1 = _grads(lambda x: A.run_max_pool2d(x, k), x)
    y2, g2 = _grads(lambda x: F.max_pool2d(x, k), x)
    torch.testing.assert_close(y1, y2)
    torch.testing.assert_close(g1[0], g2[0])


@pytest.mark.parametrize("training", [True, False])
def test_batch_norm2d(training):
    g = torch.Generator().manual_seed(2)
    x = torch.randn(4, 3, 5, 6, generator=g, dtype=torch.float64) * 3 + 1
    w, b = torch.randn(3, generator=g, dtype=torch.float64), torch.randn(3, generator=g, dtype=torch.float64)
    rm1, rv1 = torch.randn(3, generator=g, dtype=torch.float64), torch.rand(3, generator=g, dtype=torch.float64) + 0.5
    rm2, rv2 = rm1.clone(), rv1.clone()
    y1, g1 = _grads(lambda x, w, b: A.run_batch_norm2d(x, w, b, rm1, rv1, training, 0.1, 1e-5), x, w, b)
    y2, g2 = _grads(lambda x, w, b: F.batch_norm(x, rm2, rv2, w, b, training, 0.1, 1e-5), x, w, b)
    torch.testing.assert_close(y1, y2)
    for a, b_ in zip(g1, g2):
        torch.testing.assert_close(a, b_)
    torch.testing.assert_close(rm1, rm2)                                    # 运行统计量也要一样地更新
    torch.testing.assert_close(rv1, rv2)


def test_log_softmax_stable():
    x = torch.tensor([[1000.0, 1001.0, 999.0], [-1000.0, 0.0, 1000.0]], dtype=torch.float64)
    y = A.run_log_softmax(x, -1)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, F.log_softmax(x, -1))
    z = torch.randn(3, 4, 5, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(A.run_log_softmax(z, 1), F.log_softmax(z, 1))


@pytest.mark.parametrize("ls", [0.0, 0.1])
def test_cross_entropy(ls):
    g = torch.Generator().manual_seed(4)
    logits = torch.randn(16, 62, generator=g, dtype=torch.float64) * 5
    t = torch.randint(0, 62, (16,), generator=g)
    y1, g1 = _grads(lambda z: A.run_cross_entropy(z, t, ls), logits)
    y2, g2 = _grads(lambda z: F.cross_entropy(z, t, label_smoothing=ls), logits)
    assert y1.shape == ()
    torch.testing.assert_close(y1, y2)
    torch.testing.assert_close(g1[0], g2[0])
    big = torch.tensor([[1e4, 0.0], [0.0, 1e4]])
    assert torch.isfinite(A.run_cross_entropy(big, torch.tensor([1, 0]), 0.0))
