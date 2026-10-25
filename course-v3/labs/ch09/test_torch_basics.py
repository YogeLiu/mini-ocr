"""第 9 章的自动检查。"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from miniocr.models.linear_np import softmax_loss_grad
from miniocr.nn.micrograd import MLP, cross_entropy_values
from miniocr.torch_basics import GlyphDataset, MyLinear, softmax_regression_grads


def test_my_linear_matches_nn_linear():
    torch.manual_seed(0)
    mine, ref = MyLinear(5, 3), nn.Linear(5, 3)
    assert mine.weight.shape == ref.weight.shape == (3, 5)
    assert len(list(mine.parameters())) == 2
    with torch.no_grad():
        ref.weight.copy_(mine.weight)
        ref.bias.copy_(mine.bias)
    x = torch.randn(4, 7, 5)
    assert torch.allclose(mine(x), ref(x))
    assert set(mine.state_dict()) == {"weight", "bias"}
    assert MyLinear(5, 3, bias=False)(x).shape == (4, 7, 3)
    w = MyLinear(100, 10).weight
    assert w.abs().max() <= 0.1 + 1e-6                         # U(-1/√100, 1/√100)


def test_glyph_dataset_and_loader():
    ds = GlyphDataset(seed=0, chars="abc", n_per_class=4, size=16, align=True)
    assert len(ds) == 12
    x, y = ds[1]
    assert x.shape == (1, 16, 16) and x.dtype == torch.float32 and y.item() == 1 and y.dtype == torch.int64
    xb, yb = next(iter(DataLoader(ds, batch_size=5, shuffle=False)))
    assert xb.shape == (5, 1, 16, 16) and yb.tolist() == [0, 1, 2, 0, 1]


def test_autograd_matches_hand_derived():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(10, 6)), rng.integers(0, 4, 10)
    W, b = rng.normal(size=(6, 4)), rng.normal(size=4)
    L1, gW1, gb1 = softmax_regression_grads(X, y, W, b)
    L2, gW2, gb2 = softmax_loss_grad(X, y, W, b)
    assert np.isclose(L1, L2) and np.allclose(gW1, gW2) and np.allclose(gb1, gb2)


def test_micrograd_matches_torch():
    """第 8 章的 Value MLP 与同样权重的 PyTorch 网络，梯度逐个相等。"""
    net = MLP([4, 3, 2], np.random.default_rng(0))
    x, label = [0.5, -1.0, 2.0, 0.1], 1
    cross_entropy_values(net(x), label).backward()
    l1, l2 = net.layers
    W1 = torch.tensor([[p.data for p in n.w] for n in l1.neurons], dtype=torch.float64, requires_grad=True)
    b1 = torch.tensor([n.b.data for n in l1.neurons], dtype=torch.float64, requires_grad=True)
    W2 = torch.tensor([[p.data for p in n.w] for n in l2.neurons], dtype=torch.float64, requires_grad=True)
    b2 = torch.tensor([n.b.data for n in l2.neurons], dtype=torch.float64, requires_grad=True)
    h = torch.tanh(torch.tensor(x, dtype=torch.float64) @ W1.T + b1)
    loss = torch.nn.functional.cross_entropy((h @ W2.T + b2)[None], torch.tensor([label]))
    loss.backward()
    assert np.allclose(W1.grad.numpy(), [[p.grad for p in n.w] for n in l1.neurons])
    assert np.allclose(b2.grad.numpy(), [n.b.grad for n in l2.neurons])
