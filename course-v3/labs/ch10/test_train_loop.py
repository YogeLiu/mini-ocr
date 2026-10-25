"""第 10 章的自动检查。都在 CPU 上运行，几秒内完成。"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from miniocr.linalg import mlp_param_count
from miniocr.models.mlp import MLP
from miniocr.train.loop import confusion_matrix, evaluate, fit, overfit_one_batch, train_one_epoch

CPU = torch.device("cpu")


def _toy(n=200, seed=0):
    """两个高斯团：类别 0 在 (-1,-1)，类别 1 在 (+1,+1)。"""
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % 2
    x = torch.randn(n, 2, generator=g) * 0.5 + (y[:, None] * 2 - 1).float()
    return TensorDataset(x, y)


def test_mlp_structure():
    m = MLP([1024, 256, 62])
    assert sum(p.numel() for p in m.parameters()) == mlp_param_count([1024, 256, 62])
    assert m(torch.zeros(5, 1, 32, 32)).shape == (5, 62)
    kinds = [type(l).__name__ for l in MLP([8, 4, 4, 2], dropout=0.5).net]
    assert kinds == ["Linear", "ReLU", "Dropout", "Linear", "ReLU", "Dropout", "Linear"]
    assert [type(l).__name__ for l in MLP([8, 2]).net] == ["Linear"]


def test_evaluate_weighted_mean_and_no_grad():
    ds = _toy(10)
    model = nn.Linear(2, 2)
    loss, acc = evaluate(model, DataLoader(ds, batch_size=3), CPU)   # 批大小 3、3、3、1
    x, y = ds.tensors
    with torch.no_grad():
        ref = nn.functional.cross_entropy(model(x), y).item()
        ref_acc = (model(x).argmax(1) == y).float().mean().item()
    assert np.isclose(loss, ref) and np.isclose(acc, ref_acc)
    assert all(p.grad is None for p in model.parameters())          # 评测不应产生梯度


def test_train_one_epoch_reduces_loss_and_zeroes_grad():
    torch.manual_seed(0)
    ds = _toy()
    model = MLP([2, 8, 2])
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    loader = DataLoader(ds, batch_size=20, shuffle=True, generator=torch.Generator().manual_seed(0))
    before, _ = evaluate(model, loader, CPU)
    losses = [train_one_epoch(model, loader, opt, CPU) for _ in range(5)]
    after, acc = evaluate(model, loader, CPU)
    assert after < before * 0.5 and acc > 0.95 and losses[-1] < losses[0]
    # 只训练一批、lr=0：如果忘了 zero_grad，第二次调用后的 .grad 会是两倍
    opt0 = torch.optim.SGD(model.parameters(), lr=0.0)
    one = DataLoader(ds, batch_size=len(ds))
    train_one_epoch(model, one, opt0, CPU)
    g1 = model.net[0].weight.grad.clone()
    train_one_epoch(model, one, opt0, CPU)
    assert torch.allclose(model.net[0].weight.grad, g1)


def test_fit_history_and_restores_best():
    torch.manual_seed(0)
    tr, va = _toy(200, 0), _toy(100, 1)
    model = MLP([2, 8, 2])
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    hist = fit(model, DataLoader(tr, batch_size=20), DataLoader(va, batch_size=50), opt, 4, CPU, log=None)
    assert [h["epoch"] for h in hist] == [1, 2, 3, 4]
    assert set(hist[0]) == {"epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"}
    _, acc = evaluate(model, DataLoader(va, batch_size=50), CPU)
    assert np.isclose(acc, max(h["val_acc"] for h in hist))


def test_confusion_matrix():
    y = np.array([0, 0, 1, 1, 1, 2])
    p = np.array([0, 1, 1, 1, 0, 2])
    C = confusion_matrix(p, y, 3)
    assert C.tolist() == [[1, 1, 0], [1, 2, 0], [0, 0, 1]]
    assert C.sum() == len(y) and np.trace(C) == 4


def test_overfit_one_batch():
    torch.manual_seed(0)
    xb, yb = torch.randn(16, 1, 4, 4), torch.arange(16) % 4          # 随机输入、随机标签也应当能背下来
    model = MLP([16, 64, 4])
    losses = overfit_one_batch(model, xb, yb, torch.optim.Adam(model.parameters(), lr=1e-2), steps=300)
    assert losses[0] > 1.0 and losses[-1] < 0.01


def test_distort_renders_varied_glyphs():
    from miniocr.data.render import render_glyph
    a, box = render_glyph("A", np.random.default_rng(0), distort=1.0)
    b, _ = render_glyph("A", np.random.default_rng(0), distort=1.0)
    c, _ = render_glyph("A", np.random.default_rng(1), distort=1.0)
    assert a.shape == (32, 32) and a.dtype == np.uint8 and (a == b).all() and not (a == c).all()
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= 32 and 0 <= y0 < y1 <= 32 and (a < 128).sum() > 10
