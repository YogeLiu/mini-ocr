"""第 12 章的自动检查。"""
import numpy as np
import torch
import torch.nn.functional as F

from miniocr.models.cnn import ResBlock, SmallCNN, random_translate
from miniocr.nn.conv import conv2d_im2col, conv2d_loops, conv_out_size, im2col, maxpool2d, receptive_field


def test_out_size_and_receptive_field():
    assert conv_out_size(32, 3, 1, 1) == 32 and conv_out_size(32, 2, 2) == 16
    assert conv_out_size(7, 3, 2, 0) == 3 and conv_out_size(5, 5) == 1
    assert receptive_field([(3, 1)]) == 3 and receptive_field([(3, 1), (3, 1)]) == 5
    assert receptive_field([(3, 1), (2, 2), (3, 1), (2, 2), (3, 1), (2, 2)]) == 22


def test_conv_hand_example():
    x = np.arange(16, dtype=np.float64).reshape(1, 1, 4, 4)
    w = np.array([[[[1, 0], [0, -1]]]], dtype=np.float64)        # 左上 − 右下
    out = conv2d_loops(x, w)
    assert out.shape == (1, 1, 3, 3) and np.all(out == -5)
    assert conv2d_loops(x, w, np.array([2.0]), stride=2).tolist() == [[[[-3.0, -3.0], [-3.0, -3.0]]]]


def test_conv_loops_and_im2col_match_torch():
    rng = np.random.default_rng(0)
    x, w, b = rng.normal(size=(2, 3, 7, 6)), rng.normal(size=(4, 3, 3, 3)), rng.normal(size=4)
    for s, p in ((1, 0), (1, 1), (2, 1)):
        ref = F.conv2d(torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(b), stride=s, padding=p).numpy()
        assert np.allclose(conv2d_loops(x, w, b, s, p), ref)
        assert np.allclose(conv2d_im2col(x, w, b, s, p), ref)
    assert im2col(x, 3, 3).shape == (2 * 5 * 4, 27)


def test_maxpool():
    x = np.arange(32, dtype=np.float64).reshape(1, 2, 4, 4)
    ref = F.max_pool2d(torch.from_numpy(x), 2).numpy()
    assert np.array_equal(maxpool2d(x, 2), ref)
    assert maxpool2d(np.zeros((1, 1, 5, 5)), 2).shape == (1, 1, 2, 2)


def test_small_cnn_and_resblock():
    m = SmallCNN(62, 32)
    assert m(torch.zeros(3, 1, 32, 32)).shape == (3, 62)
    n = sum(p.numel() for p in m.parameters())
    assert n == 288 + 64 + 32 * 64 * 9 + 128 + 64 * 128 * 9 + 256 + 128 * 62 + 62
    assert SmallCNN(10, 8, pool="flatten")(torch.zeros(2, 1, 32, 32)).shape == (2, 10)
    r = ResBlock(8)
    x = torch.randn(2, 8, 6, 6)
    assert r(x).shape == x.shape
    with torch.no_grad():                              # 第二个 BN 的 γ 置 0 时，残差块退化为 ReLU(x)
        r.bn2.weight.zero_()
    r.eval()
    assert torch.allclose(r(x), F.relu(x))


def test_random_translate():
    x = torch.zeros(4, 1, 8, 8)
    x[:, 0, 3, 4] = 1.0
    g = torch.Generator().manual_seed(0)
    y = random_translate(x, 2, g)
    assert y.shape == x.shape and x.sum() == 4 and torch.all(y.sum((1, 2, 3)) == 1)
    pos = [tuple(torch.nonzero(y[b, 0])[0].tolist()) for b in range(4)]
    assert all(abs(r - 3) <= 2 and abs(c - 4) <= 2 for r, c in pos)
    d = torch.randint(-2, 3, (4, 2), generator=torch.Generator().manual_seed(0))
    assert pos == [(3 + int(d[b, 0]), 4 + int(d[b, 1])) for b in range(4)]   # 正的 dy 向下、正的 dx 向右
    assert torch.equal(random_translate(x, 0), x)
