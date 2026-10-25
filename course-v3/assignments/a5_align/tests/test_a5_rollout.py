"""A5 第四部分：温度采样（KV cache、可复现）、每题 G 个回答的顺序、(ids, mask) 的整理、整条序列的对数概率。"""
import numpy as np
import pytest
import torch

from ._impl import adapters as A

V, IMG, EOS = 12, 3, 2


def _model(scale=1.0):
    torch.manual_seed(0)
    m = A.get_vlm(V, IMG).eval()
    with torch.no_grad():                                                               # 放大 logit：分布不那么均匀，T = 1 与 T = 2 才分得开
        m.lm.state_dict()["tok.weight"].mul_(scale)
    return m


def _prompt(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    text = torch.randint(4, V, (B, 4), generator=g)
    return torch.cat([text[:, :1], torch.full((B, 4), IMG), text[:, 1:]], 1)            # 32×32 的图、2×2 合并：4 个视觉 token


def _greedy(m, imgs, ids, n):
    seq = ids
    with torch.no_grad():
        for _ in range(n):
            seq = torch.cat([seq, A.run_vlm_logits(m, imgs, seq)[:, -1].argmax(-1, keepdim=True)], 1)
    return seq[:, ids.shape[1]:]


def test_temperature_zero_is_greedy():
    m = _model(3.0)
    imgs, P = torch.randn(2, 3, 32, 32), _prompt()
    out = A.run_sample(m, imgs, P, 6, EOS, 0.0, None)
    ref = _greedy(m, imgs, P, out.shape[1])
    assert torch.equal(out, ref)


def test_sampling_reproducible():
    m = _model()
    imgs, P = torch.randn(2, 3, 32, 32), _prompt()
    a = A.run_sample(m, imgs, P, 8, -1, 1.0, torch.Generator().manual_seed(5))
    b = A.run_sample(m, imgs, P, 8, -1, 1.0, torch.Generator().manual_seed(5))
    c = A.run_sample(m, imgs, P, 8, -1, 1.0, torch.Generator().manual_seed(6))
    assert a.shape == (2, 8) and torch.equal(a, b) and not torch.equal(a, c)


@pytest.mark.parametrize("T", [1.0, 2.0])
def test_first_token_distribution(T):
    m = _model(3.0)
    imgs, P = torch.randn(1, 3, 32, 32), _prompt(1)
    with torch.no_grad():
        p = torch.softmax(A.run_vlm_logits(m, imgs, P)[0, -1] / T, -1)
    n = 3000
    out = A.run_sample(m, imgs.expand(n, -1, -1, -1), P.expand(n, -1), 1, -1, T, torch.Generator().manual_seed(0))
    freq = torch.bincount(out[:, 0], minlength=V).float() / n
    assert (freq - p).abs().max() < 4 * (p * (1 - p) / n).sqrt().max() + 1e-3


def test_banned_tokens_never_sampled():
    m = _model()
    imgs, P = torch.randn(2, 3, 32, 32), _prompt()
    out = A.run_sample(m, imgs.repeat(50, 1, 1, 1), P.repeat(50, 1), 6, -1, 3.0, torch.Generator().manual_seed(0),
                       banned=(0, 1, IMG))
    assert not torch.isin(out, torch.tensor([0, 1, IMG])).any()


def test_rollout_order():
    m = _model(3.0)
    rng = np.random.default_rng(0)
    items = [(rng.integers(0, 256, (1, 3, 32, 32), dtype=np.uint8), _prompt(1, s)[0].tolist()) for s in range(3)]
    items.append((items[0][0], _prompt(1, 9)[0, :7].tolist()))                          # 提示长度不同的一题
    resp, fin = A.run_rollout(m, items, 3, 5, EOS, 0.0, None)
    assert len(resp) == len(fin) == 12
    for i, (img, p) in enumerate(items):
        x = torch.from_numpy(img).float() / 127.5 - 1
        g = _greedy(m, x, torch.tensor([p]), 5)[0].tolist()
        g = g[:g.index(EOS)] if EOS in g else g
        for j in range(3):
            assert resp[i * 3 + j] == g                                                 # 温度 0：同一题的 G 个回答相同，且顺序对
            assert fin[i * 3 + j] == (len(g) < 5 or EOS in _greedy(m, x, torch.tensor([p]), 5)[0].tolist())


def test_make_batch():
    ids, mask = A.run_make_batch([[1, 5, 6], [1, 7]], [[8, 9], [10, 11, 4]], [True, False], EOS, 0)
    assert ids.tolist() == [[1, 5, 6, 8, 9, EOS], [1, 7, 10, 11, 4, 0]]
    assert mask.tolist() == [[0, 0, 1, 1, 1], [0, 1, 1, 1, 0]]


def test_policy_logprobs():
    m = _model()
    imgs = [torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)]
    ids = _prompt()
    lp = A.run_policy_logprobs(m, imgs, ids)
    logits = A.run_vlm_logits(m, torch.cat(imgs), ids)
    ref = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    assert lp.shape == (2, ids.shape[1] - 1) and torch.allclose(lp, ref, atol=1e-5)
