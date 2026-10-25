"""A4 第四部分：token 的身份、标签与权重、两种平均、打包、打包的 mask、打包后的前向与逐条前向相同。"""
import pytest
import torch
import torch.nn.functional as F

from ._impl import adapters as A

FMT, IMG_R, USER, ANS = 0, 1, 2, 3
enc = lambda s: [ord(c) for c in s]                                                   # noqa: E731


def test_chat_roles():
    msgs = [{"role": "user", "content": "<image>Q?"}, {"role": "assistant", "content": "ab"},
            {"role": "user", "content": "R"}, {"role": "assistant", "content": "c"}]
    ids, roles = A.run_chat_roles(msgs, 2, enc, 1, 2, 3)
    expect = ([1] + enc("User: ") + [3, 3] + enc("Q?") + enc("\n") + enc("Assistant: ") + enc("ab") + [2]
              + enc("User: ") + enc("R") + enc("\n") + enc("Assistant: ") + enc("c") + [2])
    assert ids == expect
    r = ([FMT] + [FMT] * 6 + [IMG_R] * 2 + [USER] * 2 + [FMT] + [FMT] * 11 + [ANS] * 3
         + [FMT] * 6 + [USER] + [FMT] + [FMT] * 11 + [ANS] * 2)
    assert roles == r


def test_chat_roles_per_image_counts():
    msgs = [{"role": "user", "content": "A<image>B<image>"}, {"role": "assistant", "content": "x"}]
    ids, roles = A.run_chat_roles(msgs, [2, 5], enc, 1, 2, 3)
    assert ids.count(3) == 7
    k = ids.index(3)
    assert ids[k:k + 2] == [3, 3] and ids[k + 2] == ord("B") and ids[k + 3:k + 8] == [3] * 5


def test_sft_targets():
    ids = [1, 10, 11, 3, 12, 20, 21, 2]
    roles = [FMT, FMT, USER, IMG_R, USER, ANS, ANS, ANS]
    tg, w = A.run_sft_targets(ids, roles, 0.0, 2, True)
    assert tg == [-100, -100, -100, -100, 20, 21, 2, -100]
    assert w == [0, 0, 0, 0, 1, 1, 1, 0]
    tg, w = A.run_sft_targets(ids, roles, 0.1, 2, True)
    assert tg == [10, 11, -100, 12, 20, 21, 2, -100]                                   # 图像占位符永远不算
    assert w == pytest.approx([0.1, 0.1, 0, 0.1, 1, 1, 1, 0])
    tg, w = A.run_sft_targets(ids, roles, 0.0, 2, False)
    assert tg[6] == -100 and w[6] == 0                                                 # 不学 eos


def _rand(B=3, T=6, V=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, T, V, generator=g, requires_grad=True)
    tg = torch.randint(0, V, (B, T), generator=g)
    tg[0, :4] = -100
    tg[2, :] = -100
    w = torch.rand(B, T, generator=g)
    return logits, tg, w


def test_sft_loss_token_and_sample():
    logits, tg, w = _rand()
    ce = F.cross_entropy(logits.reshape(-1, 9), tg.clamp(min=0).reshape(-1), reduction="none").view(3, 6)
    ww = w * (tg != -100)
    assert torch.isclose(A.run_sft_loss(logits, tg, w, "token"), (ww * ce).sum() / ww.sum(), atol=1e-6)
    per = (ww * ce).sum(1) / ww.sum(1).clamp(min=1e-8)
    assert torch.isclose(A.run_sft_loss(logits, tg, w, "sample"), per[:2].mean(), atol=1e-6)    # 第 2 条没有可算的位置
    assert torch.isclose(A.run_sft_loss(logits, tg, None, "token"), F.cross_entropy(logits.reshape(-1, 9), tg.reshape(-1)),
                         atol=1e-6)


def test_sft_loss_all_masked():
    logits, tg, _ = _rand()
    loss = A.run_sft_loss(logits, torch.full_like(tg, -100), None, "token")
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def _ex(n, base):
    return list(range(base, base + n)), [base + 100 + i for i in range(n)], [1.0] * n


def test_pack():
    exs = [_ex(3, 10), _ex(4, 20), _ex(2, 30), _ex(5, 40)]
    P = A.run_pack(exs, 7, 0)
    assert P["order"] == [[0, 1], [2, 3]]
    assert P["ids"].tolist() == [[10, 11, 12, 20, 21, 22, 23], [30, 31, 40, 41, 42, 43, 44]]
    assert P["seg"].tolist() == [[1, 1, 1, 2, 2, 2, 2], [1, 1, 2, 2, 2, 2, 2]]
    assert P["pos"].tolist() == [[0, 1, 2, 0, 1, 2, 3], [0, 1, 0, 1, 2, 3, 4]]
    P = A.run_pack([_ex(2, 10), _ex(3, 20)], 6, 0)
    assert P["ids"].tolist() == [[10, 11, 20, 21, 22, 0]]
    assert P["seg"][0, -1] == 0 and P["targets"][0, -1] == -100 and P["weights"][0, -1] == 0
    with pytest.raises(ValueError):
        A.run_pack([_ex(8, 0)], 7, 0)


def test_packed_mask():
    seg = torch.tensor([[1, 1, 2, 2, 2, 0], [1, 1, 1, 1, 0, 0]])
    m = A.run_packed_mask(seg)
    assert m.shape == (2, 1, 6, 6) and m.dtype == torch.bool
    for b in range(2):
        for i in range(6):
            for j in range(6):
                expect = (j <= i and seg[b, i] == seg[b, j] != 0) or (i == j)
                assert bool(m[b, 0, i, j]) == expect


def test_packed_forward_equals_separate():
    torch.manual_seed(0)
    lm = A.get_lm(30, 64, 2, 2, 32).eval()
    seqs = [torch.randint(0, 30, (n,)).tolist() for n in (5, 7, 4)]
    exs = [(s, s[1:] + [-100], [1.0] * len(s)) for s in seqs]
    P = A.run_pack(exs, 16, 0)
    assert len(P["order"]) == 1
    with torch.no_grad():
        packed = A.run_lm_logits(lm, P["ids"], P["pos"], A.run_packed_mask(P["seg"]))
        s = 0
        for q in seqs:
            alone = A.run_lm_logits(lm, torch.tensor([q]))
            assert torch.allclose(packed[0, s:s + len(q)], alone[0], atol=1e-5)
            s += len(q)


def test_packed_loss_equals_padded():
    torch.manual_seed(0)
    lm = A.get_lm(30, 64, 2, 2, 32)
    seqs = [torch.randint(1, 30, (n,)).tolist() for n in (6, 3, 8, 5)]
    exs = [(s, s[1:] + [-100], [1.0] * (len(s) - 1) + [0.0]) for s in seqs]
    ids = torch.zeros(4, 8, dtype=torch.long)
    tg = torch.full((4, 8), -100)
    for b, (i, t, _) in enumerate(exs):
        ids[b, :len(i)] = torch.tensor(i)
        tg[b, :len(t)] = torch.tensor(t)
    padded = A.run_sft_loss(A.run_lm_logits(lm, ids), tg, None, "token")
    P = A.run_pack(exs, 12, 0)
    packed = A.run_sft_loss(A.run_lm_logits(lm, P["ids"], P["pos"], A.run_packed_mask(P["seg"])), P["targets"],
                            P["weights"], "token")
    assert torch.isclose(padded, packed, atol=1e-5)
