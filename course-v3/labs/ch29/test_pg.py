"""第 29 章的自动检查。"""
import functools

import torch

from miniocr.data.corpus import text_vocab
from miniocr.models.gpt import GPT, GPTConfig
from miniocr.models.vit import ViT, ViTConfig
from miniocr.models.vlm import VLM, MLPProjector
from miniocr.rl.pg import clipped_objective, group_advantages, kl_estimate, loo_advantages, zero_variance_fraction
from miniocr.rl.rollout import make_batch, policy_logprobs, rollout


@functools.cache
def vocab():
    return text_vocab()


def test_group_advantages():
    r = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.3, 0.3, 0.3, 0.3, 0.0, 2.0, 4.0, 6.0])
    a = group_advantages(r, 4).view(3, 4)
    # 手算：均值 0.5，偏差 ±0.5，无偏标准差 √(4 × 0.25 / 3) = 0.577，优势 ±0.5 / 0.577 = ±0.866
    assert torch.allclose(a[0], torch.tensor([0.866, -0.866, -0.866, 0.866]), atol=1e-3)
    assert torch.allclose(a.mean(1), torch.zeros(3), atol=1e-5)
    assert torch.all(a[1] == 0)                                       # 组内全相同：优势为 0
    assert torch.allclose(a[2].std(), torch.tensor(1.0), atol=1e-4)   # 标准化之后标准差为 1
    b = group_advantages(10 * r + 7, 4)                               # 整体平移、缩放不改变优势
    assert torch.allclose(b, a.view(-1), atol=1e-4)
    assert torch.allclose(group_advantages(r, 4, std=False).view(3, 4)[2], torch.tensor([-3.0, -1.0, 1.0, 3.0]))
    assert abs(zero_variance_fraction(r, 4) - 1 / 3) < 1e-6


def test_loo_advantages():
    torch.manual_seed(0)
    r = torch.rand(5 * 6)
    a = loo_advantages(r, 6).view(5, 6)
    rr = r.view(5, 6)
    for g in range(5):
        for i in range(6):
            others = torch.cat([rr[g, :i], rr[g, i + 1:]])
            assert abs(a[g, i] - (rr[g, i] - others.mean())) < 1e-6
    # 留一基线的优势 = G / (G − 1) × (r − 组内均值)
    assert torch.allclose(a, (rr - rr.mean(1, keepdim=True)) * 6 / 5, atol=1e-6)


def test_kl_estimators():
    g = torch.Generator().manual_seed(0)
    pi = torch.softmax(torch.randn(20, generator=g), 0)
    ref = torch.softmax(torch.log(pi) + 0.3 * torch.randn(20, generator=g), 0)
    exact = (pi * (pi.log() - ref.log())).sum()
    x = torch.multinomial(pi, 200_000, replacement=True, generator=g)
    lp, rlp = pi.log()[x], ref.log()[x]
    k1, k2, k3 = (kl_estimate(lp, rlp, k) for k in ("k1", "k2", "k3"))
    assert abs(k1.mean() - exact) < 0.003 and abs(k3.mean() - exact) < 0.003
    assert k3.min() >= 0 and k1.min() < 0
    assert k3.std() < k1.std()
    assert torch.allclose(k2, (rlp - lp) ** 2 / 2)
    same = kl_estimate(lp, lp, "k3")
    assert torch.all(same == 0)


def test_clipped_objective_gradients():
    old = torch.zeros(6)
    ratios = torch.tensor([0.5, 0.9, 1.0, 1.1, 1.5, 3.0])
    for A in (1.0, -1.0):
        logp = torch.log(ratios).clone().requires_grad_(True)
        obj, clipped = clipped_objective(logp, old, torch.tensor(A))
        obj.sum().backward()
        g = logp.grad
        if A > 0:      # 好回答：ρ > 1.2 之后不再推高（梯度 0）；ρ 很小时照常推高
            assert torch.equal(clipped, torch.tensor([False, False, False, False, True, True]))
            assert g[4] == 0 and g[5] == 0 and g[0] > 0
        else:          # 坏回答：ρ < 0.8 之后不再压低；ρ 很大时照常压低
            assert torch.equal(clipped, torch.tensor([True, False, False, False, False, False]))
            assert g[0] == 0 and g[5] < 0
        assert torch.allclose(g[2], torch.tensor(A))                   # ρ = 1 时梯度就是 A（普通的策略梯度）
    # 不对称的裁剪（第 31 章 DAPO 的 clip-higher）
    logp = torch.log(torch.tensor([1.25])).requires_grad_(True)
    obj, clipped = clipped_objective(logp, torch.zeros(1), torch.tensor(1.0), eps_low=0.2, eps_high=0.28)
    assert not clipped.item()


def tiny_vlm():
    torch.manual_seed(0)
    V = vocab()
    vit = ViT(ViTConfig(img_size=16, patch=8, d_model=32, n_layer=1, n_head=4, pool="mean"))
    lm = GPT(GPTConfig(vocab_size=len(V), block_size=64, n_layer=1, n_head=2, d_model=32))
    return VLM(vit, MLPProjector(32, 32), lm, V.stoi["<img>"])


def test_make_batch_mask():
    ids, mask = make_batch([[1, 2, 3], [1, 2]], [[7, 8], [9]], [True, False], eos=0, pad=5)
    assert ids.tolist() == [[1, 2, 3, 7, 8, 0], [1, 2, 9, 5, 5, 5]]
    # mask[t] = 1 当且仅当 ids[t + 1] 是回答（含 eos）
    assert mask.tolist() == [[0, 0, 1, 1, 1], [0, 1, 0, 0, 0]]


def test_rollout_and_logprobs():
    V = vocab()
    m = tiny_vlm()
    eos, pad = V.stoi["<eos>"], V.stoi["<pad>"]
    img = torch.randn(2, 3, 16, 16)
    prompts = [[V.stoi["<bos>"]] + [V.stoi["<img>"]] * 4 + V.encode("Q?"), [V.stoi["<bos>"]] + [V.stoi["<img>"]] * 4 + V.encode("QQ?")]
    resp, fin = rollout(m, img, prompts, G=3, max_new=6, eos=eos, generator=torch.Generator().manual_seed(0))
    assert len(resp) == 6 and all(len(r) <= 6 for r in resp)
    assert all(f == (len(r) < 6) or f for r, f in zip(resp, fin))
    again, _ = rollout(m, img, prompts, G=3, max_new=6, eos=eos, generator=torch.Generator().manual_seed(0))
    assert again == resp                                               # 同一个随机数生成器：结果可复现
    rows = [i // 3 for i in range(6)]
    ids, mask = make_batch([prompts[i] for i in rows], resp, fin, eos, pad)
    assert mask.sum().item() == sum(len(r) + f for r, f in zip(resp, fin))
    with torch.no_grad():
        lp = policy_logprobs(m, img[rows], ids)
    for b in range(6):                                                 # 每个回答的对数概率可以逐条单独算出来，补齐不影响
        n = len(prompts[rows[b]]) + len(resp[b]) + fin[b]
        with torch.no_grad():
            one = policy_logprobs(m, img[rows[b]:rows[b] + 1], ids[b:b + 1, :n])
        assert torch.allclose(lp[b, :n - 1] * mask[b, :n - 1], one[0] * mask[b, :n - 1], atol=1e-5)
