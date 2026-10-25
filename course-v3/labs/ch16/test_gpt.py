"""第 16 章的自动检查。"""
import math

import torch

from miniocr.models.gpt import GPT, GPTConfig, RMSNorm, apply_rope, rope_cache


def small(**kw):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=2, d_model=32, **kw))


def test_rmsnorm():
    x = torch.randn(3, 8) * 5 + 2
    n = RMSNorm(8)
    y = n(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(3), atol=1e-4)            # 均方根变成 1
    assert torch.allclose(y, x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt(), atol=1e-6)
    assert not torch.allclose(y.mean(-1), torch.zeros(3), atol=1e-2)               # 不减均值


def test_rope_rotation_and_relative_position():
    cos, sin = rope_cache(20, 8)
    assert cos.shape == (20, 8) and torch.allclose(cos[0], torch.ones(8)) and torch.allclose(sin[0], torch.zeros(8))
    torch.manual_seed(0)
    q, k = torch.randn(8), torch.randn(8)
    rq = apply_rope(q.expand(20, 8), cos, sin)
    rk = apply_rope(k.expand(20, 8), cos, sin)
    assert torch.allclose(rq.norm(dim=-1), q.norm().expand(20), atol=1e-5)        # 旋转不改变长度
    d1 = rq[5] @ rk[2]                                                             # 位置 5 的 q 与位置 2 的 k
    d2 = rq[13] @ rk[10]                                                           # 相差同样是 3
    assert math.isclose(d1.item(), d2.item(), rel_tol=1e-4, abs_tol=1e-5)          # 点积只取决于相对位置
    assert not math.isclose(d1.item(), (rq[5] @ rk[4]).item(), rel_tol=1e-3)


def test_gpt_shapes_loss_and_tying():
    m = small()
    idx = torch.randint(0, 50, (2, 10))
    logits, loss = m(idx, targets=torch.randint(0, 50, (2, 10)))
    assert logits.shape == (2, 10, 50)
    assert abs(loss.item() - math.log(50)) < 0.2                                   # 随机目标：初始损失 ≈ ln V
    # 注意：目标若等于输入本身，损失会明显低于 ln V（共享权重时 logit_t ≈ ⟨norm(e_t), e_t⟩，偏爱"重复当前 token"，第 16 章）
    assert m.head.weight is m.tok.weight
    assert m.n_params() == sum(p.numel() for p in set(m.parameters()))
    t = idx.clone(); t[:, :5] = -100
    _, l2 = m(idx, targets=t)
    _, l3 = m(idx, targets=idx, loss_mask=torch.arange(10).expand(2, 10) >= 5)
    assert torch.allclose(l2, l3)


def test_gpt_is_causal():
    m = small().eval()
    a = torch.randint(0, 50, (1, 12))
    b = a.clone(); b[0, 7:] = (b[0, 7:] + 1) % 50
    la, lb = m(a)[0], m(b)[0]
    assert torch.allclose(la[:, :7], lb[:, :7], atol=1e-5)
    assert not torch.allclose(la[:, 7:], lb[:, 7:])


def test_kv_cache_matches_full_forward():
    m = small().eval()
    idx = torch.randint(0, 50, (2, 12))
    full = m(idx)[0]
    caches = [dict() for _ in m.blocks]
    out = [m(idx[:, :5], caches=caches)[0]]                                        # 先一次处理 5 个（预填充）
    for t in range(5, 9):                                                          # 再逐个
        out.append(m(idx[:, t:t + 1], caches=caches, pos0=t)[0])
    out.append(m(idx[:, 9:], caches=caches, pos0=9)[0])                            # 最后一次处理 3 个
    assert torch.allclose(torch.cat(out, 1), full, atol=1e-5)
