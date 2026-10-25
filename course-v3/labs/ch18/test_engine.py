"""第 18 章的自动检查。"""
import copy

import torch

from miniocr.models.gpt import GPT, GPTConfig
from miniocr.train.engine import flops_per_token, load_checkpoint, save_checkpoint, train_step
from miniocr.train.lm import get_batch


def small():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=40, block_size=32, n_layer=2, n_head=2, d_model=32))


def test_grad_accumulation_equals_big_batch():
    m1 = small(); m2 = copy.deepcopy(m1)
    o1 = torch.optim.SGD(m1.parameters(), lr=0.1); o2 = torch.optim.SGD(m2.parameters(), lr=0.1)
    x = torch.randint(0, 40, (16, 12)); y = torch.randint(0, 40, (16, 12))
    l1 = train_step(m1, o1, [(x, y)], clip=1e9)
    l2 = train_step(m2, o2, [(x[i:i + 4], y[i:i + 4]) for i in range(0, 16, 4)], clip=1e9)
    assert abs(l1 - l2) < 1e-5
    for a, b in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(a, b, atol=1e-6)
    assert all(p.grad is None for p in m1.parameters())                       # 做完一步后梯度清空


def run(model, opt, g, steps, ids):
    out = []
    for _ in range(steps):
        x, y = get_batch(ids, 4, 16, g)
        out.append(train_step(model, opt, [(x, y)]))
    return out


def test_checkpoint_resume_is_exact(tmp_path):
    ids = torch.randint(0, 40, (2000,))
    m = small(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); g = torch.Generator().manual_seed(0)
    ref = run(m, opt, g, 6, ids)
    m = small(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); g = torch.Generator().manual_seed(0)
    first = run(m, opt, g, 3, ids)
    save_checkpoint(tmp_path / "ck.pt", m, opt, 3, g)
    m2 = small(); torch.manual_seed(123)                                       # 换一个新模型、新优化器、乱掉的生成器
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3); g2 = torch.Generator().manual_seed(999)
    step, _ = load_checkpoint(tmp_path / "ck.pt", m2, opt2, g2)
    assert step == 3
    assert first + run(m2, opt2, g2, 3, ids) == ref                            # 逐位相同


def test_flops_per_token():
    assert flops_per_token(1000, 2, 8, 16) == 6000 + 12 * 2 * 16 * 8
