"""A4 第五部分：LoRA 层、装进模型、合并、参数计数；静态内存估算、保存下来的激活值、梯度检查点。"""
import pytest
import torch

from ._impl import adapters as A


def test_lora_linear():
    torch.manual_seed(0)
    W, b = torch.randn(6, 5), torch.randn(6)
    layer = A.get_lora_linear(W, b, 3, 6.0)
    x = torch.randn(4, 5)
    assert layer.A.shape == (3, 5) and layer.B.shape == (6, 3)
    assert torch.allclose(layer(x), x @ W.T + b, atol=1e-6)                            # B = 0：与原层相同
    with torch.no_grad():
        layer.B.copy_(torch.randn(6, 3))
    expect = x @ W.T + b + (6.0 / 3) * (x @ layer.A.T) @ layer.B.T
    assert torch.allclose(layer(x), expect, atol=1e-5)
    trainable = {id(p) for p in layer.parameters() if p.requires_grad}
    assert trainable == {id(layer.A), id(layer.B)}
    assert A.get_lora_linear(W, None, 2, 2.0)(x).shape == (4, 6)


def _lm():
    torch.manual_seed(0)
    return A.get_lm(40, 32, 2, 2, 16)


def test_apply_lora():
    lm = _lm()
    total0, _ = A.run_count_params(lm)
    x = torch.randint(0, 40, (2, 9))
    before = A.run_lm_logits(lm, x).detach()
    names = A.run_apply_lora(lm, ["qkv", "o"], 4, 8.0)
    assert sorted(names) == ["blocks.0.attn.o", "blocks.0.attn.qkv", "blocks.1.attn.o", "blocks.1.attn.qkv"]
    assert torch.allclose(A.run_lm_logits(lm, x), before, atol=1e-6)
    d = 16
    lora_n = 2 * (4 * (d + 3 * d) + 4 * (d + d))                                       # 每层 qkv：r·(in + out)，o 同理
    total, trainable = A.run_count_params(lm)
    assert trainable == lora_n and total == total0 + lora_n
    assert A.run_apply_lora(lm, ["qkv", "o"], 4, 8.0) == []                            # 已经包过的不再包


def test_merge_lora():
    lm = _lm()
    keys = set(lm.state_dict())
    A.run_apply_lora(lm, ["qkv", "gate", "down"], 2, 4.0)
    with torch.no_grad():
        for n, p in lm.named_parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.1)
    x = torch.randint(0, 40, (2, 9))
    before = A.run_lm_logits(lm, x).detach()
    assert A.run_merge_lora(lm) == 6
    assert torch.allclose(A.run_lm_logits(lm, x), before, atol=1e-5)
    assert set(lm.state_dict()) == keys                                                 # 合并后和原模型的结构一样


def test_static_memory():
    s = A.run_static_memory(1000, 100, "fp32", "adamw", False)
    assert (s["params"], s["grads"], s["optim"], s["master"], s["total"]) == (4000, 400, 800, 0, 5200)
    s = A.run_static_memory(1000, 1000, "bf16", "adamw", True)
    assert (s["params"], s["grads"], s["optim"], s["master"], s["total"]) == (2000, 2000, 8000, 4000, 16000)
    assert A.run_static_memory(10, 10, "fp32", "sgd", True)["total"] == 80                 # fp32 不需要主权重


def test_saved_tensors():
    a = torch.randn(256, 256, requires_grad=True)
    with A.get_saved_tensors() as s:
        b = a.exp()                                                                     # 保存结果 b
        c = b[:10].sin()                                                                # 保存输入：b 的视图，同一块存储
    assert s.bytes == 256 * 256 * 4
    c.sum().backward()
    lin = torch.nn.Linear(256, 256)
    x = torch.randn(8, 256, requires_grad=True)
    with A.get_saved_tensors(lin) as s:
        y = torch.relu(lin(x))                                                          # 保存 x（参数不算）和 relu 的输出
    assert s.bytes == 8 * 256 * 4 * 2
    y.sum().backward()


def test_checkpoint_blocks():
    torch.manual_seed(0)
    lm1 = A.get_lm(40, 64, 3, 2, 32)
    lm2 = A.get_lm(40, 64, 3, 2, 32)
    lm2.load_state_dict(lm1.state_dict())
    assert A.run_checkpoint_blocks(lm2.blocks) == 3
    assert A.run_checkpoint_blocks(lm2.blocks) == 0
    lm1.train(), lm2.train()
    x = torch.randint(0, 40, (4, 48))
    with A.get_saved_tensors(lm1) as s1:
        l1 = A.run_lm_logits(lm1, x).logsumexp(-1).mean()
    with A.get_saved_tensors(lm2) as s2:
        l2 = A.run_lm_logits(lm2, x).logsumexp(-1).mean()
    l1.backward()
    l2.backward()
    assert torch.isclose(l1, l2, atol=1e-6)
    g1 = [p.grad for p in lm1.parameters()]
    g2 = [p.grad for p in lm2.parameters()]
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(g1, g2))
    assert s2.bytes < s1.bytes / 2
