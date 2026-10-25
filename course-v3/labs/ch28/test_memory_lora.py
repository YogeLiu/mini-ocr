"""第 28 章的自动检查。"""
import copy

import torch
import torch.nn as nn

from miniocr.models.gpt import GPT, GPTConfig
from miniocr.models.lora import LoRALinear, apply_lora, count_params, merge_lora
from miniocr.train.memory import SavedTensors, checkpoint_blocks, static_memory


def tiny(seed=0):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=2, d_model=32))


def test_lora_linear_init_and_shapes():
    torch.manual_seed(0)
    base = nn.Linear(6, 4)
    lo = LoRALinear(copy.deepcopy(base), r=2, alpha=4)
    assert lo.A.shape == (2, 6) and lo.B.shape == (4, 2) and lo.scale == 2.0
    x = torch.randn(3, 6)
    assert torch.allclose(lo(x), base(x))                                # B = 0：一开始与原来的层相同
    with torch.no_grad():
        lo.B.normal_()
    assert torch.allclose(lo(x), base(x) + 2.0 * x @ (lo.B @ lo.A).T, atol=1e-5)
    assert not lo.base.weight.requires_grad and lo.A.requires_grad


def test_apply_lora_counts_and_grads():
    m = tiny()
    total0, _ = count_params(m)
    names = apply_lora(m, targets=("qkv", "o"), r=4, alpha=8)
    assert names == ["blocks.0.attn.qkv", "blocks.0.attn.o", "blocks.1.attn.qkv", "blocks.1.attn.o"]
    total, trainable = count_params(m)
    assert trainable == 2 * (4 * (32 + 96) + 4 * (32 + 32))            # 每层 r·(in + out)：qkv 32→96，o 32→32
    assert total == total0 + trainable
    assert apply_lora(m, targets=("qkv", "o")) == []                    # 不会重复包
    ids = torch.randint(0, 50, (2, 10))
    loss = m(ids, targets=ids)[1]
    loss.backward()
    for n, p in m.named_parameters():
        assert (p.grad is not None) == p.requires_grad, n


def test_merge_equals_unmerged():
    m = tiny()
    apply_lora(m, targets=("qkv", "o", "gate", "up", "down"), r=4)
    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            with torch.no_grad():
                mod.B.normal_(0, 0.1)
    ids = torch.randint(0, 50, (2, 10))
    with torch.no_grad():
        a = m(ids)[0]
        assert merge_lora(m) == 10
        b = m(ids)[0]
    assert torch.allclose(a, b, atol=1e-4)
    assert not any(isinstance(x, LoRALinear) for x in m.modules())


def test_static_memory():
    s = static_memory(1000, 100)
    assert s == dict(params=4000, grads=400, optim=800, master=0, total=5200)
    s = static_memory(1000, 1000, "bf16", master_fp32=True)
    assert s["params"] == 2000 and s["grads"] == 2000 and s["optim"] == 8000 and s["master"] == 4000


def test_saved_tensors_and_checkpoint():
    lin = nn.Linear(8, 8, bias=False)
    x = torch.randn(4, 8, requires_grad=True)
    with SavedTensors(lin) as s:
        y = torch.relu(lin(x))
    assert s.bytes == 2 * 4 * 8 * 4                                     # matmul 存 x，relu 存输出；权重不算
    m = tiny()
    ids = torch.randint(0, 50, (2, 16))
    with SavedTensors(m) as s1:
        l1 = m(ids, targets=ids)[1]
    l1.backward()
    g1 = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    checkpoint_blocks(m.blocks)
    m.train()
    with SavedTensors(m) as s2:
        l2 = m(ids, targets=ids)[1]
    l2.backward()
    assert s2.bytes < s1.bytes / 2                                      # 检查点之后保存的激活值少得多
    assert abs(l1.item() - l2.item()) < 1e-6
    for a, p in zip(g1, m.parameters()):
        assert torch.allclose(a, p.grad, atol=1e-6)                     # 梯度完全相同
