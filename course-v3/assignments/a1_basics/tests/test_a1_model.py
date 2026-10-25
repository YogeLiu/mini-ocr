"""A1 问题 char_cnn、checkpoint。"""
import io

import torch
import torch.nn as nn

from ._impl import adapters as A


def _keys(channels):
    ks = {"head.weight", "head.bias"}
    for i in range(len(channels)):
        ks |= {f"convs.{i}.weight"} | {f"bns.{i}.{n}" for n in ("weight", "bias", "running_mean", "running_var")}
    return ks


def _reference(num_classes, channels):
    """用 torch.nn 搭的同一结构（测试里可以用现成的层），state_dict 的键与说明书一致。"""
    class Ref(nn.Module):
        def __init__(self):
            super().__init__()
            cins = (1,) + tuple(channels[:-1])
            self.convs = nn.ModuleList(nn.Conv2d(a, b, 3, padding=1, bias=False) for a, b in zip(cins, channels))
            self.bns = nn.ModuleList(nn.BatchNorm2d(c) for c in channels)
            self.head = nn.Linear(channels[-1], num_classes)

        def forward(self, x):
            for c, b in zip(self.convs, self.bns):
                x = nn.functional.max_pool2d(torch.relu(b(c(x))), 2)
            return self.head(x.mean(dim=(2, 3)))
    return Ref()


def test_state_dict_keys_and_param_count():
    ch = (8, 16, 32)
    m = A.get_char_cnn(62, ch)
    sd = m.state_dict()
    assert {k for k in sd if not k.endswith("num_batches_tracked")} == _keys(ch)
    n = 9 * (1 * 8 + 8 * 16 + 16 * 32) + 2 * (8 + 16 + 32) + 32 * 62 + 62
    assert sum(p.numel() for p in m.parameters()) == n
    assert m(torch.zeros(5, 1, 32, 32)).shape == (5, 62)


def test_char_cnn_matches_reference():
    torch.manual_seed(0)
    ch = (8, 16, 32)
    ref = _reference(62, ch)
    for b in ref.bns:                                                 # 随机的运行统计量，确认评测模式用的是它们
        b.running_mean.normal_()
        b.running_var.uniform_(0.5, 2.0)
        b.weight.data.normal_()
        b.bias.data.normal_()
    sd = {k: v for k, v in ref.state_dict().items() if not k.endswith("num_batches_tracked")}
    x = torch.rand(3, 1, 32, 32)
    torch.testing.assert_close(A.run_char_cnn(62, ch, sd, x), ref.eval()(x), atol=1e-5, rtol=1e-4)


def test_checkpoint_resume_is_exact():
    torch.manual_seed(0)
    AdamW = A.get_adamw_cls()
    a = A.get_char_cnn(10, (4, 8, 8))
    oa = AdamW(a.parameters(), lr=1e-2, weight_decay=0.1)
    g = torch.Generator().manual_seed(0)
    data = [(torch.rand(6, 1, 32, 32, generator=g), torch.randint(0, 10, (6,), generator=g)) for _ in range(6)]

    def step(m, o, x, y):
        o.zero_grad()
        nn.functional.cross_entropy(m(x), y).backward()
        o.step()

    for x, y in data[:3]:
        step(a, oa, x, y)
    buf = io.BytesIO()
    A.run_save_checkpoint(a, oa, 3, buf)
    buf.seek(0)
    b = A.get_char_cnn(10, (4, 8, 8))
    ob = AdamW(b.parameters(), lr=1e-2, weight_decay=0.1)
    assert A.run_load_checkpoint(buf, b, ob) == 3
    for x, y in data[3:]:
        step(a, oa, x, y)
        step(b, ob, x, y)
    for (k, p), q in zip(a.state_dict().items(), b.state_dict().values()):
        torch.testing.assert_close(p, q, msg=k)
