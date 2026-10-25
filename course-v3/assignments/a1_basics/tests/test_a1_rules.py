"""A1 的限制：a1/ 里不能用现成的层、损失、优化器和 im2col（检查器见 assignments/_common/rules.py）。"""
from assignments._common.rules import check_dir, violations

from ._impl import CODE_DIR


def test_no_builtin_layers_losses_optimizers():
    bad = check_dir(CODE_DIR)
    assert not bad, "a1/ 里用了被禁止的现成实现（见说明书第 1 节）：\n" + "\n".join(bad)


def test_checker_catches_examples():
    src = """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from .nn import conv2d
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(3, 4)
        self.p = nn.Parameter(torch.zeros(3))
def f(x, w):
    a = x.unfold(2, 3, 1)
    b = torch.log_softmax(x, -1)
    return conv2d(x, w), F.relu(x)
"""
    bad = " ".join(violations(src, "t.py"))
    for s in ("torch.nn.functional", "from torch.optim import AdamW", "torch.nn.Linear", ".unfold", "torch.log_softmax"):
        assert s in bad, s
    assert "Parameter" not in bad and "torch.nn.Module" not in bad and "conv2d" not in bad.replace("torch.", "")
