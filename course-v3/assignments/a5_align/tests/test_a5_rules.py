"""A5 的限制：a5/ 里不能用现成的层、损失、优化器、注意力、插值（检查器见 assignments/_common/rules.py）。"""
from assignments._common.rules import check_dir

from ._impl import CODE_DIR


def test_no_builtin_layers_losses_optimizers():
    bad = check_dir(CODE_DIR)
    assert not bad, "a5/ 里用了被禁止的现成实现（见说明书第 1 节）：\n" + "\n".join(bad)
