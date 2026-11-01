"""A4 的限制：rules.toml 列出的机制实现目录里不能用现成的层、损失、优化器、注意力（检查器见 assignments/_common/rules.py）。"""
from assignments._common.rules import check_assignment, check_dir, violations

from ._impl import ASSIGNMENT_DIR


def test_no_builtin_layers_losses_optimizers():
    bad = check_assignment(ASSIGNMENT_DIR)
    assert not bad, "mechanism/ 里用了被禁止的现成实现（见 handout.md 1.1 节）：\n" + "\n".join(bad)


def test_checker_catches_common_shortcuts():
    """检查器自身的正反测试：这些写法必须被抓到，那些必须放行。"""
    for src in ["import torch.nn.functional as F\ny = F.scaled_dot_product_attention(q, k, v)",
                "import torch\ny = torch.nn.Linear(3, 4)",
                "import torch\ny = x.softmax(-1)",
                "from torch.optim import AdamW",
                "from miniocr.models.gpt import GPT",
                "import miniocr.tokenizer",
                "from assignments.b1.mechanism.nn import Linear"]:
        assert violations(src, "x.py"), src
    for src in ["import torch\nclass M(torch.nn.Module): pass\np = torch.nn.Parameter(torch.zeros(3))",
                "import torch\ny = torch.exp(x - x.amax(-1, keepdim=True))",
                "import torch\nclass O(torch.optim.Optimizer): pass",
                "from torch.optim import Optimizer",
                "from miniocr.data.zh_en import load_pairs",
                "from miniocr.device import pick_device",
                "from .nn import Linear"]:
        assert not violations(src, "x.py"), src


def test_checker_scans_subpackages(tmp_path):
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "deep.py").write_text("import torch.nn.functional as F\n", encoding="utf-8")
    assert check_dir(str(tmp_path)), "子目录里的违规没有被检查到"


def test_assignment_manifest_scans_forbidden_imports(tmp_path):
    (tmp_path / 'rules.toml').write_text('mechanism = ["mechanism"]\n')
    mechanism = tmp_path / 'mechanism'
    mechanism.mkdir()
    for src in ('import torch.nn.functional as F', 'from torch.nn import Linear',
                'from miniocr.models.gpt import GPT', 'from miniocr.eval.rewards import layered_reward',
                'from assignments.a3.mechanism import model'):
        (mechanism / 'answer.py').write_text(src)
        assert check_assignment(str(tmp_path)), src
    (mechanism / 'answer.py').write_text('import miniocr.data\nimport miniocr.device\n')
    assert not check_assignment(str(tmp_path))
