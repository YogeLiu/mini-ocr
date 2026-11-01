import pytest
from assignments._common.rules import check_assignment, check_dir, violations
from ._impl import ASSIGNMENT_DIR


def test_assignment_rules():
    assert not check_assignment(str(ASSIGNMENT_DIR))


@pytest.mark.parametrize('source', ['import torch.nn.functional as F', 'from torch.nn import Linear',
    'import torch\nx=torch.nn.Embedding(4,8)', 'from miniocr.nn.lora import LoRALinear',
    'from miniocr.models.gpt import GPT', 'import miniocr', 'from assignments.b2.mechanism import model'])
def test_banned_imports(source,tmp_path):
    d=tmp_path/'mechanism'/'nested'; d.mkdir(parents=True)
    (tmp_path/'rules.toml').write_text('mechanism = ["mechanism"]')
    (d/'bad.py').write_text(source)
    assert check_assignment(str(tmp_path)) and check_dir(str(d))


@pytest.mark.parametrize('source', ['from miniocr.data.sft import SFTExample', 'from miniocr.device import pick_device',
    'import torch\nclass M(torch.nn.Module): pass\nx=torch.nn.Parameter(torch.ones(2))', 'from .layers import Linear'])
def test_allowed_imports(source):
    assert not violations(source,'ok.py')
