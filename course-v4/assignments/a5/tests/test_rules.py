from pathlib import Path
import pytest
from assignments._common.rules import check_assignment, violations
from ._impl import ASSIGNMENT_DIR


def test_mechanism_rules():
    assert not check_assignment(str(ASSIGNMENT_DIR))
    assert (Path(ASSIGNMENT_DIR)/'rules.toml').read_text().strip() == 'mechanism = ["mechanism"]'


@pytest.mark.parametrize('src',[
    'import torch.nn.functional as F', 'from torch.nn import Conv2d',
    'import torch\nx=torch.nn.Linear(2,3)', 'from torch.nn import LayerNorm',
    'from miniocr.nn.vit import TinyViT', 'import miniocr.models.formula_ocr',
    'from assignments.a2.mechanism import GPT', 'import torch\nx=x.softmax(-1)',
])
def test_rules_reject_nested_imports(tmp_path,src):
    (tmp_path/'rules.toml').write_text('mechanism = ["mechanism"]')
    sub=tmp_path/'mechanism'/'nested'; sub.mkdir(parents=True)
    (sub/'bad.py').write_text(src)
    assert check_assignment(str(tmp_path))


@pytest.mark.parametrize('src',[
    'from miniocr.data import formula_images','from miniocr.device import pick_device',
    'import torch\np=torch.nn.Parameter(torch.ones(3))', 'from .layers import Linear',
])
def test_rules_allow(src):
    assert not violations(src,'ok.py')
