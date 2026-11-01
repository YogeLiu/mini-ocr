from assignments._common.rules import check_assignment, check_dir, violations
from ._impl import ASSIGNMENT_DIR
import pytest


def test_mechanism_rules():
    assert not check_assignment(str(ASSIGNMENT_DIR))


@pytest.mark.parametrize('src', [
    'import torch.nn.functional as F', 'from torch.nn import Linear',
    'import torch\nx=torch.nn.Sequential(torch.nn.Linear(2,3))',
    'from torch.optim import AdamW', 'from miniocr.models.gpt import GPT',
    'from miniocr.autograd import Value', 'import miniocr.nn.losses',
    'from assignments.b2.mechanism import model',
    'import torch\ny=x.softmax(-1)'])
def test_banned_imports(src,tmp_path):
    assert violations(src,'bad.py')
    sub = tmp_path/'nested'; sub.mkdir()
    (sub/'bad.py').write_text(src)
    assert check_dir(str(tmp_path))


@pytest.mark.parametrize('src', [
    'from miniocr.data.symbols import make_dataset, group_split, cross_duplicates, nearest_pixel_distance',
    'from miniocr.device import pick_device',
    'import torch\nclass M(torch.nn.Module): pass\np=torch.nn.Parameter(torch.ones(2))',
    'from .layers import Linear'])
def test_allowed_imports(src):
    assert not violations(src,'ok.py')
