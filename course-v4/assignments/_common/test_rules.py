"""数据工具的命名空间不能成为机制参考答案或框架算子的旁路。"""
import pytest

from assignments._common.rules import check_assignment, violations


@pytest.mark.parametrize('symbol', [
    'latex_lm.train_tokenizer', 'latex_lm.load_latex_lm',
    'latex_lm.BPETokenizer', 'latex_lm.train_bpe',
    'latex_lm.evaluate_nll', 'latex_lm.evaluate_lm', 'latex_lm.evaluate_latex',
    'sft.encode_sft', 'sft.collate_sft',
    'vlm_sft.answer_labels', 'vlm_sft.FormulaCollator', 'vlm_sft.audit_samples',
    'latex_corpus.tokenize_latex', 'latex_corpus.apply_token_rules',
    'latex_corpus.parse_latex', 'latex_corpus.serialize',
    'latex_corpus.normalization_key', 'postcorrect.normalization_key',
    'formula_images.normalization_key', 'contrastive_pairs.normalization_key',
    'image_tensor_corpus.render', 'sft.render_chat', 'sft.zero_shot',
    'latex_lm.bits_per_byte',
])
@pytest.mark.parametrize('style', ['direct', 'module', 'package'])
def test_data_reference_shortcuts(symbol, style):
    module, name = symbol.split('.')
    source = {
        'direct': f'from miniocr.data.{module} import {name} as answer',
        'module': f'import miniocr.data.{module} as data\nanswer = data.{name}',
        'package': f'from miniocr import data\nanswer = data.{symbol}',
    }[style]
    assert violations(source, 'answer.py'), source


@pytest.mark.parametrize('source', [
    'from miniocr.data.latex_lm import *',
    'from miniocr.data import *',
    'from miniocr.data.symbols import F\nx = F.relu(x)',
    'import miniocr.data.latex_lm as lm\nx = lm.F.cross_entropy(x, y)',
    'from miniocr.data.sft import torch as t\nx = t.nn.Linear(2, 3)',
    'from miniocr.data import image_tensors as im\nx = im.F.interpolate(x)',
    'from miniocr.data.contrastive_pairs import torch as t\nx = t.optim.AdamW(p)',
])
def test_data_reexported_framework_shortcuts(source):
    assert violations(source, 'answer.py')


@pytest.mark.parametrize('source', [
    'from miniocr.data.latex_corpus import standard_split, clean_formula, load_formulas',
    'from miniocr.data.latex_lm import encode_formulas, FormulaWindows, LatexData',
    'from miniocr.data.token_file import cache_key, write_token_file, load_token_file',
    'from miniocr.data.symbols import make_dataset, group_split, cross_duplicates, nearest_pixel_distance',
    'from miniocr.data.sft import SFTExample',
    'from miniocr.data.postcorrect import build_split, build_dataset, PostCorrectExample',
    'from miniocr.data import formula_images',
    'from miniocr.data.vlm_sft import prepare_images',
    'from miniocr.data.image_tensors import image_to_chw, normalize, resize_chw',
    'from miniocr.data.zh_en import load_pairs',
    'import miniocr.data\nimport miniocr.device',
    'from miniocr.data.sft import torch as t\nx = t.zeros(2)',
    'from .layers import Linear',
])
def test_needed_data_tools_remain_available(source):
    assert not violations(source, 'answer.py')


@pytest.mark.parametrize('assignment', ['a1', 'a2', 'a3', 'a4', 'a5'])
def test_manifest_checks_data_shortcuts_only_in_mechanism(tmp_path, assignment):
    root = tmp_path / assignment
    nested = root / 'mechanism' / 'nested'
    nested.mkdir(parents=True)
    (root / 'rules.toml').write_text('mechanism = ["mechanism"]\n')
    source = 'from miniocr.data.latex_lm import train_tokenizer\n'
    (root / 'evaluate.py').write_text(source)
    assert not check_assignment(root)
    (nested / 'answer.py').write_text(source)
    assert check_assignment(root)
