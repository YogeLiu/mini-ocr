"""生成真实学习者目录，锁定跨函数、类方法及常量旁的练习边界。"""
import ast
from pathlib import Path
import subprocess
import sys

import pytest

from tools.make_student import strip_exercises

ROOT = Path(__file__).resolve().parents[1]


def test_all_exercises_compile():
    for path in (ROOT / 'miniocr').rglob('*.py'):
        result, _ = strip_exercises(path.read_text())
        compile(result, str(path), 'exec')


@pytest.mark.parametrize('path, hollow, intact', [
    ('nn/contrastive.py', ['clip_loss', 'siglip_loss'], ['DualEncoder.__init__']),
    ('eval/retrieval.py', ['retrieval_metrics'], []),
    ('train/grpo.py', ['grpo_loss'], []),
    ('train/distill.py', ['masked_kl', 'answer_labels', 'quality_check'], []),
    ('train/policy_gradient.py', ['reinforce_loss', 'categorical_score_gradient',
                                'kl_estimators', 'ppo_surrogate', 'sequence_log_prob'], []),
    ('train/vlm_lora.py', ['adapter_spec'], ['attach_lora', 'parameter_audit', 'move_batch']),
    ('train/vlm_grpo.py', ['group_advantages', 'grpo_loss', 'probe_rewards'],
                        ['completion_mask', 'token_logprobs', 'degenerate']),
    ('data/image_tensors.py', ['image_to_chw', 'normalize', 'area_weights', 'transform_image'],
                            ['chw_to_image', 'resize_chw', 'restore_coordinates', 'pixel_metrics']),
    ('data/vlm_sft.py', ['answer_labels'], ['FormulaCollator.__call__', 'audit_samples']),
    ('models/formula_ocr.py', ['Connector.__init__', 'Connector.forward', 'FormulaOCR.forward', 'token_saliency'],
                            ['targets_for', 'FormulaOCR.__init__', 'FormulaOCR.generate']),
    ('models/vlm_anatomy.py', ['smart_resize', 'replica_processor', 'MLPConnector.__init__',
                             'MLPConnector.forward', 'QueryConnector.__init__', 'QueryConnector.forward',
                             'expand_image_placeholder'], ['load_image_only_processor']),
    ('models/multimodal.py', ['prefix_mask', 'VisualPrefix.forward', 'CrossAttention.forward'],
                           ['VisualPrefix.__init__', 'CrossAttention.__init__']),
])
def test_chapter_exercise_boundaries(path, hollow, intact):
    source = (ROOT / 'miniocr' / path).read_text()
    result, _ = strip_exercises(source)

    def functions(text):
        found = {}
        for node in ast.parse(text).body:
            if isinstance(node, ast.FunctionDef):
                found[node.name] = node
            elif isinstance(node, ast.ClassDef):
                found.update({node.name + '.' + m.name: m for m in node.body
                              if isinstance(m, ast.FunctionDef)})
        return found

    before, after = functions(source), functions(result)
    for name in hollow:
        body = after[name].body
        if ast.get_docstring(after[name]) is not None:
            body = body[1:]
        assert len(body) == 1 and isinstance(body[0], ast.Raise), name
        assert body[0].exc.func.id == 'NotImplementedError', name
    for name in intact:
        assert ast.dump(before[name]) == ast.dump(after[name]), name
    assert not any(isinstance(n, ast.Raise) for n in ast.parse(result).body)


def test_generated_course_imports_and_raises(tmp_path):
    dest = tmp_path / 'my-course'
    subprocess.run([sys.executable, '-m', 'tools.make_student', str(dest)], cwd=ROOT, check=True)
    assert not (dest / 'assignments/solutions').exists()
    assert not (dest / 'outputs').exists()
    assert all((dest / 'assignments' / f'a{i}').is_dir() for i in range(1, 6))
    assert (dest / 'book').is_dir()
    subprocess.run([sys.executable, '-c', '''
import importlib
from pathlib import Path
for path in Path('miniocr').rglob('*.py'):
    name = '.'.join(path.with_suffix('').parts)
    importlib.import_module(name.removesuffix('.__init__'))
from miniocr.nn.contrastive import clip_loss
from miniocr.train.grpo import grpo_loss
from miniocr.models.multimodal import VisualPrefix, CrossAttention
from miniocr.train.vlm_grpo import PROBE_OUTPUTS, probe_rewards
assert PROBE_OUTPUTS
for func, args in [(clip_loss, (None,)), (grpo_loss, (None,)*5),
                   (VisualPrefix().forward, (None, None)),
                   (CrossAttention().forward, (None, None)), (probe_rewards, ('x',))]:
    try:
        func(*args)
    except NotImplementedError as exc:
        assert '练习' in str(exc)
    else:
        raise AssertionError(func)
'''], cwd=dest, check=True)


@pytest.mark.parametrize('present', [False, True])
def test_optional_project_and_solutions_exclusion(tmp_path, monkeypatch, present):
    from tools import make_student as M
    source = tmp_path/'source'; source.mkdir()
    (source/'tools').mkdir()
    (source/'assignments/solutions').mkdir(parents=True)
    (source/'assignments/solutions/secret.py').write_text('secret = 1')
    for item in M.COPY:
        if item == 'project': continue
        p = source/item
        if not p.exists():
            if p.suffix: p.write_text('')
            else: p.mkdir()
    if present:
        (source/'project').mkdir()
        (source/'project/README.md').write_text('project content')
    dest = tmp_path/'student'
    monkeypatch.setattr(M, '__file__', str(source/'tools/make_student.py'))
    monkeypatch.setattr(sys, 'argv', ['make_student', str(dest)])
    M.main()
    assert not (dest/'assignments/solutions').exists()
    assert (dest/'project').exists() == present
    if present: assert (dest/'project/README.md').read_text() == 'project content'
