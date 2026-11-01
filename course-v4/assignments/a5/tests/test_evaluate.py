"""门禁测试仅用合成 checkpoint，不加载任何 test 数据。"""
import json
from pathlib import Path
import pytest
import torch
from assignments.a5 import evaluate as E


def setup_plan(tmp_path):
    ckpt=tmp_path/'model.pt'; torch.save({'cfg': {'max_length': 16}}, ckpt)
    plan=tmp_path/'plan.json'
    plan.write_text(json.dumps(dict(candidates={'base':str(ckpt)},data_dir=str(tmp_path/'data'),test_n=2,device='cpu',batch_size=1,seed=0)))
    run=tmp_path/'run'
    E.freeze(plan,run)
    return run,ckpt


def test_test_gate_and_repeat(tmp_path):
    run,ckpt=setup_plan(tmp_path)
    with pytest.raises(ValueError,match='final-test'): E.begin_test(run,'base',False)
    with pytest.raises(ValueError,match='preregistered'): E.begin_test(run,'other',True)
    assert not list(run.glob('TEST_STARTED*'))
    frozen,entry=E.begin_test(run,'base',True)
    assert entry['sha256']==E.digest(ckpt)
    marker=list(run.glob('TEST_STARTED*'))
    assert len(marker)==1 and json.loads(marker[0].read_text())['candidate']=='base'
    # 即使后续模型加载/数据读取失败，也不能重复开始。
    with pytest.raises(FileExistsError): E.begin_test(run,'base',True)


@pytest.mark.parametrize('what',['checkpoint','registry','code'])
def test_frozen_mutations_rejected(tmp_path,monkeypatch,what):
    run,ckpt=setup_plan(tmp_path)
    if what=='checkpoint': ckpt.write_bytes(b'changed')
    elif what=='registry': (run/'FROZEN.json').write_text('{}')
    else: monkeypatch.setattr(E,'code_digest',lambda:'changed')
    with pytest.raises(ValueError): E.begin_test(run,'base',True)
    assert not list(run.glob('TEST_STARTED*'))


def test_each_preregistered_candidate_once(tmp_path):
    plan=tmp_path/'plan.json'
    paths={}
    for name in ('base','ablation'):
        p=tmp_path/(name+'.pt'); torch.save({'cfg': {'max_length': 16}, 'name': name}, p); paths[name]=str(p)
    plan.write_text(json.dumps(dict(candidates=paths,data_dir=str(tmp_path/'data'),test_n=2,device='cpu',batch_size=1,seed=0)))
    run=tmp_path/'run'; E.freeze(plan,run)
    assert json.loads((run/'FROZEN.json').read_text())['max_length'] == 16
    for name in paths: E.begin_test(run,name,True)
    assert len(list(run.glob('TEST_STARTED*')))==2
    with pytest.raises(FileExistsError): E.freeze(plan,run)


def test_alias_checkpoint_rejected(tmp_path):
    p=tmp_path/'x.pt'; p.write_bytes(b'x')
    plan=tmp_path/'plan.json'
    plan.write_text(json.dumps(dict(candidates={'a':str(p),'b':str(p)},data_dir=str(tmp_path/'data'),test_n=2,device='cpu',batch_size=1,seed=0)))
    with pytest.raises(ValueError,match='duplicate'): E.freeze(plan,tmp_path/'run')


def test_cli_rejects_test_before_loading(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('model/data must not be read')
    monkeypatch.setattr(E,'load_model',forbidden)
    with pytest.raises(ValueError,match='final-test'):
        E.main(['--split','test','--run-dir',str(tmp_path)])


def test_shared_outputs_rejected():
    with pytest.raises(ValueError): E.safe_path(E.ROOT/'outputs'/'a5-new')


def test_panel_hand_metrics():
    summary=E.score(['ab','cd'],['ab','xd'],'val')
    assert summary['n']==2
    assert summary['em_norm']['value']==.5
    assert summary['cer']['value']==.25
    assert set(summary)=={'n','em_raw','em_norm','cer','parse_rate'}
    assert summary['em_norm']['interval'][0]<.5<summary['em_norm']['interval'][1]


@pytest.mark.parametrize('lengths', [(16, 32), (32, 16)])
def test_freeze_rejects_different_lengths_before_consuming(tmp_path, lengths):
    paths = {}
    for i, length in enumerate(lengths):
        p = tmp_path / f'{i}.pt'
        torch.save({'cfg': {'max_length': length}}, p)
        paths[str(i)] = str(p)
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps(dict(candidates=paths, data_dir=str(tmp_path/'data'),
                                   test_n=2, device='cpu', batch_size=1, seed=0)))
    run = tmp_path / 'run'
    with pytest.raises(ValueError, match='max_length'):
        E.freeze(plan, run)
    assert not list(run.glob('*.json'))


@pytest.mark.parametrize('solutions', ['0', '1'])
def test_training_snapshot_and_mutations(tmp_path, monkeypatch, solutions):
    import shutil
    root = tmp_path / 'course'
    for rel in ['assignments/a5', 'miniocr']:
        shutil.copytree(E.ROOT/rel, root/rel, ignore=shutil.ignore_patterns('__pycache__'))
    monkeypatch.setattr(E, 'ROOT', root)
    monkeypatch.setattr(E, '__file__', str(root/'assignments/a5/evaluate.py'))
    monkeypatch.setenv('A_SOLUTIONS', solutions)
    training = root/'assignments/a5/training'
    if training.exists(): shutil.rmtree(training)
    run, _ = setup_plan(tmp_path)
    frozen = json.loads((run/'FROZEN.json').read_text())
    assert frozen['training_code']['status'] == 'missing'
    missing = E.code_digest()
    training.mkdir()
    assert E.code_digest() != missing
    with pytest.raises(ValueError, match='code changed'):
        E.begin_test(run, 'base', True)
    assert not list(run.glob('TEST_STARTED*'))
    nested = training/'nested'; nested.mkdir()
    source = nested/'model.py'; source.write_text('x = 1')
    assert E.training_code() == {'status': 'present', 'files': {'nested/model.py': E.digest(source)}}
    before = E.code_digest()
    source.write_text('x = 2')
    assert E.code_digest() != before
    before = E.code_digest()
    source.rename(nested/'renamed.py')
    assert E.code_digest() != before
    before = E.code_digest()
    (nested/'renamed.py').unlink()
    assert E.code_digest() != before
    second = tmp_path/'second'; second.mkdir()
    run, _ = setup_plan(second)
    assert json.loads((run/'FROZEN.json').read_text())['training_code'] == {'status': 'present', 'files': {}}
