"""评测协议用人工哨兵数据测试，不读取真实 test 样本或错例。"""
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from assignments.a1 import evaluate as E


def setup_registry(tmp_path,monkeypatch):
    paths = {}
    for name in ('one','two'):
        path = tmp_path/(name+'.pt'); path.write_bytes(name.encode()); paths[name] = path
    extra = dict(model_cfg=dict(input_dim=1024,hidden_dim=2,n_classes=24),data_cfg=E.SMOKE_DATA)
    monkeypatch.setattr(E,'load_model',lambda *a:(object(),extra))
    monkeypatch.setattr(E,'implementation_hash',lambda:'source-hash')
    E.freeze(tmp_path/'run',paths)
    return tmp_path/'run',paths


def test_final_gate_registration_and_repeat(tmp_path,monkeypatch):
    root,paths = setup_registry(tmp_path,monkeypatch)
    accessed = []
    def data(cfg,split):
        # 包括 train 的访问也必须在启动记录之后。
        assert list(root.glob('TEST_STARTED_*'))
        accessed.append(split)
        return dict(split=split)
    monkeypatch.setattr(E,'dataset',data)
    monkeypatch.setattr(E,'metrics',lambda *a:dict(n=4,accuracy=.5,accuracy_ci95=[0.,1.]))
    monkeypatch.setattr(E,'leakage_summary',lambda *a:dict(shared_groups=0))
    with pytest.raises(KeyError): E.final_test(root,'unknown')
    assert not accessed
    result = E.main(['--final-test','--candidate','one','--run-dir',str(root)])
    assert accessed == ['train','test','fresh']
    assert set(result['test']) == {'n','accuracy','accuracy_ci95','leakage'}
    with pytest.raises(FileExistsError): E.final_test(root,'one')
    assert len(accessed) == 3
    E.final_test(root,'two')
    assert len(accessed) == 6
    started = json.loads(next(root.glob('TEST_STARTED_*')).read_text())
    assert started['utc'] and started['checkpoint_sha256'] and started['frozen_sha256']


def test_failed_start_is_consumed(tmp_path,monkeypatch):
    root,_ = setup_registry(tmp_path,monkeypatch)
    def fail(*args): raise RuntimeError('simulated read failure')
    monkeypatch.setattr(E,'dataset',fail)
    with pytest.raises(RuntimeError): E.final_test(root,'one')
    with pytest.raises(FileExistsError): E.final_test(root,'one')


def test_changed_checkpoint_rejected_before_read(tmp_path,monkeypatch):
    root,paths = setup_registry(tmp_path,monkeypatch)
    paths['one'].write_bytes(b'changed')
    monkeypatch.setattr(E,'dataset',lambda *a:pytest.fail('must not read data'))
    with pytest.raises(ValueError,match='checkpoint changed'): E.final_test(root,'one')
    assert not list(root.glob('TEST_STARTED_*'))


def test_frozen_registry_and_implementation(tmp_path,monkeypatch):
    root,_ = setup_registry(tmp_path,monkeypatch)
    monkeypatch.setattr(E,'implementation_hash',lambda:'new-source')
    with pytest.raises(ValueError,match='implementation'): E.final_test(root,'one')
    monkeypatch.setattr(E,'implementation_hash',lambda:'source-hash')
    (root/'REGISTRY_SHA256').write_text('different-registry')
    monkeypatch.setattr(E,'dataset',lambda *a:pytest.fail('must not read data'))
    with pytest.raises(ValueError,match='registry changed'): E.final_test(root,'two')


def test_duplicate_alias_and_refreeze(tmp_path,monkeypatch):
    root,paths = setup_registry(tmp_path,monkeypatch)
    with pytest.raises(FileExistsError): E.freeze(root,paths)
    with pytest.raises(ValueError,match='duplicate'): E.freeze(tmp_path/'other',dict(a=paths['one'],b=paths['one']))


@pytest.mark.parametrize('args',[[],['--split','test'],['--final-test'],
    ['--final-test','--candidate','a','--run-dir','x','--device','cpu']])
def test_cli_requires_explicit_mode(args):
    with pytest.raises(SystemExit): E.main(args)


def test_dev_never_requests_test(monkeypatch):
    extra = dict(data_cfg=E.SMOKE_DATA)
    monkeypatch.setattr(E,'load_model',lambda *a:(object(),extra))
    accessed = []
    def data(cfg,split):
        assert split in {'train','val'}
        accessed.append(split)
        return {}
    monkeypatch.setattr(E,'dataset',data)
    monkeypatch.setattr(E,'sha256',lambda *a:'ckpt-hash')
    monkeypatch.setattr(E,'metrics',lambda *a:{})
    monkeypatch.setattr(E,'leakage_summary',lambda *a:{})
    E.main(['--ckpt','dummy'])
    assert accessed == ['val','train']


def test_group_bootstrap_hand():
    assert E.grouped_interval([1,1,1],[0,0,1]) == [1,1]
    assert E.grouped_interval([0,0],[1,2]) == [0,0]
    # 一个来源全对，另一个全错：抽2组只能得到0、1/2、1。
    assert E.grouped_interval([1,1,0,0],[0,0,1,1]) == [0,1]
    with pytest.raises(ValueError): E.grouped_interval([],[])


def test_leakage_hand_and_group_split():
    train = dict(images=np.stack([np.zeros((32,32)),np.ones((32,32))]),groups=np.array([0,1]))
    held = dict(images=np.stack([np.ones((32,32)),np.full((32,32),.5)]),groups=np.array([2,3]))
    result = E.leakage_summary(train,held,chunk=1)
    # 均匀图aHash相同；像素RMS分别为0、0.5。
    assert result['hash_hit_rate'] == 1 and result['pixel_hit_rate'] == .5 and result['shared_groups'] == 0
    groups = np.repeat(np.arange(10),2)
    parts = E.group_split(groups,seed=0)
    assert [len(p) for p in parts] == [16,2,2]
    assert sorted(np.concatenate(parts).tolist()) == list(range(20))
    assert not (set(groups[parts[0]]) & set(groups[parts[1]]))
    assert not (set(groups[parts[0]]) & set(groups[parts[2]]))


def test_metrics_weighting_eval_and_no_grad():
    class Model(torch.nn.Module):
        def forward(self,x):
            assert not self.training and not torch.is_grad_enabled()
            return x
    data = dict(images=np.zeros((3,1,2),dtype=np.float32),labels=np.array([0,1,0]),groups=np.array([0,0,1]))
    result = E.metrics(Model(),data,batch_size=2)
    assert result['n'] == 3 and result['n_groups'] == 2
    assert result['accuracy'] == pytest.approx(2/3)
    assert result['loss'] == pytest.approx(np.log(2))
    assert 'predictions' not in result


def test_outputs_write_rejected():
    with pytest.raises(ValueError): E.safe_path('outputs/a1/new.json')



def test_student_adapter_is_bound_in_reference_mode(tmp_path, monkeypatch):
    import shutil
    from assignments.a1 import evaluate as module
    source = Path(module.__file__).parent
    copy = tmp_path/'assignments/a1'
    shutil.copytree(source, copy, ignore=shutil.ignore_patterns('__pycache__'))
    monkeypatch.setattr(module, '__file__', str(copy/'evaluate.py'))
    (tmp_path/'miniocr/data').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source.parents[1]/'miniocr/data/symbols.py', tmp_path/'miniocr/data/symbols.py')
    identity = module.implementation_hash
    before = identity()
    adapter = copy/'tests/adapters.py'
    adapter.write_text(adapter.read_text() + '\n# changed learner wiring\n')
    assert identity() != before
