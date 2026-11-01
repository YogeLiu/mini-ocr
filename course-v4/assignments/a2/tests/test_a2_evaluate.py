from pathlib import Path
"""评测协议用玩具数据验证；不读取任何真实 test 数据。"""
import json
import math
from types import SimpleNamespace
import pytest
import torch
from assignments.a2 import evaluate as E
from ._impl import adapters as A


def test_registry_freezes_and_refuses_retries(tmp_path):
    p=tmp_path/'registry.json'
    E.registry_action(p,'a',{'hash':'a'})
    E.registry_action(p,'b',{'hash':'b'})
    with pytest.raises(ValueError): E.registry_action(p,'a',{'hash':'changed'},True)
    E.registry_action(p,'a',{'hash':'a'},True)
    assert 'started' in json.loads(p.read_text())['candidates']['a']
    with pytest.raises(ValueError): E.registry_action(p,'a',{'hash':'a'},True)
    with pytest.raises(ValueError): E.registry_action(p,'c',{'hash':'c'})
    E.registry_action(p,'b',{'hash':'b'},True)


def test_uniform_score_counts_utf8_eos_and_windows():
    class Tok:
        def encode(self,text,allow_special):
            assert not allow_special
            return [1]*len(text.encode())
    class Uniform(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.p=torch.nn.Parameter(torch.zeros(1)); self.calls=[]
        def forward(self,x):
            self.calls.append(x.clone())
            return torch.zeros(*x.shape,5),None
    m=Uniform()
    out=E.score(m,Tok(),['中','a'],0,4,2,batch=2)
    assert out['n_bytes']==4 and out['n_targets']==6
    assert out['bits_per_byte']==pytest.approx(3.) # 6 targets * log2(4) / 4 bytes
    assert out['nll_sum']==pytest.approx(6*math.log(4))
    assert all(x.shape[1]==2 for x in m.calls)


def test_test_gate_precedes_data_access(monkeypatch,tmp_path):
    monkeypatch.setattr(E,'standard_split',lambda:pytest.fail('不得读数据'))
    with pytest.raises(SystemExit):
        E.main(['--tokenizer','missing','--ckpt','missing','--final-test'])


def test_cpu_train_save_load_and_evaluate(tmp_path,capsys):
    torch.manual_seed(1)
    cfg=dict(vocab_size=258,block_size=8,n_layer=1,n_head=2,d_model=8,d_ff=16,
             rope_base=10000.,tie_weights=True)
    model=A.get_gpt(cfg)
    x=torch.tensor([[256,120,61,49]]); y=torch.tensor([[120,61,49,257]])
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    loss=model(x,y)[1]; loss.backward(); opt.step()
    ck=tmp_path/'model.pt'; tok=tmp_path/'bpe.json'
    torch.save(dict(cfg=cfg,state_dict=model.state_dict()),ck)
    tok.write_text(json.dumps(dict(merges=[],specials=['<|bos|>','<|eos|>'])))
    E.main(['--tokenizer',str(tok),'--ckpt',str(ck),'--smoke','--samples','2','--max-new','4'])
    out=json.loads(capsys.readouterr().out)
    assert out['split']=='smoke' and math.isfinite(out['metrics']['bits_per_byte'])
    assert out['compilation']['count']==2 and out['threshold']=='待校准'


def test_failed_final_start_is_consumed_before_data_read(tmp_path,monkeypatch):
    registry=tmp_path/'registry.json'
    tok=tmp_path/'bpe.json'; ck=tmp_path/'model.pt'
    tok.write_text('{}'); torch.save({'cfg':{'vocab_size':3,'block_size':2},'state_dict':{}},ck)
    model=torch.nn.Module()
    monkeypatch.setattr(E,'REGISTRY',registry)
    monkeypatch.setattr(E,'corpus_available',lambda:True)
    monkeypatch.setattr(E,'candidate_identity',lambda args:{'toy':'frozen'})
    monkeypatch.setattr(E,'load_tokenizer',lambda path:(None,0,2,3))
    monkeypatch.setattr(E.A,'get_gpt',lambda cfg:model)
    def fail():
        assert 'started' in json.loads(registry.read_text())['candidates']['toy']
        raise RuntimeError('simulated failure, no real test data')
    monkeypatch.setattr(E,'standard_split',fail)
    E.registry_action(registry,'toy',{'toy':'frozen'})
    args=['--tokenizer',str(tok),'--ckpt',str(ck),'--candidate','toy','--final-test','--max-new','2']
    with pytest.raises(RuntimeError,match='simulated'): E.main(args)
    with pytest.raises(ValueError,match='重复'): E.main(args)



def test_student_adapter_is_bound_in_reference_mode(tmp_path, monkeypatch):
    import shutil
    from assignments.a2 import evaluate as module
    source = Path(module.__file__).parent
    copy = tmp_path/'a2'
    shutil.copytree(source, copy, ignore=shutil.ignore_patterns('__pycache__'))
    monkeypatch.setattr(module, '__file__', str(copy/'evaluate.py'))
    from types import SimpleNamespace
    artifact = tmp_path/'artifact'; artifact.write_text('synthetic')
    args = SimpleNamespace(ckpt=artifact, tokenizer=artifact, device='cpu', smoke=True, samples=1, max_new=1, seed=0)
    identity = lambda: module.candidate_identity(args)
    before = identity()
    adapter = copy/'tests/adapters.py'
    adapter.write_text(adapter.read_text() + '\n# changed learner wiring\n')
    assert identity() != before
