"""全部使用临时合成数据；绝不读取项目 test。"""
import json
from pathlib import Path
import pytest
from assignments.a4 import evaluate as E


def load_copy(checkpoint,device,settings):
    assert Path(checkpoint).read_text()=='synthetic checkpoint'
    assert device=='cpu'
    return lambda sources:list(sources)


def files(tmp_path):
    ck=tmp_path/'model.txt'; ck.write_text('synthetic checkpoint')
    cfg=tmp_path/'run.json'
    cfg.write_text(json.dumps(dict(factory=__name__+':load_copy',checkpoint=str(ck),settings={},device='cpu',protocol_version='synthetic-v1')))
    dev=tmp_path/'dev.jsonl'
    dev.write_text(json.dumps(dict(id='synthetic',split='val',source='x',target='x',is_clean=True))+'\n')
    data=tmp_path/'synthetic-test.jsonl'
    data.write_text(json.dumps(dict(id='synthetic',split='test',source='x',target='x',is_clean=True))+'\n')
    identity=dict(config_sha256=E.sha256(cfg),checkpoint_sha256=E.sha256(ck),factory_sha256=E.sha256(__file__))
    reg=tmp_path/'frozen.json'
    reg.write_text(json.dumps(dict(frozen=True,test_sha256=E.sha256(data),candidates={'one':identity,'alias':identity})))
    return cfg,dev,data,reg,identity


def test_dev_end_to_end(tmp_path,capsys):
    cfg,dev,_,_,_=files(tmp_path)
    result=E.main(['--config',str(cfg),'--data',str(dev)])
    assert result['panel']['em_norm']['value']==1
    assert result['reward_mean']==1 and result['training_pass'] is None
    assert 'static_probe_rewards' in result and 'model_diagnostics' in result
    assert 'rows' not in capsys.readouterr().out


def test_test_guard_precedes_any_file_read(tmp_path):
    with pytest.raises(ValueError,match='final-test'):
        E.main(['--config','missing','--data','missing','--split','test'])


def test_claim_freezes_registry_and_rejects_alias(tmp_path,monkeypatch):
    _,_,_,reg,identity=files(tmp_path)
    monkeypatch.setattr(E,'LEDGER',tmp_path/'ledger')
    E.claim_final(reg,'one',identity)
    for candidate in ('one','alias'):
        with pytest.raises(ValueError,match='拒绝重复'):
            E.claim_final(reg,candidate,identity)
    changed=json.loads(reg.read_text()); changed['candidates']['new']=identity
    reg.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='不能改登记表'):
        E.claim_final(reg,'new',identity)


def test_distinct_preregistered_candidates_each_once(tmp_path,monkeypatch):
    _,_,_,reg,identity=files(tmp_path)
    monkeypatch.setattr(E,'LEDGER',tmp_path/'ledger')
    second={**identity,'checkpoint_sha256':'a'*64}
    d=json.loads(reg.read_text()); d['candidates']['second']=second; reg.write_text(json.dumps(d))
    E.claim_final(reg,'one',identity)
    E.claim_final(reg,'second',second)
    assert len((E.LEDGER/'events.jsonl').read_text().splitlines())==2


def test_failure_records_start_before_data_open_and_consumes_candidate(tmp_path,monkeypatch):
    cfg,_,data,reg,_=files(tmp_path)
    monkeypatch.setattr(E,'LEDGER',tmp_path/'ledger')
    data.unlink()
    args=['--config',str(cfg),'--data',str(data),'--split','test','--final-test','--registry',str(reg),'--candidate','one']
    with pytest.raises(RuntimeError,match='已占用'):
        E.main(args)
    events=[json.loads(s)['event'] for s in (E.LEDGER/'events.jsonl').read_text().splitlines()]
    assert events==['started','failed']
    with pytest.raises(ValueError,match='拒绝重复'):
        E.main(args)


def test_final_summary_only(tmp_path,monkeypatch,capsys):
    cfg,_,data,reg,_=files(tmp_path)
    monkeypatch.setattr(E,'LEDGER',tmp_path/'ledger')
    result=E.main(['--config',str(cfg),'--data',str(data),'--split','test','--final-test','--registry',str(reg),'--candidate','one'])
    assert result['panel']['em_norm']['value']==1
    assert 'by_error_type' not in result['panel']
    assert 'static_probe_rewards' not in result and 'model_diagnostics' not in result
    assert 'target' not in capsys.readouterr().out


@pytest.mark.parametrize('change',['unregistered','unfrozen','checkpoint'])
def test_bad_registration(tmp_path,monkeypatch,change):
    _,_,_,reg,identity=files(tmp_path)
    monkeypatch.setattr(E,'LEDGER',tmp_path/'ledger')
    candidate='one'
    if change=='unregistered': candidate='unknown'
    if change=='unfrozen':
        d=json.loads(reg.read_text()); d['frozen']=False; reg.write_text(json.dumps(d))
    if change=='checkpoint': identity={**identity,'checkpoint_sha256':'b'*64}
    with pytest.raises(ValueError): E.claim_final(reg,candidate,identity)
    assert not E.LEDGER.exists()
