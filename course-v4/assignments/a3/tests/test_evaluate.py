import json
import pytest
from assignments.a3 import evaluate as E


def setup_registry(tmp_path):
    cfg=tmp_path/'config.json'; cfg.write_text('{}')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(protocol=E.protocol_id(),data_sha256='0'*64,
                                       candidates={'one':E.digest(cfg),'alias':E.digest(cfg),'other':'1'*64})))
    ledger=tmp_path/'ledger'; E.freeze(manifest,ledger)
    return cfg,ledger


def test_no_flag_before_data_access(tmp_path):
    with pytest.raises(ValueError,match='final-test'):
        E.evaluate(tmp_path/'missing',tmp_path/'missing',split='test')


def test_repeat_alias_and_failed_attempt_consumed(tmp_path):
    cfg,ledger=setup_registry(tmp_path)
    # 不加载任何 test 数据：第一次故意指向不存在文件，仍须留下启动记录。
    with pytest.raises(FileNotFoundError):
        E.evaluate(tmp_path/'missing',cfg,split='test',final_test=True,candidate='one',ledger=ledger)
    assert len(list(ledger.glob('*.STARTED.json'))) == 1
    for name in ['one','alias']:
        with pytest.raises(FileExistsError): E.start_test(cfg,name,ledger)


def test_unregistered_changed_config_and_registry(tmp_path):
    cfg,ledger=setup_registry(tmp_path)
    with pytest.raises(ValueError): E.start_test(cfg,'missing',ledger)
    cfg.write_text('{"changed":true}')
    with pytest.raises(ValueError): E.start_test(cfg,'one',ledger)
    with pytest.raises(FileExistsError): E.freeze(tmp_path/'manifest.json',ledger)
    assert not list(ledger.glob('*.STARTED.json'))


def test_val_hand_panel_and_no_test_rows(tmp_path,monkeypatch):
    from miniocr.data.postcorrect import PostCorrectExample,NoiseEdit,save_jsonl
    rows=[PostCorrectExample('a','val','x','x'),PostCorrectExample('b','val','O','0',(NoiseEdit('confusable',0,'0','O'),))]
    data=tmp_path/'val.jsonl'; save_jsonl(rows,data)
    config=tmp_path/'model.json'; config.write_text('{}')
    monkeypatch.setattr(E,'predict',lambda sources,cfg: ['x','0'] if cfg==config else sources)
    baseline=tmp_path/'prompt.json'; baseline.write_text('{}')
    result=E.evaluate(data,config,baseline=baseline)
    assert result['model']['em_norm']['value'] == 1
    assert result['model']['cer']['value'] == 0
    assert result['model']['broken_rate']['value'] == 0
    assert result['model']['fix_rate']['value'] == 1
    assert result['copy_baseline']['em_norm']['value'] == .5
    assert result['em_norm_gain_over_prompt'] == .5
    assert set(result['model']) == {'n','em_norm','cer','broken_rate','fix_rate'}
    assert result['model']['cer']['interval'] == (0.,0.)


def test_wrong_split_and_duplicate_ids(tmp_path):
    from miniocr.data.postcorrect import PostCorrectExample,save_jsonl
    cfg=tmp_path/'c.json'; cfg.write_text('{}'); data=tmp_path/'val.jsonl'
    for rows in [[PostCorrectExample('x','train','x','x')], [PostCorrectExample('x','val','x','x')]*2]:
        save_jsonl(rows,data)
        with pytest.raises(ValueError): E.evaluate(data,cfg,baseline=cfg)


def test_outputs_write_rejected():
    with pytest.raises(ValueError): E.writable(E.ROOT/'outputs'/'a3.json')


def test_distinct_registered_candidates_each_start_once(tmp_path):
    a=tmp_path/'a.json'; b=tmp_path/'b.json'
    a.write_text('{"run_id":"a"}'); b.write_text('{"run_id":"b"}')
    manifest=tmp_path/'manifest.json'; ledger=tmp_path/'ledger'
    manifest.write_text(json.dumps(dict(protocol=E.protocol_id(),data_sha256='0'*64,
                                       candidates={'a':E.digest(a),'b':E.digest(b)})))
    E.freeze(manifest,ledger)
    E.start_test(a,'a',ledger); E.start_test(b,'b',ledger)
    assert len(list(ledger.glob('*.STARTED.json'))) == 2
    for config,name in [(a,'a'),(b,'b')]:
        with pytest.raises(FileExistsError): E.start_test(config,name,ledger)
