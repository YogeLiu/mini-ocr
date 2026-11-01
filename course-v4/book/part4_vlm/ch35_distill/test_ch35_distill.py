"""CPU 机制回归；不自动下载模型或数据。"""
import json
from types import SimpleNamespace
import pytest
import torch
from miniocr.train.distill import masked_kl, answer_labels, quality_check
from book.part4_vlm.ch35_distill.tour import training_pool, main


def test_kl_matches_distribution_definition_and_detaches():
    s=torch.tensor([[[1.,2.,-1.],[2.,0.,1.]]],requires_grad=True)
    t=torch.tensor([[[0.,1.,2.],[4.,2.,1.]]],requires_grad=True)
    mask=torch.tensor([[1,0]])
    loss=masked_kl(s,t,mask,2.)
    prob=torch.softmax(t.detach()[0,0]/2,-1)
    expected=4*(prob*(prob.log()-torch.log_softmax(s[0,0]/2,-1))).sum()
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert t.grad is None
    assert s.grad[0,1].abs().sum()==0
    assert s.grad[0,0].abs().sum()>0


def test_empty_mask_and_equal_logits():
    s=torch.randn(2,3,7,requires_grad=True)
    assert masked_kl(s,s,torch.ones(2,3)).abs()<1e-6
    z=masked_kl(s,s+2,torch.zeros(2,3)); z.backward()
    assert z==0 and torch.equal(s.grad,torch.zeros_like(s))
    with pytest.raises(ValueError): masked_kl(s,s,torch.ones(2,3),0)
    with pytest.raises(ValueError): masked_kl(s,s[:,:,:4],torch.ones(2,3))


def test_answer_mask_preserves_eos_and_rejects_prefix_mismatch():
    ids=torch.tensor([[10,11,12,13,2,0]])
    labels=answer_labels(ids,torch.tensor([[10,11]]),torch.tensor([[1,1,1,1,1,0]]))
    assert labels.tolist()==[[-100,-100,12,13,2,-100]]
    assert ids[0,0]==10
    with pytest.raises(ValueError): answer_labels(ids,torch.tensor([[10,99]]))
    with pytest.raises(ValueError): answer_labels(ids,ids)


def test_quality_rules_and_false_positive():
    assert quality_check('x^2','x^2')["accepted"]
    assert not quality_check('', 'x')["accepted"]
    assert not quality_check('x'*400, 'x')["accepted"]
    assert not quality_check('x', 'x', truncated=True)["accepted"]
    assert not quality_check(r'\frac{', 'x')["accepted"]
    assert not quality_check('y', 'x', render_tolerance=0)["accepted"]
    # 可解析的错误可以通过宽松质检，展示假阳性，不把质检当 oracle。
    assert quality_check('y', 'x', render_tolerance=1)["accepted"]
    unknown=quality_check('x',r'\begin{pmatrix} x \end{pmatrix}')
    assert unknown['render_match'] is None and unknown['accepted']


def test_training_pool_prevents_overlap_and_nontrain(tmp_path):
    row=dict(id="a",split="train",target="x",image="a.png")
    (tmp_path/'base.json').write_text(json.dumps([row]))
    pseudo=dict(row,id="b",pred="y",accepted=True,scale=.75)
    (tmp_path/'quality.json').write_text(json.dumps(dict(rows=[pseudo])))
    args=SimpleNamespace(data=tmp_path,teacher_dir=tmp_path,n_train=1,arm="distill")
    assert [r['label'] for r in training_pool(args)]==['x','y']
    pseudo['split']='val'
    (tmp_path/'quality.json').write_text(json.dumps(dict(rows=[pseudo])))
    with pytest.raises(ValueError): training_pool(args)
    pseudo.update(split="train",id="a")
    (tmp_path/'quality.json').write_text(json.dumps(dict(rows=[pseudo])))
    with pytest.raises(ValueError): training_pool(args)


def test_default_cpu_tour(tmp_path):
    main(['--out',str(tmp_path)])
    result=json.loads((tmp_path/'kl.json').read_text())
    assert result['final']<result['initial']*.05


def test_final_test_requires_explicit_freeze(tmp_path):
    with pytest.raises(ValueError,match="final-test"):
        main(['5','--out',str(tmp_path)])
    assert not (tmp_path/'final_test.lock').exists()


def test_real_processor_answer_prefix_if_cached():
    from miniocr.models.vlm_anatomy import local_snapshot, load_image_only_processor
    from PIL import Image
    path=local_snapshot('smol')
    if path is None: pytest.skip('SmolVLM weights absent; offline-only test')
    p=load_image_only_processor(path,'smol')
    messages=[dict(role='user',content=[dict(type='image'),dict(type='text',text='Read formula.')])]
    image=Image.new('RGB',(32,32),'white')
    prefix=p(text=p.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),images=[image],return_tensors='pt')
    messages.append(dict(role='assistant',content=[dict(type='text',text='x^2')]))
    full=p(text=p.apply_chat_template(messages,tokenize=False,add_generation_prompt=False),images=[image],return_tensors='pt')
    labels=answer_labels(full['input_ids'],prefix['input_ids'])
    assert (labels!=-100).sum()>0


def test_existing_final_lock_blocks_before_model_load(tmp_path):
    (tmp_path/'final_test.lock').write_text('previous evaluation')
    with pytest.raises(FileExistsError):
        main(['5','--final-test','--adapter',str(tmp_path/'adapter'),'--out',str(tmp_path)])
    assert (tmp_path/'final_test.lock').read_text()=='previous evaluation'


def test_prepare_uses_original_png_and_disjoint_keys(tmp_path,monkeypatch):
    import io
    from PIL import Image
    import pyarrow as pa
    import pyarrow.parquet as pq
    import miniocr.data.distill_images as module
    payload=io.BytesIO(); Image.new('RGB',(17,9),'white').save(payload,format='PNG')
    texts=['x','y','z','w']
    path=tmp_path/'source.parquet'
    pq.write_table(pa.Table.from_pylist([dict(latex_formula=t,image=dict(bytes=payload.getvalue())) for t in texts]),path)
    monkeypatch.setattr(module,'_parquet_paths',lambda:[path])
    monkeypatch.setattr(module,'standard_split',lambda **kw:dict(train=['x','x','y'],val=['z'],test=['w']))
    out=tmp_path/'data'
    meta=module.prepare_distill_images(out,n_train=1,n_teacher=1,n_eval=1)
    assert meta['counts']==dict(base=1,teacher=1,val=1,test=1)
    all_rows=[json.loads((out/(s+'.json')).read_text())[0] for s in ('base','teacher','val','test')]
    assert len({r['id'] for r in all_rows})==4
    assert all(Image.open(out/r['image']).size==(17,9) for r in all_rows)


def test_selection_ignores_test_and_locks_result(tmp_path):
    from book.part4_vlm.ch35_distill.summarize import summarize
    for arm,cer in [('original',.6),('distill',.5)]:
        directory=tmp_path/arm; (directory/'adapter').mkdir(parents=True)
        metrics=dict(n=2,em_norm=dict(value=0.),cer=dict(value=cer),parse_rate=dict(value=1.))
        (directory/(arm+'_val.json')).write_text(json.dumps(dict(metrics=metrics,truncated=0)))
        (directory/'training.json').write_text(json.dumps(dict(seconds=1.,steps=1,n_pool=2)))
        (directory/'adapter/adapter_model.safetensors').write_bytes(b'fixture')
    (tmp_path/'final').mkdir()
    (tmp_path/'final/final_test.json').write_text('MUST NOT READ TEST DURING SELECTION')
    summarize(tmp_path,select=True)
    selection=json.loads((tmp_path/'selection.json').read_text())
    assert selection['selected']=='distill'
    with pytest.raises(FileExistsError): summarize(tmp_path,select=True)


def test_val_only_blocks_test_before_creating_outputs(tmp_path):
    with pytest.raises(SystemExit):
        main(['5', '--val-only', '--out', str(tmp_path/'blocked')])
    assert not (tmp_path/'blocked').exists()


def test_val_only_preparation_never_materializes_test(tmp_path, monkeypatch):
    import io
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    import miniocr.data.distill_images as module
    image = io.BytesIO()
    Image.new('RGB', (12, 8), 'white').save(image, format='PNG')
    path = tmp_path/'source.parquet'
    pq.write_table(pa.Table.from_pylist([
        dict(latex_formula=t, image=dict(bytes=image.getvalue())) for t in ['x', 'y', 'z']
    ]), path)
    monkeypatch.setattr(module, '_parquet_paths', lambda: [path])
    monkeypatch.setattr(module, 'standard_split', lambda **kw: dict(train=['x','y'], val=['z'], test=['DO NOT MATERIALIZE']))
    out = tmp_path/'data'
    meta = module.prepare_distill_images(out, 1, 1, 1, val_only=True)
    assert meta['counts'] == dict(base=1, teacher=1, val=1)
    assert not (out/'test.json').exists()
    assert len(list(out.glob('*.png'))) == 3


@pytest.mark.parametrize('problem', ['missing','wrong_arm','hash'])
def test_final_rejects_unfrozen_adapter_before_model_load(tmp_path, problem, monkeypatch):
    from book.part4_vlm.ch35_distill import tour
    monkeypatch.setattr(tour, 'load', lambda *a: pytest.fail('must reject before model load'))
    adapter=tmp_path/'original/adapter'; adapter.mkdir(parents=True)
    artifact=adapter/'adapter_model.safetensors'; artifact.write_bytes(b'weights')
    import hashlib
    selection=dict(selected='original',adapter_sha256=hashlib.sha256(b'weights').hexdigest())
    if problem=='wrong_arm': selection['selected']='distill'
    if problem=='hash': selection['adapter_sha256']='0'*64
    if problem!='missing': (tmp_path/'selection.json').write_text(json.dumps(selection))
    with pytest.raises((ValueError,FileNotFoundError)):
        main(['5','--final-test','--adapter',str(adapter),'--out',str(tmp_path/'final')])
    assert not (tmp_path/'final/final_test.lock').exists()


def test_final_accepts_matching_selection(tmp_path, monkeypatch):
    import hashlib
    from book.part4_vlm.ch35_distill import tour
    adapter=tmp_path/'original/adapter'; adapter.mkdir(parents=True)
    (adapter/'adapter_model.safetensors').write_bytes(b'weights')
    (tmp_path/'selection.json').write_text(json.dumps(dict(selected='original',
        adapter_sha256=hashlib.sha256(b'weights').hexdigest())))
    class ReachedLoad(Exception): pass
    def load(*a): raise ReachedLoad
    monkeypatch.setattr(tour,'load',load)
    with pytest.raises(ReachedLoad):
        main(['5','--final-test','--adapter',str(adapter),'--out',str(tmp_path/'final')])
    assert (tmp_path/'final/final_test.lock').exists()
