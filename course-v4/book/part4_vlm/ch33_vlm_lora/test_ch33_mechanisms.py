"""无权重 CPU 回归；真实 processor 仅在本地缓存齐备时运行。"""
import json
import pytest
import torch
from miniocr.data.vlm_sft import answer_labels, FormulaCollator
from miniocr.train.vlm_lora import adapter_spec, move_batch
from miniocr.models.vlm_anatomy import local_snapshot, load_image_only_processor
from book.part4_vlm.ch33_vlm_lora.tour import toy_model, main


def test_answer_mask_padding_and_image_inside_answer():
    ids = torch.tensor([[0,10,99,11,20,99,21],[10,99,11,22,23,0,0]])
    attention = torch.tensor([[0,1,1,1,1,1,1],[1,1,1,1,1,0,0]])
    actual = answer_labels(ids,attention,[3,3],[99])
    assert actual.tolist() == [[-100,-100,-100,-100,20,-100,21],[-100,-100,-100,22,23,-100,-100]]
    assert ids[0,0] == 0


def test_padding_id_can_also_be_supervised_eos():
    assert answer_labels(torch.tensor([[4,0,0]]),torch.tensor([[1,1,0]]),[1]).tolist() == [[-100,0,-100]]


@pytest.mark.parametrize('prefix', [[3],[-1],[]])
def test_missing_answer_and_bad_lengths_rejected(prefix):
    with pytest.raises(ValueError): answer_labels(torch.tensor([[1,2,3]]),torch.ones(1,3),prefix)


def test_shifted_mask_has_no_loss_gradient():
    ids = torch.tensor([[5,99,6,7,8]])
    labels = answer_labels(ids,torch.ones_like(ids),[3],[99])
    logits = torch.randn(1,5,100,requires_grad=True)
    torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,100),labels[:,1:].reshape(-1)).backward()
    assert torch.count_nonzero(logits.grad[:,:2]) == 0
    assert logits.grad[:,2:4].abs().sum() > 0


def test_target_selection_includes_deltanet_and_excludes_saved_connector():
    m = toy_model()
    language = adapter_spec(m,'language'); connector = adapter_spec(m,'connector'); vision = adapter_spec(m,'vision')
    assert any('in_proj_qkv' in n for n in language['target_modules'])
    assert not any('.visual.' in n for n in language['target_modules'])
    assert connector['modules_to_save'] == ['model.visual.merger']
    assert 'model.visual.qkv' in vision['target_modules']
    assert not any('.merger.' in n for n in vision['target_modules'])


def test_move_preserves_integer_labels():
    result=move_batch({'labels':torch.tensor([-100,4]),'pixels':torch.ones(2)},'cpu',torch.bfloat16)
    assert result['labels'].dtype == torch.long
    assert result['pixels'].dtype == torch.bfloat16


def test_default_tour_cpu(tmp_path):
    main(['--out',str(tmp_path)])
    assert json.loads((tmp_path/'mask.json').read_text())['masked_gradient_zero']
    assert (tmp_path/'strategies.json').exists()


def test_test_protocol_requires_final_flag_and_adapter(tmp_path):
    from argparse import Namespace
    from book.part4_vlm.ch33_vlm_lora.tour import part5
    with pytest.raises(ValueError): part5(Namespace(split='test',final_test=False,adapter=None))


def test_real_processor_prefix_and_answer(tmp_path):
    path = local_snapshot('qwen35')
    if path is None: pytest.skip('缺少本地 Qwen3.5 权重/processor，不下载')
    from PIL import Image
    p = load_image_only_processor(path,'qwen35')
    p.image_processor.size={'shortest_edge':16384,'longest_edge':65536}
    Image.new('RGB',(128,32),'white').save(tmp_path/'probe.png')
    batch = FormulaCollator(p,tmp_path)([dict(id='probe',image='probe.png',target='x^2'),
        dict(id='second',image='probe.png',target='y')])
    assert batch['input_ids'].shape[0] == 2
    assert (batch['labels'][batch['attention_mask']==0] == -100).all()
    assert (batch['labels'][batch['input_ids']==p.image_token_id] == -100).all()
    decoded = p.decode(batch['labels'][0][batch['labels'][0]!=-100],skip_special_tokens=True)
    assert 'x^2' in decoded


def test_test_lock_cannot_be_bypassed_by_new_output(tmp_path):
    from argparse import Namespace
    from book.part4_vlm.ch33_vlm_lora.tour import part5
    lock=tmp_path/'ch33_final_test.lock'; lock.write_text('already evaluated')
    with pytest.raises(FileExistsError):
        part5(Namespace(split='test',final_test=True,adapter='adapter',data=tmp_path,out=tmp_path/'different'))
    assert lock.read_text() == 'already evaluated'


def test_selection_rejects_test_metrics(tmp_path):
    from book.part4_vlm.ch33_vlm_lora.gpu_suite import ranking
    path=tmp_path/'result.json'; path.write_text(json.dumps({'split':'test','metrics':{}}))
    with pytest.raises(ValueError,match='test'): ranking(path)


def test_resume_cannot_silently_change_configuration(tmp_path,monkeypatch):
    from book.part4_vlm.ch33_vlm_lora import gpu_suite
    monkeypatch.setattr(gpu_suite,'ROOT',tmp_path)
    out=tmp_path/'probe'; out.mkdir()
    (out/'timing.json').write_text('{}')
    (out/'config.json').write_text(json.dumps({'parts':[5],'n':40}))
    with pytest.raises(ValueError,match='参数 n 不同'): gpu_suite.run('probe',5,'--n',100)
    gpu_suite.run('probe',5,'--n',40)  # completed runs do not touch CUDA or re-evaluate


def test_supplement_rejects_test_before_any_io(monkeypatch,tmp_path):
    from book.part4_vlm.ch33_vlm_lora import supplement
    monkeypatch.setattr(supplement,'ROOT',tmp_path/'absent')
    with pytest.raises(ValueError,match='split=val'):
        supplement.run('forbidden',5,'--split','test')
    with pytest.raises(ValueError,match='split=val'):
        supplement.run('forbidden',5,'--final-test')
    assert not (tmp_path/'absent').exists()


def test_prepare_supplement_never_extracts_test_images(monkeypatch,tmp_path):
    from io import BytesIO
    from PIL import Image
    import pyarrow as pa
    import pyarrow.parquet as pq
    from miniocr.data import vlm_sft
    image = BytesIO(); Image.new('RGB',(8,8),'white').save(image,format='PNG')
    path = tmp_path/'source.parquet'
    pq.write_table(pa.table({'latex_formula':['x','y','z'],
        'image':[{'bytes':image.getvalue()}]*3}),path)
    monkeypatch.setattr(vlm_sft,'_parquet_paths',lambda:[path])
    monkeypatch.setattr(vlm_sft,'standard_split',lambda **kw:dict(train=['x'],val=['y'],test=['z']))
    out = tmp_path/'prepared'
    result = vlm_sft.prepare_images(out,1,1,include_test=False)
    assert result['counts'] == {'train':1,'val':1}
    assert not (out/'test.json').exists()
    assert len(list(out.glob('*.png'))) == 2


def test_batched_generation_counts_only_until_first_eos():
    from book.part4_vlm.ch33_vlm_lora.tour import trim_generated
    assert trim_generated(torch.tensor([10,11,2,0,0]),[2]).tolist() == [10,11,2]
    assert trim_generated(torch.tensor([10,3,2,0]),[2,3]).tolist() == [10,3]
    assert trim_generated(torch.tensor([10,11,12]),[2]).tolist() == [10,11,12]


def test_paired_intervals_preserve_alignment_and_difference_direction():
    from book.part4_vlm.ch33_vlm_lora.summarize_supplement import paired
    def result(preds):
        return {'split':'val','observations':[dict(id=str(i),target=t,pred=p)
            for i,(t,p) in enumerate(zip(['x','z','a','c'],preds))]}
    baseline=result(['y','z','b','c']); candidate=result(['x','w','a','c'])
    delta=paired(baseline,candidate,repeats=100)
    assert delta['em_gained']==2 and delta['em_lost']==1
    assert delta['metrics']['em_norm']['value']==.25
    assert delta['metrics']['cer']['value']==-.25
    assert delta['metrics']['parse_rate']['value']==0
    candidate['observations'].reverse()
    with pytest.raises(AssertionError): paired(baseline,candidate,repeats=100)


@pytest.mark.parametrize('model_key,expected', [('qwen35',[248046,248044]),('qwen3vl',151645)])
def test_generation_eos_alignment(model_key, expected):
    from types import SimpleNamespace
    from book.part4_vlm.ch33_vlm_lora.tour import align_generation_eos
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=151645))
    align_generation_eos(model_key, model)
    assert model.generation_config.eos_token_id == expected


def test_evaluate_passes_dual_eos_and_counts_first_stop(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    from book.part4_vlm.ch33_vlm_lora import tour
    Image.new('RGB',(8,8)).save(tmp_path/'x.png')
    class Processor:
        tokenizer = SimpleNamespace(padding_side='right')
        image_processor = SimpleNamespace(to_dict=lambda:{})
        def apply_chat_template(self, *a, **kw): return 'prompt'
        def __call__(self, **kw): return {'input_ids':torch.tensor([[1]])}
        def decode(self, tokens, **kw):
            assert tokens.tolist() == [9,248046]
            return 'x'
    class Model:
        generation_config = SimpleNamespace(eos_token_id=248044)
        def eval(self): pass
        def generate(self, **kw):
            assert kw['eos_token_id'] == [248046,248044]
            assert self.generation_config.eos_token_id == [248046,248044]
            return torch.tensor([[1,9,248046,248044]])
    monkeypatch.setattr(tour,'local_snapshot',lambda *a:None)
    args=SimpleNamespace(model='qwen35',model_cache=None,data=tmp_path,device='cpu',adapter=None,max_new_tokens=3)
    result,_=tour.evaluate(args,Model(),Processor(),torch.float32,[dict(id='x',image='x.png',target='x')])
    assert result['truncated'] == 0
    assert result['observations'][0]['generated_tokens'] == 2
    assert result['eos_token_id'] == [248046,248044]


def test_training_restores_dual_eos_after_trl_alignment(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from book.part4_vlm.ch33_vlm_lora import tour
    model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=248044),save_pretrained=lambda p:None)
    processor=SimpleNamespace()
    monkeypatch.setattr(tour,'load',lambda a:(model,processor,torch.float32))
    monkeypatch.setattr(tour,'attach_lora',lambda *a:(model,{}))
    monkeypatch.setattr(tour,'parameter_audit',lambda m:dict(trainable=1,groups={}))
    monkeypatch.setattr(tour,'rows_for',lambda *a:[])
    monkeypatch.setattr(tour,'FormulaCollator',lambda *a:None)
    class Trainer:
        def __init__(self, **kw):
            assert kw['args'].eos_token == '<|im_end|>'
            self.model=kw['model']; self.model.generation_config.eos_token_id=248046
            self.state=SimpleNamespace(log_history=[])
        def train(self):
            assert self.model.generation_config.eos_token_id == [248046,248044]
            return SimpleNamespace(metrics={})
    monkeypatch.setitem(sys.modules,'trl',SimpleNamespace(SFTConfig=SimpleNamespace,SFTTrainer=Trainer))
    monkeypatch.setitem(sys.modules,'datasets',SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda x:x)))
    tour.main(['4','--data',str(tmp_path),'--out',str(tmp_path)])
    assert json.loads((tmp_path/'config.json').read_text())['eval_batch_size'] == 1
