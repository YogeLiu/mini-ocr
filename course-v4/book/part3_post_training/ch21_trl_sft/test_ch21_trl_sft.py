"""CPU 快测：验证监督合同、真正的 TRL 更新以及保存/重载。"""
import copy

import pytest
import torch

from miniocr.data.sft import SFTExample, collate_sft
from miniocr.models.qwen3 import find_local_qwen3
from miniocr.train.trl_sft import dataset_rows, audit_batch, tiny_fixture, make_trainer, loss_step_parity


@pytest.fixture
def fixture():
    pytest.importorskip('trl')
    pytest.importorskip('peft')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    return tiny_fixture()


def test_ch21_invalid_labels():
    with pytest.raises(ValueError): dataset_rows([])
    with pytest.raises(ValueError): dataset_rows([SFTExample((3,4),(-100,-100),2)])


def test_ch21_collation_masks_eos_and_padding(fixture,tmp_path):
    model,tok,examples=fixture
    trainer=make_trainer(model,tok,examples,tmp_path)
    batch=trainer.data_collator(dataset_rows(examples))
    reference=collate_sft(examples,tok.pad_token_id)
    for key in reference: torch.testing.assert_close(batch[key],reference[key])
    assert audit_batch(batch,tok,examples)['supervised_tokens']==6
    broken=copy.deepcopy(batch)
    broken['labels'][0,1]=4
    with pytest.raises(AssertionError): audit_batch(broken,tok,examples)
    broken=copy.deepcopy(batch)
    broken['labels'][0,3]=-100
    with pytest.raises(AssertionError): audit_batch(broken,tok,examples)


def test_ch21_fused_loss_gradient_and_adam_parity(fixture,tmp_path):
    model,tok,examples=fixture
    trainer=make_trainer(model,tok,examples,tmp_path)
    before={k:v.clone() for k,v in trainer.model.state_dict().items()}
    result=loss_step_parity(trainer,[collate_sft([e],0) for e in examples])
    assert result['microbatches']==2 and result['max_parameter_delta']>0
    for k,v in trainer.model.state_dict().items(): torch.testing.assert_close(v,before[k],atol=0,rtol=0)


def test_ch21_trainer_update_and_adapter_reload(fixture,tmp_path):
    from peft import PeftModel
    model,tok,examples=fixture
    base=copy.deepcopy(model)
    trainer=make_trainer(model,tok,examples,tmp_path/'train',steps=1,accumulation=2)
    frozen={k:p.clone() for k,p in trainer.model.named_parameters() if not p.requires_grad}
    before={k:p.clone() for k,p in trainer.model.named_parameters() if p.requires_grad}
    trainer.train()
    assert any(not torch.equal(p,before[k]) for k,p in trainer.model.named_parameters() if p.requires_grad)
    for k,p in trainer.model.named_parameters():
        if not p.requires_grad: torch.testing.assert_close(p,frozen[k],atol=0,rtol=0)
    trainer.model.save_pretrained(tmp_path/'adapter')
    loaded=PeftModel.from_pretrained(base,tmp_path/'adapter').eval()
    batch=collate_sft(examples,0)
    inputs={k:v for k,v in batch.items() if k!='labels'}
    trainer.model.eval()
    with torch.no_grad(): torch.testing.assert_close(trainer.model(**inputs).logits,loaded(**inputs).logits)


@pytest.mark.skipif(find_local_qwen3() is None,reason='本地没有 Qwen3 大权重')
def test_ch21_real_tokenizer_contract():
    from transformers import AutoTokenizer
    from miniocr.data.sft import encode_sft
    tok=AutoTokenizer.from_pretrained(str(find_local_qwen3()),local_files_only=True)
    ex=encode_sft(tok,'x^2','x^{2}')
    assert audit_batch(collate_sft([ex],tok.pad_token_id),tok,[ex])['completion_only']


def test_ch21_test_guard_never_loads_model_without_permission(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from book.part3_post_training.ch21_trl_sft import tour
    def forbidden(*args):
        raise AssertionError('不应加载模型或数据')
    monkeypatch.setattr(tour,'selected',forbidden)
    monkeypatch.setattr(tour,'data',forbidden)
    args=SimpleNamespace(out=tmp_path,final_test=False)
    assert 'skipped' in tour.part5(args)
    args.final_test=True
    with pytest.raises(ValueError,match='重载验证'): tour.part5(args)
    (tmp_path/'reload-verified.json').write_text('{}')
    (tmp_path/'FINAL_TEST_STARTED.json').write_text('{}')
    with pytest.raises(ValueError,match='禁止反复'): tour.part5(args)
