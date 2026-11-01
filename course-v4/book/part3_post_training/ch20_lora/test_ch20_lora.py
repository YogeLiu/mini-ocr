"""CPU 快测：非零增量、输入梯度、冻结基座、递归替换和状态恢复。"""
import copy
import pytest
import torch
from torch import nn
from miniocr.nn.lora import (LoRALinear, inject_lora, merge_lora, parameter_counts,
                            adapter_state, memory_estimate)
from book.part3_post_training.ch20_lora.tour import mechanism, parity, tiny_qwen, main


def test_initialization_and_first_gradient():
    result=mechanism()
    assert result['initial_max_error']==0
    assert result['first_A_grad_norm']==0
    assert result['first_B_grad_norm']>0


@pytest.mark.parametrize('shape',[(5,11),(2,3,11)])
def test_formula_gradient_and_merge(shape):
    torch.manual_seed(3)
    layer=LoRALinear(nn.Linear(11,7,bias=True).double(),3,7)
    with torch.no_grad(): layer.B.normal_()
    x=torch.randn(*shape,dtype=torch.double,requires_grad=True)
    y=x.detach().clone().requires_grad_()
    output=layer(x)
    explicit=nn.functional.linear(y,layer.base.weight+layer.scale*(layer.B@layer.A),layer.base.bias)
    torch.testing.assert_close(output,explicit)
    output.sum().backward(); explicit.sum().backward()
    torch.testing.assert_close(x.grad,y.grad)
    merged=layer.merged()
    torch.testing.assert_close(merged(x),output)
    torch.testing.assert_close(layer.merged().weight,merged.weight)
    assert not merged.weight.requires_grad and layer.base.weight.grad is None
    torch.testing.assert_close(merge_lora(layer)(x),output)


def test_attach_counts_alpha_and_atomic_error():
    model=nn.ModuleDict({'a':nn.Linear(5,7),'b':nn.Linear(7,2)})
    original=copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError): inject_lora(model,['a','missing'])
    assert all(p.requires_grad for p in model.parameters())
    inject_lora(model,['a','b'],4,8)
    assert parameter_counts(model)['formula']==4*(5+7+7+2)
    assert parameter_counts(model)['trainable']==84
    with pytest.raises(ValueError): inject_lora(model,['a'])
    assert torch.equal(model['a'].base.weight,original['a.weight'])
    count=parameter_counts(model)
    model['a'].alpha=99
    assert parameter_counts(model)==count


def test_adapter_roundtrip_and_recursive_merge():
    torch.manual_seed(8)
    model=nn.Sequential(nn.Linear(4,4),nn.Sequential(nn.Linear(4,3)))
    other=copy.deepcopy(model)
    inject_lora(model,['0'],2,4)
    inject_lora(other,['0'],2,4)
    for m in model.modules():
        if isinstance(m,LoRALinear):
            with torch.no_grad(): m.B.normal_()
    state=adapter_state(model)
    other.load_state_dict(state,strict=False)
    x=torch.randn(5,4)
    torch.testing.assert_close(model(x),other(x))
    expected=model(x)
    merge_lora(model); merge_lora(model)
    torch.testing.assert_close(model(x),expected)
    assert not any(isinstance(m,LoRALinear) for m in model.modules())


def test_peft_forward_and_gradients():
    pytest.importorskip('peft')
    assert max(parity().values())<2e-6


def test_tiny_qwen_forward_backward_and_memory():
    torch.set_num_threads(2)
    model=tiny_qwen().eval()
    tokens=torch.tensor([[1,2,3,4]])
    original=model(tokens).logits.detach()
    inject_lora(model,['q_proj','v_proj'],4,8)
    torch.testing.assert_close(model(tokens).logits,original)
    model(tokens).logits.square().mean().backward()
    assert all(p.grad is None for n,p in model.named_parameters() if not n.endswith(('.A','.B')))
    counts=parameter_counts(model)
    assert counts['trainable']==counts['formula']
    e=memory_estimate(model,1,32)
    e2=memory_estimate(model,2,64)
    assert e2['weights_bytes']==e['weights_bytes']
    assert e2['activation_proxy_bytes']==4*e['activation_proxy_bytes']
    assert e['adam_m_v_bytes']==8*counts['trainable']


def test_cli_safe_default_and_test_guard(tmp_path):
    main(['1','2','7','8','--out',str(tmp_path)])
    assert (tmp_path/'part1.json').exists()
    assert not (tmp_path/'FINAL_TEST_STARTED.json').exists()
    with pytest.raises(SystemExit): main(['--ranks','0'])


def test_final_test_refuses_repeated_access_before_loading(tmp_path):
    from book.part3_post_training.ch20_lora.tour import part8
    from types import SimpleNamespace
    (tmp_path/'selection.json').write_text('{}')
    (tmp_path/'FINAL_TEST_STARTED.json').write_text('{}')
    with pytest.raises(ValueError,match='已开始过 test'):
        part8(SimpleNamespace(final_test=True,out=tmp_path))


def test_real_tokenizer_mask_when_cached():
    from miniocr.models.qwen3 import find_local_qwen3
    from miniocr.data.sft import encode_sft
    path=find_local_qwen3()
    if path is None: pytest.skip('本地没有Qwen3权重')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
    sample=encode_sft(tok,'x_2','x^2')
    assert sample is not None
    assert all(x==-100 for x in sample.labels[:sample.prompt_length])
    assert sample.labels[-1]==tok.eos_token_id


@pytest.mark.parametrize('rank,alpha',[(0,8),(-1,8),(2,0),(2,float('nan'))])
def test_invalid_hyperparameters(rank,alpha):
    with pytest.raises(ValueError): LoRALinear(nn.Linear(2,3),rank,alpha)
