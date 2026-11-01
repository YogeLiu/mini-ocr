"""独立手算、梯度对拍与错误输入；不依赖参考模型输出作唯一判据。"""
import math
import pytest
import torch
import torch.nn.functional as F
from ._impl import adapters as A
from .test_a2_transformer import _gpt, _rope_complex


def compare_grads(actual, expected, inputs, atol=1e-5):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=1e-5)
    ga=torch.autograd.grad(actual.square().sum(),inputs,retain_graph=True)
    gb=torch.autograd.grad(expected.square().sum(),inputs)
    for a,b in zip(ga,gb):
        torch.testing.assert_close(a,b,atol=atol,rtol=1e-4)


def test_rms_hand_and_gradients():
    x=torch.tensor([[3.,4.]],dtype=torch.double,requires_grad=True)
    w=torch.tensor([2.,3.],dtype=torch.double,requires_grad=True)
    y=A.run_rmsnorm(2,0,w,x)
    compare_grads(y,x*w/(x.square().mean(-1,keepdim=True)).sqrt(),(x,w))
    torch.testing.assert_close(y,torch.tensor([[6.,12.]],dtype=torch.double)/math.sqrt(12.5))


def test_rope_gradients():
    x=torch.randn(1,2,3,4,requires_grad=True)
    compare_grads(A.run_rope(x,100.,3),_rope_complex(x,100.,3),(x,))


def test_attention_hand_and_gradients():
    q=torch.zeros(1,1,2,2,requires_grad=True)
    k=torch.zeros(1,1,2,2,requires_grad=True)
    v=torch.tensor([[[[2.,4.],[6.,8.]]]],requires_grad=True)
    mask=torch.ones(2,2,dtype=torch.bool).tril()
    y=A.run_attention(q,k,v,mask)
    torch.testing.assert_close(y,torch.tensor([[[[2.,4.],[4.,6.]]]]))
    compare_grads(y,F.scaled_dot_product_attention(q,k,v,attn_mask=mask),(q,k,v))


def test_swiglu_hand_and_gradients():
    x=torch.tensor([[[0.,1.]]],dtype=torch.double,requires_grad=True)
    weights=[torch.eye(2,dtype=torch.double,requires_grad=True) for _ in range(3)]
    gate,up,down=weights
    y=A.run_swiglu(x,*weights)
    torch.testing.assert_close(y,torch.tensor([[[0.,1/(1+math.exp(-1))]]],dtype=torch.double))
    compare_grads(y,(F.silu(F.linear(x,gate))*F.linear(x,up))@down.T,(x,*weights))


def test_multihead_cross_gradients():
    g=torch.Generator().manual_seed(33)
    x=torch.randn(2,3,4,generator=g,dtype=torch.double,requires_grad=True)
    kv=torch.randn(2,5,4,generator=g,dtype=torch.double,requires_grad=True)
    ws=[torch.randn(4,4,generator=g,dtype=torch.double,requires_grad=True) for _ in range(4)]
    mask=torch.ones(3,5,dtype=torch.bool).tril(2)
    # 按头循环独立计算，避免复用被测实现的 reshape/transposes。
    q,k,v=F.linear(x,ws[0]),F.linear(kv,ws[1]),F.linear(kv,ws[2])
    heads=[F.scaled_dot_product_attention(q[...,i:i+2],k[...,i:i+2],v[...,i:i+2],attn_mask=mask) for i in (0,2)]
    want=F.linear(torch.cat(heads,-1),ws[3])
    compare_grads(A.run_multihead(x,kv,2,*ws,mask),want,(x,kv,*ws))


def test_gpt_gradients_independent_forward():
    from assignments.a2.tests.reference_forward import independent_forward
    fx=_gpt(); m=A.get_gpt(fx['cfg']); m.load_state_dict(fx['state_dict'])
    # state_dict 默认 detach；手动保留共享参数对象，检查梯度两条路径之和。
    sd=dict(m.named_parameters()); sd['head.weight']=sd['tok.weight']
    actual=m(fx['idx'])[0]
    expected=independent_forward(sd,fx['cfg'],fx['idx'])
    compare_grads(actual,expected,tuple(m.parameters()),atol=0.02)


class Scripted(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.blocks=[None]; self.calls=[]
    def forward(self,x,caches=None,pos0=0):
        self.calls.append((x.shape[1],pos0))
        logits=torch.full((*x.shape,4),-100.)
        logits[...,0]=100.  # banned，不能选
        logits[...,2]=10.
        logits[...,3]=torch.where(x==2,20.,0.)
        return logits,None


def test_generate_eos_banned_and_cache_calls():
    m=Scripted()
    out=A.run_generate(m,torch.tensor([[1],[2]]),4,3,[0],use_cache=True)
    assert out.tolist()==[[1,2,3],[2,3,3]]
    assert m.calls==[(1,0),(1,1)]
    fx=_gpt(); model=A.get_gpt(fx['cfg']); model.load_state_dict(fx['state_dict'])
    prompt=fx['idx'][:,:3]
    a=A.run_generate(model,prompt,4,63,[0],seed=7,greedy=False)
    b=A.run_generate(model,prompt,4,63,[0],seed=7,greedy=False)
    assert torch.equal(a,b)
    torch.testing.assert_close(A.run_generate(model,prompt,4,63,[0]),A.run_generate(model,prompt,4,63,[0],use_cache=False))


@pytest.mark.parametrize('case',range(12))
def test_bad_inputs(case):
    x=torch.ones(1,1,2,4); w=torch.eye(4)
    calls=[lambda:A.run_train_bpe(['ab'],-1,[]),
           lambda:A.run_rmsnorm(3,1e-6,torch.ones(3),x),
           lambda:A.run_rope(x[...,:3],100.,0),
           lambda:A.run_rope(x,100.,-1),
           lambda:A.run_attention(x,x,x,torch.zeros(2,2,dtype=torch.bool)),
           lambda:A.run_multihead(x[:,0],x[:,0],3,w,w,w,w),
           lambda:A.run_swiglu(x,w,w,torch.ones(3,4)),
           lambda:A.get_gpt(dict(_gpt()['cfg'],n_head=3)),
           lambda:A.run_gpt_incremental(_gpt()['cfg'],_gpt()['state_dict'],_gpt()['idx'],0),
           lambda:A.run_process_logits(torch.ones(1,4),0,0,1.),
           lambda:A.run_get_batch(torch.arange(4),1,4,torch.Generator()),
           lambda:A.run_generate(Scripted(),torch.ones(1,0,dtype=torch.long),2,3,[])]
    with pytest.raises(ValueError): calls[case]()


def test_special_literal_and_bpe_hand():
    vocab,merges=A.run_train_bpe(['ab ab abc abc abc'],3,['<|bos|>','<|eos|>'])
    assert merges==[(b'a',b'b'),(b' ',b'ab'),(b' ab',b'c')]
    tok=A.get_tokenizer(vocab,merges,['<|bos|>','<|eos|>'])
    assert tok.encode(' abcab')==[258,256]
    assert tok.encode('<|bos|>',allow_special=False)==list(b'<|bos|>')


def test_topk_ties_and_no_mutation():
    x=torch.tensor([[3.,2.,2.,0.]])
    y=A.run_process_logits(x,1.,2,1.)
    assert torch.isfinite(y).tolist()==[[True,True,True,False]]
    assert x.tolist()==[[3.,2.,2.,0.]]


@pytest.mark.parametrize('changes',[{'pos_emb':'none'},{'pos_emb':'learned'},{'norm':'layernorm'},{'tie_weights':False},{'d_ff':0}])
def test_book_config_variants(changes):
    cfg=dict(_gpt()['cfg'],**changes)
    m=A.get_gpt(cfg).eval()
    idx=_gpt()['idx']
    out=m(idx)[0]
    assert out.shape==(*idx.shape,cfg['vocab_size'])
    torch.testing.assert_close(A.run_gpt_incremental(cfg,m.state_dict(),idx,5),out,atol=1e-5,rtol=1e-4)
    if changes.get('d_ff')==0:
        assert m.state_dict()['blocks.0.mlp.gate.weight'].shape==(128,32)
    if changes.get('tie_weights') is False:
        assert m.state_dict()['head.weight'].data_ptr()!=m.state_dict()['tok.weight'].data_ptr()


def test_training_tools_invalid_inputs():
    m=A.get_gpt(_gpt()['cfg'])
    with pytest.raises(ValueError): A.run_accumulate_grads(m,[])
    with pytest.raises(ValueError): A.run_nll_sum(m,torch.tensor([1]),0)
