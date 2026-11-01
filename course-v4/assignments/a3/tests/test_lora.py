import copy
import pytest
import torch
from ._impl import adapters as A


def layer():
    m = A.get_lora(torch.tensor([[1.,2.],[3.,4.]],dtype=torch.double),torch.tensor([1.,-1.],dtype=torch.double),rank=1,alpha=2)
    with torch.no_grad():
        m.A.copy_(torch.tensor([[1.,-1.]])); m.B.copy_(torch.tensor([[2.],[3.]]))
    return m


def test_hand_forward_gradients_and_merge():
    m = layer(); x = torch.tensor([[2.,1.]],dtype=torch.double,requires_grad=True)
    assert m(x).shape == (1,2)
    torch.testing.assert_close(m(x),torch.tensor([[9.,15.]],dtype=torch.double))
    m(x).sum().backward()
    torch.testing.assert_close(x.grad,torch.tensor([[14.,-4.]],dtype=torch.double))
    torch.testing.assert_close(m.A.grad,torch.tensor([[20.,10.]],dtype=torch.double))
    torch.testing.assert_close(m.B.grad,torch.tensor([[2.],[2.]],dtype=torch.double))
    assert m.base.weight.grad is None and m.base.bias.grad is None
    old = copy.deepcopy(m.state_dict()); m.eval()
    first, second = m.merged(),m.merged()
    for merged in (first,second):
        assert not merged.training and not any(p.requires_grad for p in merged.parameters())
        torch.testing.assert_close(merged.weight,torch.tensor([[5.,-2.],[9.,-2.]],dtype=torch.double))
        torch.testing.assert_close(merged(x),m(x))
    with torch.no_grad(): first.weight.zero_()
    torch.testing.assert_close(second(x),m(x))
    for k,v in old.items(): torch.testing.assert_close(v,m.state_dict()[k])
    assert torch.autograd.gradcheck(m,(x.detach().requires_grad_(),))


def test_zero_init_freezing_and_first_step():
    torch.manual_seed(5)
    w = torch.randn(3,4); saved=w.clone()
    m = A.get_lora(w,rank=2,alpha=7)
    assert m.A.shape == (2,4) and m.B.shape == (3,2)
    assert m.A.count_nonzero() and not m.B.count_nonzero()
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 14
    with torch.no_grad(): w.zero_()
    x = torch.randn(2,3,4,requires_grad=True)
    torch.testing.assert_close(m(x),x@saved.T)
    m(x).square().sum().backward()
    assert not m.A.grad.count_nonzero() and m.B.grad.count_nonzero()
    assert x.grad is not None and m.base.weight.grad is None


@pytest.mark.parametrize('weight,bias,rank,alpha',[(torch.ones(2),None,1,1),(torch.ones(2,3),torch.ones(3),1,1),
    (torch.ones(2,3),None,0,1),(torch.ones(2,3),None,1.5,1),(torch.ones(2,3),None,1,0),
    (torch.ones(2,3),None,1,float('nan')),(torch.ones(2,3),None,1,float('inf'))])
def test_invalid(weight,bias,rank,alpha):
    with pytest.raises(ValueError): A.get_lora(weight,bias,rank=rank,alpha=alpha)


def test_peft_tiny_random_model_forward_backward_and_merge():
    # 必需依赖，不用 importorskip；整个随机网络无 from_pretrained/网络访问。
    from peft import LoraConfig, get_peft_model
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.proj=torch.nn.Linear(3,2)
        def forward(self,x): return self.proj(x).tanh()
    torch.manual_seed(20)
    base=Tiny(); ours=A.get_lora(base.proj.weight,base.proj.bias,rank=2,alpha=5)
    ref=get_peft_model(base,LoraConfig(r=2,lora_alpha=5,target_modules=['proj'],lora_dropout=0,bias='none'))
    p=ref.base_model.model.proj
    with torch.no_grad():
        ours.A.copy_(torch.randn_like(ours.A)*.2); ours.B.copy_(torch.randn_like(ours.B)*.2)
        p.lora_A.default.weight.copy_(ours.A); p.lora_B.default.weight.copy_(ours.B)
    x=torch.randn(2,4,3,requires_grad=True); z=x.detach().clone().requires_grad_()
    y=ours(x).tanh(); expected=ref(z)
    torch.testing.assert_close(y,expected)
    y.square().sum().backward(); expected.square().sum().backward()
    for a,b in [(x.grad,z.grad),(ours.A.grad,p.lora_A.default.weight.grad),(ours.B.grad,p.lora_B.default.weight.grad)]:
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    merged=ref.merge_and_unload()
    torch.testing.assert_close(ours.merged()(x).tanh(),merged(x),atol=1e-6,rtol=1e-5)
