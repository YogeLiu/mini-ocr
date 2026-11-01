"""独立手算/有限差分/PyTorch 对拍；不使用参考实现生成期望值。"""
import math
import random
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from ._impl import adapters as A


def test_value_hand_calculation_and_repeated_backward():
    a, b, c = [A.make_value(v) for v in (2, -3, 10)]
    loss = (a*b+c)**2
    assert loss.data == 16
    loss.backward()
    assert [v.grad for v in (a,b,c)] == [-24,16,8]
    loss.backward()
    assert [v.grad for v in (a,b,c)] == [-48,32,16]
    a.grad = 0
    a.backward()
    a.backward()
    assert a.grad == 2


def test_value_shared_graph():
    x = A.make_value(3)
    u = x*x
    y = u+u+x
    y.backward()
    assert y.data == 21 and x.grad == 13


@pytest.mark.parametrize("v", [-2., 0., 2.])
def test_value_relu(v):
    x = A.make_value(v)
    y = x.relu()
    y.backward()
    assert y.data == max(0,v) and x.grad == float(v>0)


def test_value_composite_gradient():
    x = A.make_value(1.3)
    y = (3-x).tanh() + 2/x + x/3 - (-x) + x**-2 + 2*x
    y.backward()
    f = lambda t: math.tanh(3-t)+2/t+t/3+t+t**-2+2*t
    assert y.data == pytest.approx(f(1.3))
    assert x.grad == pytest.approx((f(1.300001)-f(1.299999))/2e-6, rel=1e-7)
    z = A.make_value(0)
    out = z**0
    out.backward()
    assert out.data == 1 and z.grad == 0


def test_value_errors():
    with pytest.raises((TypeError, ValueError)):
        A.make_value(object())
    x = A.make_value(1)
    with pytest.raises((TypeError, AssertionError)):
        x ** x
    with pytest.raises(ZeroDivisionError):
        x / 0


def test_numerical_gradient():
    xs = [2., -3.]
    assert A.run_numerical_grad(lambda v: v[0]**2*v[1], xs) == pytest.approx([-12,4])
    assert xs == [2,-3]
    with pytest.raises(ValueError):
        A.run_numerical_grad(sum, xs, 0)


def test_softmax_hand_and_stability():
    z = torch.tensor([[0., math.log(2), math.log(3)]], dtype=torch.double, requires_grad=True)
    p = A.run_softmax(z)
    torch.testing.assert_close(p, torch.tensor([[1/6,2/6,3/6]],dtype=z.dtype))
    torch.testing.assert_close(A.run_softmax(z+10000), p)
    assert torch.autograd.gradcheck(A.run_softmax, (z,))
    big = torch.zeros(1,70000,dtype=torch.float16)
    out = A.run_softmax(big)
    assert out.dtype == big.dtype and torch.isfinite(out).all()
    assert out.float().sum().item() == pytest.approx(1,abs=.003)


def test_ce_hand_gradient_and_extreme():
    z = torch.zeros(2,2,dtype=torch.double,requires_grad=True)
    y = torch.tensor([0,1])
    loss = A.run_cross_entropy(z,y)
    assert loss.shape == () and loss.item() == pytest.approx(math.log(2))
    loss.backward()
    torch.testing.assert_close(z.grad, torch.tensor([[-.25,.25],[.25,-.25]],dtype=z.dtype))
    z = torch.tensor([[1000.,-1000.],[-3.,2.]],dtype=torch.double,requires_grad=True)
    actual = A.run_cross_entropy(z,y.flip(0))
    expected = F.cross_entropy(z,y.flip(0))
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,z)[0], torch.autograd.grad(expected,z)[0])


@pytest.mark.parametrize("case", ["empty", "rank", "integer", "nan", "label_shape", "label_type", "label_range"])
def test_loss_invalid(case):
    z, y = torch.ones(2,3), torch.tensor([0,1])
    if case == "empty": z,y = z[:0],y[:0]
    if case == "rank": z = z[0]
    if case == "integer": z = z.long()
    if case == "nan": z[0,0] = float('nan')
    if case == "label_shape": y = y[:,None]
    if case == "label_type": y = y.float()
    if case == "label_range": y[0] = 3
    with pytest.raises(ValueError): A.run_cross_entropy(z,y)
    if case in {"empty","rank","integer","nan"}:
        with pytest.raises(ValueError): A.run_softmax(z)


def test_linear_manual_gradient():
    X = torch.tensor([[1.,2.],[3.,4.]],dtype=torch.double)
    W = torch.zeros(2,2,dtype=torch.double)
    y = torch.tensor([0,1])
    # no_grad ensures the submitted dW cannot depend on building a backward graph.
    with torch.no_grad(): loss,dw = A.run_softmax_classifier(W,X,y)
    assert loss.item() == pytest.approx(math.log(2))
    torch.testing.assert_close(dw,torch.tensor([[.5,-.5],[.5,-.5]],dtype=W.dtype))
    W = torch.tensor([[.2,-.3],[.7,.1]],dtype=torch.double,requires_grad=True)
    ref = F.cross_entropy(X@W,y)+.13*(W**2).sum()
    with torch.no_grad(): loss,dw = A.run_softmax_classifier(W.detach(),X,y,.13)
    torch.testing.assert_close(loss,ref)
    torch.testing.assert_close(dw,torch.autograd.grad(ref,W)[0])
    with pytest.raises(ValueError): A.run_softmax_classifier(W,X[:,:1],y)
    with pytest.raises(ValueError): A.run_softmax_classifier(W,X,y,-1)


def test_mlp_hand_and_all_gradients():
    ts = [torch.tensor(t,dtype=torch.double,requires_grad=True) for t in
          ([[1.,-2.]], [[1.,-1.],[0.,1.]], [0.,0.], [[2.,3.],[4.,5.]], [1.,-1.])]
    out = A.run_mlp(*ts)
    torch.testing.assert_close(out, torch.tensor([[3.,2.]],dtype=torch.double))
    assert torch.autograd.gradcheck(A.run_mlp,tuple(ts))
    ref = F.linear(F.relu(F.linear(ts[0],ts[1].T,ts[2])),ts[3].T,ts[4])
    for a,b in zip(torch.autograd.grad(out.sum(),ts),torch.autograd.grad(ref.sum(),ts)):
        torch.testing.assert_close(a,b)
    with pytest.raises(ValueError): A.run_mlp(ts[0],ts[1],ts[2][:1],ts[3],ts[4])


@pytest.mark.parametrize("hidden", [0,3])
def test_classifier_module(hidden):
    m = A.get_classifier(2,hidden,2).double()
    sd = m.state_dict()
    assert set(sd) == ({'w','b'} if not hidden else {'w1','b1','w2','b2'})
    assert all(torch.count_nonzero(v)==0 for k,v in sd.items() if k.startswith('b'))
    assert all(torch.count_nonzero(v)>0 for k,v in sd.items() if k.startswith('w'))
    assert sum(p.numel() for p in m.parameters()) == (6 if not hidden else 17)
    with torch.no_grad():
        for p in m.parameters(): p.fill_(.5)
    x = torch.tensor([[1.,2.],[-1.,-2.]],dtype=torch.double,requires_grad=True)
    expected = torch.tensor([[3.5,3.5],[.5,.5]] if hidden else [[2.,2.],[-1.,-1.]],dtype=torch.double)
    torch.testing.assert_close(m(x),expected)
    m(x).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    with pytest.raises(ValueError): m(torch.zeros(1,3,dtype=torch.double))
    with pytest.raises(ValueError): A.get_classifier(0,hidden,2)


def test_adamw_hand_and_none_grad():
    p = torch.nn.Parameter(torch.tensor([2.],dtype=torch.double))
    q = torch.nn.Parameter(torch.tensor([3.],dtype=torch.double))
    opt = A.get_adamw([p,q],lr=.1,betas=(.5,.5),eps=.1,weight_decay=.2)
    p.grad = torch.tensor([4.],dtype=torch.double)
    opt.step()
    assert p.item() == pytest.approx(2*.98-.1*4/4.1)
    assert q.item() == 3 and p.is_leaf
    opt.zero_grad()
    assert p.grad is None


def test_adamw_multistep_torch():
    p = [torch.nn.Parameter(torch.tensor([1.,-2.],dtype=torch.double)) for _ in range(2)]
    q = [torch.nn.Parameter(x.detach().clone()) for x in p]
    args = dict(lr=.03,betas=(.7,.8),eps=1e-5,weight_decay=.2)
    a,b = A.get_adamw(p,**args),torch.optim.AdamW(q,**args)
    for step in range(5):
        a.lr = .03/(step+1)
        b.param_groups[0]['lr'] = a.lr
        for j in range(2):
            g = None if j==1 and step%2==0 else torch.tensor([step-2.,.1],dtype=torch.double)
            p[j].grad = None if g is None else g.clone()
            q[j].grad = None if g is None else g.clone()
        a.step(); b.step()
        for x,y in zip(p,q): torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize("kwargs", [dict(lr=-1),dict(betas=(1.,.9)),dict(eps=0),dict(weight_decay=-1)])
def test_adamw_invalid(kwargs):
    with pytest.raises(ValueError): A.get_adamw([torch.nn.Parameter(torch.ones(1))],**kwargs)


def test_schedule_hand_boundaries():
    assert [A.run_schedule(s,6,2,1.,.2) for s in (0,1,2,4,6,7)] == pytest.approx([.5,1,1,.6,.2,.2])
    assert A.run_schedule(0,4,0,1) == 1
    assert A.run_schedule(3,4,4,1) == 1
    assert A.run_schedule(4,4,4,1) == 0


@pytest.mark.parametrize("args", [(-1,5,1,1,0),(0,0,0,1,0),(0,5,6,1,0),(0,5,1,.1,.2),(0,5,1,float('nan'),0)])
def test_schedule_invalid(args):
    with pytest.raises(ValueError): A.run_schedule(*args)


def test_clip_global_norm():
    ps = [torch.nn.Parameter(torch.zeros(n,dtype=torch.double)) for n in (2,1,1)]
    ps[0].grad, ps[1].grad = torch.tensor([3.,4.],dtype=torch.double),torch.tensor([12.],dtype=torch.double)
    assert A.run_clip_grad_norm(ps,1) == 13
    torch.testing.assert_close(ps[0].grad,torch.tensor([3.,4.],dtype=torch.double)/(13+1e-6))
    assert ps[1].grad.item() == pytest.approx(12/(13+1e-6))
    old = ps[0].grad.clone()
    A.run_clip_grad_norm(ps,2)
    torch.testing.assert_close(old,ps[0].grad)
    assert A.run_clip_grad_norm([],1) == 0
    with pytest.raises(ValueError): A.run_clip_grad_norm(ps,-1)


def test_checkpoint_exact_resume(tmp_path):
    torch.manual_seed(9); random.seed(9); np.random.seed(9)
    m = A.get_classifier(2,3,2)
    opt = A.get_adamw(m.parameters(),lr=.02,betas=(.6,.8),weight_decay=.3)
    gens = {k:torch.Generator().manual_seed(s) for k,s in [('batch',4),('aug',5)]}
    def step(model, optimizer, generators, s):
        x = torch.randn(3,2,generator=generators['batch'])
        x += torch.rand(3,2,generator=generators['aug'])*.01 + random.random()*.01 + np.random.rand()*.01 + torch.rand(1)*.01
        optimizer.lr = A.run_schedule(s,6,1,.02)
        optimizer.zero_grad()
        loss = A.run_cross_entropy(model(x),torch.tensor([0,1,0]))
        loss.backward(); A.run_clip_grad_norm(model.parameters(),1); optimizer.step()
        return loss.detach().clone()
    step(m,opt,gens,0)
    path = str(tmp_path/'checkpoint.pt')
    A.save_checkpoint(path,m,opt,1,gens,{'note':'resume'})
    expected = [step(m,opt,gens,s) for s in range(1,4)]
    n = A.get_classifier(2,3,2)
    other = A.get_adamw(n.parameters())
    gs = {k:torch.Generator().manual_seed(123) for k in gens}
    completed,extra = A.load_checkpoint(path,n,other,gs)
    assert completed == 1 and extra == {'note':'resume'}
    actual = [step(n,other,gs,s) for s in range(completed,4)]
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
    assert all(torch.equal(v,n.state_dict()[k]) for k,v in m.state_dict().items())
    with pytest.raises(FileNotFoundError): A.load_checkpoint(str(tmp_path/'missing'),n)


def test_mlp_relu_zero_gradient():
    x = torch.zeros(1,1,requires_grad=True)
    w1 = torch.ones(1,1,requires_grad=True)
    b1 = torch.zeros(1,requires_grad=True)
    out = A.run_mlp(x,w1,b1,torch.ones(1,1),torch.zeros(1))
    out.sum().backward()
    assert x.grad.item() == 0 and b1.grad.item() == 0
