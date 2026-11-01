"""期望值来自正文手算、有限分布求和或 PyTorch 独立 oracle，无参考答案快照。"""
import math
import pytest
import torch
import torch.nn.functional as F
from ._impl import adapters as A


def t(x, grad=False):
    return torch.tensor(x, dtype=torch.float64, requires_grad=grad)


def test_bt_values_extremes_and_gradient():
    c, r = t([2., -1000., 1000.], True), t([.5, 0, 0], True)
    loss = A.bradley_terry_loss(c, r)
    torch.testing.assert_close(loss, t([math.log1p(math.exp(-1.5)), 1000., 0.]))
    assert torch.autograd.gradcheck(A.bradley_terry_loss, (c, r))


def test_dpo_hand_values_and_detach():
    c, r, rc, rr = (t(v, True) for v in ([-5., -3.], [-6., -7.], [-5., -5.], [-6., -6.]))
    loss, wc, wr = A.dpo_loss(c, r, rc, rr, beta=.5)
    torch.testing.assert_close(loss, t([math.log(2), math.log1p(math.exp(-1.5))]))
    torch.testing.assert_close(wc, t([0., 1.]))
    torch.testing.assert_close(wr, t([0., -.5]))
    loss.sum().backward()
    torch.testing.assert_close(c.grad, t([-.25, -.5/(1+math.exp(1.5))]))
    torch.testing.assert_close(r.grad, -c.grad)
    assert rc.grad is None and rr.grad is None
    assert torch.autograd.gradcheck(lambda x,y: A.dpo_loss(x,y,rc,rr,.5)[0], (c,r))


def test_dpo_initial_beta_point_one():
    c = t([-5.], True)
    loss = A.dpo_loss(c,t([-6.]),c.detach(),t([-6.]))[0]
    loss.sum().backward()
    assert c.grad.item() == pytest.approx(-.05)


def test_response_shift_sum_mask_and_gradient():
    x = t([[[0, math.log(3)], [math.log(3), 0], [0, 0], [4, -2]]], True)
    labels = torch.tensor([[-100, 1, -100, 0]])
    y = A.response_logprob(x, labels)
    torch.testing.assert_close(y, t([math.log(.75)+math.log(.5)]))
    oracle = -F.cross_entropy(x[:,:-1].reshape(-1,2), labels[:,1:].reshape(-1), ignore_index=-100, reduction='sum')
    torch.testing.assert_close(torch.autograd.grad(y.sum(), x, retain_graph=True)[0], torch.autograd.grad(oracle, x)[0])
    assert torch.autograd.gradcheck(lambda z: A.response_logprob(z,labels), (x,))
    torch.testing.assert_close(A.response_logprob(x, torch.full_like(labels,-100)), t([0.]))


def test_advantages_ch25_hand_table_and_groups():
    r = t([1,.1,.1,0, 7,7,7,7], True)
    want = t([.7,-.2,-.2,-.3,0,0,0,0])/(math.sqrt(.165)+.0001)
    out = A.group_advantages(r,4)
    assert out.shape == r.shape and out.dtype == r.dtype
    torch.testing.assert_close(out,want)
    torch.testing.assert_close(A.group_advantages(r,4,scale=False),t([.7,-.2,-.2,-.3,0,0,0,0]))
    assert torch.autograd.gradcheck(lambda x: A.group_advantages(x,4), (r[:4],))
    torch.testing.assert_close(A.group_advantages(t([2,3]),1),t([0,0]))


def test_ch34_sparse_reward_amplification():
    out = A.group_advantages(t([0]*7+[.2]),8,eps=1e-12)
    torch.testing.assert_close(out, t([-1/math.sqrt(7)]*7+[math.sqrt(7)]))


def test_kl_exact_expectation_direction_and_gradient():
    p,q = t([.2,.8]),t([.6,.4])
    x = (q/p).log().requires_grad_()
    k = A.kl_estimators(x)
    assert set(k) == {'k1','k2','k3'}
    torch.testing.assert_close(k['k1'],t([-math.log(3),math.log(2)]))
    torch.testing.assert_close(k['k2'],t([math.log(3)**2/2,math.log(2)**2/2]))
    torch.testing.assert_close(k['k3'],t([2-math.log(3),-.5+math.log(2)]))
    exact = .2*math.log(1/3)+.8*math.log(2)
    for key in ('k1','k3'):
        assert (p*k[key]).sum().item() == pytest.approx(exact)
    assert (k['k3'] >= 0).all()
    assert not math.isclose((p*k['k2']).sum().item(),exact,rel_tol=.01)
    for key in k:
        assert torch.autograd.gradcheck(lambda z: A.kl_estimators(z)[key], (x,))
    assert A.kl_estimators(t([1e-8]))['k3'].item() == pytest.approx(5e-17,rel=1e-7,abs=1e-24)


def batch(updated):
    mask = torch.tensor([[1,1,0],[1,1,1],[1,0,0],[1,1,0]])
    ratios = t([[1,1.5,1],[1,1,1],[1.1,1,1],[.5,1,1]]) if updated else torch.ones(4,3,dtype=torch.float64)
    new = (ratios.log()-2).requires_grad_()
    old = torch.full_like(new,-2,requires_grad=True)
    ref = old.detach().clone().requires_grad_()
    adv = (t([.7,-.2,-.2,-.3])/(math.sqrt(.165)+.0001)).requires_grad_()
    return new,old,ref,adv,mask


@pytest.mark.parametrize('updated',[False,True])
def test_ch25_complete_loss_and_gradient_tables(updated):
    new,old,ref,adv,mask = batch(updated)
    loss,d = A.grpo_loss(new,old,ref,adv,mask)
    assert loss.ndim == 0
    if updated:
        a = adv.detach().tolist()
        k15,k11,k05 = 1/1.5+math.log(1.5)-1,1/1.1+math.log(1.1)-1,1+math.log(.5)
        expected = (-1.1*a[0]+.02*k15-a[1]-1.1*a[2]+.04*k11-.9*a[3]+.02*k05)/4
        assert loss.item() == pytest.approx(expected,abs=1e-12)
        assert loss.item() == pytest.approx(-.0473,abs=5e-5)
        grad = t([[-a[0]/8,.04*(1-1/1.5)/8,0],[-a[1]/12]*3,
                  [(-1.1*a[2]+.04*(1-1/1.1))/4,0,0],[-.005,-a[3]/8,0]])
        assert d == pytest.approx(dict(clip_fraction=.25,kl_mean=(k15+k11+k05)/8,ratio_mean=1.0125))
    else:
        assert loss.item() == pytest.approx(0,abs=1e-14)
        grad = -adv.detach()[:,None]*mask/(4*mask.sum(-1)[:,None])
        assert d == pytest.approx(dict(clip_fraction=0,kl_mean=0,ratio_mean=1))
    loss.backward()
    torch.testing.assert_close(new.grad,grad)
    assert old.grad is None and ref.grad is None and adv.grad is None
    assert torch.autograd.gradcheck(lambda z: A.grpo_loss(z,old,ref,adv,mask)[0], (new,))


@pytest.mark.parametrize('adv,ratio,want',[(1,1.5,0),(1,.5,-.5),(-1,.5,0),(-1,1.5,1.5)])
def test_clipping_four_quadrants(adv,ratio,want):
    x = t([[math.log(ratio)]],True)
    loss,d = A.grpo_loss(x,t([[0]]),t([[0]]),t([adv]),t([[1]]),beta=0)
    loss.backward()
    assert x.grad.item() == pytest.approx(want)
    assert d['clip_fraction'] == 1  # 越界率，不是只统计有利方向饱和


def test_zero_advantage_keeps_kl_empty_rows_and_mask():
    x = t([[math.log(2),1000],[1000,1000]],True)
    loss,d = A.grpo_loss(x,torch.zeros_like(x),torch.zeros_like(x),t([0,0]),t([[1,0],[0,0]]))
    assert loss.item() == pytest.approx(.04*(math.log(2)-.5)/2)
    loss.backward()
    torch.testing.assert_close(x.grad,t([[.01,0],[0,0]]))
    z,diag = A.grpo_loss(x,torch.zeros_like(x),torch.zeros_like(x),t([0,0]),torch.zeros_like(x))
    assert z.item() == 0 and all(v == 0 for v in diag.values())


@pytest.mark.parametrize('case',range(17))
def test_invalid_inputs(case):
    v=t([0,1]); m=t([[0,1]])
    calls = [lambda:A.bradley_terry_loss(v,v[:1]),lambda:A.dpo_loss(v,v,v,v,beta=0),
             lambda:A.dpo_loss(v,v,v,v,beta=float('nan')),lambda:A.dpo_loss(v,v,v,v[:1]),
             lambda:A.group_advantages(v,3),lambda:A.group_advantages(v,0),
             lambda:A.group_advantages(v,2,eps=0),lambda:A.group_advantages(m,2),
             lambda:A.kl_estimators(t([float('nan')])),lambda:A.kl_estimators(t([])),
             lambda:A.grpo_loss(m,m[:,:1],m,t([1]),torch.ones_like(m)),
             lambda:A.grpo_loss(m,m,m,t([1]),t([[1,.5]])),
             lambda:A.grpo_loss(m,m,m,t([1]),m,epsilon=1),
             lambda:A.grpo_loss(m,m,m,t([1]),m,beta=-1),
             lambda:A.grpo_loss(m,m,m,t([[1]]),m),
             lambda:A.response_logprob(t([[[0,1],[0,1]]]),torch.tensor([[0,2]])),
             lambda:A.response_logprob(t([[[0,1],[0,1]]]),torch.tensor([[0]]))]
    with pytest.raises(ValueError): calls[case]()
