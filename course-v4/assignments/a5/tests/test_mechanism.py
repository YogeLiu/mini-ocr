"""小整数手算 + 独立 PyTorch 算子 + float64 中心差分；无需二进制快照。"""
import math
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from ._impl import adapters as A


def rand(*shape):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(sum(shape)), dtype=torch.double, requires_grad=True)


def compare_grads(y, ref, xs):
    torch.testing.assert_close(y, ref)
    g = torch.arange(y.numel(), dtype=y.dtype).reshape(y.shape) + .3
    actual = torch.autograd.grad(y, xs, g, retain_graph=True)
    expected = torch.autograd.grad(ref, xs, g, retain_graph=True)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-6)


def test_conv_hand_values_and_overlap():
    x = np.arange(1.,10.).reshape(1,1,3,3)
    w = np.ones((1,1,2,2)); b = np.array([.5])
    y,c = A.conv_forward(x,w,b)
    np.testing.assert_allclose(y, [[[[12.5,16.5],[24.5,28.5]]]])
    dx,dw,db = A.conv_backward(np.ones_like(y),c)
    np.testing.assert_allclose(dx, [[[[1,2,1],[2,4,2],[1,2,1]]]])
    np.testing.assert_allclose(dw, [[[[12,16],[24,28]]]])
    np.testing.assert_allclose(db,[4])
    x[:] = 99; w[:] = 0
    for actual,expected in zip(A.conv_backward(np.ones_like(y),c),(dx,dw,db)):
        np.testing.assert_allclose(actual,expected)


@pytest.mark.parametrize('stride,padding', [(1,0),(2,1),(3,2)])
def test_conv_oracle_and_numerical_gradient(stride,padding):
    rng = np.random.default_rng(27)
    xs = [rng.normal(size=s) for s in ((2,2,3,4),(2,2,2,3),(2,))]
    y,c = A.conv_forward(*xs,stride,padding)
    ts = [torch.tensor(v,requires_grad=True) for v in xs]
    ref = F.conv2d(*ts,stride=stride,padding=padding)
    np.testing.assert_allclose(y,ref.detach(),atol=1e-12)
    upstream = rng.normal(size=y.shape)
    grads = A.conv_backward(upstream,c)
    expected = torch.autograd.grad(ref,ts,torch.tensor(upstream))
    for k,(x,g,e) in enumerate(zip(xs,grads,expected)):
        assert g.shape == x.shape
        np.testing.assert_allclose(g,e,atol=1e-10)
        numerical = np.zeros_like(x)
        for idx in np.ndindex(x.shape):
            old = x[idx]
            x[idx] = old+1e-5
            plus = (A.conv_forward(*xs,stride,padding)[0]*upstream).sum()
            x[idx] = old-1e-5
            minus = (A.conv_forward(*xs,stride,padding)[0]*upstream).sum()
            x[idx] = old
            numerical[idx] = (plus-minus)/2e-5
        np.testing.assert_allclose(g,numerical,atol=1e-8,rtol=1e-6)


@pytest.mark.parametrize('kind', ['channels','bias','stride','padding','kernel','dout'])
def test_conv_invalid(kind):
    x,w,b = np.ones((1,1,3,3)),np.ones((1,1,2,2)),np.zeros(1)
    with pytest.raises(ValueError):
        if kind == 'channels': A.conv_forward(x,np.ones((1,2,2,2)),b)
        elif kind == 'bias': A.conv_forward(x,w,np.zeros(2))
        elif kind == 'stride': A.conv_forward(x,w,b,0)
        elif kind == 'padding': A.conv_forward(x,w,b,padding=-1)
        elif kind == 'kernel': A.conv_forward(x,np.ones((1,1,4,4)),b)
        else:
            _,c = A.conv_forward(x,w,b)
            A.conv_backward(np.ones((1,1,1,1)),c)


def test_patch_hand_order_and_grid():
    x = torch.arange(1.,17.).reshape(1,2,2,4)
    torch.testing.assert_close(A.patchify(x,2),torch.tensor([[[1.,2,5,6,9,10,13,14],[3,4,7,8,11,12,15,16]]]))
    w = torch.tensor([[[[0.,1],[0,0]]]])
    torch.testing.assert_close(A.patch_embed(x[:,:1],w,torch.tensor([.5])),torch.tensor([[[2.5],[4.5]]]))
    assert A.patch_grid(5,7,2,True) == (3,4)
    assert A.patch_grid(4,6,2) == (2,3)


@pytest.mark.parametrize('bias', [False,True])
def test_patch_gradients(bias):
    x,w,b = rand(1,2,4,6),rand(3,2,2,2),rand(3)
    xs = (x,w,b) if bias else (x,w)
    fn = lambda *z: A.patch_embed(*z)
    y = fn(*xs)
    ref = F.conv2d(x,w,b if bias else None,stride=2).flatten(2).transpose(1,2)
    compare_grads(y,ref,xs)
    assert torch.autograd.gradcheck(fn,xs,fast_mode=True)
    assert torch.autograd.gradcheck(lambda z:A.patchify(z,2),(x,),fast_mode=True)


@pytest.mark.parametrize('cls', [False,True])
def test_positions_oracle_axes_and_gradients(cls):
    p = rand(1,6+int(cls),2)
    assert A.interpolate_positions(p,(2,3),(2,3),cls) is p
    y = A.interpolate_positions(p,(2,3),(3,4),cls)
    spatial = p[:,int(cls):].reshape(1,2,3,2).permute(0,3,1,2)
    ref = F.interpolate(spatial,size=(3,4),mode='bicubic',align_corners=False).flatten(2).transpose(1,2)
    ref = torch.cat((p[:,:int(cls)],ref),1)
    compare_grads(y,ref,(p,))
    assert torch.autograd.gradcheck(lambda z:A.interpolate_positions(z,(2,3),(3,4),cls),(p,),fast_mode=True)
    constant = torch.full((1,7,1),3.,dtype=torch.double); constant[:,0]=99
    out = A.interpolate_positions(constant,(2,3),(4,2))
    torch.testing.assert_close(out[:,0],constant[:,0])
    torch.testing.assert_close(out[:,1:],torch.full((1,8,1),3.,dtype=torch.double))


@pytest.mark.parametrize('case',range(9))
def test_patch_positions_invalid(case):
    with pytest.raises(ValueError):
        if case == 0: A.patch_grid(3,4,2)
        elif case == 1: A.patch_grid(4,4,0)
        elif case == 2: A.patchify(torch.zeros(1,3,4),2)
        elif case == 3: A.patchify(torch.zeros(1,1,3,4),2)
        elif case == 4: A.patch_embed(torch.zeros(1,1,4,4),torch.zeros(2,2,2,2))
        elif case == 5: A.patch_embed(torch.zeros(1,1,4,4),torch.zeros(2,1,2,3))
        elif case == 6: A.patch_embed(torch.zeros(1,1,4,4),torch.zeros(2,1,2,2),torch.zeros(3))
        elif case == 7: A.interpolate_positions(torch.zeros(1,5,2),(2,3),(3,4))
        else: A.interpolate_positions(torch.zeros(1,5,2),(2,2),(0,4))


def block_oracle(x,s,heads):
    d = x.shape[-1]
    z = F.layer_norm(x,(d,),s['norm1.weight'],s['norm1.bias'],1e-5)
    q,k,v = F.linear(z,s['qkv.weight'],s['qkv.bias']).reshape(*x.shape[:2],3,heads,d//heads).permute(2,0,3,1,4).unbind(0)
    z = F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape_as(x)
    x = x+F.linear(z,s['proj.weight'],s['proj.bias'])
    z = F.layer_norm(x,(d,),s['norm2.weight'],s['norm2.bias'],1e-5)
    z = F.gelu(F.linear(z,s['mlp.0.weight'],s['mlp.0.bias']))
    return x+F.linear(z,s['mlp.2.weight'],s['mlp.2.bias'])


def test_block_forward_backward_bidirectional():
    torch.manual_seed(4)
    m = A.get_vision_block(4,2).double()
    x = rand(1,3,4)
    s = dict(m.named_parameters())
    compare_grads(m(x),block_oracle(x,s,2),(x,*s.values()))
    assert torch.autograd.gradcheck(m,(x,),fast_mode=True)
    perm = [2,0,1]
    torch.testing.assert_close(m(x[:,perm]),m(x)[:,perm])
    changed = x.detach().clone(); changed[:,2,0]+=3
    assert not torch.allclose(m(changed)[:,0],m(x)[:,0])
    with torch.no_grad():
        for p in m.parameters(): p.zero_()
    torch.testing.assert_close(m(x),x)  # 残差手算


def test_vit_full_oracle_roundtrip_gradient():
    torch.manual_seed(9)
    cfg = dict(image_size=4,patch=2,width=4,depth=1,heads=2,classes=3)
    m = A.get_vit(cfg).double()
    x = rand(1,1,4,6)
    s = dict(m.named_parameters())
    z = F.conv2d(x,s['weight'],s['bias'],stride=2).flatten(2).transpose(1,2)
    pos = s['pos']
    spatial = F.interpolate(pos[:,1:].reshape(1,2,2,4).permute(0,3,1,2),size=(2,3),mode='bicubic',align_corners=False).flatten(2).transpose(1,2)
    z = torch.cat((s['cls'].expand(1,-1,-1),z),1)+torch.cat((pos[:,:1],spatial),1)
    z = block_oracle(z,{k.removeprefix('blocks.0.'):v for k,v in s.items() if k.startswith('blocks.0.')},2)
    ref = F.linear(F.layer_norm(z,(4,),s['norm.weight'],s['norm.bias'])[:,0],s['head.weight'],s['head.bias'])
    assert m(x).shape == (1,3)
    compare_grads(m(x),ref,(x,*s.values()))
    assert torch.autograd.gradcheck(m,(x,),fast_mode=True)
    clone = A.get_vit(cfg).double(); clone.load_state_dict(m.state_dict())
    torch.testing.assert_close(m(x),clone(x))
    with torch.no_grad():
        for p in m.parameters(): p.zero_()
        m.load_state_dict({**m.state_dict(),'head.bias':torch.tensor([1.,2.,3.],dtype=torch.double)})
    torch.testing.assert_close(m(x),torch.tensor([[1.,2.,3.]],dtype=torch.double))


def test_connector_oracle_token_independence():
    torch.manual_seed(5)
    m = A.get_connector(4).double(); x=rand(2,3,4)
    s=dict(m.named_parameters())
    z=F.layer_norm(x,(4,),s['net.0.weight'],s['net.0.bias'])
    ref=F.linear(F.gelu(F.linear(z,s['net.1.weight'],s['net.1.bias'])),s['net.3.weight'],s['net.3.bias'])
    compare_grads(m(x),ref,(x,*s.values()))
    assert torch.autograd.gradcheck(m,(x,),fast_mode=True)
    changed=x.detach().clone(); changed[:,1]+=10
    torch.testing.assert_close(m(changed)[:,[0,2]],m(x)[:,[0,2]])
    with torch.no_grad():
        for p in m.parameters(): p.zero_()
        m.load_state_dict({**m.state_dict(),'net.3.bias':torch.tensor([1.,2.,3.,4.],dtype=torch.double)})
    torch.testing.assert_close(m(x),torch.tensor([1.,2.,3.,4.],dtype=torch.double).expand(2,3,4))


@pytest.mark.parametrize('case',range(7))
def test_modules_invalid(case):
    with pytest.raises(ValueError):
        if case==0: A.get_vision_block(3,2)
        elif case==1: A.get_vision_block(4,0)
        elif case==2: A.get_connector(0)
        elif case==3: A.get_connector(4)(torch.zeros(1,3,5))
        elif case==4: A.get_vit(dict(image_size=4,patch=2,width=4,depth=0,heads=2,classes=3))
        else:
            m=A.get_vit(dict(image_size=4,patch=2,width=4,depth=1,heads=2,classes=3))
            m(torch.zeros(1,2 if case==5 else 1,3 if case==6 else 4,4))


def test_clip_hand_and_gradient():
    s=torch.zeros(4,4,dtype=torch.double,requires_grad=True)
    loss=A.clip_loss(s)
    assert loss.ndim==0 and loss.item()==pytest.approx(math.log(4))
    torch.testing.assert_close(torch.autograd.grad(loss,s)[0],(torch.full((4,4),.25,dtype=torch.double)-torch.eye(4,dtype=torch.double))/4)
    assert A.clip_loss(torch.tensor([[123.]],dtype=torch.double)).item()==pytest.approx(0.)
    s=rand(3,3)
    labels=torch.arange(3)
    ref=(F.cross_entropy(s,labels)+F.cross_entropy(s.T,labels))/2
    compare_grads(A.clip_loss(s),ref,(s,))
    assert torch.autograd.gradcheck(A.clip_loss,(s,))
    torch.testing.assert_close(A.clip_loss(s+10000),ref)
    torch.testing.assert_close(A.clip_loss(s.T),ref)
    torch.testing.assert_close(A.clip_loss(s[[2,0,1]][:,[2,0,1]]),ref)
    extreme=torch.tensor([[1000.,-1000.],[1000.,-1000.]],requires_grad=True)
    assert torch.isfinite(A.clip_loss(extreme))
    assert torch.isfinite(torch.autograd.grad(A.clip_loss(extreme),extreme)[0]).all()


@pytest.mark.parametrize('shape',[(0,0),(2,3),(3,),(1,2,2)])
def test_clip_invalid(shape):
    with pytest.raises(ValueError): A.clip_loss(torch.zeros(shape))
