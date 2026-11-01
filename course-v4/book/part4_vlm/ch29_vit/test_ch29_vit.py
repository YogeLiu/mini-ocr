"""CPU 机制检查与离线 CLI 冒烟，无网络/权重/GPU 依赖。"""
import json
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from miniocr.nn.vit import patchify, patch_grid, patch_embed, interpolate_positions, VisionBlock, TinyViT
from book.part4_vlm.ch29_vit import tour


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_patch_order_multichannel_rectangular():
    x = torch.arange(2*4*6).reshape(1,2,4,6)
    result = patchify(x, 2)
    expected = torch.stack([x[0,:,r:r+2,c:c+2].flatten() for r in (0,2) for c in (0,2,4)])
    assert torch.equal(result[0], expected)
    with pytest.raises(ValueError):
        patchify(x, 4)
    assert patch_grid(5,7,4,pad=True) == (2,2)


@pytest.mark.parametrize('patch', [1,2,4])
def test_patch_forward_backward_and_gradcheck(patch):
    torch.manual_seed(1)
    values = [torch.randn(1,2,4,8,dtype=torch.double),
              torch.randn(3,2,patch,patch,dtype=torch.double), torch.randn(3,dtype=torch.double)]
    inputs = [[v.clone().requires_grad_() for v in values] for _ in range(2)]
    a, b = [patch_embed(*xs, method=m) for xs,m in zip(inputs,('reshape','conv'))]
    torch.testing.assert_close(a,b)
    weight = torch.randn_like(a)
    for result in (a,b):
        (result*weight).sum().backward()
    for left,right in zip(*inputs):
        torch.testing.assert_close(left.grad,right.grad)
    assert torch.autograd.gradcheck(lambda *xs:patch_embed(*xs), tuple(inputs[0]), fast_mode=True)


def test_position_interpolation_rectangular_cls_constant_and_gradient():
    pos = torch.ones(1,7,3,requires_grad=True)
    with torch.no_grad():
        pos[:,0] = 9
    out = interpolate_positions(pos,(2,3),(5,7))
    assert out.shape == (1,36,3)
    torch.testing.assert_close(out[:,0],pos[:,0])
    torch.testing.assert_close(out[:,1:],torch.ones_like(out[:,1:]))
    out.sum().backward()
    assert torch.isfinite(pos.grad).all() and (pos.grad != 0).all()
    assert interpolate_positions(pos,(2,3),(2,3)) is pos
    assert interpolate_positions(pos[:,1:],(2,3),(1,4),has_cls=False).shape == (1,4,3)
    with pytest.raises(ValueError):
        interpolate_positions(pos,(3,3),(4,4))


def test_positions_preserve_spatial_axes():
    table = torch.tensor([[[99.],[0.],[1.],[2.],[10.],[11.],[12.]]])
    out = interpolate_positions(table,(2,3),(4,6))[0,1:,0].reshape(4,6)
    assert (out[:,1:] > out[:,:-1]).all()
    assert (out[1:] > out[:-1]).all()


def test_attention_matches_independent_sdpa():
    torch.manual_seed(0)
    block = VisionBlock(12,3).double()
    x = torch.randn(2,5,12,dtype=torch.double,requires_grad=True)
    z = block.norm1(x)
    q,k,v = block.qkv(z).chunk(3,dim=-1)
    q,k,v = [t.reshape(2,5,3,4).transpose(1,2) for t in (q,k,v)]
    y = F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(2,5,12)
    ref = x + block.proj(y)
    ref = ref + block.mlp(block.norm2(ref))
    torch.testing.assert_close(block(x),ref)
    # 双向 block 不依赖顺序；位置表在 block 外引入。
    perm = torch.tensor([4,2,0,1,3])
    torch.testing.assert_close(block(x[:,perm]),block(x)[:,perm])


def test_tinyvit_variable_resolution_and_roundtrip(tmp_path):
    torch.manual_seed(1)
    model = TinyViT(width=16,depth=1)
    x = torch.randn(3,1,32,48)
    output = model(x)
    assert output.shape == (3,24)
    F.cross_entropy(output,torch.tensor([1,2,3])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    torch.save(model.state_dict(),tmp_path/'model.pt')
    fresh = TinyViT(width=16,depth=1)
    fresh.load_state_dict(torch.load(tmp_path/'model.pt',weights_only=True))
    torch.testing.assert_close(output,fresh(x))


def test_subset_groups_nested_and_shift_no_wrap():
    groups = np.repeat(np.arange(16),2)
    labels = groups//4
    train = np.arange(32)
    low = tour.train_subset(train,labels,groups,.5)
    full = tour.train_subset(train,labels,groups,1.)
    assert set(low) <= set(full) and len(low)==16
    assert all(sum(groups[low]==g)==2 for g in np.unique(groups[low]))
    x = torch.zeros(1,1,8,8); x[0,0,-1,-1]=1
    assert tour.shifted(x,2).sum()==0


def test_patch_occupancy_padding():
    ink = np.zeros((5,7),np.float32); ink[-1,-1]=1
    stats = tour.patch_statistics(ink,4)
    assert stats['tokens']==4 and stats['occupied']==1


def test_cli_parts_and_final_test_protocol(tmp_path,monkeypatch):
    def fake_data(sources,per_source,seed):
        rng = np.random.default_rng(seed)
        return dict(images=rng.random((48,32,32),dtype=np.float32),
                    labels=np.arange(48)%24,groups=np.arange(48))
    monkeypatch.setattr(tour,'make_dataset',fake_data)
    monkeypatch.setattr(tour,'load_formula_images',lambda n:[])
    tour.main(['1','2','3','4','5','--steps','2','--eval-every','1','--batch-size','4',
               '--width','8','--depth','1','--cnn-width','2','--fractions','1','--out',str(tmp_path)])
    run = json.loads((tmp_path/'run.json').read_text())
    assert run['results']['4']['status'].startswith('skipped')
    assert run['results']['5']['final_test']==[]
    assert not (tmp_path/'final_test.json').exists()
    # 冻结配置时才最终评测，并验证不能复用目录覆盖。
    final = tmp_path/'final'
    tour.main(['5','--steps','1','--batch-size','4','--width','8','--depth','1',
               '--cnn-width','2','--fractions','1','--final-test','--out',str(final)])
    assert len(json.loads((final/'final_test.json').read_text()))==2
    with pytest.raises(SystemExit):
        tour.main(['5','--out',str(final)])
