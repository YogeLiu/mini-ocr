import pytest
import torch
from torch.nn import functional as F
from ._impl import adapters as A


def test_targets_hand_utf8_padding():
    x,y=A.targets_for(['x','é'],3)
    assert x.tolist()==[[257,121,0,0],[257,196,170,0]]
    assert y.tolist()==[[121,258,-100,-100],[196,170,258,-100]]


@pytest.mark.parametrize('text',['','abcd'])
def test_targets_reject(text):
    with pytest.raises(ValueError): A.targets_for([text],3)


def test_ocr_training_mask_causality_and_roundtrip():
    torch.manual_seed(42)
    cfg=dict(width=8,vision_depth=1,decoder_depth=1,patch=16,max_length=3)
    m=A.get_ocr(cfg)
    x=torch.rand(2,1,32,160)
    inp,lab=A.targets_for(['x','ab'],3)
    logits,loss=m(x,inp,lab)
    assert logits.shape==(2,4,259)
    torch.testing.assert_close(loss,F.cross_entropy(logits.reshape(-1,259),lab.flatten(),ignore_index=-100))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    altered=inp.clone(); altered[:,2:]=30
    torch.testing.assert_close(m(x,altered)[0][:,:2],logits[:,:2])
    assert not torch.allclose(m(torch.zeros_like(x),inp)[0],logits)
    m2=A.get_ocr(cfg); m2.load_state_dict(m.state_dict())
    torch.testing.assert_close(m(x,inp)[0],m2(x,inp)[0])
    assert m.generate(x)==m2.generate(x)
