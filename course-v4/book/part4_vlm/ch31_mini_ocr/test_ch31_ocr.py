"""CPU checks for multimodal masking, causality, cached decoding and attribution."""
import pytest
import torch
import torch.nn.functional as F

from miniocr.models.formula_ocr import FormulaOCR, targets_for, token_saliency, BOS, EOS


@pytest.fixture
def model():
    torch.manual_seed(31)
    torch.set_num_threads(2)
    return FormulaOCR(width=16,vision_depth=1,decoder_depth=1,max_length=16).eval()


def test_utf8_shift_padding_and_rejection():
    inputs,labels = targets_for(['x','α'],3)
    assert inputs[0].tolist() == [BOS,ord('x')+1,0,0]
    assert labels[0].tolist() == [ord('x')+1,EOS,-100,-100]
    assert bytes((inputs[1,1:3]-1).tolist()).decode() == 'α'
    with pytest.raises(ValueError): targets_for(['abcd'],3)
    with pytest.raises(ValueError): targets_for([''],3)


def test_visual_and_padding_loss_are_excluded(model):
    x = torch.rand(2,1,32,160)
    inputs,labels = targets_for(['x','y+1'],16)
    logits,loss = model(x,inputs,labels)
    manual = F.cross_entropy(logits[labels!=-100],labels[labels!=-100])
    torch.testing.assert_close(loss,manual)
    # Suffix padding embeddings cannot influence valid causal positions.
    changed = inputs.clone(); changed[labels==-100] = 100
    altered,other = model(x,changed,labels)
    torch.testing.assert_close(other,loss)
    torch.testing.assert_close(altered[labels!=-100],logits[labels!=-100])
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.connector.net[-1].weight.grad.abs().sum() > 0
    assert model.patch_weight.grad.abs().sum() > 0


def test_causality_and_image_dependence(model):
    x = torch.rand(1,1,32,160)
    inputs,labels = targets_for(['abc'],16)
    a,_ = model(x,inputs)
    other = inputs.clone(); other[:,3] = ord('z')+1
    b,_ = model(x,other)
    torch.testing.assert_close(a[:,:3],b[:,:3])
    c,_ = model(torch.zeros_like(x),inputs)
    assert not torch.allclose(a,c)


def test_cached_generation_matches_full_prefix(model):
    image = torch.rand(2,1,32,160)
    actual = model.generate(image,max_new_tokens=6)
    tokens = torch.full((2,1),BOS,dtype=torch.long)
    rows = [[] for _ in range(2)]
    done = [False,False]
    with torch.no_grad():
        for _ in range(6):
            logits,_ = model(image,tokens)
            scores = logits[:,-1].clone(); scores[:,[0,BOS]] = -float('inf')
            new = scores.argmax(-1)
            for i,t in enumerate(new.tolist()):
                if t == EOS: done[i] = True
                if not done[i]: rows[i].append(t-1)
            tokens = torch.cat((tokens,new[:,None]),1)
    assert actual == [bytes(row).decode('utf-8',errors='replace') for row in rows]


def test_saliency_finite_nonzero_without_parameter_grad(model):
    x = torch.rand(1,1,32,160)
    inputs,labels = targets_for(['x'],16)
    sal = token_saliency(model,x,inputs,0,int(labels[0,0]))
    assert sal.shape == (32,160) and torch.isfinite(sal).all() and sal.sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_learns_two_image_conditioned_examples(model):
    x = torch.stack((torch.zeros(1,32,160),torch.ones(1,32,160)))
    inputs,labels = targets_for(['x','y'],16)
    opt = torch.optim.AdamW(model.parameters(),lr=.01)
    first = float(model(x,inputs,labels)[1].detach())
    for _ in range(60):
        opt.zero_grad(); loss = model(x,inputs,labels)[1]; loss.backward(); opt.step()
    assert float(loss.detach()) < first*.15
    assert model.generate(x) == ['x','y']


def test_test_gate_does_not_load_data(tmp_path):
    from argparse import Namespace
    from book.part4_vlm.ch31_mini_ocr.tour import part5
    assert 'skipped' in part5(Namespace(final_test=False))['status']
