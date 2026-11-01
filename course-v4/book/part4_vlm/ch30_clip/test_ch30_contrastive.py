"""Fast CPU checks of the mechanism, mask and held-out experiment protocol."""
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from miniocr.data.contrastive_pairs import encode_bytes, image_tensor, prepare_pairs, open_pairs
from miniocr.eval.retrieval import retrieval_metrics
from miniocr.nn.contrastive import DualEncoder, clip_loss, siglip_loss
from book.part4_vlm.ch30_clip.tour import main


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('kind', ['clip','siglip'])
def test_loss_forward_backward_reference_and_gradcheck(kind):
    torch.manual_seed(30)
    scores = torch.randn(4,4,dtype=torch.double,requires_grad=True)
    bias = torch.tensor(-1.5,dtype=torch.double,requires_grad=True)
    if kind == 'clip':
        actual = clip_loss(scores)
        ref = (F.cross_entropy(scores,torch.arange(4))+F.cross_entropy(scores.T,torch.arange(4)))/2
        params = (scores,)
        assert torch.autograd.gradcheck(clip_loss,params)
    else:
        actual = siglip_loss(scores,bias)
        ref = F.binary_cross_entropy_with_logits(scores+bias,torch.eye(4,dtype=torch.double),reduction='sum')/4
        params = (scores,bias)
        assert torch.autograd.gradcheck(siglip_loss,params)
    torch.testing.assert_close(actual,ref)
    for a,b in zip(torch.autograd.grad(actual,params),torch.autograd.grad(ref,params)):
        torch.testing.assert_close(a,b)


def test_normalization_shift_permutation_and_extremes():
    b = 8
    z = torch.zeros(b,b)
    assert clip_loss(z).item() == pytest.approx(math.log(b))
    assert siglip_loss(z).item() == pytest.approx(b*math.log(2))
    s = torch.randn(b,b)
    torch.testing.assert_close(clip_loss(s+10),clip_loss(s))
    assert not torch.isclose(siglip_loss(s+10),siglip_loss(s))
    permutation = torch.randperm(b)
    for fn in (clip_loss,siglip_loss):
        torch.testing.assert_close(fn(s[permutation][:,permutation]),fn(s))
        extreme = (torch.eye(b)*2-1).mul(10000).requires_grad_()
        fn(extreme).backward()
        assert torch.isfinite(extreme.grad).all()
        with pytest.raises(ValueError): fn(torch.zeros(3,4))
        with pytest.raises(ValueError): fn(torch.zeros(0,0))


def test_bias_balances_zero_score_positive_negative_counts():
    b = 16
    bias = torch.tensor(-math.log(b-1),requires_grad=True)
    siglip_loss(torch.zeros(b,b),bias).backward()
    assert abs(bias.grad.item()) < 1e-6


def test_temperature_and_wrong_pairing():
    aligned = torch.eye(8)
    assert clip_loss(aligned/.03) < clip_loss(aligned/.2)
    assert clip_loss(aligned) < clip_loss(aligned.roll(1,1))


def test_padding_does_not_change_embeddings_or_receive_gradient():
    torch.manual_seed(30)
    net = DualEncoder(16,32)
    net.eval()
    short = encode_bytes(['x+1','x^2'],8)
    long = F.pad(short,(0,12))
    torch.testing.assert_close(net.encode_text(short),net.encode_text(long),atol=1e-6,rtol=1e-5)
    scores = net(torch.rand(2,1,32,160),long)
    siglip_loss(scores,net.bias).backward()
    assert net.embedding.weight.grad[0].count_nonzero() == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    with pytest.raises(ValueError): net.encode_text(torch.zeros(1,8,dtype=torch.long))


def test_retrieval_direction_gallery_and_ties():
    perfect = retrieval_metrics(torch.eye(12))
    assert perfect['i2t_r1'] == perfect['t2i_r1'] == perfect['i2t_r10'] == 1
    shifted = retrieval_metrics(torch.eye(12).roll(1,1))
    assert shifted['i2t_r1'] == shifted['t2i_r1'] == 0
    tied = retrieval_metrics(torch.zeros(12,12))
    assert tied['i2t_r1'] == pytest.approx(1/12)
    assert tied['i2t_r5'] == pytest.approx(5/12)
    small = retrieval_metrics(torch.eye(3))
    assert small['t2i_r10'] == 1


def test_data_has_disjoint_keys_and_no_truncation(tmp_path):
    data = prepare_pairs(tmp_path,{'train':16,'val':8},max_length=32)
    assert not {r['key_sha256'] for r in data['train']['manifest']} & {r['key_sha256'] for r in data['val']['manifest']}
    assert data['train']['images'].shape == (16,1,32,160)
    with pytest.raises(ValueError): prepare_pairs(tmp_path,{'train':17},max_length=32)
    with pytest.raises(ValueError): prepare_pairs(tmp_path,{'test':4})
    with pytest.raises(ValueError): open_pairs({'test':1})
    with pytest.raises(ValueError): encode_bytes(['abcdef'],3)
    with pytest.raises(ValueError): image_tensor(np.zeros((3,3)))


def test_open_pairs_respects_shared_split_and_deduplicates(tmp_path,monkeypatch):
    import io
    from PIL import Image
    pq = pytest.importorskip('pyarrow.parquet')
    import pyarrow as pa
    import miniocr.data.contrastive_pairs as pairs
    buffer = io.BytesIO()
    Image.fromarray(np.array([[255,0],[255,255]],dtype='uint8')).save(buffer,format='PNG')
    payload = buffer.getvalue()
    rows = [{'latex_formula':text,'image':{'bytes':image}} for text,image in
            [('x^2',payload),('x^{2}',payload),('z+2',b'invalid-test-image'),
             ('y+1',payload),('a+3',payload)]]
    path = tmp_path/'data.parquet'
    pq.write_table(pa.Table.from_pylist(rows),path)
    monkeypatch.setattr(pairs,'_parquet_paths',lambda:[path])
    calls = []
    def split(**kwargs):
        calls.append(kwargs)
        return dict(train=['x^2','x^{2}','a+3'],val=['y+1'],test=['z+2'])
    monkeypatch.setattr(pairs,'standard_split',split)
    data = open_pairs({'train':2,'val':1})
    assert calls == [dict(n_train=None)]
    assert [r['formula'] for r in data['train']['manifest']] == ['x^2','a+3']
    assert len(data['val']['manifest']) == 1


def test_complete_cpu_pipeline_and_once_only_test(tmp_path):
    run, data = tmp_path/'run',tmp_path/'data'
    common = ['--run-dir',str(run),'--data-dir',str(data),'--train-n','16','--val-n','4',
              '--test-n','4','--width','8','--max-length','32','--batches','4',
              '--steps','2','--eval-every','2','--n','4']
    main(['--out',str(run),*common,'--final-test'])
    frozen = json.loads((run/'frozen.json').read_text())
    final = json.loads((run/'final_test.json').read_text())
    assert final['test_evaluated_once'] and final['winner'] == frozen['winner']
    assert final['metrics']['gallery_n'] == 4
    with pytest.raises(FileExistsError,match='test already started'):
        main(['5','--out',str(tmp_path/'retry'),*common,'--final-test'])
