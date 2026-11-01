import math
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from ._impl import adapters as A


class ToyTokenizer:
    """人工词表：提示=[1,2]，a=[3]，bb=[4,5]；不依赖下载。"""
    eos_token_id = 9
    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{'role': 'system', 'content': 'Correct OCR errors in the following LaTeX formula.'},
                            {'role': 'user', 'content': 'noisy'}]
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return 'P:'
    def encode(self, text, **kwargs):
        assert kwargs == {'add_special_tokens': False}
        return {'P:': [1, 2], 'P:a': [1, 2, 3], 'P:bb': [1, 2, 4, 5],
                'P:merge': [1, 7], 'P:empty': [1, 2]}[text].copy()


def test_encoding_hand_values_and_truncation():
    e = A.run_encode_sft(ToyTokenizer(), 'noisy', 'a', max_length=4)
    assert e.input_ids == (1, 2, 3, 9)
    assert e.labels == (-100, -100, 3, 9) and e.prompt_length == 2
    assert A.run_encode_sft(ToyTokenizer(), 'noisy', 'a', max_length=3) is None


@pytest.mark.parametrize('target,limit,eos', [(' ', 8, 9), ('merge', 8, 9), ('empty', 8, 9), ('a', 1, 9), ('a', 8, None)])
def test_encoding_invalid(target, limit, eos):
    tok = ToyTokenizer(); tok.eos_token_id = eos
    with pytest.raises(ValueError):
        A.run_encode_sft(tok, 'noisy', target, max_length=limit)


def ex(ids=(1, 2, 9), labels=(-100, 2, 9)):
    return SimpleNamespace(input_ids=ids, labels=labels)


def test_collate_hand_values_eos_and_immutability():
    examples = [ex(), ex((1, 9), (-100, 9))]
    b = A.run_collate(iter(examples), 9, multiple=4)
    expected = {'input_ids': [[1,2,9,9], [1,9,9,9]],
                'labels': [[-100,2,9,-100], [-100,9,-100,-100]],
                'attention_mask': [[1,1,1,0], [1,1,0,0]]}
    for key, value in expected.items():
        assert b[key].dtype == torch.long and b[key].device.type == 'cpu'
        assert b[key].tolist() == value
    assert examples[0].input_ids == (1,2,9)


@pytest.mark.parametrize('examples,pad,multiple', [([],9,1),([ex()],None,1),([ex()],9,0),
    ([ex((1,),(-100,))],9,1),([ex((1,2),(-100,))],9,1),([ex((1,2),(2,-100))],9,1)])
def test_collate_invalid(examples,pad,multiple):
    with pytest.raises(ValueError): A.run_collate(examples,pad,multiple=multiple)


def test_loss_hand_value_and_gradient():
    x = torch.zeros(1,4,2, requires_grad=True)
    y = torch.tensor([[-100,-100,1,0]])
    loss = A.run_causal_loss(x,y)
    assert loss.item() == pytest.approx(math.log(2))
    assert A.run_causal_loss(x,y,reduction='sum').item() == pytest.approx(2*math.log(2))
    loss.backward()
    torch.testing.assert_close(x.grad, torch.tensor([[[0.,0.],[.25,-.25],[-.25,.25],[0.,0.]]]))


def test_loss_pytorch_gradient_and_token_weighting():
    torch.manual_seed(19)
    x = (torch.randn(2,5,4)*100).requires_grad_()
    y = torch.tensor([[-100,1,-100,-100,-100],[-100,2,3,1,0]])
    ref = F.cross_entropy(x[:,:-1].reshape(-1,4), y[:,1:].reshape(-1), ignore_index=-100)
    got = A.run_causal_loss(x,y)
    torch.testing.assert_close(got,ref)
    torch.testing.assert_close(torch.autograd.grad(got,x,retain_graph=True)[0],torch.autograd.grad(ref,x)[0])
    weighted = sum(A.run_causal_loss(x[i:i+1],y[i:i+1],reduction='sum') for i in range(2))/5
    torch.testing.assert_close(weighted,got)


@pytest.mark.parametrize('shape,labels,reduction', [((1,3,2),[[-100,-100,-100]],'mean'),
    ((1,3,2),[[-100,2,0]],'mean'),((1,3,2),[[-100,-2,0]],'mean'),
    ((1,3,2),[[-100,1]],'mean'),((1,3),[[-100,1,0]],'mean'),((1,3,2),[[-100,1,0]],'none')])
def test_loss_invalid(shape,labels,reduction):
    with pytest.raises(ValueError): A.run_causal_loss(torch.zeros(shape),torch.tensor(labels),reduction=reduction)
