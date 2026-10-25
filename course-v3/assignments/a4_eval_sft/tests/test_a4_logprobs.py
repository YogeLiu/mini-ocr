"""A4 第三部分：每个 token 的对数概率（错位）、熵、求和与平均、两种 mask 约定。"""
import torch
import torch.nn.functional as F

from ._impl import adapters as A


def test_token_logprobs_offset():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 11)
    ids = torch.randint(0, 11, (2, 6))
    lp = A.run_token_logprobs(logits, ids)
    assert lp.shape == (2, 5)
    for b in range(2):
        for t in range(5):
            assert torch.isclose(lp[b, t], F.log_softmax(logits[b, t], -1)[ids[b, t + 1]], atol=1e-6)


def test_token_logprobs_bf16_uses_float32():
    torch.manual_seed(0)
    logits = (torch.randn(1, 4, 50) * 5).bfloat16()
    ids = torch.randint(0, 50, (1, 4))
    lp = A.run_token_logprobs(logits, ids)
    ref = F.log_softmax(logits.float()[:, :-1], -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    assert lp.dtype == torch.float32 and torch.allclose(lp, ref, atol=1e-5)


def test_entropy():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7) * 3
    h = A.run_token_entropy(logits)
    assert h.shape == (2, 4)
    assert torch.allclose(h, torch.distributions.Categorical(logits=logits[:, :-1]).entropy(), atol=1e-5)
    big = torch.full((1, 2, 7), -1e4)
    big[..., 3] = 1e4
    assert torch.allclose(A.run_token_entropy(big), torch.zeros(1, 1), atol=1e-6)        # 完全确定，不能出 nan


def test_seq_logprob():
    lp = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -0.5, -9.0]])
    mask = torch.tensor([[0.0, 1, 1], [1, 1, 0]])
    assert A.run_seq_logprob(lp, mask, "sum").tolist() == [-5.0, -1.0]
    assert A.run_seq_logprob(lp, mask, "mean").tolist() == [-2.5, -0.5]


def test_mask_conventions():
    ids = torch.tensor([[1, 5, 6, 7, 2, 0], [1, 8, 9, 2, 0, 0]])
    mask = A.run_mask_from_lengths([3, 2], [5, 4], 6)
    assert mask.tolist() == [[0, 0, 1, 1, 0], [0, 1, 1, 0, 0]]
    tg = A.run_mask_to_labels(ids, mask)
    assert tg.tolist() == [[-100, -100, 7, 2, -100, -100], [-100, 9, 2, -100, -100, -100]]
    assert torch.equal(A.run_labels_to_mask(tg), mask)
