"""第 27 章的自动检查。"""
import functools
import math

import numpy as np
import torch

from miniocr.data.corpus import text_vocab
from miniocr.eval.calib import auroc, ece, reliability
from miniocr.models.gpt import GPT, GPTConfig
from miniocr.rl.logprobs import (labels_to_mask, mask_from_lengths, mask_to_labels, response_mask, seq_logprob,
                                 token_entropy, token_logprobs)
from miniocr.train.sft import chat_roles, sft_loss, sft_targets


@functools.cache
def vocab():
    return text_vocab()


def tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=len(vocab()), block_size=64, n_layer=2, n_head=2, d_model=32)).eval()


def test_token_logprobs_by_hand():
    logits = torch.log(torch.tensor([[[0.5, 0.25, 0.25], [0.1, 0.8, 0.1], [1 / 3, 1 / 3, 1 / 3]]]))
    ids = torch.tensor([[2, 0, 1]])                      # 位置 0 预测 ids[1] = 0，位置 1 预测 ids[2] = 1
    lp = token_logprobs(logits, ids)
    assert lp.shape == (1, 2)
    assert torch.allclose(lp, torch.log(torch.tensor([[0.5, 0.8]])), atol=1e-6)


def test_logprobs_match_model_loss_and_padding():
    lm = tiny()
    ids = torch.randint(0, len(vocab()), (1, 12), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        logits, loss = lm(ids, targets=torch.cat([ids[:, 1:], torch.full((1, 1), -100)], 1))
        lp = token_logprobs(logits, ids)
        assert abs(-lp.mean().item() - loss.item()) < 1e-5            # 语言模型的损失 = −平均对数概率
        padded = torch.cat([ids, torch.zeros(1, 4, dtype=torch.long)], 1)
        lp2 = token_logprobs(lm(padded)[0], padded)
        assert torch.allclose(lp, lp2[:, :11], atol=1e-5)             # 右侧补齐不改变前面的对数概率


def test_masks_and_conventions():
    V = vocab()
    msgs = [{"role": "user", "content": "<image>Hi"}, {"role": "assistant", "content": "ok"}]
    ids, roles = chat_roles(V, msgs, 2)
    m = response_mask(roles)
    tg, _ = sft_targets(ids, roles)
    t = torch.tensor([tg])
    assert torch.equal(labels_to_mask(t)[0], m)
    assert torch.equal(mask_to_labels(torch.tensor([ids]), m[None]), t)
    assert m.sum().item() == 3                                        # 'o' 'k' <eos>
    assert mask_from_lengths([3], [7], 10).tolist() == [[0, 0, 1, 1, 1, 1, 0, 0, 0]]
    lm = tiny()
    with torch.no_grad():
        logits = lm(torch.tensor([ids]))[0]
        lp = token_logprobs(logits, torch.tensor([ids]))
        assert abs(-seq_logprob(lp, m[None], "mean").item() - sft_loss(logits, t).item()) < 1e-5   # SFT 损失 = 回答的 −平均对数概率
        assert abs(seq_logprob(lp, m[None]).item() - (lp[0] * m).sum().item()) < 1e-6


def test_entropy_bounds():
    lm = tiny()
    ids = torch.randint(0, len(vocab()), (2, 10), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        h = token_entropy(lm(ids)[0])
    assert h.shape == (2, 9) and (h >= 0).all() and (h <= math.log(len(vocab())) + 1e-5).all()
    one_hot = torch.full((1, 2, 5), -1e9)
    one_hot[0, 0, 3] = 0.0
    assert token_entropy(one_hot)[0, 0].item() < 1e-6                 # 完全确定的分布，熵为 0
    assert abs(token_entropy(torch.zeros(1, 2, 5))[0, 0].item() - math.log(5)) < 1e-6


def test_ece_and_auroc():
    assert ece([0.9] * 10, [1] * 9 + [0]) < 1e-9                     # 说 90%，对了 90%
    assert abs(ece([1.0] * 4, [1, 0, 1, 0]) - 0.5) < 1e-9
    r = reliability([0.05, 0.15, 0.95, 1.0], [0, 1, 1, 1])
    assert [b[4] for b in r] == [1, 1, 2] and r[-1][0] == 0.9           # 1.0 落在最后一段
    assert auroc([0.9, 0.8, 0.3, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5
    rng = np.random.default_rng(0)
    s, y = rng.normal(size=200), rng.integers(0, 2, 200)
    brute = np.mean([(a > b) + 0.5 * (a == b) for a, ya in zip(s, y) if ya for b, yb in zip(s, y) if not yb])
    assert abs(auroc(s, y) - brute) < 1e-9
    assert math.isnan(auroc([0.1, 0.2], [1, 1]))
