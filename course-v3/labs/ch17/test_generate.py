"""第 17 章的自动检查。"""
import math

import torch

from miniocr.generate import beam_search, generate, process_logits
from miniocr.models.gpt import GPT, GPTConfig


def small():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=30, block_size=64, n_layer=2, n_head=2, d_model=32)).eval()


def test_temperature_topk_topp():
    lg = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    p = process_logits(lg, temperature=2.0).softmax(-1)
    assert torch.allclose(p, (lg / 2).softmax(-1)) and p[0, 0] < 0.5                 # 温度 > 1 更平
    p = process_logits(lg, top_k=2).softmax(-1)
    assert torch.allclose(p, torch.tensor([[0.625, 0.375, 0.0, 0.0]]), atol=1e-6)
    p = process_logits(lg, top_p=0.7).softmax(-1)                                     # 0.5 不够，0.5+0.3 ≥ 0.7
    assert torch.allclose(p, torch.tensor([[0.625, 0.375, 0.0, 0.0]]), atol=1e-6)
    p = process_logits(lg, top_p=0.4).softmax(-1)                                     # 第一个就够了
    assert torch.allclose(p, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    lg2 = torch.tensor([[2.0, -1.0, 0.5]])
    out = process_logits(lg2, repetition_penalty=2.0, prev=torch.tensor([[0, 1]]))
    assert torch.allclose(out, torch.tensor([[1.0, -2.0, 0.5]]))


def test_generate_cache_matches_no_cache():
    m = small()
    idx = torch.randint(0, 30, (3, 5))
    a = generate(m, idx, 12, greedy=True, use_cache=True)
    b = generate(m, idx, 12, greedy=True, use_cache=False)
    assert a.shape == (3, 17) and torch.equal(a, b)
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    assert torch.equal(generate(m, idx, 8, top_k=5, generator=g1), generate(m, idx, 8, top_k=5, generator=g2, use_cache=False))


def test_generate_eos_and_embeds():
    m = small()
    idx = torch.randint(0, 30, (2, 4))
    ref = generate(m, idx, 6, greedy=True)
    out = generate(m, idx, 6, greedy=True, inputs_embeds=m.tok(idx))
    assert torch.equal(out, ref)
    eos = int(ref[0, 5])
    out = generate(m, idx, 6, greedy=True, eos=eos)
    assert (out[0, 6:] == eos).all()


def test_beam_search_at_least_as_good_as_greedy():
    m = small()
    idx = torch.randint(0, 30, (1, 4))

    def logp(seq):
        lp = torch.log_softmax(m(seq[:, :-1])[0], -1)
        return lp[0, torch.arange(3, seq.shape[1] - 1), seq[0, 4:]].sum().item()

    g = generate(m, idx, 6, greedy=True)
    b = beam_search(m, idx, 6, beam=4)
    assert b.shape == g.shape and logp(b) >= logp(g) - 1e-5
    assert torch.equal(beam_search(m, idx, 6, beam=1), g)
