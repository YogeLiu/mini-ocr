"""练习 11.1、11.2：独立前向参考和反例；全部 CPU，不依赖语料缓存。"""
from dataclasses import replace
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from miniocr.models.gpt import GPT, GPTConfig
from book.part2_lm.ch11_gpt.tour import (
    future_difference, last_logits, parameter_count, parameter_terms,
    reinitialize, total_params, visible_future,
)


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(previous)


def make_model(**kwargs):
    values = dict(vocab_size=19, block_size=16, n_layer=2, n_head=2, d_model=16)
    values.update(kwargs)
    return GPT(GPTConfig(**values)).eval()


def norm_reference(x, norm):
    return (x.float() / (x.float().square().mean(-1, keepdim=True)+norm.eps).sqrt()
            * norm.weight).to(x.dtype)


def attention_reference(attn, x, pos0=0):
    """显式分头、三角屏蔽、softmax；不用被测forward/attention/apply_rope。"""
    b, t, _ = x.shape
    q, k, v = [F.linear(x, proj.weight).reshape(b, t, heads, attn.dh).transpose(1, 2)
               for proj, heads in ((attn.q, attn.h), (attn.k, attn.hk), (attn.v, attn.hk))]
    if attn.q_norm is not None:
        q, k = norm_reference(q, attn.q_norm), norm_reference(k, attn.k_norm)
    angles = torch.tensor([[(pos0+p)*10000**(-2*i/attn.dh)
                            for i in range(attn.dh//2)] for p in range(t)])
    def rotate(z):
        a, b = z.chunk(2, -1)
        return torch.cat((a*angles.cos()-b*angles.sin(), a*angles.sin()+b*angles.cos()), -1)
    q, k = rotate(q), rotate(k)
    k, v = [z.repeat_interleave(attn.h//attn.hk, dim=1) for z in (k, v)]
    scores = q@k.transpose(-2, -1)/math.sqrt(attn.dh)
    scores = scores.masked_fill(torch.arange(t)[None, :] > torch.arange(t)[:, None], -torch.inf)
    out = scores.softmax(-1)@v
    return F.linear(out.transpose(1, 2).reshape(b, t, -1), attn.o.weight)


def forward_reference(model, ids, *, final_norm=True, residual=True):
    x = F.embedding(ids, model.tok.weight)
    for block in model.blocks:
        branch = attention_reference(block.attn, norm_reference(x, block.n1))
        x = x+branch if residual else branch
        z = norm_reference(x, block.n2)
        gate = F.linear(z, block.mlp.gate.weight)
        branch = F.linear(gate*torch.sigmoid(gate)*F.linear(z, block.mlp.up.weight), block.mlp.down.weight)
        x = x+branch if residual else branch
    if final_norm:
        x = norm_reference(x, model.norm)
    return F.linear(x, model.head.weight)


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("vocab,d,layers,ff", [(19, 16, 1, 0), (33, 32, 3, 96), (101, 64, 4, 0)])
def test_parameter_formula(vocab, d, layers, ff, tied):
    model = make_model(vocab_size=vocab, d_model=d, n_layer=layers, d_ff=ff, tie_weights=tied)
    expected = vocab*d + layers*(4*d*d+3*d*model.cfg.ff_dim()+2*d)+d + (0 if tied else vocab*d)
    assert parameter_count(model.cfg) == model.n_params() == expected
    assert model.n_params(non_embedding=True) == expected-vocab*d


def test_gpt2_exact_parameter_count():
    terms = parameter_terms(50257, 768, 12, 3072, gelu=True, bias=True,
                            layer_norm=True, learned_positions=1024)
    assert terms["每块注意力"] == 2362368
    assert terms["每块前馈"] == 4722432
    assert total_params(terms, 12) == 124439808


@pytest.mark.parametrize("tied", [True, False])
def test_weight_storage_and_counting(tied):
    model = make_model(tie_weights=tied)
    assert (model.head.weight is model.tok.weight) == tied
    assert (model.head.weight.data_ptr() == model.tok.weight.data_ptr()) == tied
    extra = sum(p.numel() for p in model.state_dict().values())-model.n_params()
    assert extra == (19*16 if tied else 0)


def test_shared_gradient_is_sum_of_two_paths():
    model = make_model()
    independent = GPT(replace(model.cfg, tie_weights=False)).eval()
    independent.load_state_dict(model.state_dict())
    x, y = torch.randint(19, (2, 2, 7))
    model(x, y)[1].backward()
    independent(x, y)[1].backward()
    a, b = independent.tok.weight.grad, independent.head.weight.grad
    assert a.norm() > 0 and b.norm() > 0
    torch.testing.assert_close(model.tok.weight.grad, a+b)


@pytest.mark.parametrize("tied", [True, False])
def test_zero_head_loss_equals_log_vocab(tied):
    model = make_model(tie_weights=tied)
    nn.init.zeros_(model.head.weight)
    logits, loss = model(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))
    assert torch.count_nonzero(logits) == 0
    assert loss.item() == pytest.approx(math.log(19), abs=1e-6)


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
@pytest.mark.parametrize("gqa,qknorm", [(False, False), (True, True)])
def test_attention_matches_independent_reference(impl, gqa, qknorm):
    model = make_model(attn_impl=impl, n_kv_head=1 if gqa else 0, qk_norm=qknorm)
    attn = model.blocks[0].attn
    x = torch.randn(2, 5, 16)
    # 非零偏移同时检查 RoPE 的切片，不只比较相同原点。
    actual = attn(x, model.cos[3:8], model.sin[3:8])
    torch.testing.assert_close(actual, attention_reference(attn, x, 3), atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_causality_and_fault_detection(impl):
    model = make_model(attn_impl=impl)
    ids = torch.tensor([[1, 4, 6, 9, 2, 3, 7, 8]])
    assert future_difference(model, ids) == 0
    with visible_future(model):
        assert future_difference(model, ids) > 1e-5
    assert future_difference(model, ids) == 0


@pytest.mark.parametrize("mask_kind", ["ignore", "mask", "both"])
def test_loss_mask_and_ignore_index(mask_kind):
    model = make_model()
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    targets = torch.tensor([[2, 3, 4], [5, 6, 7]])
    mask = torch.tensor([[True, False, True], [False, True, True]])
    if mask_kind in ("ignore", "both"):
        targets[0, 2] = -100
    supplied = mask if mask_kind in ("mask", "both") else None
    before = targets.clone()
    logits, loss = model(ids, targets, loss_mask=supplied)
    keep = targets != -100
    if supplied is not None:
        keep &= mask
    expected = -logits.float().log_softmax(-1)[keep].gather(1, targets[keep, None]).mean()
    torch.testing.assert_close(loss, expected)
    assert torch.equal(before, targets)


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_pos0_chunk_cache_matches_full_forward(impl):
    model = make_model(attn_impl=impl)
    ids = torch.tensor([[1, 2, 4, 7, 8, 3, 9]])
    full = model(ids)[0]
    caches, wrong = model.new_caches(), model.new_caches()
    pieces = []
    for start, end in ((0, 2), (2, 5), (5, 7)):
        pieces.append(model(ids[:, start:end], caches=caches, pos0=start)[0])
    torch.testing.assert_close(torch.cat(pieces, 1), full, atol=1e-6, rtol=1e-5)
    model(ids[:, :5], caches=wrong)
    bad = model(ids[:, 5:], caches=wrong, pos0=0)[0]
    assert (bad-full[:, 5:]).abs().max() > 1e-5


def test_gpt_reference_rejects_missing_norm_and_residual():
    model = make_model()
    # 非单位 gamma 可以抓住将 final norm 写死为无参数归一化的错误。
    with torch.no_grad():
        model.norm.weight.copy_(torch.linspace(.5, 1.5, 16))
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    actual, loss = model(ids)
    assert loss is None
    torch.testing.assert_close(actual, forward_reference(model, ids), atol=1e-6, rtol=1e-5)
    for kwargs in ({"final_norm": False}, {"residual": False}):
        assert (actual-forward_reference(model, ids, **kwargs)).abs().max() > .01


def test_next_token_loss_uses_supplied_shifted_targets_without_second_shift():
    model = make_model()
    sequence = torch.tensor([[1, 2, 4, 6, 8, 10], [3, 5, 7, 9, 11, 13]])
    x, y = sequence[:, :-1], sequence[:, 1:]
    logits, loss = model(x, y)
    expected_logits = forward_reference(model, x)
    expected = -expected_logits.log_softmax(-1).gather(-1, y[..., None]).mean()
    copied = -expected_logits.log_softmax(-1).gather(-1, x[..., None]).mean()
    torch.testing.assert_close(logits, expected_logits, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(loss, expected)
    assert abs(loss.item()-copied.item()) > .05
    # API并不自动猜测标签是否错位；错误数据会得到另一个明确的loss。
    torch.testing.assert_close(model(x, x)[1], copied)


def test_inputs_embeds_and_last_projection_match_full_forward():
    model = make_model()
    x = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]])
    lengths = torch.tensor([2, 4])
    logits = model(x)[0]
    torch.testing.assert_close(model(inputs_embeds=model.tok(x))[0], logits)
    torch.testing.assert_close(last_logits(model, x, lengths), logits[torch.arange(2), lengths-1])


def test_block_size_boundary_includes_pos0():
    model = make_model(block_size=4)
    assert model(torch.ones(1, 4, dtype=torch.long))[0].shape == (1, 4, 19)
    with pytest.raises(AssertionError, match="超过 block_size"):
        model(torch.ones(1, 5, dtype=torch.long))
    with pytest.raises(AssertionError, match="超过 block_size"):
        model(torch.ones(1, 2, dtype=torch.long), pos0=3)


def test_reinitialization_preserves_tie_and_scales_only_residual_outputs():
    model = make_model()
    plain = make_model()
    torch.manual_seed(91)
    reinitialize(model, .02, scaled=True)
    torch.manual_seed(91)
    reinitialize(plain, .02, scaled=False)
    for (name, a), (_, b) in zip(model.named_parameters(), plain.named_parameters()):
        factor = math.sqrt(2*model.cfg.n_layer) if name.endswith(("attn.o.weight", "mlp.down.weight")) else 1
        torch.testing.assert_close(a*factor, b)
    assert model.head.weight is model.tok.weight
