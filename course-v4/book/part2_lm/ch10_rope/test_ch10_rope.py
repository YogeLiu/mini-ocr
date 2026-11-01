"""练习10.1：独立标量旋转参考，避免仅检验形状或自洽性质。"""
import math
import pytest
import torch
from miniocr.models.gpt import GPT, GPTConfig, apply_rope, rope_cache, rotate_half
from book.part2_lm.ch10_rope.tour import (
    PositionGPT, adjacent_rope, cache_diagnostic, half_to_adjacent, value_shift_demo,
)


def manual_rotation(x, positions, base=10000):
    """Python 标量 sin/cos，不调用被测频率表、配对和旋转实现。"""
    out = torch.empty_like(x)
    d = x.shape[-1]
    for t, pos in enumerate(positions):
        for i in range(d//2):
            angle = pos * base ** (-2*i/d)
            c, s = math.cos(angle), math.sin(angle)
            a, b = x[..., t, i], x[..., t, i+d//2]
            out[..., t, i] = a*c-b*s
            out[..., t, i+d//2] = a*s+b*c
    return out


@pytest.mark.parametrize("base", [100., 10000., 1e6])
@pytest.mark.parametrize("shift", [1, 100, 1000])
def test_relative_position(base, shift):
    q, k = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(37))
    cos, sin = rope_cache(shift+20, 8, base)
    def at(x, pos):
        return apply_rope(x, cos[pos], sin[pos])
    for m, n in ((0, 0), (2, 5), (9, 1)):
        a = (at(q, m)*at(k, n)).sum(-1)
        b = (at(q, m+shift)*at(k, n+shift)).sum(-1)
        torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_norm_preserved(dtype):
    x = torch.randn(2, 3, 16, 8, generator=torch.Generator().manual_seed(4)).to(dtype)
    y = apply_rope(x, *rope_cache(16, 8))
    assert y.dtype == dtype
    tol = 0.03 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(y.float().norm(dim=-1), x.float().norm(dim=-1), atol=tol, rtol=tol)


def test_rotate_half_values_and_square():
    x = torch.tensor([[1., 2., 3., 4., 5., 6.], [-1., 0., 2., 3., -4., 5.]])
    want = torch.tensor([[-4., -5., -6., 1., 2., 3.], [-3., 4., -5., -1., 0., 2.]])
    torch.testing.assert_close(rotate_half(x), want, atol=0, rtol=0)
    torch.testing.assert_close(rotate_half(rotate_half(x)), -x, atol=0, rtol=0)


@pytest.mark.parametrize("dh", [2, 8, 32])
def test_matches_independent_2d_rotation(dh):
    x = torch.randn(2, 3, 7, dh, generator=torch.Generator().manual_seed(dh))
    y = apply_rope(x, *rope_cache(7, dh, 137.))
    torch.testing.assert_close(y, manual_rotation(x, range(7), 137.), atol=2e-6, rtol=2e-6)


def test_hand_calculated_two_dimensional_example():
    q, k = torch.tensor([[1., 0.]]), torch.tensor([[0.6, 0.8]])
    def rotate(x, position):
        angle = torch.tensor([[position*0.5]*2])
        return apply_rope(x, angle.cos(), angle.sin())[0]
    expected = 0.6*math.cos(1.5)-0.8*math.sin(1.5)
    for m, n in ((2, 5), (10, 13)):
        assert (rotate(q, m)@rotate(k, n)).item() == pytest.approx(expected, abs=1e-6)


def test_zero_is_exact_identity():
    x = torch.randn(2, 3, 1, 8)
    cos, sin = rope_cache(1, 8)
    assert torch.equal(cos, torch.ones(1, 8))
    assert torch.equal(sin, torch.zeros(1, 8))
    assert torch.equal(apply_rope(x, cos, sin), x)


def test_frequency_exponent_and_repeated_halves():
    cos, sin = rope_cache(4, 8)
    angles = torch.tensor([[t*f for f in (1., .1, .01, .001)*2] for t in range(4)])
    torch.testing.assert_close(cos, angles.cos(), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(sin, angles.sin(), atol=1e-7, rtol=1e-6)


def test_conventions_equivalent_after_permutation():
    x = torch.randn(2, 3, 11, 8, generator=torch.Generator().manual_seed(3))
    p = half_to_adjacent(8)
    half = apply_rope(x, *rope_cache(11, 8))
    adjacent = adjacent_rope(x[..., p], torch.arange(11))
    torch.testing.assert_close(half[..., p], adjacent, atol=0, rtol=0)
    torch.testing.assert_close(adjacent[..., p.argsort()], half, atol=0, rtol=0)


@pytest.mark.parametrize("mistake", ["wrong_exponent", "cos_only", "adjacent"])
def test_oracle_rejects_common_mistakes(mistake):
    x = torch.arange(1., 25.).reshape(3, 8)
    c, s = rope_cache(3, 8)
    if mistake == "wrong_exponent":
        angle = torch.arange(3.)[:, None]*10000**(-torch.arange(4.)/8)
        angle = torch.cat((angle, angle), -1)
        wrong = apply_rope(x, angle.cos(), angle.sin())
    elif mistake == "cos_only":
        wrong = x*c
    else:
        wrong = adjacent_rope(x, torch.arange(3))
    expected = manual_rotation(x, range(3))
    torch.testing.assert_close(apply_rope(x, c, s), expected, atol=2e-6, rtol=1e-6)
    assert (wrong-expected).abs().max() > .1


@pytest.mark.parametrize("dh", [0, 1, 7])
def test_invalid_head_dimension(dh):
    with pytest.raises(ValueError, match="正偶数"):
        rope_cache(3, dh)
    with pytest.raises(ValueError, match="正偶数"):
        rotate_half(torch.ones(2, dh))
    with pytest.raises(ValueError, match="正偶数"):
        apply_rope(torch.ones(2, dh), torch.ones(2, dh), torch.zeros(2, dh))


@pytest.mark.parametrize("mode", ["rope", "learned", "none"])
def test_last_logits_matches_full_and_context_boundary(mode):
    torch.manual_seed(0)
    model = PositionGPT(GPTConfig(vocab_size=17, block_size=8, n_layer=1, n_head=2, d_model=16), mode).eval()
    idx = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    lengths = torch.tensor([2, 4])
    torch.testing.assert_close(model.last_logits(idx, lengths), model(idx)[0][torch.arange(2), lengths-1])
    if mode == "learned":
        with pytest.raises(ValueError, match="无定义"):
            model.set_context(16)
    else:
        model.set_context(16)
        assert model(torch.ones(1, 16, dtype=torch.long))[0].shape == (1, 16, 17)


def test_default_tour_variant_matches_gpt_and_partial_keeps_other_pairs():
    cfg = dict(vocab_size=17, block_size=8, n_layer=1, n_head=2, d_model=16)
    torch.manual_seed(0)
    original = GPT(GPTConfig(**cfg)).eval()
    torch.manual_seed(0)
    variant = PositionGPT(GPTConfig(**cfg)).eval()
    idx = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(original(idx)[0], variant(idx)[0], atol=0, rtol=0)
    partial = PositionGPT(GPTConfig(**cfg), partial=0.5)
    x = torch.randn(8, 8)
    y = apply_rope(x, partial.cos, partial.sin)
    assert torch.equal(x[:, [2, 3, 6, 7]], y[:, [2, 3, 6, 7]])
    full = apply_rope(x, *rope_cache(8, 8))
    torch.testing.assert_close(y[:, [0, 1, 4, 5]], full[:, [0, 1, 4, 5]])


def test_value_rotation_breaks_shift_invariance():
    a, b, c, d = value_shift_demo(0)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    assert (c-d).abs().max() > .1


def test_cache_position_offset():
    full, good, bad = cache_diagnostic(0)
    torch.testing.assert_close(full, good, atol=1e-6, rtol=1e-5)
    assert (bad-good).abs().max() > 1e-5
