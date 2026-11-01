"""附录 B：输出对拍标准注意力；访问次数和 FLOPs 用闭式手算核对。"""
import math

import pytest
import torch
import torch.nn.functional as F

from . import tour


@pytest.mark.parametrize("Br,Bc", [(1, 1), (4, 4), (8, 16), (5, 7), (32, 32)])
def test_matches_standard(Br, Bc):
    q, k, v = tour.random_inputs(32, 8)
    reference = F.scaled_dot_product_attention(q, k, v)
    assert (tour.standard_attention(q, k, v)-reference).abs().max() < 1e-10
    assert (tour.flash_attention(q, k, v, Br, Bc)-reference).abs().max() < 1e-10


def test_matches_standard_with_causal_mask():
    q, k, v = tour.random_inputs(32, 8)
    mask = tour.causal(32)
    reference = tour.standard_attention(q, k, v, mask)
    assert (tour.flash_attention(q, k, v, 4, 4, mask)-reference).abs().max() < 1e-10


def test_online_softmax_hand_example():
    # 分数 [1,2,3,0]，两块。第二块把最大值从 2 改成 3，旧的 l 乘 e^(2−3)。
    s = torch.tensor([[1., 2., 3., 0.]], dtype=torch.float64)
    v = torch.eye(4, dtype=torch.float64)
    m = torch.full((1,), -math.inf, dtype=torch.float64)
    l = torch.zeros(1, dtype=torch.float64)
    o = torch.zeros(1, 4, dtype=torch.float64)
    m, l, o = tour.online_softmax_update(s[:, :2], m, l, o, v[:2])
    assert m.item() == 2 and l.item() == pytest.approx(1+math.exp(-1))
    m, l, o = tour.online_softmax_update(s[:, 2:], m, l, o, v[2:])
    assert m.item() == 3 and l.item() == pytest.approx(1+math.exp(-1)+math.exp(-2)+math.exp(-3))
    assert (o/l.unsqueeze(-1))[0].tolist() == pytest.approx(s.softmax(-1)[0].tolist())


def test_hand_counts_n16():
    a, b = tour.count_pair(16, 4, 4, 4)
    # 标准：4·16·4 + 4·256 = 1280；分块：2·16·4 + 2·256·4/4 = 640。
    assert a.accesses == tour.standard_accesses(16, 4) == 1280
    assert b.accesses == tour.flash_accesses(16, 4, 4) == 640
    # FLOPs：标准 4096+1536；分块 4096+1280+13·64+64。
    assert a.flops == tour.standard_flops(16, 4) == 5632
    assert b.flops == tour.flash_flops(16, 4, 4) == 6272
    # 快速存储：2·4·4 + 2·4·4 + 2·16 + 8 = 104。
    assert b.peak_fast == tour.fast_needed(4, 4, 4) == 104


@pytest.mark.parametrize("Br,Bc", [(2, 4), (4, 8), (8, 2), (32, 32)])
def test_counter_matches_closed_form(Br, Bc):
    a, b = tour.count_pair(32, 8, Br, Bc)
    assert b.accesses == tour.flash_accesses(32, 8, Br)
    assert b.flops == tour.flash_flops(32, 8, Bc)
    assert a.flops == tour.standard_flops(32, 8)


def test_saves_accesses_not_flops():
    a, b = tour.count_pair(64, 8, 8, 8)
    assert b.accesses*2 == a.accesses
    assert b.flops > a.flops
    # 主项 4N²D 不变：大 N 大 Bc 时 FLOPs 比趋近 1，不会降到 1 以下很多。
    assert tour.flash_flops(1024, 64, 32)/tour.standard_flops(1024, 64) == pytest.approx(1, abs=.02)


def test_too_small_block_is_worse():
    a, b = tour.count_pair(64, 8, 1, 1)
    assert b.accesses > a.accesses


def test_choose_blocks_respects_capacity():
    for M in (256, 1024, 4096):
        Br, Bc = tour.choose_blocks(64, 8, M)
        assert tour.fast_needed(Br, Bc, 8) <= M
    with pytest.raises(ValueError):
        tour.choose_blocks(64, 8, 4)


def test_standard_fits_when_memory_is_large():
    M = tour.fast_needed(64, 64, 8)
    assert tour.standard_accesses(64, 8, M) == 2048 == tour.flash_accesses(64, 8, 64)
    assert tour.standard_accesses(64, 8, M-1) == 18432


@pytest.mark.parametrize("mask", [False, True])
def test_gradients_match_with_recompute(mask):
    m = tour.causal(16) if mask else None
    std, g0 = tour.saved_bytes(lambda q, k, v: tour.standard_attention(q, k, v, m), 16)
    redo, g1 = tour.saved_bytes(lambda q, k, v: tour.flash_attention(q, k, v, 4, 4, m, recompute=True), 16)
    assert not g1.isnan().any()
    assert (g0-g1).abs().max() < 1e-10


def test_recompute_saves_less_as_n_grows():
    fn = lambda q, k, v: tour.flash_attention(q, k, v, 8, 8, recompute=True)
    assert tour.saved_bytes(fn, 128)[0] < tour.saved_bytes(tour.standard_attention, 128)[0]
