"""附录 C：量化公式手算；误差不超过 scale/2；GPTQ、AWQ 的小矩阵。"""
import pytest
import torch

from . import tour


def tensor(*values):
    return torch.tensor(values, dtype=torch.float64)


def test_int4_symmetric_by_hand():
    x = tensor(-1., 0., .5, 1.)
    restored, scale, zp = tour.roundtrip(x, 4, True)
    assert scale.item() == pytest.approx(1/7) and zp.item() == 0
    assert tour.quantize(x, scale, zp, 4).tolist() == [-7, 0, 4, 7]  # .5·7=3.5，舍入到偶数 4
    assert restored[2].item() == pytest.approx(4/7)


def test_int8_asymmetric_keeps_zero_exact():
    x = tensor(-1., 0., 3.)
    restored, scale, zp = tour.roundtrip(x, 8)
    assert scale.item() == pytest.approx(4/255)
    assert zp.item() == round(-128+255/4)
    assert restored[1].item() == 0


def test_positive_only_range_includes_zero():
    scale, zp = tour.quant_params(tensor(1., 2.), 8)
    assert scale.item() == pytest.approx(2/255)
    assert zp.item() == -128


def test_constant_zero_tensor_does_not_divide_by_zero():
    restored, scale, _ = tour.roundtrip(torch.zeros(3, dtype=torch.float64), 8)
    assert scale.item() == 1 and restored.abs().max() == 0


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("symmetric", [False, True])
def test_error_bound_on_model(bits, symmetric):
    model = tour.tiny_model()
    for p in model.parameters():
        if p.ndim >= 2:
            stats = tour.error_stats(p.detach().double(), bits, symmetric)
            assert stats["最大误差"] <= stats["上界"]*(1+1e-9)


def test_int4_scale_is_17_times_int8():
    w = tour.tiny_model().tok.weight.detach().double()
    assert tour.quant_params(w, 4)[0]/tour.quant_params(w, 8)[0] == pytest.approx(255/15)


def test_histogram_sums_to_one():
    w = tour.tiny_model().tok.weight.detach().double()
    assert tour.error_histogram(w, 8).sum().item() == pytest.approx(1)


def test_outlier_inflates_scale():
    w = tour.tiny_model().blocks[0].mlp.gate.weight.detach().double()
    spiked = w.clone()
    spiked[0, 0] = w.abs().max()*10
    assert tour.error_stats(spiked, 8, True)["scale"] == pytest.approx(tour.error_stats(w, 8, True)["scale"]*10)
    assert (tour.per_row_roundtrip(spiked, 8)-spiked).square().mean() < tour.error_stats(spiked, 8, True)["MSE"]


def test_gptq_hand_example():
    w = tensor(.6, .6).unsqueeze(0)
    H = torch.tensor([[1., .5], [.5, 1.]], dtype=torch.float64)
    plain = tour.round_to_grid(w, 1.)
    assert plain.tolist() == [[1., 1.]]
    assert tour.output_error(plain-w, H) == pytest.approx(.48)
    # e=.4；第二列更新 −.4·(.5/1) = −.2，所以 .6→.4，再取整到 0。
    one = tour.gptq_step(w, H, 0, 1.)
    assert one[0].tolist() == pytest.approx([1., .4])
    full = tour.gptq(w, H, 1.)
    assert full.tolist() == [[1., 0.]]
    assert tour.output_error(full-w, H) == pytest.approx(.28)


def test_gptq_step_uses_no_update_when_uncorrelated():
    w = tensor(.6, .6).unsqueeze(0)
    assert tour.gptq_step(w, torch.eye(2, dtype=torch.float64), 0, 1.).tolist() == [[1., .6]]


def test_awq_hand_example():
    w = tensor(.1, 1.).unsqueeze(0)
    x = torch.tensor([[10., 0.], [0., .1]], dtype=torch.float64)
    H = x @ x.T
    plain = tour.awq_roundtrip(w, x, 0., 4)
    assert plain[0].tolist() == pytest.approx([1/7, 1.])
    assert tour.output_error(plain-w, H) == pytest.approx((1/7-.1)**2*100)
    scaled = tour.awq_roundtrip(w, x, 1., 4)
    assert scaled[0, 0].item() == pytest.approx(.1)            # 重要通道：无误差
    assert scaled[0, 1].item() == pytest.approx(10/7)          # 不重要通道：误差变大
    assert tour.output_error(scaled-w, H) == pytest.approx((10/7-1)**2*.01)


def test_awq_alpha_zero_is_plain():
    w = torch.randn(4, 8, dtype=torch.float64)
    x = torch.randn(8, 64, dtype=torch.float64)
    assert torch.equal(tour.awq_roundtrip(w, x, 0., 4), tour.per_row_roundtrip(w, 4))
