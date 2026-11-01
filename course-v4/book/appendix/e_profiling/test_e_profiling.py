"""附录 E：只断言故意放的慢点排第一和调用次数，不断言用时。"""
import pytest
import torch

from . import tour


@pytest.fixture(autouse=True)
def two_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_slow_and_fast_agree():
    model = tour.make_setup()[0]
    assert tour.slow_weight_norm(model) == pytest.approx(tour.fast_weight_norm(model), rel=1e-12)


def test_profiler_finds_deliberate_slow_spot():
    prof = tour.profile_loop(tour.slow_weight_norm)
    rows = tour.span_rows(prof)
    assert rows[0][0] == "span/weight_norm"
    assert {row[2] for row in rows} == {3}                       # active=3：每段 3 次
    # 128×64 个元素，3 步：每个元素一次 .item()。
    assert tour.op_count(prof, "aten::item") >= 3*128*64


def test_fixing_removes_python_loop_signature():
    prof = tour.profile_loop(tour.fast_weight_norm)
    assert {row[2] for row in tour.span_rows(prof)} == {3}
    # 向量化后 .item() 的次数远少于元素个数。
    assert tour.op_count(prof, "aten::item") < 128*64


def test_arithmetic_intensity_by_hand():
    assert tour.daxpy_intensity() == pytest.approx(1/6)
    assert tour.matmul_intensity(256) == pytest.approx(256/6)
