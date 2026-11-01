"""附录 D：块分配和批处理步数，全部确定性，可手算。"""
import pytest

from . import tour


def test_fragmentation_by_hand():
    row = tour.fragmentation([23, 16, 1], 16)
    assert row["blocks"] == 2+1+1
    assert row["slots"] == 64 and row["wasted"] == 9+0+15


def test_wasted_slots_below_one_block_per_request():
    lengths = [p+o for p, o in tour.REQUESTS]
    for block in (4, 16, 64):
        assert tour.fragmentation(lengths, block)["wasted"] < block*len(lengths)


def test_reservation():
    assert tour.reservation([10, 20], 128)["wasted"] == 226
    with pytest.raises(ValueError):
        tour.reservation([129], 128)


def test_pool_allocates_on_demand_and_recycles():
    pool = tour.BlockPool(3, 4)
    pool.add("a")
    pool.append("a", 4)
    assert pool.used() == 1
    pool.append("a")
    assert pool.used() == 2 and pool.lengths["a"] == 5
    pool.add("b")
    pool.append("b", 4)
    with pytest.raises(MemoryError):
        pool.append("b")
    pool.release("a")
    assert pool.used() == 1
    pool.append("b")
    assert len(pool.tables["b"]) == 2


def test_prefix_sharing_and_refcount():
    pool = tour.BlockPool(16, 16)
    pool.add("x")
    pool.append("x", 40)
    pool.add("y", shared_blocks=pool.tables["x"][:2])
    pool.append("y", 8)
    assert pool.used() == 4          # 3 块给 x，1 块给 y 自己的尾巴
    assert pool.lengths["y"] == 40
    pool.release("x")
    assert pool.used() == 3          # 共享的 2 块还在，加 y 的尾巴
    pool.release("y")
    assert pool.used() == 0


def test_batching_hand_example():
    lengths = [3, 8, 2, 5]
    static, cont = tour.static_batching(lengths, 2), tour.continuous_batching(lengths, 2)
    assert static["steps"] == 8+5 == 13
    assert cont["steps"] == 10
    assert static["utilization"] == pytest.approx(18/26)
    assert cont["utilization"] == pytest.approx(18/20)
    # 静态：前两条第 8 步返回，后两条第 13 步。连续：3、8、5、10。
    assert static["mean_finish"] == pytest.approx((8+8+13+13)/4)
    assert cont["mean_finish"] == pytest.approx((3+8+5+10)/4)


def test_batch_one_equals_sequential():
    outputs = [o for _, o in tour.REQUESTS]
    assert tour.static_batching(outputs, 1)["steps"] == tour.continuous_batching(outputs, 1)["steps"] == sum(outputs)


@pytest.mark.parametrize("B", [2, 4, 8])
def test_continuous_never_worse_and_respects_bound(B):
    outputs = [o for _, o in tour.REQUESTS]
    static, cont = tour.static_batching(outputs, B), tour.continuous_batching(outputs, B)
    assert cont["lower_bound"] <= cont["steps"] <= static["steps"]


def test_serving_hand_example():
    # 块大小 4，池 2 块。两条请求：提示 4，输出 1。第一条要长到第 2 块；第二条进不来。
    row = tour.simulate_serving([(4, 1), (4, 1)], 2, 4, 2)
    assert row["peak_running"] == 2 and row["preemptions"] == 1
    assert row["steps"] == 2


def test_small_pool_preempts_but_finishes():
    row = tour.simulate_serving(tour.REQUESTS, 8, 16, 16)
    assert row["preemptions"] > 0
    assert row["steps"] >= tour.simulate_serving(tour.REQUESTS, 64, 16, 16)["steps"]


def test_preempting_itself_when_it_is_the_latest():
    # 池 3 块，块大小 4。三条都占 1 块；第三条先被抢占，第一条仍能走完。
    row = tour.simulate_serving([(4, 5), (4, 5), (4, 5)], 3, 4, 3)
    assert row["preemptions"] >= 1


def test_reservation_blocks_concurrency():
    assert tour.simulate_serving([(4, 1), (4, 1)], 2, 4, 2, reserve_output=4)["peak_running"] == 1
    with pytest.raises(ValueError):
        tour.simulate_serving([(4, 9)], 2, 4, 2, reserve_output=4)
    with pytest.raises(ValueError):
        tour.simulate_serving([(4, 9)], 2, 4, 2)


def test_plenty_of_memory_matches_continuous():
    outputs = [o for _, o in tour.REQUESTS]
    row = tour.simulate_serving(tour.REQUESTS, 1000, 16, 4)
    assert row["preemptions"] == 0
    assert row["steps"] == tour.continuous_batching(outputs, 4)["steps"]


def test_adapter_bytes():
    row = tour.adapter_bytes(100, 1, 3)
    assert row == dict(merged=600, mounted=206)
