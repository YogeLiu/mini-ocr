"""附录 A：显存账可手算；梯度等价用单进程模拟。"""
import pytest
import torch

from . import tour


@pytest.mark.parametrize("method,expected", [
    ("ddp", 16), ("zero1", 4+4+8/4), ("zero2", 4+(4+8)/4), ("zero3", 16/4), ("fsdp", 16/4)])
def test_fp32_bytes_per_param(method, expected):
    assert tour.static_per_card(1, 4, method)["total"] == pytest.approx(expected)


@pytest.mark.parametrize("method,expected", [
    ("ddp", 16), ("zero1", 2+2+12/4), ("zero2", 2+(2+12)/4), ("zero3", 16/4)])
def test_mixed_bytes_per_param(method, expected):
    assert tour.static_per_card(1, 4, method, "mixed")["total"] == pytest.approx(expected)


def test_single_card_all_methods_same():
    for method in tour.SHARDED:
        assert tour.static_per_card(10, 1, method)["total"] == 160


def test_invalid_arguments():
    with pytest.raises(ValueError):
        tour.static_per_card(1, 0, "ddp")
    with pytest.raises(ValueError):
        tour.static_per_card(1, 2, "zero4")


def test_tensor_and_pipeline_split():
    from dataclasses import replace
    from miniocr.models.gpt import GPT
    from miniocr.train.scaling import ladder_config
    model = GPT(replace(ladder_config("tiny", 64, 16), attn_impl="manual"))
    # tiny：d=64，两层，每层 d_ff=192；K=2 时矩阵减半，一维 norm 不切。
    norm = sum(p.numel() for p in model.parameters() if p.ndim == 1)
    assert tour.tensor_parallel_params(model, 2) == (model.n_params()-norm)//2+norm
    block = sum(p.numel() for p in model.blocks[0].parameters())
    emb = 64*64
    assert tour.pipeline_parallel_params(model, 2) == [block+emb, block+64+emb]
    assert tour.pipeline_parallel_params(model, 1) == [model.n_params()]
    with pytest.raises(ValueError):
        tour.pipeline_parallel_params(model, 3)


def test_hand_example():
    per = tour.linear_example()
    assert per == [2, 8, 18, 32]
    full = sum(per)/4
    assert tour.averaged([torch.tensor(5.), torch.tensor(25.)]).item() == full == 15
    d = sum(per[1:])/3
    assert tour.averaged([torch.tensor(2.), torch.tensor(d)]).item() == pytest.approx(32/3)
    assert tour.averaged([torch.tensor(2.), torch.tensor(d)], [1, 3]).item() == pytest.approx(15)


def test_equal_shards_match_full_batch():
    model = tour.tiny_model()
    x, y = tour.batch()
    full = tour._grad(model, x, y)
    assert tour.relative(tour.averaged(tour.shard_gradients(model, x, y, (2, 2, 2, 2))), full) < 1e-12


def test_unequal_shards_need_weights():
    model = tour.tiny_model()
    x, y = tour.batch()
    full = tour._grad(model, x, y)
    sizes = (5, 1, 1, 1)
    grads = tour.shard_gradients(model, x, y, sizes)
    assert tour.relative(tour.averaged(grads), full) > 1e-3
    assert tour.relative(tour.averaged(grads, sizes), full) < 1e-12


def test_distributed_sampler_pads_or_drops():
    pad = tour.sampler_indices(10, 4)
    assert [len(p) for p in pad] == [3]*4
    assert sum(pad, []).count(0) == 2 and sum(pad, []).count(1) == 2
    drop = tour.sampler_indices(10, 4, True)
    assert [len(p) for p in drop] == [2]*4
    assert sorted(sum(drop, [])) == list(range(8))


def test_batchnorm_breaks_equivalence():
    full, mean = tour.bn_gradients()
    assert tour.relative(mean, full) > 1e-3


def test_replicas_stay_in_sync_only_with_shared_gradient():
    same = tour.trajectory(steps=2)
    assert same[0] == 0
    assert same[1] < 1e-10
    assert tour.trajectory(steps=2, broken=True)[0] > 1e-6
