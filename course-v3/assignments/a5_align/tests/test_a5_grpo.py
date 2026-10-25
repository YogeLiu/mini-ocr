"""A5 第三部分：三种汇总方式（长度偏差）、GRPO 损失、软超长惩罚、动态采样、格式闸门与奖励。"""
import pytest
import torch

from ._impl import adapters as A


def _data():
    per = torch.tensor([[1.0, 2.0, 3.0, 0.0], [4.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1, 1, 0], [1, 0, 0, 0]])
    return per, mask


def test_aggregate_values():
    per, mask = _data()
    assert A.run_aggregate(per, mask, "sample", None).item() == pytest.approx((2.0 + 4.0) / 2)
    assert A.run_aggregate(per, mask, "token", None).item() == pytest.approx(10 / 4)
    assert A.run_aggregate(per, mask, "constant", None).item() == pytest.approx(10 / 8)
    assert A.run_aggregate(per, mask, "constant", 10).item() == pytest.approx(10 / 20)


def test_aggregate_per_token_weights():
    # 长度偏差：sample 模式下长回答的每个 token 权重更小
    per, mask = _data()
    for mode, w in (("sample", [[1 / 6, 1 / 6, 1 / 6, 0], [1 / 2, 0, 0, 0]]),
                    ("token", [[1 / 4, 1 / 4, 1 / 4, 0], [1 / 4, 0, 0, 0]]),
                    ("constant", [[1 / 8, 1 / 8, 1 / 8, 0], [1 / 8, 0, 0, 0]])):
        x = per.clone().requires_grad_()
        A.run_aggregate(x, mask, mode, None).backward()
        assert x.grad.flatten().tolist() == pytest.approx(sum(w, []))


def test_grpo_loss_matches_formula():
    g = torch.Generator().manual_seed(0)
    old = -torch.rand(3, 5, generator=g) * 2
    logp = (old + 0.4 * torch.randn(3, 5, generator=g)).requires_grad_()
    ref = old + 0.1 * torch.randn(3, 5, generator=g)
    adv = torch.tensor([1.0, -0.5, 0.0])
    mask = torch.tensor([[1.0, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
    loss, st = A.run_grpo_loss(logp, old, adv, mask, ref, 0.05, 0.2, 0.28, "token")
    ratio = (logp.detach() - old).exp()
    obj = torch.minimum(ratio * adv[:, None], ratio.clamp(0.8, 1.28) * adv[:, None])
    lr = ref - logp.detach()
    per = -obj + 0.05 * (lr.exp() - 1 - lr)
    assert loss.item() == pytest.approx(float((per * mask).sum() / mask.sum()), rel=1e-5)
    m = mask.bool()
    clipped = (ratio.clamp(0.8, 1.28) * adv[:, None] < ratio * adv[:, None])
    assert st["clip_frac"] == pytest.approx(float(clipped[m].float().mean()))
    assert st["ratio_max"] == pytest.approx(float(ratio[m].max()))
    assert st["kl"] == pytest.approx(float((lr.exp() - 1 - lr)[m].mean()), rel=1e-5)
    loss0, st0 = A.run_grpo_loss(logp, old, adv, mask, None, 0.0, 0.2, None, "sample")
    assert st0["kl"] == 0.0


def test_grpo_first_step_ratio_is_one():
    logp = torch.randn(2, 4)
    loss, st = A.run_grpo_loss(logp, logp.clone(), torch.tensor([1.0, -1.0]), torch.ones(2, 4), None, 0.0, 0.2, None,
                               "token")
    assert st["ratio_max"] == pytest.approx(1.0) and st["clip_frac"] == 0.0
    assert loss.item() == pytest.approx(0.0)


def test_overlong_and_keep_groups():
    assert [A.run_overlong_penalty(n, 20, 4) for n in (10, 16, 18, 20, 21)] == pytest.approx([0, 0, -0.5, -1, -1])
    r = torch.tensor([1.0, 1, 1, 0, 0, 0, 0.5, 0.2, 0.5])
    assert A.run_keep_groups(r, 3).tolist() == [False, False, True]


@pytest.mark.parametrize("task,pred,ok", [
    ("count", "3", True), ("count", "three", False), ("count", "", False), ("left", "red", True), ("left", "pink", False),
    ("exists", "yes", True), ("exists", "Yes", False), ("compare", "the same", True), ("compare", "same", False),
    ("ground", "none", True), ("ground", "[1,2,10,12]", True), ("ground", "somewhere", False),
    ("refer", "a small red circle", True), ("refer", "small red circle", False), ("caption", "", False),
    ("read", "abc", True), ("text", "", False)])
def test_well_formed(task, pred, ok):
    assert A.run_well_formed(task, pred) == ok


def test_task_reward():
    assert A.run_task_reward("count", "3", "3") == 1.0
    assert A.run_task_reward("count", "three", "3") == 0.0                                  # 格式闸门
    assert A.run_task_reward("ground", "none", "none") == 1.0
    assert A.run_task_reward("ground", "none", "[0,0,10,10]") == 0.0
    assert A.run_task_reward("ground", "[0,0,10", "[0,0,10,10]") == 0.0                    # 坏的框：先过闸门，不去算 IoU
    assert A.run_task_reward("ground", "[5,0,15,10]", "[0,0,10,10]") == pytest.approx(1 / 3)
    assert A.run_task_reward("read", "helo", "hello") == pytest.approx(0.8)
    assert A.run_task_reward("caption", "xyzxyzxyz", "a red") == 0.0                        # 1 − NED < 0 时取 0
    assert A.run_task_reward("refer", "a large blue square", "a large blue square") == 1.0
