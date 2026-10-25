"""第 31 章的自动检查。"""
import torch

from miniocr.rl.grpo import aggregate, grpo_loss, keep_groups, overlong_penalty
from miniocr.rl.rewards import doc_reward, task_reward, well_formed


def test_rewards_and_format_gate():
    cases = [("count", "2", "2", 1.0), ("count", "3", "2", 0.0), ("count", "two", "2", 0.0),
             ("left", "red", "red", 1.0), ("left", "Red", "red", 0.0), ("exists", "yes", "no", 0.0),
             ("compare", "the same", "the same", 1.0), ("refer", "a small red circle", "a small red circle", 1.0),
             ("ground", "none", "none", 1.0), ("ground", "[0,0,10,10]", "none", 0.0),
             ("ground", "none", "[0,0,10,10]", 0.0), ("ground", "[0,0,10,10]", "[0,0,10,10]", 1.0),
             ("ground", "[0,0,10,10]", "[5,0,15,10]", 50 / 150), ("ground", "[0,0,10", "[0,0,10,10]", 0.0),
             ("text", "cat", "cat", 1.0), ("text", "cot", "cat", 1 - 1 / 3), ("text", "", "cat", 0.0),
             ("caption", "a small red circle", "a small red circle", 1.0)]
    for task, pred, ref, want in cases:
        assert abs(task_reward(task, pred, ref) - want) < 1e-6, (task, pred, ref)
    assert not well_formed("count", "") and well_formed("ground", "[1,2,3,4]") and not well_formed("left", "purple")


def test_doc_reward():
    assert doc_reward("text", "abc", "abc") == 1.0 and doc_reward("text", "xyz", "abc") == 0.0
    assert abs(doc_reward("formula", r"x^{2}+1", r"x^{2}+1") - 1.0) < 1e-6
    t = "<table><tr><td>a</td><td>b</td></tr></table>"
    assert abs(doc_reward("table", t, t) - 1.0) < 1e-6
    assert doc_reward("table", "<table><tr><td>a", t) <= 1.0


def test_aggregate_modes():
    x = torch.tensor([[2.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    m = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert abs(aggregate(x, m, "sample") - 1.5) < 1e-6        # (2 + 1) / 2
    assert abs(aggregate(x, m, "token") - 1.25) < 1e-6        # (2 + 3) / 4
    assert abs(aggregate(x, m, "constant") - 5 / 6) < 1e-6    # 5 / (2 × 3)
    assert abs(aggregate(x, m, "constant", max_len=10) - 5 / 20) < 1e-6


def test_grpo_loss_on_policy_equals_policy_gradient():
    torch.manual_seed(0)
    logp = torch.randn(3, 5).requires_grad_(True)
    old = logp.detach().clone()
    adv = torch.tensor([1.0, -0.5, 0.0])
    mask = torch.tensor([[0, 1, 1, 0, 0], [0, 1, 1, 1, 1], [0, 0, 1, 1, 0]], dtype=torch.float)
    loss, st = grpo_loss(logp, old, adv, mask, mode="token")
    loss.backward()
    want = -adv[:, None] * mask / mask.sum()                   # ρ = 1：就是 −A·∇log π，按 token 平均
    assert torch.allclose(logp.grad, want, atol=1e-6)
    assert st["clip_frac"] == 0 and abs(st["ratio_max"] - 1) < 1e-6
    logp.grad = None
    loss, st = grpo_loss(logp, old, adv, mask, ref_logp=old, beta=0.1, mode="sample")
    loss.backward()                                            # 策略 = 参考模型：k3 和它的梯度都是 0
    want = -adv[:, None] * mask / mask.sum(1, keepdim=True) / 3
    assert torch.allclose(logp.grad, want, atol=1e-6) and abs(st["kl"]) < 1e-7


def test_overlong_and_keep_groups():
    assert overlong_penalty(10, 20, 4) == 0 and overlong_penalty(16, 20, 4) == 0
    assert overlong_penalty(18, 20, 4) == -0.5 and overlong_penalty(20, 20, 4) == -1.0 and overlong_penalty(21, 20, 4) == -1.0
    r = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    assert keep_groups(r, 2).tolist() == [False, True, False]
