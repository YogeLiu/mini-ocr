"""离线 CPU 机制检查：解析梯度、精确分布、mask 和 CLI 默认值。"""
import pytest
import torch

from miniocr.train.policy_gradient import (
    categorical_score_gradient, kl_estimators, ppo_surrogate,
    reinforce_loss, sequence_log_prob,
)
from book.part3_post_training.ch24_rl_basics.tour import (
    format_reward, parse_args, prompt_pool, rollout,
)
from miniocr.models.gpt import GPT, GPTConfig


def test_reinforce_exact_enumeration_and_detach():
    logits = torch.tensor([-.3, .1, .7], dtype=torch.double, requires_grad=True)
    rewards = torch.tensor([1., 2., 4.], dtype=torch.double, requires_grad=True)
    baseline = torch.tensor(1.7, requires_grad=True)
    p = logits.softmax(0)
    # 按动作概率精确求期望，而非用偶然的 MC 测试判断无偏。
    loss = -(p.detach()*logits.log_softmax(0)*(rewards-baseline).detach()).sum()
    got, = torch.autograd.grad(loss, logits)
    expected = -p.detach()*(rewards.detach()-(p.detach()*rewards.detach()).sum())
    torch.testing.assert_close(got, expected)
    loss = reinforce_loss(logits.log_softmax(0), rewards, baseline)
    loss.backward()
    assert rewards.grad is None and baseline.grad is None


def test_score_matches_autograd_and_variance_identity():
    logits = torch.tensor([-.2, .3, .7], dtype=torch.double, requires_grad=True)
    p = logits.softmax(0)
    action = torch.tensor(2)
    reward = torch.tensor(2.5)
    loss = reinforce_loss(logits.log_softmax(0)[action], reward, 1.)
    grad, = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(-grad, categorical_score_gradient(p, action, reward, 1.))
    means = torch.tensor([1., 1.5, 2.], dtype=torch.double)
    exact = p.detach()*(means-(p.detach()*means).sum())
    for baseline in (0., 1.6):
        gradients = categorical_score_gradient(p.detach(), torch.arange(3), means, baseline)
        torch.testing.assert_close((p.detach()[:, None]*gradients).sum(0), exact)


def test_kl_exact_bias_and_support_direction():
    p = torch.tensor([.1, .2, .3, .4], dtype=torch.double)
    q = p.flip(0)
    truth = (p*(p/q).log()).sum()
    values = kl_estimators((q/p).log())
    for name in ("k1", "k3"):
        torch.testing.assert_close((p*values[name]).sum(), truth)
    assert abs(((p*values["k2"]).sum()-truth).item()) > .01
    assert torch.all(values["k3"] >= 0)
    for values_at_equal in kl_estimators(torch.zeros(3)).values():
        assert torch.count_nonzero(values_at_equal) == 0


@pytest.mark.parametrize("advantage,expected", [(1., [1., 1., 0.]), (-1., [0., -1., -1.])])
def test_ppo_only_clips_favorable_direction(advantage, expected):
    ratio = torch.tensor([.5, 1., 1.5], requires_grad=True)
    adv = torch.full_like(ratio, advantage, requires_grad=True)
    ppo_surrogate(ratio, adv).sum().backward()
    torch.testing.assert_close(ratio.grad, torch.tensor(expected))
    assert adv.grad is None


def test_sequence_mask_shift_eos_padding_and_sum():
    logits = torch.zeros(2, 5, 4, requires_grad=True)
    targets = torch.tensor([[-100, -100, 1, 2, -100], [-100, 1, 2, 3, 0]])
    mask = targets != -100
    logp = sequence_log_prob(logits, targets, mask)
    torch.testing.assert_close(logp, -torch.tensor([2., 4.])*torch.tensor(4.).log())
    logp.sum().backward()
    assert logits.grad[~mask].abs().sum() == 0
    assert logits.grad[mask].abs().sum() > 0
    with pytest.raises(ValueError):
        sequence_log_prob(logits, targets[:, :2], mask)


def test_rollout_recomputation_is_same_policy():
    torch.set_num_threads(2)
    torch.manual_seed(4)
    model = GPT(GPTConfig(vocab_size=9, block_size=8, n_layer=1, n_head=2, d_model=16)).eval()
    ids = rollout(model, torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]), 4)
    logits, _ = model(ids[:, :-1])
    mask = torch.zeros_like(ids[:, 1:], dtype=torch.bool)
    mask[:, 3:] = True
    combined = sequence_log_prob(logits, ids[:, 1:], mask)
    sequential = torch.zeros(2)
    for pos in range(4, 8):
        logit, _ = model(ids[:, :pos])
        sequential += logit[:, -1].log_softmax(-1).gather(1, ids[:, pos:pos+1]).squeeze(1)
    torch.testing.assert_close(combined, sequential, atol=2e-6, rtol=1e-5)
    reinforce_loss(combined, torch.tensor([1., 0.]), .2).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_prompt_splits_and_reward_hacking_probe():
    pools = [{tuple(x) for x in prompt_pool(s).tolist()} for s in ("train", "val", "test")]
    assert sum(map(len, pools)) == 8**4
    assert not (pools[0]&pools[1] or pools[0]&pools[2] or pools[1]&pools[2])
    # 全 $ 也能得满分：明确展示奖励的盲区，不能据此声称 OCR 改善。
    torch.testing.assert_close(format_reward(torch.tensor([[0, 0, 0], [0, 1, 2], [1, 2, 3]])),
                               torch.tensor([1., .5, 0.]))


def test_cli_small_offline_defaults_and_validation():
    a = parse_args([])
    assert a.device == "cpu" and a.n == 20 and not a.final_test
    for argv in (["6"], ["--steps", "0"], ["--lr", "nan"], ["--width", "13"], ["--final-test"]):
        with pytest.raises(SystemExit):
            parse_args(argv)
