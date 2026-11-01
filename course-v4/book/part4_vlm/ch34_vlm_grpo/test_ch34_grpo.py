"""第 34 章 GRPO 机制的 CPU 回归；不加载任何模型权重，秒级完成。"""
import torch

from miniocr.train.vlm_grpo import (completion_mask, degenerate, group_advantages, grpo_loss,
                                    layered_reward, probe_rewards, token_logprobs)


def test_layered_reward_tiers():
    assert layered_reward('a+b', '$a+b$')['tier'] == 3
    assert layered_reward('a+b', '$a+b$')['reward'] == 1.0
    assert layered_reward('\\frac{a}{b}', '$\\frac{a}{c}$')['tier'] in (1, 2)
    assert layered_reward('\\frac{a}{b}', '')['tier'] == 0


def test_unparseable_output_gets_zero_reward():
    row = layered_reward('x^2', 'Explanation: this is a long sentence about things')
    assert row['tier'] == 0 and row['reward'] == 0.0


def test_group_advantages_standardize_and_exclude_constant_groups():
    rewards = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
    adv, informative = group_advantages(rewards)
    assert informative.tolist() == [True, False]
    assert torch.allclose(adv[0].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.all(adv[1] == 0)


def test_completion_mask_keeps_first_eos_and_drops_after():
    eos = 7
    tokens = torch.tensor([[3, 4, 7, 7, 5], [3, 4, 5, 6, 8], [7, 1, 2, 3, 4]])
    assert completion_mask(tokens, [eos]).tolist() == [
        [1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]]


def test_token_logprobs_match_log_softmax():
    logits = torch.randn(2, 3, 5)
    targets = torch.randint(0, 5, (2, 3))
    expected = torch.log_softmax(logits, -1).gather(-1, targets[..., None])[..., 0]
    assert torch.allclose(token_logprobs(logits, targets), expected, atol=1e-6)


def test_positive_advantage_pushes_policy_logprob_up():
    policy = torch.zeros(1, 4, requires_grad=True)
    ref = torch.zeros(1, 4)
    mask = torch.ones(1, 4)
    loss, _ = grpo_loss(policy, ref, mask, torch.tensor([1.0]), beta=0.0)
    loss.backward()
    assert torch.all(policy.grad < 0)  # 梯度下降会增大 log-prob


def test_kl_is_zero_when_policy_equals_reference():
    lp = torch.randn(2, 5)
    _, stats = grpo_loss(lp, lp.clone(), torch.ones(2, 5), torch.zeros(2), beta=0.04)
    assert abs(stats['kl']) < 1e-6


def test_zero_token_batch_rejected():
    import pytest
    with pytest.raises(ValueError):
        grpo_loss(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1))


def test_probe_outputs_are_scored_and_degenerate_detected():
    probes = probe_rewards('\\frac{a}{b}')
    assert set(probes) == {'', 'x', '\\frac{a}{b}', '\\alpha', 'a+b=c'}
    assert probes[''] == 0
    assert probes['\\frac{a}{b}'] == 3
    assert degenerate('') and degenerate('x')
    assert not degenerate('$\\frac{a}{b}$')


def test_no_parse_bonus_removes_only_unrelated_parse_reward():
    for target, pred in [('a+b+c+d+e', 'x'), ('a+b+c+d+e', 'a+b+c+d+f'), ('a+b', 'a+b'), ('a+b', '')]:
        old = layered_reward(target, pred)
        new = layered_reward(target, pred, reward_mode='no_parse_bonus')
        assert new['tier'] == old['tier']
        assert new['reward'] == (0.0 if old['tier'] == 1 else old['reward'])
    assert layered_reward('a+b+c+d+e', 'x')['reward'] == 0.2
