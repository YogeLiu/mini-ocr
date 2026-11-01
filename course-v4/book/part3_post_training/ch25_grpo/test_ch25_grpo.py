"""第 25 章快速测试：组内优势、GRPO 损失的裁剪与 KL、分层奖励与探针；不需要 GPU。"""
import math

import torch
import pytest
from types import SimpleNamespace

from miniocr.eval.rewards import hack_probes, layered_reward, parse_only_reward
from miniocr.train.grpo import group_advantages, grpo_loss


def test_group_advantages_are_zero_mean_and_zero_for_constant_groups():
    adv = group_advantages(torch.tensor([1., 0., 1., 0., 2., 2., 2., 2.]), 4, scale=False)
    assert torch.allclose(adv.view(2, 4).mean(1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(adv[4:], torch.zeros(4))


def test_clipping_saturates_in_advantage_direction():
    new = torch.tensor([[math.log(1.5)]], requires_grad=True)
    loss, _ = grpo_loss(new, torch.zeros(1, 1), new.detach(), torch.tensor([1.]), torch.ones(1, 1), beta=0.)
    loss.backward()
    assert abs(new.grad.item()) < 1e-6


def test_kl_is_zero_when_policy_equals_reference():
    _, diag = grpo_loss(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3),
                        torch.tensor([1., -1.]), torch.ones(2, 3), beta=.04)
    assert diag['kl_mean'] == 0


def test_layered_reward_levels():
    src, tgt = r'\frac{a}{b}+c', r'\frac{a}{b}+d'
    assert layered_reward('', src, tgt) == 0.0
    assert layered_reward(tgt, src, tgt) == 1.0
    assert layered_reward('Here is the answer: ' + src, src, tgt) == 0.1
    assert layered_reward(src, src, tgt) < 0.5  # 原样复制输入不得得到高分


def test_parse_only_reward_is_hackable_by_trivial_output():
    src, tgt = r'\frac{a}{b}+c', r'\frac{a}{b}+d'
    probes = dict((n, t) for n, t in hack_probes(src, tgt))
    assert parse_only_reward(probes['single_symbol'], src, tgt) == 1.0
    assert layered_reward(probes['single_symbol'], src, tgt) < 0.5


def test_final_test_requires_explicit_part4():
    from .tour import main
    with pytest.raises(SystemExit):
        main(['5', '--final-test'])


def test_final_test_refuses_previous_attempt_before_loading_model(tmp_path):
    from .tour import part4
    run = tmp_path / 'sft-continue'
    run.mkdir()
    (run / 'test-started.json').write_text('{}')
    with pytest.raises(ValueError, match='不得重复评测'):
        part4(SimpleNamespace(final_test=True, out=tmp_path, run='sft-continue'))


def test_training_seeds_share_fixed_data(monkeypatch):
    from . import tour
    calls = []
    def build(split, n, seed):
        calls.append((split, n, seed))
        return [SimpleNamespace(source='x', target='y')] * n
    monkeypatch.setattr(tour, 'build_split', build)
    monkeypatch.setattr(tour, 'render_chat', lambda tok, messages: 'prompt')
    for seed in (1, 2):
        rows = tour.prompt_rows(SimpleNamespace(steps=100, prompts=8, train_offset=4096,
                                                seed=seed, data_seed=0, tok=None))
        assert len(rows) == 800
    assert calls == [('train', 4896, 0)] * 2
