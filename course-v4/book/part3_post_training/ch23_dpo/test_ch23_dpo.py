"""第 23 章快速测试：BT/DPO 手算值、梯度方向与响应 log-prob 的 mask；不需要 GPU。"""
import math
from argparse import Namespace
import json

import torch

from miniocr.train.preference import bradley_terry_loss, dpo_loss, response_logprob


def test_tour_preserves_hand_value_after_training():
    from book.part3_post_training.ch23_dpo.tour import part1
    result = part1(Namespace(seed=0))
    assert abs(result['hand_checks']['dpo_value'] - result['hand_checks']['dpo_expected']) < 1e-6


def test_selection_saves_all_beta_paired_differences(tmp_path, monkeypatch):
    from book.part3_post_training.ch23_dpo import tour
    from miniocr.data.postcorrect import PostCorrectExample
    val = [PostCorrectExample('a', 'val', 'x', 'x'), PostCorrectExample('b', 'val', 'y', 'y')]
    monkeypatch.setattr(tour, 'build_split', lambda *a, **kw: val)
    sft = tmp_path / 'sft.jsonl'
    sft.write_text('\n'.join(json.dumps(dict(id=e.id, raw='x')) for e in val))
    for beta, predictions in ((.05, ['z', 'z']), (.3, ['x', 'y'])):
        run = tmp_path / f'beta-{beta}'
        run.mkdir()
        tour.write(run / 'val-metrics.json', dict(summary=dict(em_norm=dict(value=beta), cer={})))
        # 逆序记录，确保按 ID 而非文件行号配对。
        rows = [dict(id=e.id, raw=p) for e, p in zip(val, predictions)]
        (run / 'preds-val.jsonl').write_text('\n'.join(json.dumps(r) for r in reversed(rows)))
    result = tour.part6(Namespace(n=2, seed=0, bootstrap=100, out=tmp_path, sft_preds=sft))
    saved = json.loads((tmp_path / 'selection.json').read_text())
    assert json.loads(json.dumps(result)) == saved and saved['selected_beta'] == .3
    assert saved['per_beta']['0.05']['paired_dpo_minus_sft']['diff'] == -.5
    assert saved['per_beta']['0.3']['paired_dpo_minus_sft']['diff'] == .5


def test_bradley_terry_zero_margin_is_log2():
    assert abs(bradley_terry_loss(torch.tensor([0.]), torch.tensor([0.])).item() - math.log(2)) < 1e-6


def test_dpo_hand_value_and_rewards():
    loss, rc, rr = dpo_loss(torch.tensor([2.]), torch.tensor([-1.]), torch.tensor([0.]), torch.tensor([0.]), beta=.5)
    assert abs(rc.item() - 1.) < 1e-6 and abs(rr.item() + .5) < 1e-6
    assert abs(loss.item() - math.log(1 + math.exp(-1.5))) < 1e-6


def test_dpo_gradient_pushes_chosen_up():
    pc = torch.tensor([0.], requires_grad=True)
    loss, _, _ = dpo_loss(pc, torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]), beta=.1)
    loss.sum().backward()
    assert pc.grad.item() < 0


def test_dpo_rejects_nonpositive_beta():
    try:
        dpo_loss(torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), beta=0.)
    except ValueError:
        return
    raise AssertionError('beta=0 应报错')


def test_response_logprob_ignores_prompt_and_padding():
    logits = torch.zeros(1, 4, 5)  # 均匀分布：每个答案 token 的 log-prob 为 log(1/5)
    labels = torch.tensor([[-100, -100, 2, 1]])
    value = response_logprob(logits, labels).item()
    assert abs(value - 2 * math.log(1 / 5)) < 1e-5
