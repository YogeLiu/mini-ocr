"""第 8 章的自动检查：python -m pytest book/part2_lm/ch08_attention"""
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from miniocr.nn.attention import MultiHeadAttention, attention, causal_mask
from miniocr.data.latex_corpus import right_positions
from miniocr.models.attention_lm import AttentionLM
from miniocr.tokenizer import BPETokenizer


def test_attention_hand_example():
    q = torch.tensor([[1.0, 0.0]])
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    v = torch.tensor([[10.0], [20.0], [30.0]])
    out, w = attention(q, k, v)
    s = torch.tensor([1.0, 0.0, 1.0]) / math.sqrt(2)
    p = s.exp() / s.exp().sum()
    assert torch.allclose(w[0], p, atol=1e-6)
    assert math.isclose(out.item(), (p * torch.tensor([10.0, 20.0, 30.0])).sum().item(), rel_tol=1e-6)
    assert torch.allclose(w.sum(-1), torch.ones(1))


def test_attention_matches_torch_and_mask():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 7, 8), torch.randn(2, 3, 7, 4)
    out, _ = attention(q, k, v)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5)
    x = torch.randn(2, 3, 6, 8)
    out, w = attention(x, x, x, causal_mask(6))
    assert torch.allclose(out, F.scaled_dot_product_attention(x, x, x, is_causal=True), atol=1e-5)
    assert (w[..., 0, 1:] == 0).all() and torch.allclose(w[..., 0, 0], torch.ones(1))


def test_causal_mask():
    m = causal_mask(4)
    assert m.dtype == torch.bool and m.shape == (4, 4)
    assert m.tolist() == [[True, False, False, False], [True, True, False, False],
                          [True, True, True, False], [True, True, True, True]]


def test_mha_matches_nn_multihead_attention():
    torch.manual_seed(0)
    d, h = 16, 4
    mine = MultiHeadAttention(d, h)
    ref = nn.MultiheadAttention(d, h, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([mine.q.weight, mine.k.weight, mine.v.weight]))
        ref.in_proj_bias.copy_(torch.cat([mine.q.bias, mine.k.bias, mine.v.bias]))
        ref.out_proj.weight.copy_(mine.o.weight); ref.out_proj.bias.copy_(mine.o.bias)
    x, mem = torch.randn(2, 5, d), torch.randn(2, 9, d)
    assert torch.allclose(mine(x), ref(x, x, x)[0], atol=1e-5)
    assert torch.allclose(mine(x, kv=mem), ref(x, mem, mem)[0], atol=1e-5)          # 交叉注意力
    assert mine.last_weights.shape == (2, h, 5, 9)
    cm = MultiHeadAttention(d, h, causal=True)
    cm.load_state_dict(mine.state_dict())
    y = cm(x)
    x2 = x.clone(); x2[:, 3:] = torch.randn(2, 2, d)                                # 改动"未来"的 token
    assert torch.allclose(cm(x2)[:, :3], y[:, :3], atol=1e-6)                       # 前 3 个位置的输出不变
    assert not torch.allclose(cm(x2)[:, 3:], y[:, 3:])


def test_self_attention_is_permutation_equivariant():
    torch.manual_seed(0)
    m = MultiHeadAttention(8, 2)
    x = torch.randn(1, 6, 8)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    assert torch.allclose(m(x)[:, perm], m(x[:, perm]), atol=1e-6)


def test_padding_mask_ignores_padded_keys():
    """补齐位置（key 侧为 False）的权重必须是 0，改动补齐位置的 v 不影响输出。"""
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 4, 8), torch.randn(1, 6, 8), torch.randn(1, 6, 3)
    keep = torch.tensor([True, True, True, True, False, False])
    out, w = attention(q, k, v, keep[None, None, :])
    assert (w[..., 4:] == 0).all()
    v2 = v.clone(); v2[:, 4:] = 100.0
    assert torch.allclose(attention(q, k, v2, keep[None, None, :])[0], out, atol=1e-6)


def test_chapter_worked_example():
    """8.3–8.4 节的三 token 例子：投影、权重、输出与正文一致。"""
    from book.part2_lm.ch08_attention.tour import WK, WQ, WV, X
    Q, K, V = X @ WQ, X @ WK, X @ WV
    assert Q.tolist() == [[2, 0], [0, 3], [2, 2]] and K.tolist() == [[0, 2], [3, 0], [2, 2]]
    assert V.tolist() == [[1, 1], [2, 1], [2, 2]]
    out, w = attention(Q, K, V)
    assert torch.allclose(w[0], torch.tensor([0.0114, 0.7952, 0.1933]), atol=1e-4)
    assert torch.allclose(out, torch.tensor([[1.9886, 1.1933], [1.5036, 1.4964], [1.9546, 1.7679]]), atol=1e-4)
    out_c, w_c = attention(Q, K, V, causal_mask(3))
    assert torch.allclose(w_c[1], torch.tensor([0.9858, 0.0142, 0.0]), atol=1e-4)
    assert torch.allclose(out_c[0], V[0])


def test_multihead_padding_mask_per_sample():
    """多头时补齐 mask 要写成 (B, 1, 1, T)；B 恰好等于头数时，(B, 1, T) 会静默地把 mask 用到错误的样本上。"""
    torch.manual_seed(0)
    B, T, d, h = 2, 4, 8, 2
    m = MultiHeadAttention(d, h, causal=True)
    x = torch.randn(B, T, d)
    keep = torch.tensor([[True, True, True, True], [True, True, False, False]])
    y = m(x, mask=keep[:, None, None, :])
    assert (m.last_weights[1, :, :, 2:] == 0).all()                 # 第 2 个样本：所有头都不看补齐位置
    assert torch.allclose(y[0], m(x[:1])[0], atol=1e-6)             # 第 1 个样本没有补齐，结果和单独计算一样
    assert torch.allclose(y[1, :2], m(x[1:, :2])[0], atol=1e-6)     # 第 2 个样本的真实位置，和去掉补齐后单独计算一样


def test_right_positions_nested_delimiters_and_utf8_offsets():
    # 手写合并只把 right 合成一个 token，避免让语料训练决定测试答案。
    tok = BPETokenizer([(b'r', b'i'), (b'ri', b'g'), (b'rig', b'h'), (b'righ', b't')])
    text = r"中\left(a+\left(b\right)+\left[c\right]+\left(d\right.\right)"
    rows = right_positions(tok, text)
    assert len(rows) == 2
    expected = [(text.index('(', text.index('a+')), text.index(r'\right)')),
                (text.index('('), text.rindex(r'\right)'))]
    for (distance, target, source, ids), (opening, closing) in zip(rows, expected):
        assert ids == tok.encode(text, allow_special=False)
        assert tok.vocab[ids[target]] == b'right'
        assert b''.join(tok.vocab[i] for i in ids[:target]) == text[:closing + 1].encode()
        assert b''.join(tok.vocab[i] for i in ids[:source]) == text[:opening].encode()
        assert tok.vocab[ids[source]] == b'('
        assert distance == target - source - 1
    assert right_positions(tok, r"\left[x\right)") == []
    assert right_positions(tok, r"\right) x \left(") == []
    assert right_positions(BPETokenizer([]), r"\left(x\right)") == []


def test_right_positions_nested_arrows_preserve_pair_offsets():
    tok = BPETokenizer([(b'r', b'i'), (b'ri', b'g'), (b'rig', b'h'), (b'righ', b't')], specials=())
    text = r"模\leftarrow+\left(a+\leftrightarrow+\left(\rightarrow b\right)+\rightarrow c\right)"
    matches = right_positions(tok, text)
    assert len(matches) == 2
    # 内层先闭合；箭头命令既不能入栈，也不能弹出真正的左括号。
    openings = [text.index("(", text.index("(") + 1), text.index("(")]
    targets = [text.index(r"\right)") + 1, text.rindex(r"\right)") + 1]
    for (distance, target, source, ids), opening, closing in zip(matches, openings, targets):
        assert ids == tok.encode(text, allow_special=False)
        assert b''.join(tok.vocab[i] for i in ids[:source]) == text[:opening].encode()
        assert tok.vocab[ids[source]] == b'('
        assert b''.join(tok.vocab[i] for i in ids[:target]) == text[:closing].encode()
        assert tok.vocab[ids[target]] == b'right'
        assert distance == target - source - 1


def test_attention_lm_shape_causality_and_bos_mask():
    torch.manual_seed(7)
    model = AttentionLM(19, bos_id=17, context=8, d=16, n_heads=2)
    x = torch.randint(0, 17, (2, 8))
    before = model(x)
    assert before.shape == (2, 8, 19)
    assert (before.softmax(-1)[..., 17] == 0).all()
    assert torch.isfinite(before[..., :17]).all()
    changed = x.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 17
    after = model(changed)
    assert torch.equal(after[:, :4], before[:, :4])
    assert not torch.equal(after[:, 4:], before[:, 4:])


def test_attention_lm_padding_last_query_and_backward():
    torch.manual_seed(3)
    model = AttentionLM(19, bos_id=17, context=8, d=16)
    x = torch.randint(0, 17, (2, 8))
    lengths = torch.tensor([3, 8])
    full = model(x, lengths)
    last = model(x, lengths, last_only=True)
    assert torch.allclose(last, full[torch.arange(2), lengths - 1], atol=1e-6)
    assert (model.attn.last_weights[0, :, :, 3:] == 0).all()
    changed = x.clone()
    changed[0, 3:] = (changed[0, 3:] + 1) % 17
    assert torch.equal(last, model(changed, lengths, last_only=True))
    assert torch.allclose(last[0], model(x[:1, :3], last_only=True)[0], atol=1e-6)
    loss = F.cross_entropy(last, torch.tensor([1, 18]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError):
        model(x, torch.tensor([0, 8]))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 9, dtype=torch.long))


def test_formula_windows_preserve_all_targets_and_boundaries():
    from book.part2_lm.ch08_attention.tour import FormulaWindows

    data = FormulaWindows([[1, 2, 3, 4], [], [5]], bos=7, eos=8, context=3)
    x, lengths, y = data.batch(torch.arange(len(data)))
    assert y.tolist() == [1, 2, 3, 4, 8, 8, 5, 8]
    assert [row[:length].tolist() for row, length in zip(x, lengths)] == [
        [7], [7, 1], [7, 1, 2], [1, 2, 3], [2, 3, 4], [7], [7], [7, 5]]


@pytest.mark.parametrize(
    "context,target,source,visible,local_source,uniform",
    [
        (4, 2, 0, [126, 10, 11], 1, None),
        (12, 10, 0, [126, *range(10, 20)], 1, 1 / 11),
        (12, 11, 1, [126, *range(10, 21)], 2, 1 / 12),
        (12, 12, 0, list(range(10, 22)), 0, 1 / 12),
        (12, 13, 0, list(range(11, 23)), None, 0.0),
    ],
    ids=["short-formula", "right-padding", "bos-at-left", "source-at-left", "source-just-outside"],
)
def test_diagnose_attention_window_indices(
    monkeypatch, capsys, context, target, source, visible, local_source, uniform
):
    """直接检查诊断入口；远距离分支要求距离 >8，因此边界例子用 context=12。"""
    from book.part2_lm.ch07_lm_basics import tour as lm_tour
    from book.part2_lm.ch08_attention import tour

    ids = list(range(10, target + 12))
    record = (target - source - 1, target, source, ids)
    monkeypatch.setattr(tour, "right_positions", lambda tok, text: [record])
    exp = SimpleNamespace(val_text=["handwritten"], tok=None, bos=126, eos=127)
    expected = torch.tensor([visible + [exp.eos] * (context - len(visible))])
    assert visible[-1] == ids[target - 1]
    assert ids[target] not in expected[0].tolist()
    if exp.bos in visible:
        assert visible.index(exp.bos) == 0

    class DiagnosticModel:
        def __init__(self):
            self.context = context
            self.calls = 0
            weights = torch.zeros(1, 1, context, context)
            if local_source is not None:
                # 只有正确的查询行、源列才有非零权重，误取补齐行也会失败。
                weights[0, 0, len(visible) - 1, local_source] = 0.75
            self.attn = SimpleNamespace(last_weights=weights)

        def __call__(self, x, lengths, *, last_only):
            wanted = expected.clone()
            if self.calls and local_source is not None:
                wanted[0, local_source] = ord("x")
            assert torch.equal(x, wanted)
            assert lengths.tolist() == [len(visible)]
            assert last_only is True
            self.calls += 1
            return torch.zeros(1, 128)

    mlp_inputs = []

    def mlp_logits(model, x, bos):
        mlp_inputs.append(x.tolist())
        assert bos == exp.bos
        return torch.zeros(len(x), 128)

    monkeypatch.setattr(lm_tour, "logits", mlp_logits)
    model = DiagnosticModel()
    tour.diagnose_attention(model, exp, mlp=None)
    output = capsys.readouterr().out
    if uniform is None:  # 短公式只走概率分桶；无远距离配对。
        assert model.calls == 1 and not mlp_inputs
        assert "无距离 >8 的配对，跳过干预和注意力统计。" in output
    else:
        assert model.calls == 2
        assert mlp_inputs == [[ids[target - 8:target]]] * 2
        count = int(local_source is not None)
        hit = 0.75 if count else 0.0
        assert f"左括号仍在注意力窗口内的配对数={count}；窗口外配对数={1 - count}。" in output
        assert (f">8 全部配对：左括号平均注意力权重={hit:.4f}；"
                f"均匀分配对照={uniform:.4f}；窗口外均记零。") in output
        if count:
            assert f"均匀分配 1/可见位置数的平均值={uniform:.4f}" in output
        else:
            assert ">8 且左括号可见：" not in output


def test_attention_evaluation_counts_eos_and_weights_each_target():
    from book.part2_lm.ch08_attention.tour import FormulaWindows, evaluate_attention

    torch.manual_seed(0)
    data = FormulaWindows([[1, 2, 3, 4], [], [5]], bos=7, eos=8, context=3)
    model = AttentionLM(9, bos_id=7, context=3, d=8)
    losses = []
    for i in range(len(data)):
        x, lengths, y = data.batch(torch.tensor([i]))
        losses.append(F.cross_entropy(model(x[:, :lengths[0]], last_only=True), y).item())
    assert evaluate_attention(model, data, batch=3) == pytest.approx(sum(losses) / 8, abs=1e-6)
