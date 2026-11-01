"""第 14 章 CPU 快速检查；固定种子，模型只做前向。"""
from dataclasses import asdict
import math
from types import SimpleNamespace

import pytest
import torch

from miniocr import decoding as d
from miniocr.generate import generate, process_logits as old_process_logits
from miniocr.models.gpt import GPT, GPTConfig
from miniocr.train.pretrain import wilson_interval
from book.part2_lm.ch14_decoding import tour

FIVE = torch.tensor([[2., 1., .5, 0., -1.]])
PREV = torch.tensor([[0, 4]])
CFG = dict(vocab_size=40, block_size=64, n_layer=2, n_head=4, d_model=32)


@pytest.fixture(scope="module", autouse=True)
def two_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def kept(row):
    return torch.isfinite(row)[0].nonzero().flatten().tolist()


def test_five_token_filters_and_no_mutation():
    original = FIVE.clone()
    penalized = d.apply_repetition_penalty(FIVE, PREV, 1.2)
    assert torch.allclose(penalized, torch.tensor([[2/1.2, 1., .5, 0., -1.2]]))
    scaled = d.apply_temperature(penalized, .8)
    assert torch.allclose(scaled, torch.tensor([[2/1.2/.8, 1.25, .625, 0., -1.5]]))
    assert kept(d.top_k_filter(scaled, 4)) == [0, 1, 2, 3]
    assert kept(d.top_p_filter(d.top_k_filter(scaled, 4), .8)) == [0, 1]
    assert kept(d.min_p_filter(scaled, .1)) == [0, 1, 2, 3]
    assert torch.equal(FIVE, original)
    for out in (d.top_k_filter(FIVE, 0), d.top_k_filter(FIVE, -2), d.top_k_filter(FIVE, 9),
                d.top_p_filter(FIVE, 1), d.min_p_filter(FIVE, 0),
                d.apply_repetition_penalty(FIVE, PREV, 1),
                d.apply_repetition_penalty(FIVE, PREV[:, :0], 2)):
        assert torch.equal(out, FIVE) and out.data_ptr() != FIVE.data_ptr()


def test_top_k_ties_and_top_p_original_indices():
    assert kept(d.top_k_filter(torch.tensor([[2., 1., 1., 0.]]), 2)) == [0, 1, 2]
    assert kept(d.top_k_filter(torch.zeros(1, 5), 2)) == list(range(5))
    scrambled = FIVE[:, [4, 2, 0, 3, 1]]
    assert kept(d.top_p_filter(scrambled, .8)) == [1, 2, 4]
    assert kept(d.top_p_filter(scrambled, 1e-9)) == [2]
    assert kept(d.top_p_filter(scrambled, 1e-9, min_keep=3)) == [1, 2, 4]
    # 二进制可精确表示的边界，恰好等于 p 时不多留下一个。
    exact = torch.tensor([[.5, .25, .25]], dtype=torch.float64).log()
    assert kept(d.top_p_filter(exact, .5)) == [0]
    assert kept(d.min_p_filter(torch.tensor([[.5, .25, .125, .125]]).log(), .5)) == [0, 1]
    assert kept(d.min_p_filter(FIVE, 1.)) == [0]


def test_repetition_sign_duplicate_ids_and_encouragement():
    logits = torch.tensor([[2., -1., 0., 3.]])
    prev = torch.tensor([[0, 0, 1, 1, 2]])
    correct = d.apply_repetition_penalty(logits, prev, 2.)
    assert torch.equal(correct, torch.tensor([[1., -2., 0., 3.]]))
    assert (correct <= logits).all()  # penalty>=1：正确写法让 logit 不增。
    assert logits[0, 1] / 2 > logits[0, 1]  # 故意错：负数也除，会变大。
    assert torch.equal(d.apply_repetition_penalty(logits, prev, .5), torch.tensor([[4., -.5, 0., 3.]]))


@pytest.mark.parametrize("fn,args", [
    (d.apply_repetition_penalty, (FIVE, PREV, 0)),
    (d.apply_repetition_penalty, (FIVE, PREV, -1)),
    (d.apply_temperature, (FIVE, 0)), (d.apply_temperature, (FIVE, -1)),
    (d.top_p_filter, (FIVE, 0)), (d.top_p_filter, (FIVE, 1.1)),
    (d.top_p_filter, (FIVE, .8, 0)), (d.top_k_filter, (FIVE, 1.5)),
    (d.min_p_filter, (FIVE, -.1)), (d.min_p_filter, (FIVE, 1.1)),
])
def test_filter_validation(fn, args):
    with pytest.raises(ValueError):
        fn(*args)


@pytest.mark.parametrize("kwargs", [dict(temperature=0), dict(temperature=-1), dict(temperature=math.nan),
    dict(top_p=0), dict(top_p=1.1), dict(min_p=-.1), dict(min_p=math.nan),
    dict(repetition_penalty=0), dict(repetition_penalty=math.inf), dict(top_k=1.2)])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        d.DecodeConfig(**kwargs)
    with pytest.raises(ValueError, match="要贪心请用 greedy"):
        d.apply_temperature(FIVE, 0)


def test_order_trace_copies_and_temperature_changes_nucleus():
    trace = []
    out = d.process_logits(FIVE, d.DecodeConfig(temperature=.8, top_k=4, top_p=.8,
                                               repetition_penalty=1.2), PREV, trace)
    assert trace[0][0] == "原始"
    assert tuple(name for name, _ in trace[1:]) == d.PROCESS_ORDER
    assert kept(out) == [0, 1]
    trace[-1][1].zero_()
    assert not torch.equal(out, trace[-1][1]) and torch.equal(trace[0][1], FIVE)
    assert kept(d.top_p_filter(d.apply_temperature(FIVE, 2), .8)) == [0, 1, 2, 3]
    assert kept(d.apply_temperature(d.top_p_filter(FIVE, .8), 2)) == [0, 1, 2]


@pytest.mark.parametrize("temperature,top_k,top_p,penalty", [
    (1., 0, 1., 1.), (.5, 4, .8, 1.2), (2., 0, .95, 2.),
    (.8, 100, .01, .8), (1.3, 1, .7, 1.2), (1., -1, .6, 1.),
])
def test_process_matches_existing(temperature, top_k, top_p, penalty):
    g = torch.Generator().manual_seed(83)
    logits = torch.randn(7, 23, generator=g)
    prev = torch.randint(23, (7, 15), generator=g)
    cfg = d.DecodeConfig(temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=penalty)
    for history in (None, prev, prev[:, :0]):
        assert torch.allclose(d.process_logits(logits, cfg, history),
                              old_process_logits(logits, temperature, top_k, top_p, penalty, history))


def test_greedy_argmax_and_sampling_frequency():
    for cfg in (d.DecodeConfig(temperature=.1, top_k=1, top_p=.01),
                d.DecodeConfig(temperature=2, top_k=4, top_p=.95, min_p=.1)):
        assert torch.equal(d.process_logits(FIVE, cfg).argmax(-1), FIVE.argmax(-1))
    cfg = d.DecodeConfig(greedy=True, temperature=2, top_k=1, top_p=.01)
    trace = []
    d.process_logits(FIVE, cfg, trace=trace)
    assert [name for name, _ in trace] == ["原始", "重复惩罚"]
    assert torch.equal(d.select_next(FIVE, cfg), FIVE.argmax(-1))
    assert d.select_next(torch.ones(1, 5), cfg).item() == 0
    assert d.select_next(FIVE, d.DecodeConfig(greedy=True, repetition_penalty=3), PREV).item() == 1
    p = torch.tensor([.1, .2, .3, .4])
    samples = d.select_next(p.log().expand(20000, -1), d.DecodeConfig(),
                            generator=torch.Generator().manual_seed(11))
    assert ((samples.bincount(minlength=4)/20000)-p).abs().max() < .02


def test_sample_and_gpt_beam_one_match_old_greedy():
    torch.manual_seed(42)
    model = GPT(GPTConfig(**CFG))
    idx = torch.randint(40, (2, 4))
    expected = generate(model, idx, 6, greedy=True, use_cache=False)
    assert torch.equal(d.sample_sequences(model, idx, 6, d.DecodeConfig(greedy=True)), expected)
    h = d.beam_search(d.gpt_step_fn(model), idx[0].tolist(), 6, 1, eos=None)
    assert h.tokens == expected[0, 4:].tolist() and h.truncated and not h.finished
    assert not model.training
    g1, g2 = (torch.Generator().manual_seed(9) for _ in range(2))
    assert torch.equal(d.sample_sequences(model, idx, 4, d.DecodeConfig(), generator=g1),
                       d.sample_sequences(model, idx, 4, d.DecodeConfig(), generator=g2))


class ScriptedLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(block_size=2)
        self.inputs = []

    def forward(self, idx):
        assert not self.training and not torch.is_grad_enabled()
        self.inputs.append(idx.clone())
        # row 0 立即结束；row 1 先生成 1，下一步结束。EOS 之后本会生成 0。
        nxt = torch.where(idx[:, -1] == 0, 1, torch.where(idx[:, -1] == 2, 0, 2))
        logits = torch.full((len(idx), idx.shape[1], 4), -torch.inf)
        logits[torch.arange(len(idx)), -1, nxt] = 0
        return logits, None


def test_eos_padding_early_stop_and_window():
    model = ScriptedLM()
    idx = torch.tensor([[3, 3, 1], [3, 3, 0]])
    out = d.sample_sequences(model, idx, 5, d.DecodeConfig(greedy=True), eos=2)
    assert out.tolist() == [[3, 3, 1, 2, 2], [3, 3, 0, 1, 2]]
    assert len(model.inputs) == 2 and all(x.shape[1] == 2 for x in model.inputs)
    assert torch.equal(model.inputs[0], idx[:, -2:])


def test_repetition_uses_prompt_outside_cropped_window_and_banned():
    class ConstantLM(torch.nn.Module):
        cfg = SimpleNamespace(block_size=1)

        def forward(self, idx):
            return torch.tensor([[[3., 2., 1.]]]).expand(len(idx), idx.shape[1], -1), None

    model = ConstantLM()
    idx = torch.tensor([[0, 2]])
    out = d.sample_sequences(model, idx, 1, d.DecodeConfig(greedy=True, repetition_penalty=2))
    assert out[0, -1] == 1  # token 0 虽已不在窗口内，仍受惩罚。
    out = d.sample_sequences(model, idx, 1, d.DecodeConfig(greedy=True), banned=[0, 1])
    assert out[0, -1] == 2


def test_short_bias_hand_calculation():
    toy = d.short_bias_toy()
    assert toy.vocab == ["<bos>", "<eos>", "x", "^", "2"]
    greedy = d.beam_search(toy.step_fn, [0], 4, 1, 1)
    assert greedy.tokens == [2, 3, 4, 1]
    prefix = [0]
    for token in greedy.tokens:
        assert token == toy.logits(prefix).argmax().item()
        prefix.append(token)
    assert math.exp(greedy.logprob) == pytest.approx(.2268)
    for search in (lambda a: d.beam_search(toy.step_fn, [0], 4, 2, 1, a),
                   lambda a: d.brute_force_search(toy.step_fn, [0], 4, 1, a)):
        short, long = search(0), search(1)
        assert short.tokens == [1] and short.logprob == pytest.approx(math.log(.3))
        assert long.tokens == [2, 3, 4, 1] and long.score == pytest.approx(math.log(.2268)/1.5)
        assert long.length == 4 and long.finished and not long.truncated
    assert math.log(.2268)/1.5 > math.log(.28)/(7/6) > math.log(.3)
    assert d.gnmt_length_penalty(4, 1) == 1.5
    assert d.gnmt_length_penalty(2, .6) == pytest.approx((7/6)**.6)
    assert d.gnmt_length_penalty(99, 0) == 1


@pytest.mark.parametrize("alpha", [0., .6])
def test_wide_beam_equals_exhaustive_random_bigrams(alpha):
    g = torch.Generator().manual_seed(17)
    names = ["bos", "eos", "a", "b"]
    for _ in range(5):
        p = torch.rand(4, 4, generator=g, dtype=torch.float64)
        p /= p.sum(-1, keepdim=True)
        toy = d.BigramToyLM({a: {b: float(p[i, j]) for j, b in enumerate(names)}
                            for i, a in enumerate(names)})
        wide = d.beam_search(toy.step_fn, [0], 4, 4**4, 1, alpha, return_all=True)
        assert wide[0] == d.brute_force_search(toy.step_fn, [0], 4, 1, alpha)
        assert [h.score for h in wide] == sorted((h.score for h in wide), reverse=True)
        assert all(h.length == len(h.tokens) and h.finished != h.truncated for h in wide)
        assert any(h.truncated for h in wide)
        assert sum(math.exp(h.logprob) for h in wide) == pytest.approx(1.)


def test_finished_never_expanded_and_eos_takes_slot():
    calls = []

    def step(prefixes):
        calls.extend(prefixes)
        assert all(prefix[-1] != 1 for prefix in prefixes)
        return torch.tensor([[.1, .9]]).log().expand(len(prefixes), -1)

    best = d.beam_search(step, [0], 4, 1, 1)
    assert best.tokens == [1] and calls == [[0]]
    calls.clear()
    d.beam_search(step, [0], 4, 16, 1)
    assert len(calls) == 4
    d.brute_force_search(step, [0], 4, 1)


def test_tour_expanding_eos_wrongly_wins_with_length_penalty():
    toy = d.short_bias_toy()
    prompt, eos = [toy.stoi["<bos>"]], toy.stoi["<eos>"]
    good = d.beam_search(toy.step_fn, prompt, 4, 2, eos, alpha=1.)
    bad = tour.wrong_expand_eos(toy, prompt, eos, alpha=1.)
    assert tour.toy_text(toy, good.tokens) == "x ^ 2 <eos>"
    assert bad.tokens == [eos] * 4
    assert good.length == bad.length == 4
    assert d.gnmt_length_penalty(bad.length, 1.) == 1.5
    assert good.logprob == pytest.approx(math.log(.2268))
    assert bad.logprob == pytest.approx(math.log(.3))
    assert good.score == pytest.approx(math.log(.2268)/1.5)
    assert bad.score == pytest.approx(math.log(.3)/1.5)
    assert bad.score > good.score


def test_tour_content_report_separates_prompt_from_continuation():
    bos, eos = 0, 1
    tok = SimpleNamespace(decode=lambda ids: "".join({2: "x", 3: "y", 4: "+",
                                                   5: "a", 6: "z", 7: "w", 8: "b"}[i] for i in ids))
    prompts = [[bos, 2], [bos, 3, 4], [bos, 6], [bos, 7]]
    rows = [prompts[0] + [5, 5, eos, eos], prompts[1] + [5, 5, eos],
            prompts[2] + [eos], prompts[3] + [8]]
    report, texts = tour._content_report(rows, prompts, tok, eos)
    assert texts == ["xaa", "y+aa", "z", "wb"]
    assert report["rate"] == 1. and report["unique_ratio"] == 1.
    assert report["mean_length"] == 5/4
    assert report["distinct_1"] == 2/5 and report["distinct_2"] == 1/2
    assert report["ended_with_eos"] == 3


def test_distinct_and_report_wilson():
    seqs = [[1, 2, 1], [1, 2]]
    assert d.distinct_n(seqs, 1) == 2/5 and d.distinct_n(seqs, 2) == 2/3
    assert d.distinct_n([[], [1]], 2) is None
    assert d.unique_ratio(["x", "y", "x"]) == 2/3 and d.unique_ratio([]) is None
    report = d.decoding_report(["x", "", "x"], [[1], [], [1]], [True, False, True])
    assert report["count"] == 3 and report["successes"] == 2 and report["rate"] == 2/3
    assert report["interval"] == wilson_interval(2, 3)
    assert report["ended_with_eos"] == 2 and report["mean_length"] == 2/3
    assert report["distinct_1"] == .5 and report["distinct_2"] is None
    empty = d.decoding_report([], [], [])
    assert empty["interval"] == (None, None) and empty["rate"] is None
    with pytest.raises(ValueError):
        d.decoding_report(["x"], [], [])


def test_speculative_analytic_zero_support_equal_and_random():
    g = torch.Generator().manual_seed(2)
    pairs = [(torch.tensor([.2, .3, .5]), torch.tensor([0., .7, .3])),
             (torch.tensor([.2, .3, .5]), torch.tensor([.2, .3, .5]))]
    for _ in range(8):
        p, q = torch.rand(7, generator=g), torch.rand(7, generator=g)
        pairs.append((p/p.sum(), q/q.sum()))
    for p, q in pairs:
        out, beta = d.speculative_output_distribution(p, q)
        assert torch.allclose(out, p, atol=1e-6)
        assert beta == pytest.approx(torch.minimum(p, q).sum().item())
    p = pairs[1][0]
    assert all(d.speculative_accept_reject(p, p, g)[1] for _ in range(50))


def test_speculative_monte_carlo():
    p, q = torch.tensor([.4, .25, .15, .15, .05]), torch.tensor([0., .35, .35, .05, .25])
    g = torch.Generator().manual_seed(12)
    counts, accepts = torch.zeros(5), 0
    for _ in range(20000):
        token, accepted = d.speculative_accept_reject(p, q, g)
        counts[token] += 1
        accepts += accepted
    assert (counts/20000-p).abs().max() < .02
    assert abs(accepts/20000 - float(torch.minimum(p, q).sum())) < .02


def test_tour_cli_and_fast_part1(capsys):
    args = tour.parse_args(["1", "6", "--seed", "3"])
    assert args.parts == [1, 6] and args.device == "cpu" and args.seed == 3
    assert args.samples == 50 and args.smoke_steps == 150
    for argv in (["0"], ["8"], ["--samples", "0"], ["--prompt-tokens", "-1"], ["--max-new", "0"]):
        with pytest.raises(SystemExit) as exc:
            tour.parse_args(argv)
        assert exc.value.code == 2
    tour.part1(tour.Experiment(args))
    output = capsys.readouterr().out
    assert "阶段=原始" in output and "阶段=top-p" in output
    assert "最终采样分布=" in output and "熵_nat=" in output and "min-p=0.1" in output
    assert "logit=2.0000" in output


def test_checkpoint_identity_load_and_hash_rejection(tmp_path, monkeypatch, capsys):
    from miniocr.data.latex_lm import train_tokenizer, encode_formulas
    tok = train_tokenizer(["x+y", "x^2"], num_merges=2)
    seqs, _ = encode_formulas(tok, ["x+y", "x^2"])
    data = SimpleNamespace(tokenizer=tok, val=seqs, train=seqs,
                           train_text=["x+y", "x^2"], val_text=["x+y", "x^2"])
    requested = []
    monkeypatch.setattr(tour, "load_latex_lm", lambda n: requested.append(n) or data)
    monkeypatch.setattr(tour, "corpus_available", lambda: False)
    monkeypatch.setattr(tour, "ROOT", tmp_path)
    cfg = GPTConfig(vocab_size=len(tok), block_size=8, n_layer=1, n_head=2, d_model=16)
    identity = dict(model=asdict(cfg), data_version="latex-v1/train/n=123",
                    tokenizer_hash=tour._tokenizer_digest(tok),
                    bos=tok.special_ids["<|bos|>"], eos=tok.special_ids["<|eos|>"])
    saved = dict(identity=identity, model=GPT(cfg).state_dict(), step=7)
    path = tmp_path / "checkpoint.pt"
    torch.save(saved, path)
    args = tour.parse_args(["5", "--ckpt", str(path), "--samples", "2", "--max-new", "2"])
    exp = tour.Experiment(args)
    tour.part5(exp)
    assert requested == [123] and exp.model.cfg == cfg
    assert (tmp_path / "decoding-seed0.json").is_file()
    output = capsys.readouterr().out
    assert "训练步数=7" in output and "降级为内置样例" in output
    assert output.count("mathtext可解析数=") == 9
    args.max_new = 9
    with pytest.raises(ValueError, match="block_size"):
        tour.part5(exp)
    identity["tokenizer_hash"] = "wrong"
    torch.save(saved, path)
    with pytest.raises(ValueError, match="哈希不一致"):
        tour.Experiment(args).prepare()
