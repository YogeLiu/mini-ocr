"""几秒级 CPU 合成测试：绝不调用 load_latex_lm。"""
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from miniocr.data.latex_lm import FormulaWindows, encode_formulas, evaluate_lm
from miniocr.data.token_file import StaleCacheError
from miniocr.models.gpt import GPT, GPTConfig, RMSNorm
from miniocr.tokenizer import BPETokenizer
from miniocr.train.memory import matmul_params, train_flops
from miniocr.train.optim import warmup_cosine
from miniocr.train.pretrain import (PretrainConfig, initial_loss, leakage_overlap, parse_report,
    plot_learning_curves, reference_parse_report, sample_report, summarize_log, token_stream,
    train_pretrain, validation_metrics, wilson_interval, window_logits)
from miniocr.train.scaling import (LADDER, HOFFMANN2022_APPROACH3, budget_steps, chinchilla_grid,
    chinchilla_optimal, compare_offset_fits, compute_frontier, fit_power_law, fit_power_law_offset,
    ladder_config, ladder_table, parameter_counts, synthetic_scaling)


@pytest.fixture(autouse=True, scope="module")
def two_threads():
    prior = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(prior)


def tiny(**kw):
    return replace(GPTConfig(vocab_size=13, block_size=8, n_layer=2, d_model=16, n_head=2, d_ff=32), **kw)


def test_defaults_keys_and_legacy_rope():
    cfg = tiny()
    assert (cfg.pos_emb, cfg.norm, cfg.resid_scale, cfg.rope, cfg.tie_weights) == ("rope", "rmsnorm", True, True, True)
    model = GPT(cfg)
    block_keys = ["n1.weight", "n2.weight", "attn.q.weight", "attn.k.weight", "attn.v.weight", "attn.o.weight",
                  "mlp.gate.weight", "mlp.up.weight", "mlp.down.weight"]
    expected = {"tok.weight", "head.weight", "norm.weight"}
    expected.update(f"blocks.{i}.{k}" for i in range(2) for k in block_keys)
    assert set(model.state_dict()) == expected
    assert isinstance(model.norm, RMSNorm)
    assert model.head.weight is model.tok.weight
    torch.manual_seed(23)
    old = GPT(tiny(rope=False))
    torch.manual_seed(23)
    new = GPT(tiny(pos_emb="none"))
    x = torch.arange(8).reshape(1, 8)
    assert torch.equal(old(x)[0], new(x)[0])
    assert torch.equal(old.cos, torch.ones_like(old.cos))
    assert torch.count_nonzero(old.sin) == 0
    assert not hasattr(GPT(tiny(rope=False, pos_emb="learned")), "pos")


@pytest.mark.parametrize("changes", [{"pos_emb": "learned"}, {"norm": "layernorm"}, {"resid_scale": False},
                                     {"pos_emb": "learned", "norm": "layernorm", "resid_scale": False}])
def test_switches_forward_backward_and_cache(changes):
    model = GPT(tiny(**changes)).eval()
    x = torch.tensor([[0, 3, 4, 5, 2, 7]])
    logits, loss = model(x, (x+1) % 13)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    cache = model.new_caches()
    parts = [model(x[:, :3], caches=cache)[0], model(x[:, 3:5], caches=cache, pos0=3)[0],
             model(x[:, 5:], caches=cache, pos0=5)[0]]
    torch.testing.assert_close(torch.cat(parts, 1), logits, atol=1e-6, rtol=1e-5)
    lengths = torch.tensor([4])
    torch.testing.assert_close(window_logits(model, x, lengths), logits[:, 3])
    if changes.get("norm") == "layernorm":
        assert model.norm.bias is not None and model.blocks[0].n1.bias is not None
    if changes.get("pos_emb") == "learned":
        assert model.n_params()-model.n_params(True) == (13+8)*16
        with torch.no_grad():
            torch.testing.assert_close(model(inputs_embeds=model.tok(x))[0], logits)


def test_residual_init_scaling_and_invalid_config():
    torch.manual_seed(8)
    scaled = GPT(tiny())
    torch.manual_seed(8)
    plain = GPT(tiny(resid_scale=False))
    for name, p in scaled.named_parameters():
        q = dict(plain.named_parameters())[name]
        torch.testing.assert_close(q, p*2 if name.endswith(("attn.o.weight", "mlp.down.weight")) else p)
    with pytest.raises(ValueError, match="pos_emb"):
        GPT(tiny(pos_emb="bad"))


def test_formula_parameter_counts_and_budgets():
    for size in LADDER:
        cfg = ladder_config(size, 19, 8)
        model = GPT(cfg)
        count = parameter_counts(cfg)
        assert count == dict(total=model.n_params(), non_embedding=model.n_params(True), matmul=matmul_params(model))
        assert train_flops(count["matmul"], 123) == 6*count["matmul"]*123
    for cfg in (tiny(pos_emb="learned", norm="layernorm", tie_weights=False), tiny(n_kv_head=1, qk_norm=True)):
        model = GPT(cfg)
        assert parameter_counts(cfg) == dict(total=model.n_params(), non_embedding=model.n_params(True), matmul=matmul_params(model))
    row = ladder_table(19, 17, 8)[0]
    assert row["compute"] == row["flops_per_token"]*17
    by_tokens = budget_steps(100, 2, 8, tokens=33)
    by_compute = budget_steps(100, 2, 8, compute=6*100*33)
    assert by_tokens == by_compute
    assert by_tokens["steps"] == 3 and by_tokens["actual_tokens"] == 48


def test_power_laws_and_frontier():
    x = np.geomspace(10, 1e7, 40)
    exact = fit_power_law(x, 12*x**(-.45))
    assert exact["a"] == pytest.approx(12, rel=1e-12)
    assert exact["alpha"] == pytest.approx(.45, rel=1e-12)
    for noise, tol in ((0, 1e-6), (.003, .03)):
        x, y = synthetic_scaling(a=25, alpha=.3, Linf=1.5, noise=noise, seed=7)
        fit = fit_power_law_offset(x, y)
        assert fit["a"] == pytest.approx(25, rel=tol)
        assert fit["alpha"] == pytest.approx(.3, rel=tol)
        assert fit["Linf"] == pytest.approx(1.5, rel=tol)
    comparison = compare_offset_fits(*synthetic_scaling())
    assert comparison["power"]["alpha"] < comparison["offset"]["alpha"]-.1
    points = [(1, 5), (1, 4), (2, 3), (10, 2), (20, 1.5), (100, 1)]
    assert compute_frontier(points, bins=None) == [(1., 4.), (2., 3.), (10., 2.), (20., 1.5), (100., 1.)]
    assert compute_frontier(points, bins=2) == [(2., 3.), (100., 1.)]
    assert compute_frontier([]) == []
    with pytest.raises(ValueError):
        fit_power_law([0, 1], [1, 2])


def test_chinchilla_closed_form_against_grid():
    p = HOFFMANN2022_APPROACH3
    assert p["alpha"]/(p["alpha"]+p["beta"])+p["beta"]/(p["alpha"]+p["beta"]) == pytest.approx(1)
    for c in (1e15, 1e18, 1e24):
        exact = chinchilla_optimal(c, **p)
        grid = chinchilla_grid(c, **p)
        assert 6*exact["N_opt"]*exact["D_opt"] == pytest.approx(c, rel=1e-14)
        assert exact["N_opt"] == pytest.approx(grid["N_opt"], rel=.002)
        assert exact["L_opt"] == pytest.approx(grid["L_opt"], rel=1e-6)


def fixture_data():
    seqs = [[0, 2, 3, 1], [0, 4, 1], [0, 6, 7, 8, 9, 2, 3, 4, 5, 6, 1]]
    stream = torch.tensor([t for s in seqs*30 for t in s])
    texts = ["x^2", "y", "abcdefghij"]
    return stream, seqs, texts


def test_training_resume_bitwise_logs_and_identity(tmp_path):
    ids, seqs, texts = fixture_data()
    cfg = PretrainConfig(steps=6, batch=4, context=4, accum=2, warmup=2, eval_every=2,
                         eval_formulas=1, final_eval_formulas=3, eval_batch=8, seed=4)
    def model():
        torch.manual_seed(92)
        return GPT(tiny(dropout=.2))
    def run(m, directory, **kw):
        return train_pretrain(m, ids, seqs, texts, cfg, bos=0, eos=1, run_dir=directory,
                              data_version="synthetic-v1", tokenizer_hash="synthetic", log=None, **kw)
    full, interrupted = model(), model()
    one = run(full, tmp_path/"full")
    run(interrupted, tmp_path/"resumed", stop_after=3)  # 非验证步，必须保留累计loss
    # 模拟日志领先于检查点，包含重复/半行内容；须按字节位置删除。
    with (tmp_path/"resumed"/"metrics.jsonl").open("a") as f:
        f.write('{"type":"train","step":4}\npartial')
    with (tmp_path/"resumed"/"evals.csv").open("a") as f:
        f.write("bad trailing line\n")
    torch.rand(31)
    restored = model()
    two = run(restored, tmp_path/"resumed", resume=True)
    for p, q in zip(full.parameters(), restored.parameters()):
        assert torch.equal(p, q)
    for a, b in zip(one["evals"], two["evals"]):
        assert {k: v for k, v in a.items() if k != "tok_per_sec"} == {k: v for k, v in b.items() if k != "tok_per_sec"}
    records = [json.loads(line) for line in (tmp_path/"resumed"/"metrics.jsonl").read_text().splitlines()]
    train = [r for r in records if r["type"] == "train"]
    assert [r["step"] for r in train] == list(range(1, 7))
    assert set(train[0]) >= {"step", "lr", "train_loss", "grad_norm", "tokens_seen", "elapsed_sec"}
    assert train[-1]["tokens_seen"] == 6*4*4
    assert set(two["evals"][-1]) >= {"train_loss", "val_nat_per_token", "val_bits_per_byte", "lr", "tok_per_sec"}
    assert two["evals"][0]["eval_bytes"] == len(texts[0].encode())
    assert two["evals"][-1]["eval_bytes"] == sum(len(t.encode()) for t in texts)
    for name in ("metrics.jsonl", "evals.csv", "config.json", "checkpoint.pt"):
        assert (tmp_path/"resumed"/name).exists()
    saved = torch.load(tmp_path/"resumed"/"checkpoint.pt", weights_only=False)
    assert set(saved) >= {"scaler", "rng", "step", "identity", "log_positions"}
    with pytest.raises(ValueError, match="training.steps"):
        train_pretrain(model(), ids, seqs, texts, replace(cfg, steps=7), bos=0, eos=1,
                       run_dir=tmp_path/"resumed", data_version="synthetic-v1", tokenizer_hash="synthetic", resume=True)
    with pytest.raises(ValueError, match="data_version"):
        train_pretrain(model(), ids, seqs, texts, cfg, bos=0, eos=1,
                       run_dir=tmp_path/"resumed", data_version="changed", tokenizer_hash="synthetic", resume=True)
    assert plot_learning_curves(tmp_path/"full").stat().st_size > 1000
    assert summarize_log(tmp_path/"full")["best_step"] in (2, 4, 6)


def test_validation_last_projection_byte_denominator_and_init():
    ids, seqs, texts = fixture_data()
    m = GPT(tiny(pos_emb="learned"))
    windows = FormulaWindows([s[1:-1] for s in seqs], 0, 1, 4)
    regular = evaluate_lm(m, windows, sum(len(t.encode()) for t in texts), batch=8)
    fast = validation_metrics(m, seqs, texts, 0, 1, 4, batch=8)
    assert fast["nat_per_token"] == pytest.approx(regular["nat_per_token"], abs=1e-6)
    assert fast["bits_per_byte"] == pytest.approx(regular["bits_per_byte"], abs=1e-6)
    assert fast["n_tokens"] == sum(len(s)-1 for s in seqs)
    state = torch.get_rng_state()
    init = initial_loss(m, ids, 0, PretrainConfig(batch=2, context=4))
    assert math.isfinite(init) and torch.equal(state, torch.get_rng_state())


def test_accumulation_uses_valid_target_weight(tmp_path):
    ids, seqs, texts = fixture_data()
    cfg = PretrainConfig(steps=1, warmup=0, batch=4, context=4, eval_batch=8)
    torch.manual_seed(4)
    full = GPT(tiny())
    micro = GPT(tiny())
    micro.load_state_dict(full.state_dict())
    for m, accum in ((full, 1), (micro, 2)):
        train_pretrain(m, ids, seqs, texts, replace(cfg, accum=accum), bos=0, eos=1,
                       run_dir=tmp_path/str(accum), data_version="toy", log=None)
    for p, q in zip(full.parameters(), micro.parameters()):
        torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)


def test_cache_schedule_summary_and_parse(tmp_path):
    tok = BPETokenizer([], specials=("<|bos|>", "<|eos|>"))
    texts = ["x^2", "y"]
    ids, meta = token_stream(tok, texts, "v1", tmp_path/"tokens")
    direct, _ = token_stream(tok, texts, "v1", use_file=False)
    assert torch.equal(ids, direct)
    assert meta["formula_count"] == 2
    assert torch.equal(token_stream(tok, texts, "v1", tmp_path/"tokens")[0], ids)
    with pytest.raises(StaleCacheError):
        token_stream(tok, texts, "v2", tmp_path/"tokens")
    other = BPETokenizer([], specials=("<|eos|>", "<|bos|>"))
    with pytest.raises(StaleCacheError):
        token_stream(other, texts, "v1", tmp_path/"tokens")
    lr = lambda s: warmup_cosine(s, 10, 2, .01, .001)
    assert [lr(0), lr(1), lr(2)] == pytest.approx([.005, .01, .01])
    assert lr(6) == pytest.approx(.0055) and lr(10) == .001
    assert lr(9) > lr(10) and lr(11) == .001
    records = [dict(type="train", step=i, train_loss=2., grad_norm=1., lr=.01) for i in range(1, 61)]
    records[50]["train_loss"] = 4.
    records[55]["grad_norm"] = float("nan")
    records.append(dict(type="eval", step=60, train_loss=2., val_nat_per_token=2.5))
    (tmp_path/"config.json").write_text(json.dumps(dict(initial_loss=3., ln_vocab=3.1)))
    (tmp_path/"metrics.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    summary = summarize_log(tmp_path)
    assert summary["spike_steps"] == [51] and summary["first_nonfinite_step"] == 56
    assert summary["final_train_minus_val"] == -.5
    report = parse_report(["x^2", r"\frac{"])
    assert report["count"] == 2 and report["rate"] == .5
    assert wilson_interval(0, 0) == (None, None)
    assert wilson_interval(0, 10)[0] == pytest.approx(0, abs=1e-15)
    assert wilson_interval(10, 10)[1] == pytest.approx(1)
    ref = reference_parse_report(["x^2", r"\begin{pmatrix}x\end{pmatrix}"])
    assert ref["matrix"]["count"] == 1 and ref["matrix"]["rate"] == 0
    assert ref["non_matrix"]["rate"] == 1
    assert leakage_overlap(["x^2", "y"], ["z"]) == 0
    assert leakage_overlap(["x^2", "y"], ["z", "x^2", "x^2"]) == 1
    m = GPT(GPTConfig(vocab_size=len(tok), n_layer=1, d_model=8, n_head=2, d_ff=16, block_size=4))
    before = torch.get_rng_state()
    report = sample_report(m, tok, count=2, max_new=3, greedy=True)
    assert report["count"] == 2 and len(report["examples"]) == 2
    assert m.training and torch.equal(before, torch.get_rng_state())


def test_tour_argparse():
    result = subprocess.run([sys.executable, "-m", "book.part2_lm.ch13_pretraining.tour", "11"],
                            text=True, capture_output=True)
    assert result.returncode == 2 and "部分号须为1–10" in result.stderr
    from book.part2_lm.ch13_pretraining.tour import parse_args
    for argv in (["--steps", "0"], ["--lr", "nan"], ["--warmup", "-1"], ["--samples", "0"],
                 ["--amp", "fp16"], ["--seed", "-1"]):
        with pytest.raises(SystemExit) as exc:
            parse_args(argv)
        assert exc.value.code == 2
    args = parse_args(["--preset", "cpu", "--steps", "10", "--warmup", "2"])
    assert args.steps == 10 and args.warmup == 2 and args.context == 32


def test_ngram_landmarks_hand_calculation():
    from miniocr.train.pretrain import ngram_landmarks
    # 可预测类别 {EOS=1,a=2,b=3}；训练 a a EOS / b EOS。
    ids = torch.tensor([0, 2, 2, 1, 0, 3, 1])
    seqs, texts = [[0, 2, 3, 1], [0, 3, 1]], ["ab", "中"]
    out = ngram_landmarks(ids, seqs, texts, 0, 1, vocab_size=4)
    assert out["H1"] == pytest.approx(-2*(2/5)*math.log(2/5)-(1/5)*math.log(1/5))
    # 加一：p(a)=p(EOS)=3/8, p(b)=2/8；BOS→a/b=2/5, a→b=1/5, b→EOS=2/4。
    uni = -3*math.log(3/8)-2*math.log(2/8)
    bi = -2*math.log(2/5)-math.log(1/5)-2*math.log(2/4)
    for name, nll in (("unigram", uni), ("bigram", bi)):
        assert out[name]["nat_per_token"] == pytest.approx(nll/5)
        assert out[name]["bits_per_byte"] == pytest.approx(nll/(math.log(2)*5))
    assert (out["train_targets"], out["n_tokens"], out["n_bytes"], out["predictable_classes"]) == (5, 5, 5, 3)
    # 未见类别和未见上文；平滑不使用验证频数，空公式仍预测 EOS。
    unseen = ngram_landmarks(ids, [[0, 4, 1]], ["z"], 0, 1, k=.5, vocab_size=5)
    assert unseen["unigram"]["nat_per_token"] == pytest.approx(-math.log((.5/7)*(2.5/7))/2)
    assert unseen["bigram"]["nat_per_token"] == pytest.approx(-math.log((.5/4)*(.5/2))/2)
    empty = ngram_landmarks(ids, [[0, 1]], [" "], 0, 1, vocab_size=4)
    assert empty["bigram"]["nat_per_token"] == pytest.approx(-math.log(1/5))
    with pytest.raises(ValueError):
        ngram_landmarks(ids, seqs, texts, 0, 1, k=0)


def test_fraction_config_and_attention_budget():
    from miniocr.train.scaling import fraction_config
    cfg = PretrainConfig(steps=60, warmup=6)
    assert [(fraction_config(cfg, f).steps, fraction_config(cfg, f).warmup)
            for f in (.25, .5, 1)] == [(15, 2), (30, 3), (60, 6)]
    scaled = fraction_config(replace(cfg, steps=1, warmup=0), .25)
    assert scaled.steps == scaled.warmup == 1 and scaled.lr == cfg.lr
    for frac in (0, -1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            fraction_config(cfg, frac)
    row = ladder_table(19, 17, 8)[0]
    assert row["attention_extra_ratio"] == row["layers"]*8*row["dim"]/row["matmul"]


def test_read_ladder_final_points(tmp_path, monkeypatch, capsys):
    import csv
    from miniocr.train.scaling import read_ladder_points
    def run(size, frac, steps, *, context=4, val_hash="same", denom=3, final=True, loss=2.0):
        path = tmp_path/"ladder"/size/f"frac{frac:g}"
        path.mkdir(parents=True)
        config = dict(training=dict(steps=steps, batch=2, context=context, final_eval_formulas=3),
                      tokenizer_hash="tok", train_hash="train", val_hash=val_hash, matmul_params=100)
        (path/"config.json").write_text(json.dumps(config))
        row = dict(step=steps, tokens_seen=steps*2*context, eval_formulas=denom,
                   eval_tokens=10, eval_bytes=20, val_nat_per_token=loss)
        with (path/"evals.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(dict(row, step=1, tokens_seen=2*context, val_nat_per_token=.01))
            if final:
                writer.writerow(row)
    # 未完成 run 不建立参考协议；周期低 loss 不能混入。
    run("aaa", .25, 2, val_hash="wrong", final=False)
    run("small", .25, 2)
    run("small", .5, 4, loss=1.8)
    run("small", 1, 8, loss=1.6)
    run("tiny", .25, 2, context=8)
    run("tiny", .5, 4, denom=2)
    run("tiny", 1, 8, val_hash="different")
    messages = []
    rows = read_ladder_points(tmp_path/"ladder", log=messages.append)
    assert [(r["size"], r["frac"], r["D"], r["C"], r["val_nat_per_token"]) for r in rows] == [
        ("small", .25, 16, 9600, 2.), ("small", .5, 32, 19200, 1.8), ("small", 1., 64, 38400, 1.6)]
    assert len(messages) == 4
    from types import SimpleNamespace
    from book.part2_lm.ch13_pretraining import tour
    monkeypatch.setattr(tour, "ROOT", tmp_path)
    tour.part7(SimpleNamespace(args=SimpleNamespace(seed=0)))
    assert (tmp_path/"ladder_scaling.png").stat().st_size > 1000
    output = capsys.readouterr().out
    assert output.count("阶梯最终点=") == 3 and "阶梯偏移拟合=" in output
    assert len(compute_frontier([(r["C"], r["val_nat_per_token"]) for r in rows])) == 3


def test_tour_modes_and_scientific_output(capsys):
    from book.part2_lm.ch13_pretraining.tour import emit, parse_args
    assert parse_args([]).check_resume
    assert not parse_args(["--preset", "cpu"]).check_resume
    assert not parse_args(["--steps", "10"]).check_resume
    assert parse_args(["--steps", "10", "--check-resume"]).check_resume
    args = parse_args(["3", "--resume", "/tmp/example"])
    assert args.resume == "/tmp/example" and not args.check_resume
    assert args.ladder_fracs == [.25, .5, 1.]
    for argv in (["--ladder-fracs", "0,1"], ["--ladder-fracs", "nan"], ["--ladder-fracs", "1,1"],
                 ["--resume", "x", "--check-resume"], ["4", "--resume", "x"]):
        with pytest.raises(SystemExit):
            parse_args(argv)
    emit(lr=warmup_cosine(0, 5000, 200, .003), 正确lr=warmup_cosine(1, 5000, 200, .003),
         C=1e23, N_opt=100, D_opt=1e21, compute_6ND=10**20, flops_per_token=12, loss=2.1)
    assert capsys.readouterr().out == ("lr=1.5000e-05；正确lr=3.0000e-05；C=1.0000e+23；N_opt=1.0000e+02；"
                                      "D_opt=1.0000e+21；compute_6ND=1.0000e+20；flops_per_token=1.2000e+01；loss=2.1000\n")


def test_tour_real_resume_and_single_run(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from book.part2_lm.ch13_pretraining import tour
    monkeypatch.setattr(tour, "ROOT", tmp_path)
    ids, seqs, texts = fixture_data()
    common = ["3", "--steps", "4", "--batch", "2", "--context", "4", "--warmup", "1", "--eval-every", "2"]
    def experiment(options):
        exp = tour.Experiment(tour.parse_args(common+options))
        # 使用真实 CLI/Experiment/训练恢复路径，仅以人造语料替换加载。
        exp.data = SimpleNamespace(val=seqs, val_text=texts, tokenizer=list(range(13)))
        exp.ids, exp.bos, exp.eos = ids, 0, 1
        exp.version, exp.meta = "toy", {"tokenizer_hash": "toy"}
        def model():
            torch.manual_seed(0)
            return GPT(tiny())
        exp.model = model
        return exp
    exp = experiment(["--run-dir", str(tmp_path/"new")])
    calls = []
    train = exp.train
    def recording_train(*args, **kw):
        calls.append(kw)
        return train(*args, **kw)
    exp.train = recording_train
    tour.part3(exp)
    assert calls == [{"resume": False}]
    partial = experiment(["--run-dir", str(tmp_path/"partial")])
    partial.train(partial.model(), partial.run_dir, stop_after=2, log=None)
    resumed = experiment(["--resume", str(partial.run_dir)])
    tour.part3(resumed)
    a = torch.load(exp.run_dir/"checkpoint.pt", weights_only=False)
    b = torch.load(resumed.run_dir/"checkpoint.pt", weights_only=False)
    assert b["step"] == 4
    assert all(torch.equal(a["model"][key], value) for key, value in b["model"].items())
    records = [json.loads(line) for line in (resumed.run_dir/"metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records if r["type"] == "train"] == [1, 2, 3, 4]
