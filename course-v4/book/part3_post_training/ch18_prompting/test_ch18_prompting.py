"""CPU 快速单元测试；唯一真实模型冒烟测试只生成两个 token，不训练。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataclasses import FrozenInstanceError
from pathlib import Path
import random

import pytest
import torch

from miniocr.data.latex_corpus import SAMPLE_FORMULAS, standard_split
from miniocr.data.postcorrect import (
    DEFAULT_NOISE, ERROR_TYPES, NoiseConfig, NoiseEdit, PostCorrectExample, build_dataset,
    build_split, check_no_leak, corrupt, error_type_counts, few_shot_pool, load_jsonl,
    make_example, save_jsonl,
)
from miniocr.eval.metrics import error_rate
from miniocr.eval.postcorrect_eval import (
    copy_baseline, em_ignoring_final_punct, evaluate, extract_formula, format_issues,
    oracle_baseline, score_postcorrect, strip_final_punct_tokens,
)
from miniocr.models.qwen3 import find_local_qwen3, load_qwen3
from miniocr.prompting import (
    PARAPHRASES, STRICT, few_shot, few_shot_strict, generate_greedy, render_chat,
    render_completion, zero_shot,
)
from miniocr.train.pretrain import wilson_interval

FORMULAS = SAMPLE_FORMULAS + [rf"x_{{{i}}}+\frac{{a}}{{b}}" for i in range(100)]
KWARGS = dict(formulas=FORMULAS, n_val=12, n_test=12)


def test_deterministic_identity_and_prefix():
    texts = list(enumerate(FORMULAS))
    forward = {i: make_example(s, "val", i, 19) for i, s in texts}
    backward = {i: make_example(s, "val", i, 19) for i, s in reversed(texts)}
    assert forward == backward
    assert make_example(texts[4][1], "val", 4, 19) == forward[4]
    small = build_split("train", 5, **KWARGS)
    assert small == build_split("train", 10, **KWARGS)[:5]
    assert [e.target for e in small] == standard_split(n_train=None, **KWARGS)["train"][:5]
    assert [e.target for e in build_split("val", seed=4, **KWARGS)] == [e.target for e in build_split("val", seed=8, **KWARGS)]


@pytest.mark.parametrize("kind,text", tuple(zip(ERROR_TYPES, (
    r"lO1\times\cdot-", r"x^{2}", r"\alpha", r"\frac{a}{b}", r"x^{2}", "x+y"))))
def test_each_noise(kind, text):
    ex = make_example(text, "val", 0, config=NoiseConfig(0, {kind: 1}, 1))
    assert ex.error_types == (kind,)
    assert ex.source != ex.target
    edit = ex.errors[0]
    assert text[:edit.position] + edit.after + text[edit.position+len(edit.before):] == ex.source


def test_noise_boundaries_and_missing_nested_argument():
    assert corrupt(r"\alpha", random.Random(0), NoiseConfig(0, {"confusable": 1}, 1)) == (r"\alpha", ())
    source, edits = corrupt(r"\frac{a}{\sqrt{x}}", random.Random(0), NoiseConfig(0, {"missing_arg": 1}, 1))
    assert source in (r"\frac{a}", r"\frac{a}{\sqrt}")
    assert edits
    assert corrupt(r"\{x\}", random.Random(0), NoiseConfig(0, {"drop_brace": 1}, 1))[1] == ()
    assert corrupt("xyz", random.Random(0), NoiseConfig(0, {"missing_arg": 1}, 1))[1] == ()


def test_invariant_and_clean_fraction():
    cfg = NoiseConfig(.2, {"extra_space": 1}, 1)
    examples = [make_example("x+y", "val", i, config=cfg) for i in range(1000)]
    assert .15 < sum(e.is_clean for e in examples)/len(examples) < .25
    for ex in examples + [make_example(s, "val", i) for i, s in enumerate(FORMULAS)]:
        assert ex.is_clean == (ex.source == ex.target) == (ex.errors == ())
        assert len(ex.errors) <= DEFAULT_NOISE.max_errors
    with pytest.raises(FrozenInstanceError):
        examples[0].source = "changed"
    with pytest.raises(TypeError):
        DEFAULT_NOISE.rates["confusable"] = 1
    with pytest.raises(ValueError):
        PostCorrectExample("bad", "val", "x", "y")
    with pytest.raises(ValueError):
        PostCorrectExample("bad", "val", "x", "y", (NoiseEdit("confusable", 0, "y", "z"),))


def test_splits_pool_and_json(tmp_path):
    data = build_dataset(formulas=FORMULAS, split_n_val=12, split_n_test=12)
    assert all(data.values())
    assert check_no_leak(data)["ok"]
    contaminated = {**data, "test": data["test"] + data["train"][:1]}
    assert not check_no_leak(contaminated)["ok"]
    pool = few_shot_pool(3, **KWARGS)
    assert all(e.split == "train" and not e.is_clean for e in pool)
    assert pool == few_shot_pool(5, **KWARGS)[:3]
    assert len({e.id for e in pool}) == 3
    assert error_type_counts(pool).keys() == set(ERROR_TYPES)
    path = tmp_path / "examples.jsonl"
    save_jsonl(pool, path)
    assert load_jsonl(path) == pool


@pytest.mark.parametrize("text,expected", [
    ("$x^2$", "x^2"), ("$$x^2$$", "x^2"), (r"\[x^2\]", "x^2"), (r"\(x^2\)", "x^2"),
    ("```latex\nx^2\n```", "x^2"), ("<think>reason\nmore</think>\n$x^2$", "x^2"),
    ("Corrected: $x^2$", "x^2"), ("Output: x^2\nExplanation follows", "x^2"),
    ("修正后：x^2", "x^2"), ("\n\nx^2\nBecause...", "x^2"), ("", ""),
    ("<think>unfinished", ""), ("$$\nx^2\n$$", "x^2"),
])
def test_extract(text, expected):
    assert extract_formula(text) == expected


def test_extract_saved_n100_outputs():
    """真实 val 输出的快照，不依赖 scratch 路径或再次调用模型。"""
    import json
    rows = json.loads(Path(__file__).with_name("fixtures").joinpath("extraction_n100.json").read_text())
    for row in rows:
        assert extract_formula(row["pred"]) == row["expected"], row["case"]


@pytest.mark.parametrize("punctuation", [".", ",", ";", "。"])
def test_extract_real_output_trailing_punctuation(punctuation):
    formula = r"\beta_U(H) = \sum_{\substack{i < j \\ i, j \in U}} \beta^{ij}(H)"
    assert extract_formula(f"${formula}${punctuation}") == formula


def test_extract_real_zero_shot_explanation():
    formula = r"\beta_U(H) = \sum_{\substack{i < j \\ i, j \in U}} \beta^{ij}(H)"
    text = ("The LaTeX formula you provided is already correct. However, if you want to ensure "
            "OCR errors are corrected, here's the properly formatted formula:\n\n"
            f"$$\n{formula}\n$$\n\nLet me know if you need further adjustments!")
    assert extract_formula(text) == formula


@pytest.mark.parametrize("prefix", ["", "The corrected formula:\n\n", "formula = x\n"])
def test_extract_multiline_first_block(prefix):
    # 块外含文字（包括裸等式）时只取首块，否则连接所有块。
    expected = "= x + y" if prefix else "= x + y = z"
    assert extract_formula(prefix + "$$\n= x +\n y\n$$\n$$\n= z\n$$") == expected


@pytest.mark.parametrize("text,expected", [
    (r"$inline$ \(round\) \[display\] $$first$$ $$second$$", "inline round display first second"),
    (r"$inline$ \(round\) \[first\] \[second\]", "inline round first second"),
    (r"$inline$ \(first\) \(second\)", "inline first second"),
    (r"Before $inline$ then $$display$$", "inline"),
    ("$x$,\n$y$. ; 。\\[z\\]", "x y z"),
    (r"Before $first$ and $second$", "first"),
    ("before \\[x +\ny\\] after", "x + y"),
    ("before \\(x +\ny\\) after", "x + y"),
    ("before $x +\ny$ after", "x + y"),
    (r"\$price\$ then $x + \$5$.", r"x + \$5"),
    (r"\$x\$", r"\$x\$"),
    (r"\\$x$", "x"),  # 两个反斜杠是换行命令，不转义后面的美元符。
    (r"\\[not a delimiter\\] $x$", "x"),
    ("<think>$$wrong$$</think>\n```latex\nExplanation\n$$\nx +\ny\n$$\n```", "x + y"),
    ("Output: x+y.\nExplanation", "x+y"),
    ("Corrected: Output: x+y,", "x+y"),
    (r"$x\,.$", r"x\,"),
    (r"\left(x\right.", r"\left(x\right."),
    ("Here is the corrected formula.\nCorrected: x^2\n+y\n这是修正后的公式。", "x^2 +y"),
    ("Corrected:\n\\begin{aligned}\nx &= y \\\\\n&= z\n\\end{aligned}",
     r"\begin{aligned} x &= y \\ &= z \end{aligned}"),
    ("Output: x+y\n\\text{Here are three words}\nNote that x^2 is preserved",
     r"x+y \text{Here are three words} Note that x^2 is preserved"),
])
def test_extract_delimiters_and_fallback(text, expected):
    assert extract_formula(text) == expected


@pytest.mark.parametrize("ending", [r"\ldots", r"\cdot", r"\.", r"\,", r"\;", r"\。", r"\\"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_extract_preserves_latex_endings(ending, wrapped):
    formula = "x" + ending
    assert extract_formula(f"${formula}$." if wrapped else formula) == formula


def test_format_issues():
    assert format_issues("  ") == ("empty",)
    issues = format_issues("<think>reason</think>\n```latex\n$x$\n```\nHere is the answer.")
    assert set(issues) == {"has_explanation", "dollar_wrapped", "code_fence", "think_block"}
    assert not format_issues(r"\frac{x}{y}")


@pytest.mark.parametrize("tokens,expected", [
    ([], []), ([".", ",", ";"], []),
    (["x", ",", "y", ".", ";", ","], ["x", ",", "y"]),
    (["x", r"\.", "."], ["x", r"\."]),
    (["x", r"\,", r"\;"], ["x", r"\,", r"\;"]),
    (["x", "。"], ["x", "。"]),
])
def test_strip_final_punct_tokens(tokens, expected):
    original = tokens.copy()
    assert strip_final_punct_tokens(tokens) == expected
    assert tokens == original


@pytest.mark.parametrize("target,pred,expected", [
    ("x . , ;", "x; .", True), ("x", "x,", True),
    (r"\dfrac12 .", r"\frac{1}{2},", True),
    ("x^2 .", "x^{2}", True), ("", ". , ;", True), ("", "", True),
    ("x,y.", "xy", False), ("1.2;", "12", False),
    ("x.", "y", False), ("x。", "x", False),
    (r"x\.", "x", False), (r"x\ldots", "x", False),
    ("x^{.}", "x^{}", False), ("{x.", "{x,", True),
])
def test_em_ignoring_final_punct(target, pred, expected):
    assert em_ignoring_final_punct(target, pred) is expected
    assert em_ignoring_final_punct(pred, target) is expected


def manual_examples():
    return [PostCorrectExample("a", "val", "x", "x"),
            PostCorrectExample("b", "val", "a", "b", (NoiseEdit("confusable", 0, "b", "a"),)),
            PostCorrectExample("c", "val", "x  +y", "x+y", (NoiseEdit("extra_space", 1, "", "  "),))]


def test_scoring_values_and_intervals():
    examples = manual_examples()
    preds = ["z", "b", "x  +y"]
    result = score_postcorrect(examples, preds, bootstrap_samples=50)
    s = result["summary"]
    assert s["em_raw"]["value"] == 1/3
    assert s["em_norm"]["value"] == 2/3
    assert s["broken_rate"]["value"] == 1
    assert s["broken_rate"]["count"] == 1
    assert s["fix_rate"]["value"] == 1
    assert s["fix_rate"]["count"] == 2
    assert s["mean_edit_distance"] == 1/3
    assert s["cer"]["value"] == pytest.approx(1/5)
    assert s["cer"]["value"] == error_rate([e.target for e in examples], preds, normalize=True)
    assert s["em_norm"]["interval"] == wilson_interval(2, 3)
    assert s["by_error_type"]["extra_space"]["count"] == 1
    assert score_postcorrect(examples, preds, bootstrap_samples=50) == result
    assert score_postcorrect(examples[1:], preds[1:])["summary"]["broken_rate"]["value"] is None
    assert score_postcorrect(examples[:1], preds[:1])["summary"]["fix_rate"]["value"] is None
    assert score_postcorrect([], [])["summary"]["em_raw"]["interval"] == (None, None)
    assert score_postcorrect(examples, preds, normalize=False)["summary"]["cer"]["value"] == error_rate([e.target for e in examples], preds, normalize=False)
    with pytest.raises(ValueError):
        score_postcorrect(examples, [])


def test_copy_oracle_multilabel_and_protocol():
    examples = manual_examples()
    assert copy_baseline(examples)["summary"]["broken_rate"]["value"] == 0
    assert oracle_baseline(examples)["summary"]["em_raw"]["value"] == 1
    assert oracle_baseline(examples)["summary"]["cer"]["value"] == 0
    ex = PostCorrectExample("d", "val", "1  ", "l", (
        NoiseEdit("confusable", 0, "l", "1"), NoiseEdit("extra_space", 1, "", "  ")))
    by_type = oracle_baseline([ex])["summary"]["by_error_type"]
    assert by_type["confusable"]["count"] == by_type["extra_space"]["count"] == 1
    with pytest.raises(ValueError, match="第 0.10.1 节"):
        evaluate(lambda _: pytest.fail("禁止调用预测"), split="test")
    result = evaluate(lambda sources: sources, split="test", allow_test=True, **KWARGS)
    assert result["summary"]["n"] >= 12
    assert all(r["split"] == "test" for r in result["rows"])


def test_messages_and_fake_template():
    pool = few_shot_pool(3, **KWARGS)
    assert [m["role"] for m in zero_shot("x")] == ["system", "user"]
    messages = few_shot("x", pool)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user", "assistant", "user"]
    assert messages[2]["content"] == pool[0].target
    assert STRICT in few_shot_strict("x", pool)[0]["content"]
    assert len(PARAPHRASES) >= 3 and len(set(PARAPHRASES)) == len(PARAPHRASES)
    assert render_completion("x", pool).endswith("Noisy: x\nCorrected:")
    with pytest.raises(ValueError, match="train"):
        few_shot("x", manual_examples())
    class FakeTokenizer:
        def apply_chat_template(self, msgs, **kwargs):
            assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
            return repr(msgs)
    assert render_chat(FakeTokenizer(), messages) == repr(messages)


def test_find_local_only(tmp_path):
    assert find_local_qwen3(tmp_path) is None
    snapshot = tmp_path / "models--Qwen--Qwen3-0.6B/snapshots/abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    assert find_local_qwen3(tmp_path) is None
    (snapshot / "model.safetensors").touch()
    assert find_local_qwen3(tmp_path) == snapshot
    with pytest.raises(FileNotFoundError):
        load_qwen3(tmp_path / "absent")


def test_local_tokenizer_template():
    path = find_local_qwen3()
    if path is None:
        pytest.skip("本地 Qwen3 tokenizer 缺失")
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    except OSError:
        pytest.skip("本地 Qwen3 tokenizer 文件缺失")
    text = render_chat(tok, zero_shot("x"))
    assert "<think>\n\n</think>\n\n" in text
    assert "<think>\n\n</think>\n\n" not in render_chat(tok, zero_shot("x"), True)
    assert tok.bos_token is None


def test_cpu_model_smoke():
    path = find_local_qwen3()
    if path is None:
        pytest.skip("本地 Qwen3 权重缺失")
    if not all((path / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json")):
        pytest.skip("本地 Qwen3 tokenizer 文件缺失")
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        model, tok = load_qwen3(path)
        assert not model.training
        assert next(model.parameters()).device.type == "cpu"
        sources = ["x", "x+y"]
        prompts = [render_chat(tok, zero_shot(s)) for s in sources]
        side = tok.padding_side
        result = generate_greedy(model, tok, prompts, max_new_tokens=2, batch_size=2, sources=sources)
        assert len(result.texts) == 2
        assert all(0 < n <= 2 for n in result.generated_tokens)
        assert result.prompt_tokens == tuple(len(tok.encode(p, add_special_tokens=False)) for p in prompts)
        assert tok.padding_side == side
        assert result.seconds > 0
    finally:
        torch.set_num_threads(old_threads)


def test_tour_lab_parts():
    from .tour import PARTS
    lab = Path(__file__).with_name("lab.md").read_text()
    assert set(PARTS) == set(range(1, 11))
    assert all(f"## {number}. " in lab for number in PARTS)


def test_tour_extraction_counts_and_deltas(monkeypatch):
    from types import SimpleNamespace
    from . import tour

    examples = [PostCorrectExample(str(i), "val", s, s) for i, s in enumerate(("x", "x.", "z"))]
    texts = ("$x$.", "x.", "wrong")  # 修好、改坏、仍错，各一条。
    raw = score_postcorrect(examples, texts, bootstrap_samples=0)
    extracted = score_postcorrect(examples, [extract_formula(s) for s in texts], bootstrap_samples=0)
    data = dict(result=SimpleNamespace(texts=texts), raw=raw, extracted=extracted)
    exp = SimpleNamespace(available=lambda: True, args=SimpleNamespace(n=3),
                          run=lambda *args: data, val=lambda n: examples[:n])
    output = []
    monkeypatch.setattr(tour, "emit", lambda **values: output.append(values))
    tour.part6(exp)
    assert "原始输出 em_norm 错、抽取后对" in output[0]["抽取修好定义"]
    assert "原始输出 em_norm 对、抽取后错" in output[0]["抽取改坏定义"]
    rows = [row for row in output if "格式问题" in row]
    assert len(rows) == 3
    for row in rows:
        assert row["抽取修好"] == row["抽取改坏"] == 1
        for metric in ("parse_rate", "em_norm"):
            before, after = raw["summary"][metric]["value"], extracted["summary"][metric]["value"]
            assert row[metric] == dict(抽取前=before, 抽取后=after, 变化=after-before)
    diagnostics = [row for row in output if "诊断" in row]
    assert len(diagnostics) == 4
    for row in diagnostics:
        assert row["诊断"] == "非官方指标，只用于错例分析；官方指标仍是 EM_norm"
        copy = row["方法"] == "copy"
        assert row["忽略句末标点的EM_norm"] == dict(
            value=1 if copy else 2/3, successes=3 if copy else 2,
            count=3, interval=wilson_interval(3 if copy else 2, 3))
        assert row["改坏样本中仅差句末标点的条数"] == (0 if copy else 1)
        assert row["改坏样本数"] == (0 if copy else 2)


def test_final_punct_diagnostic_broken_bucket_and_empty(monkeypatch):
    from . import tour
    examples = [PostCorrectExample("clean", "val", "x.", "x."),
                PostCorrectExample("noisy", "val", "x .", "x.",
                                   (NoiseEdit("extra_space", 1, "", " "),))]
    scored = score_postcorrect(examples, ["x", "x"], bootstrap_samples=0)
    output = []
    monkeypatch.setattr(tour, "emit", lambda **values: output.append(values))
    tour.report_final_punct_diagnostic("test", scored, scoring="extract_formula 后评分")
    assert output[-1]["忽略句末标点的EM_norm"]["successes"] == 2
    assert output[-1]["改坏样本中仅差句末标点的条数"] == 1
    # 诊断不覆盖官方评分；有噪声样本不算改坏。
    assert scored["summary"]["em_norm"]["successes"] == 0
    assert scored["summary"]["broken_rate"]["successes"] == 1
    tour.report_final_punct_diagnostic("empty", score_postcorrect([], []), scoring="空样本")
    assert output[-1]["忽略句末标点的EM_norm"] == dict(
        value=None, successes=0, count=0, interval=(None, None))
    assert output[-1]["改坏样本中仅差句末标点的条数"] == 0


@pytest.mark.parametrize("part,file_count", [(4, 2), (5, 3), (8, 3), (9, 3)])
def test_tour_out_sample_records(part, file_count, tmp_path, monkeypatch):
    """各入口实际经过 Experiment.run 的导出路径；伪生成器无需加载模型。"""
    import json
    from types import SimpleNamespace
    from . import tour

    examples = manual_examples()
    full_texts = ("$x$.\nThanks!", "$$b$$", "Explanation\n$$x+y$$")
    calls = []

    def generate(model, tok, prompts, **kwargs):
        assert kwargs["batch_size"] == 32
        calls.append(prompts)
        texts = tuple(s.split("\n", 1)[0] for s in full_texts) if kwargs["stop_at_newline"] else full_texts
        return SimpleNamespace(texts=texts, raw_texts=full_texts, prompt_tokens=(10, 11, 12),
                               generated_tokens=(3, 4, 5), length_capped=(False,) * 3, seconds=1.)

    monkeypatch.setattr(tour, "find_local_qwen3", lambda: tmp_path)
    monkeypatch.setattr(tour, "render_chat", lambda tok, messages: repr(messages))
    monkeypatch.setattr(tour, "generate_greedy", generate)
    exp = tour.Experiment(SimpleNamespace(n=3, seed=0, device="cpu", dtype="fp32", batch_size=32, sub_n=None, out=tmp_path / "out"))
    exp._val = examples
    monkeypatch.setattr(exp, "tok", lambda: object())
    monkeypatch.setattr(exp, "model", lambda: object())
    monkeypatch.setattr(exp, "pool", lambda **kwargs: few_shot_pool(3, **KWARGS))
    tour.PARTS[part][1](exp)
    # 复用生成缓存后，仍能读到完整且不重复的导出记录。
    tour.PARTS[part][1](exp)
    paths = list(exp.args.out.glob("*.jsonl"))
    assert len(paths) == len(calls) == file_count
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == len(examples)
        for row, ex, raw in zip(rows, examples, full_texts):
            assert (row["id"], row["source"], row["target"]) == (ex.id, ex.source, ex.target)
            assert row["error_types"] == list(ex.error_types)
            assert row["raw"] == raw
            assert row["pred"] == (raw.split("\n", 1)[0] if row["mode"] == "completion" else raw)
            assert row["extracted"] == extract_formula(row["pred"]) == row["score"]["pred"]
            assert row["generation_total_seconds"] == 1.
            assert row["batch_size"] == 32


@pytest.fixture
def saved_experiment(tmp_path, monkeypatch):
    """真实导出/读取路径配伪生成器；复用分支若加载模型或生成立即失败。"""
    import json
    from types import SimpleNamespace
    from . import tour

    monkeypatch.setattr(tour, "find_local_qwen3", lambda: tmp_path)
    monkeypatch.setattr(tour, "render_chat", lambda tok, messages: repr(messages))
    monkeypatch.setattr(tour.Experiment, "tok", lambda self: object())
    monkeypatch.setattr(tour.Experiment, "val", lambda self, n=None: manual_examples()[:n])
    monkeypatch.setattr(tour.Experiment, "pool", lambda self, **kwargs: few_shot_pool(3, **KWARGS))

    def make(*, reuse):
        return tour.Experiment(SimpleNamespace(n=3, seed=0, device="cpu", dtype="fp32", batch_size=4, sub_n=None, out=tmp_path, reuse=reuse))

    def forbid(*args, **kwargs):
        pytest.fail("复用已有文件时不能加载模型或生成")

    def save(mode="few_shot"):
        monkeypatch.setattr(tour.Experiment, "model", lambda self: object())
        monkeypatch.setattr(tour, "generate_greedy", lambda *a, **kw: SimpleNamespace(
            texts=("$x$", "b", "x+y"), raw_texts=("$x$\nExplanation", "b", "x+y"),
            prompt_tokens=(10, 11, 12), generated_tokens=(3, 4, 5),
            length_capped=(False, False, True), seconds=6.))
        original = make(reuse=True).run(mode, 3)  # --reuse 缺文件时允许生成。
        path = next(tmp_path.glob("*.jsonl"))
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        monkeypatch.setattr(tour.Experiment, "model", forbid)
        monkeypatch.setattr(tour, "generate_greedy", forbid)
        return make, path, rows, original

    return save


@pytest.mark.parametrize("mode", ["zero_shot", "few_shot", "few_shot_strict", "completion"])
@pytest.mark.parametrize("has_seconds", [True, False])
def test_reuse_saved_generation(saved_experiment, mode, has_seconds, capsys):
    import json
    from . import tour
    make, path, rows, original = saved_experiment(mode)
    for row in rows:
        row["extracted"] = "stale extraction"
        row["score"] = {"em_norm": False}
        if not has_seconds:
            del row["generation_total_seconds"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = path.read_bytes()
    data = make(reuse=True).run(mode, 3)
    assert data["reused_from"] == path
    assert data["raw"] == original["raw"]
    assert data["extracted"] == original["extracted"]
    result = data["result"]
    assert result.texts == original["result"].texts
    assert result.raw_texts == original["result"].raw_texts
    assert result.prompt_tokens == (10, 11, 12)
    assert result.generated_tokens == (3, 4, 5)
    assert result.length_capped == (False, False, True)
    assert result.seconds == (6. if has_seconds else None)  # 总秒数不能逐条累加。
    tour.report(mode, data)
    output = capsys.readouterr().out
    assert f"已保存生成（{path}）" in output and "原记录" in output
    assert '原生成用时=6.0' in output if has_seconds else '原生成用时="未记录"' in output
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["id", "prompt", "model_path", "dtype", "device"])
@pytest.mark.parametrize("missing", [True, False])
def test_reuse_rejects_identity_mismatch(saved_experiment, field, missing):
    import json
    make, path, rows, _ = saved_experiment()
    if missing:
        del rows[1][field]
    else:
        rows[1][field] = "different"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=f"第 2 条字段 {field}.*{'缺失' if missing else '不一致'}"):
        make(reuse=True).run("few_shot", 3)


@pytest.mark.parametrize("damage,match", [
    ("count", "记录数不一致"), ("json", "JSON 无效"),
    ("prompt_tokens", "prompt_tokens 缺失"), ("generated_tokens", "generated_tokens 缺失"),
    ("raw", "raw 缺失"), ("pred", "pred 缺失"), ("length_capped", "length_capped 缺失"),
    ("time_missing", "generation_total_seconds 各条记录不一致"),
    ("time_different", "generation_total_seconds 各条记录不一致"),
    ("time_negative", "generation_total_seconds 必须"),
])
def test_reuse_rejects_incomplete_records(saved_experiment, damage, match):
    import json
    make, path, rows, _ = saved_experiment()
    if damage == "count":
        rows.pop()
    elif damage == "time_missing":
        del rows[1]["generation_total_seconds"]
    elif damage == "time_different":
        rows[1]["generation_total_seconds"] = 7.
    elif damage == "time_negative":
        for row in rows:
            row["generation_total_seconds"] = -1.
    elif damage != "json":
        del rows[1][damage]
    path.write_text("broken\n" if damage == "json" else "".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=match):
        make(reuse=True).run("few_shot", 3)


def test_reuse_requires_out(capsys):
    from . import tour
    with pytest.raises(SystemExit, match="2"):
        tour.main(["--reuse"])
    assert "--reuse 必须搭配 --out" in capsys.readouterr().err


def test_part1_target_character_lengths(monkeypatch):
    from types import SimpleNamespace
    from . import tour
    examples = [PostCorrectExample(str(i), "val", s, s) for i, s in enumerate(("x", "x+y", r"\alpha"))]
    monkeypatch.setattr(tour, "target_splits", lambda: {"train": ["z"], "val": [e.target for e in examples], "test": ["w"]})
    exp = SimpleNamespace(args=SimpleNamespace(seed=0, n=1), val=lambda: examples)
    output = []
    monkeypatch.setattr(tour, "emit", lambda **values: output.append(values))
    tour.part1(exp)
    stats = next(row for row in output if row.get("统计范围") == "val 全集")
    assert stats["实际n"] == 3
    assert stats["目标公式字符长度均值"] == pytest.approx(10/3)
    assert stats["目标公式字符长度中位数"] == 3


def test_part1_final_punct_full_val(monkeypatch):
    from types import SimpleNamespace
    from . import tour
    texts = ("x.", "x,  ", "x;", "x", "x。", "y.")
    examples = [PostCorrectExample(str(i), "val", s, s) for i, s in enumerate(texts)]
    monkeypatch.setattr(tour, "target_splits", lambda: {"train": [], "val": texts, "test": []})
    output = []
    monkeypatch.setattr(tour, "emit", lambda **values: output.append(values))
    tour.part1(SimpleNamespace(args=SimpleNamespace(seed=0, n=1), val=lambda: examples))
    stats = next(row for row in output if row.get("统计范围") == "val 全集")
    assert stats["目标句末标点"] == {
        p: dict(count=count, total=6, value=count/6) for p, count in ((".", 2), (",", 1), (";", 1))}


def test_generation_padding_eos_counts_and_restore():
    """不用权重，验证批生成接口契约及异常后的状态恢复。"""
    from types import SimpleNamespace
    class Batch(dict):
        def to(self, device):
            return Batch({k: v.to(device) for k, v in self.items()})
    class Tokenizer:
        padding_side = "right"
        pad_token_id = 0
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return [1] * len(text)
        def __call__(self, batch, **kwargs):
            assert self.padding_side == "left"
            assert kwargs == dict(padding=True, return_tensors="pt", add_special_tokens=False)
            return Batch(input_ids=torch.tensor([[0, 1], [1, 1]]),
                         attention_mask=torch.tensor([[0, 1], [1, 1]]))
        def decode(self, ids, skip_special_tokens):
            assert skip_special_tokens
            return "x\nexplanation" if ids[0] == 7 else "y"
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(max_position_embeddings=100)
            self.generation_config = SimpleNamespace(eos_token_id=[9])
        def generate(self, **kwargs):
            assert not self.training
            assert kwargs["do_sample"] is False
            assert all(kwargs[k] is None for k in ("temperature", "top_p", "top_k"))
            output = torch.tensor([[0, 1, 7, 9, 0], [1, 1, 8, 8, 8]])
            assert kwargs["stopping_criteria"](output[:, :3], None).tolist() == [True, False]
            return output
    model, tok = Model(), Tokenizer()
    result = generate_greedy(model, tok, ["x", "yy"], max_new_tokens=3, stop_at_newline=True)
    assert result.texts == ("x", "y")
    assert result.raw_texts == ("x\nexplanation", "y")
    assert result.prompt_tokens == (1, 2)
    assert result.generated_tokens == (1, 3)
    assert result.length_capped == (False, True)
    assert model.training and tok.padding_side == "right"
    with pytest.raises(ValueError, match="上下文"):
        generate_greedy(model, tok, ["x", "yy"], max_new_tokens=100)
    assert model.training and tok.padding_side == "right"


@pytest.mark.parametrize('argv,expected', [
    ([], ('cpu', 'fp32', 20, 4, None)),
    (['--device', 'cuda', '--n', '5001', '--batch-size', '32', '--sub-n', '500'],
     ('cuda', 'fp32', 5001, 32, 500)),
    (['--device', 'cuda', '--dtype', 'bf16'], ('cuda', 'bf16', 20, 4, None)),
])
def test_tour_cli_configuration(argv, expected, monkeypatch, capsys):
    from . import tour
    seen = []
    monkeypatch.setattr(tour, '_CONTEXT', tour._CONTEXT.copy())
    monkeypatch.setattr(tour.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(tour.torch.cuda, 'get_device_name', lambda: 'Test GPU')
    monkeypatch.setattr(tour, 'PARTS', {10: ('configuration', lambda exp: seen.append(exp.args))})
    tour.main(['10', *argv])
    args = seen[0]
    assert (args.device, args.dtype, args.n, args.batch_size, args.sub_n) == expected
    output = capsys.readouterr().out
    assert f'batch_size={args.batch_size}' in output
    assert 'sub_n=' in output
    if args.device == 'cuda':
        assert 'device=cuda(Test GPU)' in output and 'GPU名称="Test GPU"' in output


@pytest.mark.parametrize('flag', ['--batch-size', '--sub-n'])
@pytest.mark.parametrize('value', ['0', '-1'])
def test_tour_positive_sizes(flag, value):
    from . import tour
    with pytest.raises(SystemExit, match='2'):
        tour.main([flag, value])


@pytest.mark.parametrize('n,sub_n,expected', [
    (20, None, (20, 20)), (100, None, (30, 50)),
    (5001, 500, (500, 500)), (20, 500, (20, 20)),
])
def test_tour_subset_prefix(n, sub_n, expected):
    from types import SimpleNamespace
    from . import tour
    exp = SimpleNamespace(args=SimpleNamespace(n=n, sub_n=sub_n))
    assert tuple(tour.subset_n(exp, default) for default in (30, 50)) == expected


def test_reuse_rejects_different_batch_size(saved_experiment):
    make, path, rows, _ = saved_experiment()
    exp = make(reuse=True)
    exp.args.batch_size = 32
    with pytest.raises(ValueError, match='batch_size.*不一致'):
        exp.run('few_shot', 3)
