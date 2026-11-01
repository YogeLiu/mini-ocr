"""第 16 章：CPU 数值对拍；不下载、不长训练，不断言计时快慢。"""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch

from miniocr.kvcache import local_model_configs
from miniocr.models.gpt import GPT, GPTConfig
from miniocr.train.memory import checkpoint_blocks, static_memory, matmul_params
from miniocr.train.precision import float_format, format_table, spacing, quantize_gradients, SimpleGradScaler
from miniocr.train.resources import (
    measure_static_bytes, activation_bytes_estimate, measure_activations, qwen_like_params,
    training_flops, count_flops, resource_report,
)
from . import tour


@pytest.fixture(autouse=True)
def cpu_two_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield
    torch.set_num_threads(old)


def config(**kwargs):
    values = dict(vocab_size=37, d_model=32, n_layer=2, n_head=2,
                  d_ff=64, block_size=64, attn_impl="manual")
    values.update(kwargs)
    return GPTConfig(**values)


@pytest.mark.parametrize("name", ["fp32", "fp16", "bf16"])
def test_format_formula(name):
    row = float_format(name)
    fi = torch.finfo(row["dtype"])
    assert row["sign_bits"]+row["exponent_bits"]+row["mantissa_bits"] == fi.bits
    for key in ("max", "tiny", "eps"):
        assert row[key] == getattr(fi, key)
    sub = torch.nextafter(torch.tensor(0., dtype=row["dtype"]), torch.tensor(1., dtype=row["dtype"]))
    assert row["subnormal"] == sub.item() == fi.tiny*fi.eps
    assert len(format_table()) == 3


@pytest.mark.parametrize("name,x,expected", [
    ("fp16", 1., 2**-10), ("bf16", 1., 2**-7),
    ("fp16", 256., .25), ("bf16", 256., 2.),
    ("fp16", 1000., .5), ("bf16", 1000., 4.),
    ("fp16", 65504., 32.), ("bf16", 65504., 256.),
])
def test_spacing(name, x, expected):
    assert spacing(x, name) == expected
    assert spacing(-x, float_format(name)["dtype"]) == expected
    # 最大有限数与未被精确表示的65504只检查binade公式，不能用到inf的距离。
    if x < 65504:
        dt = float_format(name)["dtype"]
        t = torch.tensor(x, dtype=dt)
        assert (torch.nextafter(t, torch.tensor(float("inf"), dtype=dt))-t).item() == expected


@pytest.mark.parametrize("x", [0., 1e-8, 70000., float("inf"), float("nan")])
def test_spacing_rejects_non_normal(x):
    with pytest.raises(ValueError):
        spacing(x, "fp16")


def test_scaling_restores_small_gradients():
    grad = torch.randn(10000)*1e-7
    plain, scaled = quantize_gradients(grad), quantize_gradients(grad, 2**16)
    assert plain["zero_fraction"] > .1
    assert scaled["zero_fraction"] < .001
    assert scaled["relative_error"] < .001 < plain["relative_error"]
    assert scaled["inf_fraction"] == 0


def test_scaler_torch_cpu_parity():
    try:
        reference = torch.amp.GradScaler("cpu", init_scale=2.**16, growth_interval=2)
    except (TypeError, RuntimeError, AttributeError):
        pytest.skip("本机不支持 CPU GradScaler")
    actual = tour._scaler_sequence(SimpleGradScaler(growth_interval=2))
    expected = tour._scaler_sequence(reference)
    assert len(actual) == 8
    for a, b in zip(actual, expected):
        assert a == b
    assert [r["next_scale"] for r in actual] == [65536, 32768, 16384, 16384, 32768, 32768, 65536, 65536]


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_scaler_skips_and_enforces_order(bad):
    p = torch.nn.Parameter(torch.tensor(1.))
    opt = torch.optim.SGD([p], lr=.1)
    scaler = SimpleGradScaler(init_scale=8)
    scaler.scale(p.square()).backward()
    p.grad.fill_(bad)
    scaler.unscale_(opt)
    assert scaler.found_inf
    with pytest.raises(RuntimeError):
        scaler.unscale_(opt)
    assert scaler.step(opt) is False
    assert p.item() == 1.
    with pytest.raises(RuntimeError):
        scaler.step(opt)
    scaler.update()
    assert scaler.get_scale() == 4
    assert not scaler.found_inf


def test_scaler_resume_and_implicit_unscale():
    p = torch.nn.Parameter(torch.tensor(1.))
    opt = torch.optim.SGD([p], lr=.1)
    scaler = SimpleGradScaler(init_scale=8, growth_interval=2)
    scaler.scale(p*2).backward()
    assert scaler.step(opt)
    scaler.update()
    assert p.item() == pytest.approx(.8)
    state = scaler.state_dict()
    restored = SimpleGradScaler()
    restored.load_state_dict(state)
    opt.zero_grad(set_to_none=True)
    restored.scale(p*2).backward()
    restored.step(opt)
    restored.update()
    assert restored.get_scale() == 16
    assert p.item() == pytest.approx(.6)
    try:
        reference = torch.amp.GradScaler("cpu")
    except (TypeError, RuntimeError, AttributeError):
        return
    reference.load_state_dict(state)
    assert reference.state_dict() == state


def test_clip_after_unscale():
    updates = []
    for correct in (True, False):
        p = torch.nn.Parameter(torch.ones(2))
        opt = torch.optim.SGD([p], lr=.1)
        scaler = SimpleGradScaler(init_scale=1024)
        scaler.scale((p*torch.tensor([3., 4.])).sum()).backward()
        if correct:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p], 1.)
        if not correct:
            scaler.unscale_(opt)
        updates.append(p.grad.norm().item())
        scaler.step(opt)
        scaler.update()
    assert updates[0] == pytest.approx(1., abs=1e-6)
    assert updates[1] == pytest.approx(1/1024, abs=1e-8)


@pytest.mark.parametrize("mixed", [False, True])
def test_static_memory_matches_actual_adamw(mixed):
    estimate, actual = tour._static_example(mixed)
    for key in ("params", "grads", "optim", "master"):
        assert actual[key] == estimate[key]
    assert actual["total"]-estimate["total"] == actual["step"] > 0
    assert actual["master_grads"] == 0
    assert actual["optim"] == sum(actual["optimizer_breakdown"].values())
    assert actual["optimizer_breakdown"]["exp_avg"] == actual["optimizer_breakdown"]["exp_avg_sq"]


def test_static_storage_alias_and_nested_state():
    m = torch.nn.Module()
    original = torch.arange(12, dtype=torch.float32)
    m.a = torch.nn.Parameter(original[:8])
    m.b = torch.nn.Parameter(original[4:])
    opt = torch.optim.SGD(m.parameters(), lr=.1)
    tensor = torch.ones(3)
    opt.state[m.a] = {"nested": {"moments": [tensor, tensor[:1]]}, "step": torch.tensor(1.)}
    actual = measure_static_bytes(m, opt)
    assert actual["params"] == 12*4  # 两个Parameter对象，共用整个底层storage。
    assert actual["optim"] == 3*4
    assert actual["step"] == 4
    assert actual["total"] == (12+3+1)*4


@pytest.mark.parametrize("B,T", [(1, 8), (2, 16), (4, 32)])
@pytest.mark.parametrize("dtype,bytes_per", [(torch.float32, 4), (torch.bfloat16, 2)])
def test_activation_estimate_matches_saved(B, T, dtype, bytes_per):
    cfg = config()
    model = GPT(cfg).to(dtype)
    estimate = activation_bytes_estimate(cfg, B, T, bytes_per)
    actual = measure_activations(model, B, T)
    assert abs(estimate["total"]-actual)/actual < .02
    assert sum(r["bytes"] for r in estimate["breakdown"]) == estimate["total"]
    assert sum(r["fraction"] for r in estimate["breakdown"]) == pytest.approx(1.)
    assert estimate["total"] == (estimate["linear_coefficient"]*T+
                                  estimate["quadratic_coefficient"]*T*T+estimate["constant"])


@pytest.mark.parametrize("change", [dict(attn_impl="sdpa"), dict(qk_norm=True),
                                    dict(n_kv_head=1), dict(dropout=.1), dict(norm="layernorm")])
def test_activation_unsupported_path_is_explicit(change):
    with pytest.raises(ValueError, match="待验证"):
        activation_bytes_estimate(config(**change), 2, 16)


def test_checkpoint_gradients_memory_and_flops():
    plain = GPT(config())
    checked = deepcopy(plain)
    checkpoint_blocks(checked.blocks)
    x = torch.randint(plain.cfg.vocab_size, (2, 32))
    y = x.clone()
    plain_bytes = measure_activations(plain, 2, 32)
    checked_bytes = measure_activations(checked, 2, 32)
    plain_flops = count_flops(plain, x, y)
    checked_flops = count_flops(checked, x, y)
    torch.testing.assert_close(tour._gradient(plain), tour._gradient(checked), rtol=1e-5, atol=1e-7)
    assert checked_bytes < plain_bytes
    assert 1.1 < checked_flops/plain_flops < 1.4


def test_variable_length_accumulation_and_pretrain():
    full, gradients = tour._accumulation_experiment()
    for group in ("A", "B"):
        for mode in ("token", "pretrain"):
            assert tour._relative(gradients[group, mode], full) < 2e-6
        for mode in ("micro_mean", "sample"):
            assert tour._relative(gradients[group, mode], full) > .01
        assert tour._relative(gradients[group, "pretrain"], gradients[group, "token"]) < 2e-6
    assert tour._relative(gradients["B", "token"], gradients["A", "token"]) < 2e-6
    assert tour._relative(gradients["B", "micro_mean"], gradients["A", "micro_mean"]) > .01
    assert tour._relative(gradients["B", "sample"], gradients["A", "sample"]) < 2e-6


def test_counted_flops_match_full_attention():
    cfg = config()
    model = GPT(cfg)
    x = torch.randint(cfg.vocab_size, (2, 32))
    estimated = training_flops(cfg, 2, 32)
    forward = count_flops(model, x, x.clone(), backward=False)
    both = count_flops(model, x, x.clone())
    assert estimated["matmul_params"] == matmul_params(model)
    assert .98 < both/estimated["with_attention"] < 1.02
    assert .98 < forward/estimated["forward_with_attention"] < 1.02
    assert both == 3*forward
    assert estimated["attention"] == 12*cfg.n_layer*2*32**2*cfg.d_model


def test_resource_report_consistent_units():
    cfg = config()
    report = resource_report(cfg, 2, 16, accum=3, peak_tflops=1., tokens_per_sec=100., token_budget=1000)
    n = GPT(cfg).n_params()
    assert report["static"] == static_memory(n, n)
    assert report["tokens_per_step"] == 2*16*3
    assert report["total_estimate"] == report["static"]["total"]+report["activation"]["total"]
    assert report["flops"]["six_nd"] == 6*n*2*16*3
    assert report["achieved_flops"] == 6*n*100
    assert report["mfu"] == report["achieved_flops"]/1e12
    assert report["estimated_seconds"] == 10
    assert 6*n*1000/(1e12*report["mfu"]) == pytest.approx(10)
    once = resource_report(cfg, 2, 16)
    assert once["activation"] == report["activation"]  # 累积不同时保留三个微批的图。
    assert once["mfu"] is None
    assert once["estimated_seconds"] is None
    mixed = resource_report(cfg, 2, 16, precision="mixed", trainable_fraction=.01)
    assert mixed["static"] == static_memory(n, round(n*.01), "bf16", master_fp32=True)
    assert "待验证" in mixed["activation"]["status"]
    assert mixed["total_estimate"] == mixed["static"]["total"]+mixed["activation"]["total"]
    assert mixed["activation"]["total"] < once["activation"]["total"]
    sdpa = resource_report(replace(cfg, attn_impl="sdpa"), 2, 16)
    assert sdpa["activation"]["total"] is None


@pytest.mark.parametrize("kwargs", [dict(peak_tflops=0), dict(tokens_per_sec=float("nan")),
                                    dict(trainable_fraction=1.1), dict(accum=0), dict(precision="fp16")])
def test_resource_report_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        resource_report(config(), 2, 16, **kwargs)


def test_qwen_local_tie_and_fallback(tmp_path):
    cfg = local_model_configs(tmp_path)[0]
    missing = qwen_like_params(cfg)
    assert "待验证" in missing["tie_source"]
    assert cfg.vocab_size == 151936
    assert missing["tied"] == 596049920
    assert missing["untied"] == 751632384
    assert missing["untied"]-missing["tied"] == cfg.d_model*cfg.vocab_size
    path = tmp_path/"config.json"
    path.write_text(json.dumps({"tie_word_embeddings": False}))
    local = replace(cfg, source=f"本地 config {path}")
    actual = qwen_like_params(local)
    d, dh = cfg.d_model, cfg.head_dim
    expected = (2*d*cfg.vocab_size+cfg.n_layer*(2*d*cfg.n_head*dh+
                2*d*cfg.n_kv_head*dh+2*dh+3*d*cfg.d_ff+2*d)+d)
    assert actual["total"] == expected == missing["untied"]
    assert actual["tie_word_embeddings"] is False
    assert "待验证" not in actual["tie_source"]
    assert qwen_like_params(local, True)["total"] == missing["tied"]
    with pytest.raises(ValueError):
        qwen_like_params(local_model_configs(tmp_path)[1])


@pytest.mark.parametrize("argv", [["0"], ["12"], ["1", "-1"], ["--peak-tflops", "0"],
                                 ["--peak-tflops", "nan"], ["--seed", "-1"]])
def test_parse_args_rejects_invalid(argv):
    with pytest.raises(SystemExit):
        tour.parse_args(argv)


def test_parse_args_order():
    args = tour.parse_args(["5", "1", "5", "--seed", "7", "--peak-tflops", "1"])
    assert args.parts == [5, 1, 5]
    assert args.device == "cpu"
    assert args.seed == 7
    assert args.peak_tflops == 1.


@pytest.mark.parametrize("part", [1, 5])
def test_fast_parts_run_alone(part, capsys):
    tour.main([str(part)])
    output = capsys.readouterr().out
    assert f"部分={part}；内容=" in output
    assert "总用时秒=" in output
    assert "来源=数学结论" in output
    assert "设备=cpu；线程=2" in output


def test_qwen_generation_local_config_and_memory(tmp_path):
    path = tmp_path/"models--Qwen--Qwen3-0.6B/snapshots/local/config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(num_hidden_layers=28, num_attention_heads=16,
        num_key_value_heads=8, head_dim=128, hidden_size=1024, intermediate_size=3072,
        vocab_size=151936, tie_word_embeddings=True)))
    cfg = local_model_configs(tmp_path)[0]
    counts = qwen_like_params(cfg)
    assert counts["total"] == counts["tied"] == 596049920
    assert counts["untied"] == 751632384
    assert counts["tie_word_embeddings"] is True
    assert "待验证" not in counts["tie_source"]
    assert "待验证" not in cfg.note
    assert static_memory(counts["total"], counts["total"])["total"] == 9536798720
    assert static_memory(counts["total"], round(counts["total"]*.01),
                         "bf16", master_fp32=True)["total"] == 1275546826
