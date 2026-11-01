"""第 17 章：小模型离线对拍；官方权重存在时追加短序列验收。"""
import json
import os
from dataclasses import replace
from types import SimpleNamespace

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config as HFConfig, GPT2LMHeadModel, GPT2Model

from miniocr.models.gpt2 import (
    GPT2, GPT2Config, architecture_diff_rows, config_field_notes,
    convert_hf_gpt2_state_dict, find_local_gpt2, gelu_new, generate_greedy,
    n_params_formula,
)
from .tour import Experiment, byte_decoder, hf_greedy, safetensors_accounting


@pytest.fixture(scope="module", autouse=True)
def two_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(17)
    config = HFConfig(vocab_size=97, n_positions=32, n_embd=48, n_layer=2, n_head=4,
                      attn_pdrop=0, embd_pdrop=0, resid_pdrop=0, bos_token_id=96, eos_token_id=96)
    config._attn_implementation = "eager"
    hf = GPT2LMHeadModel(config).float().eval()
    cfg = GPT2Config.from_hf_dict(config.to_dict())
    own = GPT2(cfg).eval()
    own.load_state_dict(convert_hf_gpt2_state_dict(hf.state_dict(), cfg), strict=True)
    idx = torch.randint(0, 97, (2, 9))
    return hf, own, idx


@torch.no_grad()
def test_logits_and_all_hidden(tiny):
    hf, own, idx = tiny
    ref = hf(idx, output_hidden_states=True, use_cache=False)
    actual, hidden = own(idx, return_hidden=True)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, ref.logits, atol=1e-5, rtol=1e-5)
    assert len(hidden) == len(ref.hidden_states) == own.cfg.n_layer+1
    for a, b in zip(hidden, ref.hidden_states):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(hidden[0], own.wte(idx)+own.wpe(torch.arange(idx.shape[1])))


def test_counts_and_sharing(tiny):
    hf, own, _ = tiny
    n = n_params_formula(own.cfg)
    assert n["total"] == sum(v for k, v in n.items() if k != "total")
    assert n["total"] == sum(p.numel() for p in own.parameters()) == sum(p.numel() for p in hf.parameters())
    assert own.head.weight is own.wte.weight
    assert own.head.weight.data_ptr() == own.wte.weight.data_ptr()
    assert sum(t.numel() for t in own.state_dict().values()) - n["total"] == own.cfg.vocab_size*own.cfg.n_embd
    assert n_params_formula(GPT2Config())["total"] == 124439808


@torch.no_grad()
def test_unprefixed_model_and_legacy_buffers(tiny):
    hf, own, idx = tiny
    bare = GPT2Model(hf.config).eval()
    bare.load_state_dict(hf.transformer.state_dict())
    state = bare.state_dict()
    state["h.0.attn.bias"] = torch.ones(1, 1, 32, 32, dtype=torch.bool)
    state["h.0.attn.masked_bias"] = torch.tensor(-1e4)
    other = GPT2(own.cfg).eval()
    other.load_state_dict(convert_hf_gpt2_state_dict(state, own.cfg), strict=True)
    torch.testing.assert_close(other(idx), own(idx), atol=0, rtol=0)


@pytest.mark.parametrize("bug", ["no_transpose", "qkv_order"])
@torch.no_grad()
def test_silent_bugs_load_but_change_logits(tiny, bug):
    hf, own, idx = tiny
    state = convert_hf_gpt2_state_dict(hf.state_dict(), own.cfg, bug)
    bad = GPT2(own.cfg).eval()
    bad.load_state_dict(state, strict=True)
    assert (bad(idx)-own(idx)).abs().max() > 1e-4
    if bug == "no_transpose":
        torch.testing.assert_close(state["blocks.0.attn.proj.weight"], hf.state_dict()["transformer.h.0.attn.c_proj.weight"])
        torch.testing.assert_close(state["blocks.0.attn.q.weight"], own.blocks[0].attn.q.weight)
        torch.testing.assert_close(state["blocks.0.mlp.fc.weight"], own.blocks[0].mlp.fc.weight)
    else:
        torch.testing.assert_close(state["blocks.0.attn.q.weight"], own.blocks[0].attn.k.weight)
        torch.testing.assert_close(state["blocks.0.attn.q.bias"], own.blocks[0].attn.k.bias)


def test_invalid_conversion(tiny):
    hf, own, _ = tiny
    with pytest.raises(ValueError, match="未知"):
        convert_hf_gpt2_state_dict({}, own.cfg, "typo")
    sd = dict(hf.state_dict())
    sd["lm_head.weight"] = sd["lm_head.weight"] + 1
    with pytest.raises(ValueError, match="共享"):
        convert_hf_gpt2_state_dict(sd, own.cfg)


def test_qkv_bias_slices_are_distinct(tiny):
    hf, own, _ = tiny
    sd, d = dict(hf.state_dict()), own.cfg.n_embd
    sd["transformer.h.0.attn.c_attn.bias"] = torch.arange(3*d, dtype=torch.float32)
    for bug, order in ((None, ("q", "k", "v")), ("qkv_order", ("k", "q", "v"))):
        converted = convert_hf_gpt2_state_dict(sd, own.cfg, bug)
        for i, name in enumerate(order):
            torch.testing.assert_close(converted[f"blocks.0.attn.{name}.bias"],
                                       torch.arange(i*d, (i+1)*d, dtype=torch.float32))


@torch.no_grad()
def test_causal_mask_and_boundary(tiny):
    _, own, idx = tiny
    altered = idx.clone()
    altered[:, 5:] = (altered[:, 5:]+1) % own.cfg.vocab_size
    torch.testing.assert_close(own(idx)[:, :5], own(altered)[:, :5], atol=0, rtol=0)
    assert own(torch.zeros((1, 32), dtype=torch.long)).shape == (1, 32, 97)
    with pytest.raises(ValueError, match="序列长度 33.*n_positions=32"):
        own(torch.zeros((1, 33), dtype=torch.long))
    with pytest.raises(ValueError, match="非空"):
        own(torch.zeros((1, 0), dtype=torch.long))
    with pytest.raises(ValueError, match="续写长度"):
        generate_greedy(own, idx, 32)


@torch.no_grad()
def test_greedy_tiny(tiny):
    hf, own, idx = tiny
    assert torch.equal(generate_greedy(own, idx, 3), hf_greedy(hf, idx, 3))
    assert torch.equal(generate_greedy(own, idx, 0), idx)


@torch.no_grad()
def test_exact_and_approx_gelu(tiny):
    x = torch.linspace(-5, 5, 10001, dtype=torch.float64)
    error = (gelu_new(x)-F.gelu(x)).abs().max().item()
    assert 1e-4 < error < 1e-3
    _, own, idx = tiny
    exact = GPT2(replace(own.cfg, activation="gelu")).eval()
    exact.load_state_dict(own.state_dict())
    diff = (exact(idx)-own(idx)).abs().max().item()
    assert 0 < diff < 1e-3
    with pytest.raises(ValueError, match="activation"):
        GPT2Config(activation="relu")


def test_config_notes_cover_official_config():
    path = find_local_gpt2()
    # 官方 small config 的完整键集合；无缓存仍验证覆盖，绝不联网。
    builtin = dict(activation_function="gelu_new", architectures=["GPT2LMHeadModel"],
                   attn_pdrop=0.1, bos_token_id=50256, embd_pdrop=0.1, eos_token_id=50256,
                   initializer_range=0.02, layer_norm_epsilon=1e-5, model_type="gpt2", n_ctx=1024,
                   n_embd=768, n_head=12, n_layer=12, n_positions=1024, resid_pdrop=0.1,
                   summary_activation=None, summary_first_dropout=0.1, summary_proj_to_labels=True,
                   summary_type="cls_index", summary_use_proj=True, task_specific_params={}, vocab_size=50257)
    data = json.loads((path/"config.json").read_text()) if path else builtin
    assert set(data) <= config_field_notes().keys()
    assert all(config_field_notes()[k] for k in data)
    assert GPT2Config.from_hf_dict(builtin) == GPT2Config()


def test_difference_table_has_sources_and_missing_data(tmp_path):
    required = {"位置编码", "归一化", "MLP/激活", "注意力头", "Q/K norm", "线性层偏置", "权重共享",
                "词表大小", "最大位置", "层数", "宽度", "head_dim", "norm eps", "rope_theta"}
    rows = architecture_diff_rows(hub=tmp_path)
    assert required <= {r["field"] for r in rows}
    for row in rows:
        for name in ("gpt2", "book", "qwen3"):
            assert row[name]["source"]
        assert "待验证" in str(row["qwen3"]["value"])
    assert find_local_gpt2(hub=tmp_path) is None


def test_local_discovery_and_qk_keys(tmp_path):
    from safetensors.torch import save_file
    folder = tmp_path/"models--Qwen--Qwen3-0.6B"/"snapshots"/"local"
    folder.mkdir(parents=True)
    (folder/"config.json").write_text(json.dumps(dict(num_attention_heads=4, num_key_value_heads=2, head_dim=16)))
    save_file({"model.layers.0.self_attn.q_norm.weight": torch.ones(16),
               "model.layers.0.self_attn.k_norm.weight": torch.ones(16)}, str(folder/"model.safetensors"))
    rows = {r["field"]: r for r in architecture_diff_rows(hub=tmp_path)}
    assert rows["Q/K norm"]["qwen3"]["value"] is True
    assert "keys" in rows["Q/K norm"]["qwen3"]["source"]
    assert rows["head_dim"]["qwen3"]["value"] == 16
    assert "models--Qwen--Qwen3-0.6B" in rows["head_dim"]["qwen3"]["source"]
    assert "待验证" in rows["词表大小"]["qwen3"]["value"]
    folder = tmp_path/"models--openai-community--gpt2"/"snapshots"/"local"
    folder.mkdir(parents=True)
    (folder/"config.json").write_text("{}")
    assert find_local_gpt2(tmp_path) is None
    save_file({"test": torch.ones(1)}, str(folder/"model.safetensors"))
    assert find_local_gpt2(tmp_path) == folder


@torch.no_grad()
def test_fault_context_restores_after_exception(tiny):
    hf, own, idx = tiny
    exp = Experiment(SimpleNamespace(device="cpu"))
    exp._own, exp._hf = own, hf
    before = own(idx).clone()
    for bug in ("gelu", "no_transpose", "qkv_order"):
        with pytest.raises(RuntimeError, match="演示异常"):
            with exp.bug(bug):
                raise RuntimeError("演示异常")
        torch.testing.assert_close(own(idx), before, atol=0, rtol=0)


def test_missing_weights_skip(capsys, monkeypatch):
    from . import tour
    monkeypatch.setattr(tour, "find_local_gpt2", lambda: None)
    exp = Experiment(SimpleNamespace(device="cpu"))
    for p in (1, 2, 3, 4, 5, 8, 9, 10, 11):
        tour.PARTS[p][1](exp)
    assert capsys.readouterr().out.count("跳过：未找到本地 GPT-2 权重；本章不会自动下载") == 9


@pytest.fixture(scope="module")
def official():
    if find_local_gpt2() is None:
        pytest.skip("未找到本地 GPT-2 权重；本章不会自动下载")
    return Experiment(SimpleNamespace(device="cpu"))


def test_official_safetensors_accounting():
    path = find_local_gpt2()
    if path is None:
        pytest.skip("未找到本地 GPT-2 权重；本章不会自动下载")
    # 只读文件头，不调用模型加载；独立核对正文所用原始文件的账。
    account = safetensors_accounting(path/"model.safetensors")
    assert account["tensor_count"] == 160
    assert not account["has_lm_head"]
    assert not account["has_transformer_prefix"]
    assert account["dtypes"] == ["F32"]
    assert account["attention_bias_shapes"] == {
        f"h.{i}.attn.bias": (1, 1, 1024, 1024) for i in range(12)
    }
    assert account["parameter_elements"] == 124439808
    assert account["parameter_bytes"] == account["parameter_fp32_bytes"] == 497759232
    assert account["buffer_bytes"] == 50331648
    assert account["header_bytes"] == 14283
    assert account["file_bytes"] == 548105171
    assert account["file_equation_holds"]


@torch.no_grad()
def test_official_logits_and_hidden(official):
    idx = official.ids("A short test.")
    own, hf = official.own(), official.hf()
    a, hidden = own(idx, return_hidden=True)
    b = hf(idx, output_hidden_states=True, use_cache=False)
    # 大模型累积舍入误差大于随机 tiny；显式 fp32、eager、同一输入。
    torch.testing.assert_close(a, b.logits, atol=2e-4, rtol=1e-5)
    for x, y in zip(hidden, b.hidden_states):
        torch.testing.assert_close(x, y, atol=2e-4, rtol=1e-5)
    assert sum(p.numel() for p in own.parameters()) == 124439808


@torch.no_grad()
def test_official_greedy_and_token_bytes(official):
    idx = official.ids("The quick brown fox")
    assert torch.equal(generate_greedy(official.own(), idx, 3), hf_greedy(official.hf(), idx, 3))
    decoder, text = byte_decoder(), "hello world 中文"
    tokens = official.tok().convert_ids_to_tokens(official.tok().encode(text, add_special_tokens=False))
    assert b"".join(bytes(decoder[c] for c in t) for t in tokens) == text.encode()
    assert decoder["Ġ"] == 32
    assert len(official.tok().encode("模", add_special_tokens=False)) > 1


def test_local_qwen3_generation_config():
    from miniocr.models.gpt2 import _configs
    if not _configs("models--Qwen--Qwen3-0.6B"):
        pytest.skip("未找到本地 Qwen3-0.6B 配置；本章不会自动下载")
    rows = {r["field"]: r["qwen3"] for r in architecture_diff_rows()}
    assert rows["词表大小"]["value"] == 151936
    assert rows["最大位置"]["value"] == 40960
    assert rows["权重共享"]["value"] is True
    assert "Qwen3-0.6B" in rows["词表大小"]["source"]
