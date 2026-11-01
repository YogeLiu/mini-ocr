"""第 15 章：CPU 小随机模型的对拍、故障检测与独立手算，不依赖语料或检查点。"""
from dataclasses import asdict
import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import pytest
import torch

from miniocr.generate import generate
from miniocr.kvcache import (
    cached_mask, prefill_chunks, cache_nbytes, kv_cache_bytes, kv_cache_bytes_for,
    inject_bug, local_model_configs, step_cost, intensity_sweep, time_decode, time_prefill,
)
from miniocr.models.gpt import GPT, GPTConfig
from book.part2_lm.ch15_kvcache.tour import (
    Experiment, chunk_logits, decode_difference, main, parse_args,
)


@pytest.fixture(autouse=True)
def small_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    with torch.no_grad():
        yield
    torch.set_num_threads(old)


def make_model(**kwargs):
    cfg = dict(vocab_size=37, d_model=32, n_head=4, n_kv_head=2,
               n_layer=2, block_size=64, qk_norm=True, tie_weights=False)
    cfg.update(kwargs)
    return GPT(GPTConfig(**cfg)).eval()


@pytest.mark.parametrize("new,past", [(1, 0), (1, 8), (2, 3), (4, 0), (4, 7)])
def test_cached_mask(new, past):
    mask = cached_mask(new, past)
    expected = torch.ones(new, past+new, dtype=torch.bool).tril(past)
    assert mask.dtype == torch.bool and mask.shape == (new, past+new)
    assert torch.equal(mask, expected)
    if new == 1:
        assert mask.all()
    if (new, past) == (2, 3):
        assert mask.int().tolist() == [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
@pytest.mark.parametrize("heads,norm", [(4, False), (2, False), (2, True)])
@pytest.mark.parametrize("batch", [1, 2])
def test_cache_generation_and_step_logits(impl, heads, norm, batch):
    model = make_model(attn_impl=impl, n_kv_head=heads, qk_norm=norm)
    prompt = torch.randint(37, (batch, 10))
    full = generate(model, prompt, 30, greedy=True, use_cache=False)
    cached = generate(model, prompt, 30, greedy=True, use_cache=True)
    assert torch.equal(full, cached)
    assert decode_difference(model, prompt, 30) < 1e-5


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
@pytest.mark.parametrize("sizes", [[10], [6, 4], [3, 3, 4], [4, 1, 5], [1]*10])
def test_prefill_chunks(impl, sizes):
    model = make_model(attn_impl=impl)
    idx = torch.randint(37, (2, 10))
    actual, caches = prefill_chunks(model, idx, sizes)
    assert (actual-model(idx)[0]).abs().max() < 1e-5
    assert all(c["k"].shape[2] == 10 for c in caches)


@pytest.mark.parametrize("norm", [False, True])
def test_cache_shapes_rope_and_bytes(norm):
    model = make_model(qk_norm=norm)
    idx = torch.randint(37, (2, 10))
    inputs, hooks = [], []
    for block in model.blocks:
        hooks.append(block.attn.register_forward_pre_hook(lambda module, args: inputs.append(args[0].clone())))
    try:
        _, caches = prefill_chunks(model, idx, [10])
    finally:
        for hook in hooks:
            hook.remove()
    for block, x, cache in zip(model.blocks, inputs, caches):
        attn = block.attn
        # 独立旋转式，不调用 apply_rope：缓存应等于在各自位置旋转后的 K。
        k = (x @ attn.k.weight.T).reshape(2, 10, 2, 8).transpose(1, 2)
        if norm:
            k = k * torch.rsqrt(k.square().mean(-1, keepdim=True)+attn.k_norm.eps) * attn.k_norm.weight
        angle = torch.arange(10)[:, None] * 10000.0**(-torch.arange(0, 8, 2)/8)
        a, b = k.chunk(2, -1)
        expected = torch.cat([a*angle.cos()-b*angle.sin(), a*angle.sin()+b*angle.cos()], -1)
        assert cache["k"].shape == cache["v"].shape == (2, 2, 10, 8)
        torch.testing.assert_close(cache["k"], expected, atol=1e-6, rtol=1e-5)
        assert (cache["k"]-k).abs().max() > 1e-4
        expected_v = (x @ attn.v.weight.T).reshape(2, 10, 2, 8).transpose(1, 2)
        torch.testing.assert_close(cache["v"], expected_v)
    assert cache_nbytes(caches) == kv_cache_bytes_for(model.cfg, 10, 2, 4)
    model(idx[:, :1], caches=caches, pos0=10)
    assert all(c["k"].shape == c["v"].shape == (2, 2, 11, 8) for c in caches)
    assert cache_nbytes(caches) == kv_cache_bytes_for(model.cfg, 11, 2, 4)


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
@pytest.mark.parametrize("bug", ["pos0=0", "mask_offset", "pre_rope_k", "no_cache_update"])
def test_bug_detection_and_restore(impl, bug):
    model = make_model(attn_impl=impl)
    prompt = torch.randint(37, (2, 10))
    full = model(prompt)[0]
    originals = [b.attn.forward for b in model.blocks]
    if bug == "pos0=0":
        bad, _ = chunk_logits(model, prompt, [6, 4], wrong_pos=True)
        assert (bad-full).abs().max() > 1e-4
        assert decode_difference(model, prompt, 20, wrong_pos=True) > 1e-4
    else:
        with inject_bug(model, bug):
            bad, caches = prefill_chunks(model, prompt, [6, 4])
            error = decode_difference(model, prompt, 20)
            if bug == "no_cache_update":
                # 第二块仍使用本次拼好的 KV，第三次才受“未写回”影响。
                assert (bad-full).abs().max() < 1e-5
                assert caches[0]["k"].shape[2] == 6
                assert error > 1e-4
            else:
                assert (bad-full).abs().max() > 1e-4
                if bug == "mask_offset":
                    assert error < 1e-5  # 单 token 路径不会触发这一错误。
                else:
                    assert error > 1e-4
    assert [b.attn.forward for b in model.blocks] == originals
    assert (prefill_chunks(model, prompt, [6, 4])[0]-full).abs().max() < 1e-5
    assert decode_difference(model, prompt, 3) < 1e-5


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_repeat_before_cache_correct_but_larger(impl):
    model = make_model(attn_impl=impl)
    prompt = torch.randint(37, (2, 10))
    full, normal = prefill_chunks(model, prompt, [6, 4])
    with inject_bug(model, "repeat_before_cache"):
        repeated, caches = prefill_chunks(model, prompt, [6, 4])
        assert (repeated-full).abs().max() < 1e-5
        assert decode_difference(model, prompt, 20) < 1e-5
        assert cache_nbytes(caches) == cache_nbytes(normal)*model.cfg.n_head//model.cfg.kv_heads()
        assert caches[0]["k"].shape == (2, 4, 10, 8)
    assert "forward" not in model.blocks[0].attn.__dict__


def test_injection_restores_after_exception_and_nested_context():
    model = make_model()
    original = model.blocks[0].attn.forward
    with pytest.raises(RuntimeError, match="故意中断"):
        with inject_bug(model, "mask_offset"):
            outer = model.blocks[0].attn.forward
            with inject_bug(model, "pre_rope_k"):
                assert model.blocks[0].attn.forward != outer
            assert model.blocks[0].attn.forward == outer
            raise RuntimeError("故意中断")
    assert model.blocks[0].attn.forward == original
    with pytest.raises(ValueError, match="未知故障"):
        with inject_bug(model, "unknown"):
            pass


def test_kv_hand_calculations():
    assert kv_cache_bytes(28, 8, 128, 4096, 1, 2) == 469762048
    assert kv_cache_bytes(6, 2, 256, 4096, 1, 2) == 50331648
    assert cache_nbytes([{}, {}]) == 0


def test_step_cost_hand_calculation():
    cfg = GPTConfig(vocab_size=11, n_layer=2, d_model=8, d_ff=16, n_head=2, n_kv_head=1)
    # Q/O: 各8×8；K/V: 各8×4；SwiGLU: 3×8×16；输出: 8×11。
    N = 2*(2*8*8 + 2*8*4 + 3*8*16) + 8*11
    # B=3、新块2、历史5：线性2NBT；QKᵀ与AV共4L d B T (past+T)。
    flops = 2*N*3*2 + 4*2*8*3*2*(5+2)
    # 权重读一次；全部可见 K/V 读一次，新 K/V 写一次；bf16=2字节。
    read_kv = 2*2*1*4*(5+2)*3*2
    write_kv = 2*2*1*4*2*3*2
    cost = step_cost(cfg, 2, 5, B=3)
    assert cost["matmul_params"] == N
    assert cost["flops"] == flops
    assert cost["kv_read_bytes"] == read_kv
    assert cost["kv_write_bytes"] == write_kv
    assert cost["bytes"] == N*2 + read_kv + write_kv
    assert cost["intensity"] == flops/cost["bytes"]


def test_intensity_increases_with_batch_and_prefill_length():
    cfg = make_model().cfg
    rows = intensity_sweep(cfg)
    prefill = [r["intensity"] for r in rows if r["phase"] == "预填充"]
    assert all(a < b for a, b in zip(prefill, prefill[1:]))
    for past in (512, 4096):
        decode = [r["intensity"] for r in rows if r["T_past"] == past]
        assert all(a < b for a, b in zip(decode, decode[1:]))


def test_local_configs_offline_fallback_and_text_config(tmp_path):
    qwen, hybrid = local_model_configs(tmp_path)
    assert qwen.source == hybrid.source == "内置数值（待验证）"
    assert qwen.note == "Qwen3-0.6B 内置回退配置（待验证）"
    assert qwen.vocab_size == 151936
    assert kv_cache_bytes_for(qwen, 4096) == 469762048
    assert kv_cache_bytes_for(hybrid, 4096) == 50331648
    # 显式头维与 hidden_size 不同：不能复用 GPT 的 d/n_head 假设。
    cost = step_cost(qwen, 1, 3)
    assert cost["attention_flops"] == 4*28*16*128*4
    with pytest.raises(ValueError, match="混合"):
        step_cost(hybrid, 1, 3)
    path = tmp_path/"models--Qwen--Qwen3.5-0.8B/snapshots/local/config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(text_config=dict(num_hidden_layers=4, num_attention_heads=8,
                    num_key_value_heads=2, head_dim=256, hidden_size=1024, intermediate_size=3584,
                    vocab_size=100, layer_types=["linear_attention"]*3+["full_attention"]))))
    cfg = local_model_configs(tmp_path)[1]
    assert str(path) in cfg.source
    assert cfg.n_layer == 4 and cfg.full_layers == 1
    assert kv_cache_bytes_for(cfg, 4096) == kv_cache_bytes(1, 2, 256, 4096)


@pytest.mark.parametrize("parts", [["0"], ["10"], ["1", "-1"], ["--ckpt", "/missing/ch15.pt"]])
def test_parse_args_rejects_invalid(parts):
    with pytest.raises(SystemExit):
        parse_args(parts)


def test_tour_parts_2_4(capsys):
    main(["2", "4"])
    output = capsys.readouterr().out
    assert "章节=15" in output and "总用时秒=" in output
    assert "K形状=(1, 2, 11, 16)" in output
    assert "mask_01=[[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]" in output


def test_timing_helpers_and_bounds():
    model = make_model()
    assert time_decode(model, 3, n_new=2, repeats=2) > 0
    assert time_decode(model, 3, n_new=2, use_cache=False, repeats=2) > 0
    assert time_prefill(model, 4, repeats=2) > 0
    with pytest.raises(ValueError):
        time_decode(model, 64, repeats=1)
    with pytest.raises(ValueError):
        prefill_chunks(model, torch.zeros((1, 4), dtype=torch.long), [2, 1])


def test_random_checkpoint_loading(tmp_path, capsys):
    # 临时随机检查点只验证 identity 接口，不依赖第13章产物。
    model = make_model(block_size=8)
    path = tmp_path/"checkpoint.pt"
    torch.save(dict(identity=dict(model=asdict(model.cfg), bos=2, eos=3), model=model.state_dict()), path)
    exp = Experiment(parse_args(["--ckpt", str(path)]))
    exp.prepare()
    assert exp.bos == 2 and exp.eos == 3
    for key, value in model.state_dict().items():
        assert torch.equal(exp.model.state_dict()[key], value)
    main(["9", "--ckpt", str(path)])
    assert "输出完全相同=True" in capsys.readouterr().out


def test_qwen_reads_generation_config_not_embedding(tmp_path):
    data = dict(num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
                head_dim=128, hidden_size=1024, intermediate_size=3072, vocab_size=151936)
    for repo, vocab in (("Qwen3-0.6B", 151936), ("Qwen3-Embedding-0.6B", 151669)):
        path = tmp_path/f"models--Qwen--{repo}/snapshots/local/config.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(data, vocab_size=vocab)))
    cfg = local_model_configs(tmp_path)[0]
    assert "models--Qwen--Qwen3-0.6B/" in cfg.source
    assert cfg.vocab_size == 151936
    assert (cfg.n_layer, cfg.n_head, cfg.n_kv_head, cfg.head_dim) == (28, 16, 8, 128)
    assert cfg.note == "Qwen3-0.6B 本地配置"
