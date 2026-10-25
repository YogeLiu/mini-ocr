"""第 35 章的自动检查（不需要下载模型）。"""
import pytest
import torch
import torch.nn as nn

from miniocr.hf_train import answer_mask_by_prefix, hf_answer_logp, lora_target_names, trl_reward


def test_answer_mask_by_prefix():
    assert answer_mask_by_prefix([1, 2, 3], [1, 2, 3, 7, 8, 9]) == [0, 0, 0, 1, 1, 1]
    with pytest.raises(ValueError, match="第 2 个"):
        answer_mask_by_prefix([1, 2, 3], [1, 2, 4, 7])         # 边界上的 token 被合并了：不是前缀
    with pytest.raises(ValueError):
        answer_mask_by_prefix([1, 2, 3], [1, 2])


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj, self.o_proj = nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)
        self.in_proj_qkv = nn.Linear(4, 12)


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = nn.ModuleList([_Attn()])
        self.language_model = nn.ModuleList([_Attn(), _Attn()])
        self.lm_head = nn.Linear(4, 10)


def test_lora_target_names():
    m = _Toy()
    names = lora_target_names(m, suffixes=("q_proj", "o_proj"))
    assert names == ["language_model.0.q_proj", "language_model.0.o_proj", "language_model.1.q_proj", "language_model.1.o_proj"]
    assert len(lora_target_names(m, suffixes=("q_proj",), skip=())) == 3          # 不跳过时视觉塔也算
    assert lora_target_names(m, suffixes=("in_proj_qkv",)) == ["language_model.0.in_proj_qkv", "language_model.1.in_proj_qkv"]


def test_hf_answer_logp():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(1, 4, 6, generator=g)
    ids = torch.tensor([[0, 3, 5, 2]])
    mask = torch.tensor([[0, 0, 1, 1]])                               # 回答是位置 2、3 的 token
    lp = torch.log_softmax(logits, -1)
    want = lp[0, 1, 5] + lp[0, 2, 2]                                   # 位置 t 的 logits 预测 ids[t + 1]
    assert torch.allclose(hf_answer_logp(logits, ids, mask), want.reshape(1))
    assert torch.allclose(hf_answer_logp(logits, ids, mask, "mean"), (want / 2).reshape(1))


def test_trl_reward():
    f = trl_reward("text")
    out = f(["hello", [{"role": "assistant", "content": " hallo "}],
             [{"role": "assistant", "content": [{"type": "text", "text": "hello"}]}], ""], ["hello"] * 4)
    assert out[0] == 1.0 and abs(out[1] - 0.8) < 1e-9 and out[2] == 1.0 and out[3] == 0.0   # 空回答：格式闸门
    assert trl_reward("count")(["3", "three"], ["3", "3"]) == [1.0, 0.0]
    assert f.__name__ == "reward_text"
