"""A2 的适配器：把你在 assignments/a2_sequences/a2/ 里写的代码接到测试上。这里只做"接线"，不放实质逻辑。"""
from __future__ import annotations

from typing import Iterable, Iterator, Protocol

import numpy as np
import torch


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]: ...


# ---------- 第一部分：分词 ----------
def run_pretokenize(text: str) -> list[str]:
    raise NotImplementedError


def run_train_bpe(text: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, str], list[tuple[str, str]]]:
    """返回 (vocab, merges)：vocab 是 {编号: 字符串}，merges 按学到的先后排列。"""
    raise NotImplementedError


def get_tokenizer(vocab: dict[int, str], merges: list[tuple[str, str]], special_tokens: list[str]) -> TokenizerLike:
    raise NotImplementedError


# ---------- 第二部分：CTC 与 LSTM ----------
def run_ctc_loss(log_probs: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor,
                 target_lengths: torch.Tensor, reduction: str) -> torch.Tensor:
    """log_probs (T, N, C)，targets (N, S) 补齐，空白是第 0 类。与 F.ctc_loss(..., blank=0, reduction=reduction) 相同。"""
    raise NotImplementedError


def run_ctc_greedy(log_probs: torch.Tensor) -> list[int]:
    """log_probs (T, C) → 标签。"""
    raise NotImplementedError


def run_ctc_beam(log_probs: torch.Tensor, beam: int) -> list[int]:
    """前缀束搜索，log_probs (T, C) → 最可能的标签。"""
    raise NotImplementedError


def run_lstm(d_in: int, hidden: int, weights: dict[str, torch.Tensor], x: torch.Tensor,
             h0: torch.Tensor, c0: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """单层单向 LSTM。weights 的键与 nn.LSTM 一层相同：weight_ih、weight_hh、bias_ih、bias_hh。
    x (T, B, d_in)，h0、c0 (B, hidden)。返回 (输出 (T, B, hidden), (h_T, c_T))。"""
    raise NotImplementedError


def get_crnn(n_classes: int) -> torch.nn.Module:
    """返回新初始化的 CRNN：输入 (B, 1, 32, W)，输出 (W // 4, B, n_classes) 的对数概率。"""
    raise NotImplementedError


def run_cer(preds: list[str], refs: list[str]) -> float:
    raise NotImplementedError


# ---------- 第三部分：Transformer 语言模型 ----------
def run_rmsnorm(d: int, eps: float, weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def run_rope(x: torch.Tensor, base: float, pos0: int) -> torch.Tensor:
    """x (B, h, T, dh)，第 t 个位置的编号是 pos0 + t。返回加上 RoPE 之后的 x。"""
    raise NotImplementedError


def run_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """缩放点积注意力。mask 是 bool，True 表示可以看；None 表示全都可以看。"""
    raise NotImplementedError


def get_gpt(cfg: dict) -> torch.nn.Module:
    """cfg 的键：vocab_size、block_size、n_layer、n_head、d_model、d_ff、rope_base、tie_weights。
    state_dict 的键见说明书第 5.5 节。模型的前向 model(idx) 返回 (logits, loss)，model(idx, targets) 时 loss 是交叉熵。"""
    raise NotImplementedError


def run_gpt(cfg: dict, state_dict: dict, idx: torch.Tensor) -> torch.Tensor:
    """装好权重，返回 logits (B, T, vocab_size)。"""
    raise NotImplementedError


def run_gpt_incremental(cfg: dict, state_dict: dict, idx: torch.Tensor, n_prefill: int) -> torch.Tensor:
    """先把 idx[:, :n_prefill] 一次送进去（预填充），之后用 KV cache 每次送 1 个 token。
    把每一步得到的 logits 拼起来返回，形状 (B, T, vocab_size)，应当与一次算完整个序列相同。"""
    raise NotImplementedError


def run_process_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """logits (B, V) → 处理后的 logits，不许采样的位置为 −∞。"""
    raise NotImplementedError


def run_get_batch(ids: np.ndarray, batch_size: int, context: int,
                  generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError


def run_accumulate_grads(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, micro_batches: int) -> float:
    """清零梯度，把 (x, y) 均分成 micro_batches 份逐份反向。返回整批的平均损失；参数的 .grad 应当与一次算整批相同。"""
    raise NotImplementedError


def run_nll_sum(model: torch.nn.Module, ids: np.ndarray, context: int) -> float:
    """按说明书第 5.8 节的切窗方法，返回整段 token 的总负对数似然（nat）。"""
    raise NotImplementedError
