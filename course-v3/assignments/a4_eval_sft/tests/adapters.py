"""A4 的适配器：把你在 assignments/a4_eval_sft/a4/ 里写的代码接到测试上。这里只做"接线"，不放实质逻辑。"""
from __future__ import annotations

from typing import Callable

import torch

# 树的通用表示（测试用）：(标签, [子树, …])；表格解析的结果：(tag, colspan, rowspan, text, [子节点, …])
Tree = tuple


# ---------- 第一部分：指标 ----------
def run_edit_distance(a, b) -> int:
    raise NotImplementedError


def run_align_ops(pred: str, ref: str) -> list[tuple[str, str, str]]:
    """把 pred 变成 ref 的一组最少操作 [(op, pred 的字符, ref 的字符)]，op ∈ "=" "S" "D" "I"；回溯优先级见说明书 §3.1。"""
    raise NotImplementedError


def run_cer(preds: list[str], refs: list[str]) -> float:
    raise NotImplementedError


def run_wer(preds: list[str], refs: list[str]) -> float:
    raise NotImplementedError


def run_ned(p: str, r: str) -> float:
    raise NotImplementedError


def run_bleu(preds: list[list[str]], refs: list[list[str]], max_n: int) -> float:
    raise NotImplementedError


def run_anls(preds: list[str], answers: list[list[str]], tau: float) -> float:
    raise NotImplementedError


def run_parse_table(html: str) -> Tree:
    """返回 table 节点，转换成 (tag, colspan, rowspan, text, [子节点…]) 的嵌套元组。"""
    raise NotImplementedError


def run_tree_edit_distance(a: Tree, b: Tree, rename: Callable[[str, str], float]) -> float:
    """a、b 是 (标签, [子树…]) 的嵌套元组；rename(标签1, 标签2) 是改标签的代价；插入、删除各 1。
    在这里把它们转换成你的节点类型，再调用你的树编辑距离。"""
    raise NotImplementedError


def run_teds(pred_html: str, ref_html: str, structure_only: bool) -> float:
    raise NotImplementedError


# ---------- 第二部分：统计 ----------
def run_bootstrap_ci(num, den, n_boot: int, alpha: float, seed: int) -> tuple[float, float, float]:
    raise NotImplementedError


def run_paired_bootstrap(num_a, num_b, den, n_boot: int, alpha: float, seed: int) -> tuple[float, float, float, float]:
    raise NotImplementedError


def run_mcnemar(correct_a, correct_b) -> tuple[int, int, float]:
    raise NotImplementedError


def run_reliability(conf, correct, n_bins: int) -> list[tuple]:
    raise NotImplementedError


def run_ece(conf, correct, n_bins: int) -> float:
    raise NotImplementedError


def run_auroc(scores, labels) -> float:
    raise NotImplementedError


# ---------- 第三部分：对数概率 ----------
def run_token_logprobs(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def run_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def run_seq_logprob(lp: torch.Tensor, mask: torch.Tensor, reduce: str) -> torch.Tensor:
    raise NotImplementedError


def run_labels_to_mask(targets: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def run_mask_to_labels(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def run_mask_from_lengths(prompt_lens, total_lens, T: int) -> torch.Tensor:
    raise NotImplementedError


# ---------- 第四部分：SFT ----------
def run_chat_roles(messages: list[dict], n_img, encode: Callable[[str], list[int]], bos: int, eos: int,
                   img: int) -> tuple[list[int], list[int]]:
    """roles 的编码：格式 0、图像 1、用户 2、回答 3。"""
    raise NotImplementedError


def run_sft_targets(ids: list[int], roles: list[int], prompt_weight: float, eos_id: int,
                    train_eos: bool) -> tuple[list[int], list[float]]:
    raise NotImplementedError


def run_sft_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None, mode: str) -> torch.Tensor:
    raise NotImplementedError


def run_pack(examples: list[tuple[list[int], list[int], list[float]]], max_len: int, pad_id: int) -> dict:
    """返回 dict：ids、targets、weights、seg、pos 五个 (箱子数, max_len) 的张量，order（每个箱子装的样本编号）。"""
    raise NotImplementedError


def run_packed_mask(seg: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def get_lm(vocab_size: int, block_size: int, n_layer: int, n_head: int, d_model: int) -> torch.nn.Module:
    """你的 GPT（A2/A3 的结构，tie_weights=True）。"""
    raise NotImplementedError


def run_lm_logits(lm: torch.nn.Module, ids: torch.Tensor, pos: torch.Tensor | None = None,
                  mask: torch.Tensor | None = None) -> torch.Tensor:
    """pos (B, T)：每个 token 的位置编号（None = 0, 1, 2, …）；mask (B, 1, T, T)：注意力 mask（None = 因果）。返回 logits。"""
    raise NotImplementedError


# ---------- 第五部分：LoRA 与内存 ----------
def get_lora_linear(weight: torch.Tensor, bias: torch.Tensor | None, r: int, alpha: float) -> torch.nn.Module:
    """用给定的 weight (out, in)、bias 建一个你的 Linear，再包成 LoRA 层。返回的模块要有参数 A (r, in) 和 B (out, r)。"""
    raise NotImplementedError


def run_apply_lora(model: torch.nn.Module, targets: list[str], r: int, alpha: float) -> list[str]:
    raise NotImplementedError


def run_merge_lora(model: torch.nn.Module) -> int:
    raise NotImplementedError


def run_count_params(model: torch.nn.Module) -> tuple[int, int]:
    raise NotImplementedError


def run_static_memory(n_params: int, n_trainable: int, param_dtype: str, optimizer: str, master_fp32: bool) -> dict:
    raise NotImplementedError


def get_saved_tensors(model: torch.nn.Module | None = None):
    """返回一个上下文管理器：with 块里的前向结束后，它的 .bytes 是保存下来留给反向的张量的字节数。"""
    raise NotImplementedError


def run_checkpoint_blocks(blocks: torch.nn.ModuleList) -> int:
    raise NotImplementedError
