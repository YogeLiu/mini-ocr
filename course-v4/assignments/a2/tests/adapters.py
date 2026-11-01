"""A2 的适配器：把你在 assignments/a2/mechanism/ 里写的代码接到测试上。这里只做"接线"，不放实质逻辑。
每个函数的约定以这里的文档字符串和 assignments/a2/handout.md 为准。"""
from __future__ import annotations

from typing import Iterable, Iterator, Protocol

import torch


class TokenizerLike(Protocol):
    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        """False 时特殊符号拼写当普通文本；语料编码必须用 False。"""
        raise NotImplementedError
    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError


# ---------- 第一部分：字节级 BPE ----------
def run_pretokenize(text: str) -> list[str]:
    """按第 6 章正则切词块（regex）：
    r"'(?:s|t|re|ve|m|ll|d)| ?\\p{L}+| ?\\p{N}| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"。
    拼回去等于原文。"""
    raise NotImplementedError


def run_train_bpe(texts: list[str], n_merges: int,
                  special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """返回 (vocab, merges)。编号约定：0–255 是单个字节，256 + i 是第 i 条合并的结果，特殊符号按给出的顺序排在最后。
    特殊符号不参与训练：先在每段文本里把它们切掉，切开的两边分别预分词。平局取 (bytes,bytes) 字典序最大；最高频次<2 停止。负合并数或空/重复特殊符号 ValueError。"""
    raise NotImplementedError


def get_tokenizer(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                  special_tokens: list[str]) -> TokenizerLike:
    """encode：特殊符号整段编码成一个编号（多个特殊符号重叠时取最长的）；其余部分预分词后按合并顺序编码。
    decode：字节拼起来按 UTF-8 解码，不完整的字节用 U+FFFD 代替。
    encode_iterable：逐个元素（例如文件的每一行）编码并依次产出编号；在元素边界也是预分词边界的前提下，
    结果与把所有元素拼起来再 encode 相同；encode(text, allow_special=False) 把特殊符号拼写当普通文本。"""
    raise NotImplementedError


# ---------- 第二部分：Transformer 语言模型 ----------
def run_rmsnorm(d: int, eps: float, weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """最后维 RMSNorm；保留输入/weight 梯度，低精度在 fp32 计算；形状不符或 eps<0 报 ValueError。"""
    raise NotImplementedError


def run_rope(x: torch.Tensor, base: float, pos0: int) -> torch.Tensor:
    """x (B, h, T, dh)，第 t 个位置的编号是 pos0 + t。base>0、pos0>=0、dh 正偶数，否则 ValueError。用"前后两半"约定（第 i 维与第 i + dh/2 维配对）。"""
    raise NotImplementedError


def run_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """缩放点积注意力，返回输出。mask 是 bool，True 表示可以看；None 表示全都可以看。q/k (...,T,d)、v (...,Tk,dv)；错尺寸、非 bool 或全遮蔽行 ValueError。"""
    raise NotImplementedError


def get_gpt(cfg: dict) -> torch.nn.Module:
    """cfg 的键：vocab_size、block_size、n_layer、n_head、d_model、d_ff、rope_base、tie_weights。
    结构与 state_dict 的键名见 assignments/a2/handout.md；支持 d_ff=0 自动取整，以及 pos_emb/norm/resid_scale 消融开关。model(idx) 返回 (logits, None)；model(idx, targets) 返回 (logits, loss)，
    loss 是按 token 平均的交叉熵，targets 中 −100 的位置不计入。"""
    raise NotImplementedError


def run_gpt(cfg: dict, state_dict: dict, idx: torch.Tensor) -> torch.Tensor:
    """装好权重，返回 logits (B, T, vocab_size)。"""
    raise NotImplementedError


def run_gpt_incremental(cfg: dict, state_dict: dict, idx: torch.Tensor, n_prefill: int) -> torch.Tensor:
    """先把 idx[:, :n_prefill] 一次送进去（预填充），之后用 KV cache 每次送 1 个 token。
    把每一步得到的 logits 拼起来返回，形状 (B, T, vocab_size)，应当与一次算完整个序列相同。n_prefill 不在 [1,T] 内报 ValueError。"""
    raise NotImplementedError


def run_process_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """logits (B, V) → 处理后的 logits，不许采样的位置为 −∞。顺序：温度 → top-k → top-p。temperature 有限正数，top_k>=0，0<top_p<=1，否则 ValueError；top-k 并列全保留，top-p 含跨阈值项。"""
    raise NotImplementedError


# ---------- 第三部分：训练工具 ----------
def run_get_batch(ids: torch.Tensor, batch_size: int, context: int,
                  generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """ids 是一维 LongTensor。用 generator 均匀地选起点 s ∈ [0, len(ids) − context − 1]，x = ids[s:s+context]，y 错开一位。ids 非一维 long、batch_size<1 或 context 不在 [1,len(ids)-1] 内报 ValueError。"""
    raise NotImplementedError


def run_accumulate_grads(model: torch.nn.Module, micro_batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    """先清零梯度，再对每个小批 (x, y) 前向、反向。y 中 −100 的位置不计损失。
    累积后的 .grad 必须等于把所有小批拼成一个大批、按全部有效 token 平均算出的梯度。返回大批的平均损失；无有效目标报 ValueError。"""
    raise NotImplementedError


def run_nll_sum(model: torch.nn.Module, ids: torch.Tensor, context: int) -> float:
    """从 0 起每 context 个 token 独立切窗（末窗可短，不跨窗传递缓存），返回整段 token 的总负对数似然（nat）。长度<2 或 context<1 报 ValueError；这是训练诊断工具，不是逐公式最终评测。"""
    raise NotImplementedError


def run_swiglu(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """x (...,d)，gate/up (f,d)，down (d,f)；无偏置，down(silu(gate(x))*up(x))；保留所有梯度，错形状 ValueError。"""
    raise NotImplementedError


def run_multihead(x: torch.Tensor, kv: torch.Tensor, n_heads: int, wq: torch.Tensor,
                  wk: torch.Tensor, wv: torch.Tensor, wo: torch.Tensor, mask=None) -> torch.Tensor:
    """x (B,Tq,d)、kv (B,Tk,d)，四权重 (d,d)，线性为 x@W.T，无偏置/位置编码。
    mask True 可见，返回 (B,Tq,d)，保留所有梯度；头数不整除或错形状 ValueError。
    """
    raise NotImplementedError


def run_generate(model, idx: torch.Tensor, max_new: int, eos: int, banned: list[int],
                 greedy: bool = True, seed: int = 0, use_cache: bool = True) -> torch.Tensor:
    """返回含提示的编号；禁 banned，遇 EOS 的行持续补 EOS，全结束即停。
    贪心或温度1多项式采样，seed 可复现；cache 路径预填充后只传一个 token。
    max_new<0、空提示报 ValueError；max_new=0 原样返回。
    """
    raise NotImplementedError
