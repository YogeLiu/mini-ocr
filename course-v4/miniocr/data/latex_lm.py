"""第 8–16 章共享的 LaTeX LM 数据与评测；辅助代码，不是挖空练习。

分词器遵循 latex-v1：标准训练集前 20000 条、2000 次 BPE 合并，
特殊符号只有 BOS/EOS。每次重新训练，不使用可能过期的磁盘缓存。
最终评测固定取 val 前 2000 条；EOS 是目标，BOS 只作条件且屏蔽输出；
按目标 token 加权。字节数只计清洗后原文 UTF-8，不计边界符。
长公式逐目标滑窗，每个内容 token 和 EOS 恰好计一次，不截断目标；
窗口内位置从零开始，不跨公式。此协议与随机长流训练的近似评测不同。
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from miniocr.data.latex_corpus import standard_split
from miniocr.tokenizer import BPETokenizer, train_bpe
from miniocr.train.lm import bits_per_byte


@dataclass
class LatexData:
    tokenizer: BPETokenizer
    train_text: list
    val_text: list
    train: list
    val: list
    n_bytes: int


def train_tokenizer(train_text, num_merges=2000):
    """仅传入训练划分；固定取其前 20000 条，不能传验证或测试文本。"""
    return BPETokenizer(train_bpe(train_text[:20000], num_merges),
                        specials=("<|bos|>", "<|eos|>"))


def encode_formulas(tok, texts, stream=False):
    """返回 (逐条 BOS…EOS 序列或拼接的 LongTensor, 原文 UTF-8 字节数)。

    stream=True 可交给 get_batch；跨边界的 BOS 目标须由训练方忽略，
    因为评测协议不预测 BOS。文本中的特殊符号拼写一律当作普通文字。
    """
    bos, eos = (tok.special_ids[s] for s in ("<|bos|>", "<|eos|>"))
    seqs = [[bos] + tok.encode(t, allow_special=False) + [eos] for t in texts]
    ids = torch.tensor([i for seq in seqs for i in seq], dtype=torch.long) if stream else seqs
    return ids, sum(len(t.encode("utf-8")) for t in texts)


def load_latex_lm(n_train=20000):
    """只读本地标准划分；无缓存时由 standard_split 警告并降级为样例。"""
    if n_train < 1:
        raise ValueError("n_train 必须为正")
    split = standard_split(n_train=max(20000, n_train))
    tok = train_tokenizer(split["train"])
    train_text, val_text = split["train"][:n_train], split["val"][:2000]
    train, _ = encode_formulas(tok, train_text)
    val, n_bytes = encode_formulas(tok, val_text)
    return LatexData(tok, train_text, val_text, train, val, n_bytes)


class FormulaWindows:
    """逐目标滑窗的紧凑索引；每条公式单独加 BOS/EOS，不拼接公式上文。

    seqs 是不含 BOS/EOS 的内容编号；encode_formulas 的逐条结果需先取 [1:-1]。
    每个内容 token 和 EOS 各作一次目标；长公式滑窗而不丢目标。
    批内右补齐（不计入注意力），只保留一个真实 BOS。不是挖空练习。
    """

    def __init__(self, seqs, bos, eos, context):
        if context < 1:
            raise ValueError("context 必须为正")
        flat, ends, starts = [], [], []
        for seq in seqs:
            offset = len(flat)
            flat.extend([bos] + list(seq) + [eos])
            for end in range(offset + 1, len(flat)):
                ends.append(end)
                starts.append(max(offset, end - context))
        self.ids = torch.tensor(flat, dtype=torch.long)
        self.ends = torch.tensor(ends, dtype=torch.long)
        self.starts = torch.tensor(starts, dtype=torch.long)
        self.context = context
        self.bos, self.eos = bos, eos

    def __len__(self):
        return len(self.ends)

    def batch(self, indices):
        ends, starts = self.ends[indices], self.starts[indices]
        lengths = ends - starts
        offsets = torch.arange(self.context)
        positions = starts[:, None] + offsets
        # 补齐处只重复本窗口最后一个输入，不读取目标或下一条公式。
        positions = positions.minimum(ends[:, None] - 1)
        return self.ids[positions], lengths, self.ids[ends]


@torch.no_grad()
def evaluate_nll(model, data, indices=None, batch=256, window_logits=None):
    """返回按目标数加权的 nat/token；临时 eval 并恢复模型模式。

    默认 model(x) 返回 (B,T,V) logits 或 (logits, loss)。因果模型的右补齐
    不影响此前真实位置。window_logits(model,x,lengths) 可只投影最后真实
    位置并返回 (B,V)，节省大词表的计算；两条路径评测相同目标。
    调用者须把模型和数据放在 CPU（本课程短实验的统一设备）。
    """
    indices = torch.arange(len(data)) if indices is None else indices
    if len(indices) == 0 or batch < 1:
        raise ValueError("评测目标不能为空，batch 必须为正")
    was = model.training
    model.eval()
    total = 0.0
    try:
        for chunk in indices.split(batch):
            x, lengths, y = data.batch(chunk)
            if window_logits is None:
                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                logits = logits[torch.arange(len(x)), lengths - 1]
            else:
                logits = window_logits(model, x, lengths)
            logits = logits.float().clone()
            logits[:, data.bos] = -torch.inf
            total += F.cross_entropy(logits, y, reduction="sum").item()
    finally:
        model.train(was)
    return total / len(indices)


def evaluate_lm(model, data, n_bytes, batch=256, window_logits=None):
    """完整遍历已构造的验证滑窗；n_bytes 必须与这些公式的原文对应。"""
    if n_bytes <= 0:
        raise ValueError("bits/byte 需要正的原文字节数")
    nll = evaluate_nll(model, data, batch=batch, window_logits=window_logits)
    return dict(nat_per_token=nll, bits_per_byte=bits_per_byte(nll, len(data), n_bytes),
                n_tokens=len(data), n_bytes=n_bytes)


def evaluate_latex(model, tok, val_text, context, batch=256, window_logits=None):
    """标准入口：传入标准 val 文本，固定评测前 2000 条。"""
    seqs, n_bytes = encode_formulas(tok, val_text[:2000])
    bos, eos = (tok.special_ids[s] for s in ("<|bos|>", "<|eos|>"))
    data = FormulaWindows([s[1:-1] for s in seqs], bos, eos, context)
    return evaluate_lm(model, data, n_bytes, batch, window_logits)
