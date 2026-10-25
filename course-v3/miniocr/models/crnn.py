"""第 13 章：CRNN = CNN（把图像变成一串列特征）+ RNN（沿着列看上下文）+ CTC（不需要知道每个字符在哪一列）。

    MyRNN        自己写的最简单的循环神经网络（Elman RNN），与 nn.RNN 对拍
    CRNN         输入 (B, 1, 32, W) → 输出 (T = W/4, B, K) 的对数概率，交给 F.ctc_loss
    CTCCollate   DataLoader 的 collate_fn：把一批 (图像, 文字) 变成 CTC 需要的张量
    ctc_decode   对一批输出做贪心解码，返回字符串
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ctc import greedy_decode
from ..data.text import LINE_CHARSET
from ..utils import Vocab
from .cnn import conv_block


def line_vocab():
    """文字行的字符表：第 0 个是 CTC 的空白。"""
    return Vocab(LINE_CHARSET, specials=("<blank>",))


# >>> 练习 13.6：循环神经网络
class MyRNN(nn.Module):
    """输入 x (T, B, d_in)，初始状态 h0 (B, d_h)（None 表示全 0）。对 t = 0..T−1：
        h_t = tanh(x_t W_ihᵀ + b_ih + h_{t−1} W_hhᵀ + b_hh)
    返回 (所有时间步的 h，形状 (T, B, d_h)；最后的 h_T−1，形状 (B, d_h))。
    参数名、形状与 nn.RNN 的第 0 层相同：weight_ih_l0 (d_h, d_in)、weight_hh_l0 (d_h, d_h)、bias_ih_l0、bias_hh_l0，
    都用 U(−1/√d_h, 1/√d_h) 初始化。"""

    def __init__(self, d_in, d_h):
        super().__init__()
        k = 1 / math.sqrt(d_h)
        self.d_h = d_h
        self.weight_ih_l0 = nn.Parameter(torch.empty(d_h, d_in).uniform_(-k, k))
        self.weight_hh_l0 = nn.Parameter(torch.empty(d_h, d_h).uniform_(-k, k))
        self.bias_ih_l0 = nn.Parameter(torch.empty(d_h).uniform_(-k, k))
        self.bias_hh_l0 = nn.Parameter(torch.empty(d_h).uniform_(-k, k))

    def forward(self, x, h0=None):
        h = x.new_zeros(x.shape[1], self.d_h) if h0 is None else h0
        outs = []
        for t in range(x.shape[0]):
            h = torch.tanh(x[t] @ self.weight_ih_l0.T + self.bias_ih_l0 + h @ self.weight_hh_l0.T + self.bias_hh_l0)
            outs.append(h)
        return torch.stack(outs), h
# <<<


class CRNN(nn.Module):
    """CNN 部分把高度从 32 压到 1、宽度缩小 4 倍：
        conv_block(1, 32)    → MaxPool(2, 2)    (32, 16, W/2)
        conv_block(32, 64)   → MaxPool(2, 2)    (64, 8, W/4)
        conv_block(64, 128)  → MaxPool((2, 1))  (128, 4, W/4)     只缩高度，保留宽度方向的分辨率
        conv_block(128, c)   → 对高度求平均      (c, 1, W/4)
    得到 T = W/4 个时间步，每步一个 c 维向量，即图像的一"列"（宽 4 像素、感受野更宽）。
    rnn="lstm"：再经过双向 LSTM（2 层，每个方向 hidden 个单元）；rnn="none"：直接接线性层（每列独立分类）。
    返回 log_softmax 后的 (T, B, K)。"""

    def __init__(self, n_classes, c=128, hidden=128, rnn="lstm"):
        super().__init__()
        self.cnn = nn.Sequential(conv_block(1, 32), nn.MaxPool2d(2),
                                 conv_block(32, 64), nn.MaxPool2d(2),
                                 conv_block(64, 128), nn.MaxPool2d((2, 1)),
                                 conv_block(128, c))
        self.rnn = nn.LSTM(c, hidden, num_layers=2, bidirectional=True) if rnn == "lstm" else None
        self.head = nn.Linear(2 * hidden if self.rnn is not None else c, n_classes)

    def forward(self, x):
        f = self.cnn(x).mean(dim=2)                  # (B, c, T)
        f = f.permute(2, 0, 1)                       # (T, B, c)：时间维在前，与 nn.LSTM、F.ctc_loss 的约定一致
        if self.rnn is not None:
            f, _ = self.rnn(f)
        return F.log_softmax(self.head(f), dim=-1)


class CTCCollate:
    """把 [(图像, 文字), ...] 变成 (图像 (B,1,H,W), 拼接后的标签 (ΣU,), 每个标签的长度 (B,), 原始文字列表)。
    F.ctc_loss 接受"所有标签首尾相接成一个一维张量 + 各自的长度"这种格式，省去补齐。"""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, batch):
        xs = torch.stack([b[0] for b in batch])
        texts = [b[1] for b in batch]
        ids = [self.vocab.encode(t) for t in texts]
        targets = torch.tensor([i for seq in ids for i in seq], dtype=torch.long)
        lengths = torch.tensor([len(seq) for seq in ids], dtype=torch.long)
        return xs, targets, lengths, texts


def ctc_batch_loss(log_probs, targets, lengths):
    """所有样本的输入长度都是 T。reduction="mean" 时 PyTorch 先把每个样本的损失除以它的标签长度，再对批求平均。"""
    T, B, _ = log_probs.shape
    input_lengths = torch.full((B,), T, dtype=torch.long)
    return F.ctc_loss(log_probs, targets, input_lengths, lengths, blank=0, reduction="mean", zero_infinity=True)


def ctc_decode(log_probs, vocab):
    """log_probs (T, B, K) → 贪心解码后的字符串列表。"""
    lp = log_probs.detach().cpu().numpy()
    return [vocab.decode(greedy_decode(lp[:, b])) for b in range(lp.shape[1])]
