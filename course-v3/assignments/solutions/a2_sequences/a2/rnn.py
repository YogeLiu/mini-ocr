"""A2 参考实现：LSTM 与 CRNN。"""
import math

import torch
import torch.nn as nn

from .nn import BatchNorm2d, Conv2d, Linear, log_softmax, max_pool2d, relu


class LSTMLayer(nn.Module):
    """单层、单向。参数与 nn.LSTM 的一层相同：weight_ih (4H, d_in)、weight_hh (4H, H)、bias_ih、bias_hh (4H,)，
    四个门的顺序是 i、f、g、o。初始化：均匀分布 U(−1/√H, 1/√H)。"""

    def __init__(self, d_in, hidden):
        super().__init__()
        self.hidden = hidden
        k = 1 / math.sqrt(hidden)
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden, d_in).uniform_(-k, k))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden, hidden).uniform_(-k, k))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden).uniform_(-k, k))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden).uniform_(-k, k))

    def forward(self, x, state=None):
        """x (T, B, d_in) → 输出 (T, B, H)，以及最后的 (h, c)。"""
        T, B, _ = x.shape
        if state is None:
            h = x.new_zeros(B, self.hidden)
            c = x.new_zeros(B, self.hidden)
        else:
            h, c = state
        xw = x @ self.weight_ih.T + self.bias_ih                    # 输入部分一次算完所有时间步
        out = []
        for t in range(T):
            i, f, g, o = (xw[t] + h @ self.weight_hh.T + self.bias_hh).chunk(4, -1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            out.append(h)
        return torch.stack(out), (h, c)


class BiLSTM(nn.Module):
    """多层双向：每层一个正向、一个反向（把序列倒过来跑，再倒回去），两个方向的输出拼起来给下一层。"""

    def __init__(self, d_in, hidden, layers=2):
        super().__init__()
        self.fw = nn.ModuleList(LSTMLayer(d_in if i == 0 else 2 * hidden, hidden) for i in range(layers))
        self.bw = nn.ModuleList(LSTMLayer(d_in if i == 0 else 2 * hidden, hidden) for i in range(layers))

    def forward(self, x):
        for f, b in zip(self.fw, self.bw):
            y1, _ = f(x)
            y2, _ = b(x.flip(0))
            x = torch.cat([y1, y2.flip(0)], -1)
        return x


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.bn = BatchNorm2d(c_out)

    def forward(self, x):
        return relu(self.bn(self.conv(x)))


class CRNN(nn.Module):
    """(B, 1, 32, W) → (T = W/4, B, K) 的对数概率。CNN 把高度压到 1、宽度缩小 4 倍，BiLSTM 沿宽度看上下文。"""

    def __init__(self, n_classes, c=128, hidden=128, layers=2):
        super().__init__()
        self.b1, self.b2 = ConvBlock(1, 32), ConvBlock(32, 64)
        self.b3, self.b4 = ConvBlock(64, 128), ConvBlock(128, c)
        self.rnn = BiLSTM(c, hidden, layers) if layers else None
        self.head = Linear(2 * hidden if layers else c, n_classes)

    def forward(self, x):
        x = max_pool2d(self.b1(x), 2)
        x = max_pool2d(self.b2(x), 2)
        x = max_pool2d(self.b3(x), 2, 1)                            # 只缩高度
        f = self.b4(x).mean(dim=2).permute(2, 0, 1)                 # (T, B, c)
        if self.rnn is not None:
            f = self.rnn(f)
        return log_softmax(self.head(f), dim=-1)
