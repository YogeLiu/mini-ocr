"""第 1 章：全书反复用到的小工具。每一个都短小，但后面的章节都依赖它们。

    字符表 Vocab        文字 <-> 整数编号（第 10、13、14、22 章）
    one_hot             类别编号 -> 独热向量（第 6、7 章）
    pad_sequences       长短不一的序列补齐成一个矩形数组，并给出掩码（第 13、16、22 章）
    pairwise_sq_dists   两组向量两两之间的距离平方（第 2 章的模板分类器、第 20 章）
    minibatches         把数据随机分成小批（第 7、10 章）
"""
import numpy as np


# >>> 练习 1.2：字符表
class Vocab:
    """字符表：把字符映射成整数编号，反过来也行。

    编号 0..len(specials)-1 留给特殊符号（例如 "<pad>" 补齐、"<eos>" 结束），后面是按码位排好序的普通字符。
    排序是为了让同样的字符集合总得到同样的编号——不排序的话，编号取决于字符出现的先后，换一批数据就变了。
    """

    def __init__(self, chars, specials=("<pad>",)):
        self.specials = list(specials)
        self.itos = self.specials + sorted(set(chars) - set(self.specials))   # index -> string
        self.stoi = {s: i for i, s in enumerate(self.itos)}                     # string -> index

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        """字符串 -> 编号列表。遇到字符表里没有的字符抛出 KeyError（宁可报错，也不悄悄丢掉）。"""
        return [self.stoi[c] for c in text]

    def decode(self, ids, skip_specials=True):
        """编号列表 -> 字符串。默认跳过特殊符号。"""
        n = len(self.specials)
        return "".join(self.itos[i] for i in ids if not (skip_specials and i < n))
# <<<


# >>> 练习 1.3：独热向量
def one_hot(labels, n_classes):
    """类别编号 (N,) -> 独热矩阵 (N, n_classes)，float32。第 i 行只有第 labels[i] 列是 1，其余是 0。

    用花式下标一步完成：先造全 0 矩阵，再把 out[i, labels[i]] 设成 1。
    """
    labels = np.asarray(labels)
    out = np.zeros((len(labels), n_classes), dtype=np.float32)
    out[np.arange(len(labels)), labels] = 1.0
    return out
# <<<


# >>> 练习 1.4：补齐
def pad_sequences(seqs, pad=0):
    """把若干个长度不同的整数序列补齐成 (N, L) 的数组，L 是最长序列的长度。

    返回 (arr, mask)：arr 中补上的位置填 pad；mask 是同形状的 bool 数组，真实内容处为 True、补齐处为 False。
    例如 [[5, 6, 7], [8]] -> arr [[5, 6, 7], [8, 0, 0]]，mask [[T, T, T], [T, F, F]]。
    """
    L = max((len(s) for s in seqs), default=0)
    arr = np.full((len(seqs), L), pad, dtype=np.int64)
    mask = np.zeros((len(seqs), L), dtype=bool)
    for i, s in enumerate(seqs):
        arr[i, :len(s)] = s
        mask[i, :len(s)] = True
    return arr, mask
# <<<


# >>> 练习 1.5：两两距离
def pairwise_sq_dists(A, B):
    """A (N, D)、B (M, D) -> (N, M)，第 [i, j] 个元素是 ||A[i] - B[j]||²。

    用广播：A[:, None, :] 形状 (N, 1, D)，B[None, :, :] 形状 (1, M, D)，相减得到 (N, M, D)，再对最后一维求和。
    （第 4 章会讲一个更省内存的写法：||a||² - 2a·b + ||b||²。）
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
# <<<


# >>> 练习 1.6：小批量
def minibatches(n, batch_size, rng=None, drop_last=False):
    """把下标 0..n-1 分成若干批，逐批产出（这是一个生成器，用 for 循环取）。

    rng 不为 None 时先打乱顺序（训练时要打乱，评测时不需要）。drop_last=True 时丢掉最后不满一批的部分。
    """
    idx = np.arange(n) if rng is None else rng.permutation(n)
    for start in range(0, n, batch_size):
        batch = idx[start:start + batch_size]
        if drop_last and len(batch) < batch_size:
            return
        yield batch
# <<<
