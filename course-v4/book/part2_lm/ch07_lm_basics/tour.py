"""第 7 章：语言模型基础。只读本地语料，全程 CPU，不保存模型权重。

    python -m book.part2_lm.ch07_lm_basics.tour [部分号 ...]
    python -m book.part2_lm.ch07_lm_basics.tour 6 --contexts 1 3 8 16
    python -m book.part2_lm.ch07_lm_basics.tour --small --steps 20

输出 outputs/book/ch07/；辅助函数不是挖空练习。
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ch07-matplotlib")

import numpy as np
import torch
import torch.nn.functional as F

from miniocr.data.latex_corpus import SAMPLE_FORMULAS, corpus_available, right_positions, standard_split
from miniocr.models.ngram import (
    BigramLM, MLPLM, NGramLM, make_examples, make_window_examples,
    nats_to_bits, nats_to_bits_per_byte, perplexity,
)
from miniocr.tokenizer import BPETokenizer, show_tokens, train_bpe
from miniocr.train.lm import bits_per_byte

OUT = Path("outputs/book/ch07")
KS = (1.0, 0.1, 0.01, 0.001)


def window_dataset(seqs, context, bos, eos):
    """公式间重置；每个内容 token 和 EOS 恰好作为一次目标。"""
    pairs = [make_window_examples([bos] * context + list(s) + [eos], context) for s in seqs]
    return torch.cat([x for x, _ in pairs]), torch.cat([y for _, y in pairs])


def logits(model, x, bos):
    """与计数模型采用同一输出词表：BOS 只作条件。"""
    result = model(x)
    result[:, bos] = -torch.inf
    return result


@torch.no_grad()
def evaluate(model, data, bos, batch=1024):
    """全量、按目标数加权评测；不使用随机窗口近似最终分数。"""
    x, y = data
    was = model.training
    model.eval()
    total = sum(F.cross_entropy(logits(model, x[i:i + batch], bos), y[i:i + batch],
                                reduction="sum").item() for i in range(0, len(y), batch))
    model.train(was)
    return total / len(y)


@torch.no_grad()
def sample_mlp(model, bos, eos, n=200, seed=0, max_tokens=128):
    """批量祖先采样，温度为一，不作 top-k 截断；EOS 后不再记录。"""
    generator = torch.Generator().manual_seed(seed)
    history = torch.full((n, model.context), bos, dtype=torch.long)
    alive = torch.ones(n, dtype=torch.bool)
    result = [[] for _ in range(n)]
    for _ in range(max_tokens):
        p = logits(model, history, bos).softmax(-1)
        target = torch.multinomial(p, 1, generator=generator).squeeze(1)
        alive &= target != eos
        for i in alive.nonzero().flatten().tolist():
            result[i].append(target[i].item())
        history = torch.cat((history[:, 1:], target[:, None]), dim=1)
        if not alive.any():
            break
    return result


def braces_balanced(text):
    """仅检查未转义花括号的前缀深度及最终深度；不是 LaTeX 语法检查。"""
    depth = slashes = 0
    for char in text:
        if char in "{}" and slashes % 2 == 0:
            depth += 1 if char == "{" else -1
            if depth < 0:
                return False
        slashes = slashes + 1 if char == "\\" else 0
    return depth == 0


class Experiment:
    def __init__(self, args):
        self.args = args
        self.ready = False
        self.count_models, self.count_results = {}, {}
        self.models, self.results, self.histories = {}, {}, {}

    def prepare(self):
        if self.ready:
            return
        start = time.perf_counter()
        if self.args.small or not corpus_available():
            print("  提示：使用 SAMPLE_FORMULAS 降级；数字只用于流程检查。")
            split = standard_split(n_train=max(20000, self.args.n_train), n_val=4, n_test=4,
                                   formulas=SAMPLE_FORMULAS)
        else:
            split = standard_split(n_train=max(20000, self.args.n_train))
        self.train_text = split["train"][:self.args.n_train]
        self.val_text = split["val"][:2000]
        print(f"  标准划分=latex-v1；划分种子=0；模型训练条数={len(self.train_text)}；"
              f"固定验证条数={len(split['val'])}；本章验证条数={len(self.val_text)}；测试集不参与。")
        t0 = time.perf_counter()
        # 数据量实验也固定同一分词器，避免把分词变化误当作模型的收益。
        merges = train_bpe(split["train"][:20000], 2000)
        self.tok = BPETokenizer(merges, specials=("<|bos|>", "<|eos|>"))
        self.bos, self.eos = (self.tok.special_ids[s] for s in ("<|bos|>", "<|eos|>"))
        self.train = [self.tok.encode(s, allow_special=False) for s in self.train_text]
        self.val = [self.tok.encode(s, allow_special=False) for s in self.val_text]
        self.n_tokens = sum(len(s) + 1 for s in self.val)
        self.n_bytes = sum(len(s.encode("utf-8")) for s in self.val_text)
        print(f"  共用 BPE：训练条数={len(split['train'][:20000])}；请求合并次数=2000；"
              f"实际合并次数={len(merges)}；词表大小={len(self.tok)}；训练及编码用时秒={time.perf_counter()-t0:.4f}")
        print(f"  评测口径：验证目标 token 数（含 EOS）={self.n_tokens}；原文 UTF-8 字节数={self.n_bytes}；"
              "BOS 不作目标；边界符不贡献字节；每条公式重置上文。")
        self.ready = True
        print(f"  公共准备用时秒={time.perf_counter()-start:.4f}")

    def counts(self, n):
        self.prepare()
        if n not in self.count_models:
            model = (BigramLM(len(self.tok), self.bos, self.eos) if n == 2 else
                     NGramLM(len(self.tok), self.bos, self.eos, n=n))
            model.fit(self.train)
            stats = model.score_counts(self.val)
            trials = []
            for k in KS:
                model.k = k
                trials.append((model.nll_from_counts(stats), k))
            val, model.k = min(trials)
            train = model.nll_per_token(self.train)
            self.count_models[n] = model
            self.count_results[n] = dict(train=train, val=val, k=model.k,
                                         unseen=float(stats[2][stats[1] == 0].sum()/stats[2].sum()),
                                         contexts=len(model.counts), trials=trials)
        return self.count_models[n]

    def metrics(self, name, nll):
        print(f"  模型={name}；验证 nat/token={nll:.4f}；验证困惑度={perplexity(nll):.4f}；"
              f"验证 bits/byte={bits_per_byte(nll, self.n_tokens, self.n_bytes):.4f}")

    def samples(self, name, samples):
        texts = [self.tok.decode(s) for s in samples]
        for i, text in enumerate(texts[:5], 1):
            print(f"  模型={name}；采样编号={i}；公式={text!r}")
        print(f"  模型={name}；采样条数={len(texts)}；花括号配平比例={np.mean([braces_balanced(s) for s in texts]):.4f}；"
              f"采到上限比例={np.mean([len(s) == 128 for s in samples]):.4f}")
        print(f"  真实验证公式花括号配平比例={np.mean([braces_balanced(s) for s in self.val_text]):.4f}")

    def mlp(self, context):
        self.prepare()
        if context in self.models:
            return self.models[context]
        start = time.perf_counter()
        torch.manual_seed(self.args.seed + context)
        model = MLPLM(len(self.tok), context, self.args.embed_dim, self.args.hidden_dim)
        train = window_dataset(self.train, context, self.bos, self.eos)
        val = window_dataset(self.val, context, self.bos, self.eos)
        g = torch.Generator().manual_seed(self.args.seed)
        selected = torch.randperm(len(val[1]), generator=torch.Generator().manual_seed(1234))[:4096]
        probe = val[0][selected], val[1][selected]
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        print(f"  MLP 上下文长度={context}；嵌入维度={self.args.embed_dim}；隐藏层维度={self.args.hidden_dim}；"
              f"参数量={sum(p.numel() for p in model.parameters())}；批量大小=256；训练步数={self.args.steps}；学习率={3e-3:.4f}")
        history, running = [], []
        for step in range(1, self.args.steps + 1):
            idx = torch.randint(len(train[1]), (256,), generator=g)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(logits(model, train[0][idx], self.bos), train[1][idx])
            loss.backward()
            opt.step()
            running.append(loss.item())
            if step % max(1, self.args.steps // 4) == 0 or step == self.args.steps:
                vl = evaluate(model, probe, self.bos)
                tr = float(np.mean(running))
                history.append((step, tr, vl))
                print(f"  MLP 上下文长度={context}；训练步={step}；区间训练 nat/token={tr:.4f}；"
                      f"固定子集验证 nat/token={vl:.4f}；子集目标数={len(probe[1])}")
                running.clear()
        final = evaluate(model, val, self.bos)
        self.models[context], self.results[context], self.histories[context] = model, final, history
        self.metrics(f"MLP，上下文长度={context}，全量验证", final)
        print(f"  MLP 上下文长度={context}；构造样本、训练及评测用时秒={time.perf_counter()-start:.4f}")
        return model


def part1(exp):
    model = exp.counts(2)
    best = model.k
    model.k = 1.0
    ids = exp.tok.encode(r"x^{2}", allow_special=False)
    print(f"  手算公式={r'x^{2}'!r}；含边界 token 序列={[exp.bos] + ids + [exp.eos]}；平滑参数 k={model.k:.4f}；可预测类别数={len(exp.tok)-1}")
    total = 0.0
    for i, (context, target) in enumerate(model.events(ids), 1):
        p = model.prob(context, target)
        total += math.log(p)
        print(f"  预测位置={i}；上文={show_tokens([exp.tok.vocab[context[0]]])}；"
              f"目标={show_tokens([exp.tok.vocab[target]])}；相邻计数={model.counts.get(context, {}).get(target, 0)}；"
              f"上文计数={model.totals.get(context, 0)}；条件概率={p:.4f}；对数概率={math.log(p):.4f}")
    nll = -total/(len(ids)+1)
    print(f"  整条对数概率之和={total:.4f}；整条概率（科学记数）={math.exp(total):.4e}；"
          f"每 token 平均 NLL={nll:.4f}；困惑度={perplexity(nll):.4f}")
    print("  手算公式：条件概率=(相邻计数+k)/(上文计数+k×可预测类别数)；整条概率用科学记数，尾数保留四位小数，避免舍入成零。")
    model.k = best


def part2(exp):
    exp.prepare()
    text = r"\frac{a+b}{\sqrt{x^2+1}}"
    content = exp.tok.encode(text, allow_special=False)
    ids = [exp.bos] + content + [exp.eos]
    print(f"  原文={text!r}；token id 序列={ids}")
    x, y = make_examples(ids, 8)
    for i in range(min(3, len(x))):
        print(f"  整段错位样本编号={i+1}；窗口长度=8；输入 X={x[i].tolist()}；目标 Y={y[i].tolist()}")
    x, y = make_window_examples([exp.bos]*3 + content + [exp.eos], 3)
    for i in range(min(5, len(x))):
        print(f"  固定窗口样本编号={i+1}；上文长度=3；上下文={x[i].tolist()}；目标={y[i].item()}")
    print("  BOS 左补齐让首 token 也能作为目标；补齐的 BOS 不作为目标，公式间不拼接。")


def part3(exp):
    seqs = [[0, 1, 0, 1, 1, 0], [1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0]]
    model = BigramLM(4, 2, 3, k=0).fit(seqs)
    print(f"  玩具内容序列={seqs}；内容 token 总数={sum(map(len, seqs))}；BOS 编号=2；EOS 编号=3")
    print("  矩阵列顺序=[0, 1, BOS, EOS]；BOS 列恒零。无平滑的 EOS 行没有后继，约定全零。")
    for t in range(4):
        print(f"  计数矩阵行编号={t}；计数={[model.counts.get((t,), {}).get(j, 0) for j in range(4)]}")
    for k in (0.0, 1.0):
        model.k = k
        for t in range(4):
            print(f"  平滑参数 k={k:.4f}；概率矩阵行编号={t}；条件概率=[{', '.join(f'{model.prob((t,),j):.4f}' for j in range(4))}]")
        p = model.prob((0,), 0)
        print(f"  未见相邻对=(0, 0)；平滑参数 k={k:.4f}；概率={p:.4f}；loss={-math.log(p) if p else math.inf:.4f}")


def part4(exp):
    model = exp.counts(2)
    rec = exp.count_results[2]
    for val, k in rec["trials"]:
        exp.metrics(f"bigram，平滑参数 k={k:.4f}", val)
    print(f"  bigram 最佳平滑参数 k={model.k:.4f}；训练 nat/token={rec['train']:.4f}；验证 nat/token={rec['val']:.4f}")
    exp.counts(1)
    exp.metrics(f"unigram，最佳平滑参数 k={exp.count_results[1]['k']:.4f}", exp.count_results[1]["val"])
    exp.samples("bigram", model.sample(200, exp.args.seed))


def part5(exp):
    for n in range(1, 5):
        exp.counts(n)
        r = exp.count_results[n]
        print(f"  n-gram 阶数={n}；候选平滑参数与验证 nat/token="
              + ", ".join(f"k={k:.4f}: {loss:.4f}" for loss, k in r["trials"]))
        print(f"  n-gram 阶数={n}；最佳平滑参数 k={r['k']:.4f}；训练 nat/token={r['train']:.4f}；"
              f"训练困惑度={perplexity(r['train']):.4f}；验证 nat/token={r['val']:.4f}；"
              f"验证困惑度={perplexity(r['val']):.4f}；未见上文位置比例={r['unseen']:.4f}；不同上文种数={r['contexts']}")
    print("  这里没有回退或插值。未见上文用均匀分布；加 k 平滑还会惩罚低频上文，不能保证训练 loss 随阶数单调下降。")


def part6(exp):
    started = time.perf_counter()
    for context in exp.args.contexts:
        model = exp.mlp(context)
        exp.samples(f"MLP，上下文长度={context}", sample_mlp(model, exp.bos, exp.eos, seed=exp.args.seed))
    print(f"  本部分 MLP 训练、评测及采样总用时秒={time.perf_counter()-started:.4f}")
    for n in range(1, 5):
        exp.counts(n)
    best_n = min(exp.count_results, key=lambda n: exp.count_results[n]["val"])
    print("  同表比较：模型 | 验证 nat/token | 验证困惑度 | 验证 bits/byte")
    exp.metrics("bigram", exp.count_results[2]["val"])
    exp.metrics(f"最佳计数模型，阶数={best_n}", exp.count_results[best_n]["val"])
    for c, loss in exp.results.items():
        exp.metrics(f"MLP，上下文长度={c}", loss)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, rows in exp.histories.items():
        steps, train, val = zip(*rows)
        ax.plot(steps, train, label=f"context={c} train")
        ax.plot(steps, val, "--", label=f"context={c} val probe")
    ax.set(xlabel="Step", ylabel="NLL (nat/token)", title="Fixed-window MLP language models")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mlp_loss.png", dpi=140)
    plt.close(fig)
    print("  loss 曲线图=outputs/book/ch07/mlp_loss.png；虚线为固定验证子集，最终分数用全量验证。")


@torch.no_grad()
def part7(exp):
    bigram = exp.counts(2)
    # 训练需要 autograd，不能让诊断的 no_grad 装饰器覆盖它。
    with torch.enable_grad():
        models = {c: exp.mlp(c) for c in (3, 8)}
    records = [r for s in exp.val_text for r in right_positions(exp.tok, s)]
    print(f"  诊断目标=完整 right token；稳定定位的配对数={len(records)}；距离=左圆括号 token 与 right 之间的 token 数。")
    if not records:
        print("  本次验证集没有可定位配对，跳过诊断，不补造验证样本。")
        return
    for label, select in [("≤3", lambda d: d <= 3), ("4–8", lambda d: 4 <= d <= 8), (">8", lambda d: d > 8)]:
        rows = [r for r in records if select(r[0])]
        if not rows:
            print(f"  距离分桶={label}；样本位置数=0；平均概率=无样本")
            continue
        probs = [bigram.prob((ids[t-1],), ids[t]) for _, t, _, ids in rows]
        print(f"  距离分桶={label}；样本位置数={len(rows)}；模型=bigram；目标平均概率={np.mean(probs):.4f}")
        for c, model in models.items():
            x = torch.tensor([([exp.bos]*c+ids[:t])[-c:] for _, t, _, ids in rows])
            y = torch.tensor([ids[t] for _, t, _, ids in rows])
            p = logits(model, x, exp.bos).softmax(-1).gather(1, y[:, None]).mean().item()
            print(f"  距离分桶={label}；样本位置数={len(rows)}；模型=MLP，上下文长度={c}；目标平均概率={p:.4f}")
    distant = next((r for r in records if r[0] > 8), None)
    if distant:
        _, target, source, ids = distant
        altered = ids.copy()
        altered[source] = ord("x")
        for c, model in models.items():
            x = torch.tensor([ids[target-c:target], altered[target-c:target]])
            p = logits(model, x, exp.bos).softmax(-1)[:, ids[target]]
            print(f"  控制演示：只替换窗口外左括号 token；MLP 上下文长度={c}；目标概率绝对差={abs(p[0]-p[1]).item():.4f}")
        print(f"  控制演示：只替换窗口外左括号 token；模型=bigram；目标概率绝对差={abs(bigram.prob((ids[target-1],), ids[target])-bigram.prob((altered[target-1],), ids[target])):.4f}")
    print("  分桶平均数还受局部语法、频率影响，不证明模型记住了左括号；窗口外替换后的不变性才是结构限制。")


def part8(exp):
    exp.mlp(8)
    loss = exp.results[8]
    bpt = exp.n_bytes/exp.n_tokens
    print(f"  换算模型=MLP，上下文长度=8；nat/token={loss:.4f}；bits/token={nats_to_bits(loss):.4f}；"
          f"困惑度={perplexity(loss):.4f}；bits/byte={nats_to_bits_per_byte(loss,bpt):.4f}；每目标 token 平均字节数={bpt:.4f}")
    print("  换算公式：bits/token=nat/token÷ln(2)；困惑度=exp(nat/token)；bits/byte=bits/token÷每目标 token 平均字节数。")
    print(f"  手算给定 nat/token={2.0:.4f}；每 token 字节数={3.0:.4f}；bits/byte={nats_to_bits_per_byte(2.0,3.0):.4f}")


def elman_step(h, x, w, u):
    """最小 Elman 单元，辅助演示代码，无偏置。"""
    return torch.tanh(h @ w.T + x @ u.T)


def part9(exp):
    torch.manual_seed(exp.args.seed)
    width = 32
    w = torch.linalg.qr(torch.randn(width, width))[0]
    u = torch.randn(width, width)/math.sqrt(width)
    h = torch.zeros(1, width)
    for t in range(1, 7):
        h = elman_step(h, torch.randn(1, width), w, u)
        print(f"  顺序计算时间步={t}；隐藏状态范数={h.norm().item():.4f}")
    print("  梯度控制实验：随机正交 W 缩放到指定谱半径；输入及初态为零，使 tanh 保持在线性点；终点标量=隐藏状态求和÷sqrt(维度)。")
    for rho in (0.5, 1.0, 1.5):
        for t in (1, 5, 10, 20, 50):
            initial = torch.zeros(1, width, dtype=torch.float64, requires_grad=True)
            h = initial
            for _ in range(t):
                h = elman_step(h, torch.zeros_like(h), (rho*w).double(), u.double())
            gradient, = torch.autograd.grad(h.sum()/math.sqrt(width), initial)
            norm = gradient.norm().item()
            print(f"  W 谱半径={rho:.4f}；时间步数={t}；初态梯度范数={norm:.4f}；梯度范数 log10={math.log10(norm):.4f}")
    x = torch.randn(32, 256, width)
    def sequential():
        h = torch.zeros(32, width)
        for t in range(256):
            h = elman_step(h, x[:, t], w, u)
        return h
    def parallel_projection():
        return torch.tanh(x @ u.T)
    for name, fn in [("RNN 逐步循环", sequential), ("各位置并行投影", parallel_projection)]:
        with torch.no_grad():
            fn()
            times = []
            for _ in range(7):
                start = time.perf_counter()
                fn()
                times.append((time.perf_counter()-start)*1000)
        print(f"  计时操作={name}；序列长度=256；批量大小=32；隐藏维度={width}；前向中位用时毫秒={np.median(times):.4f}")
    print("  这只是直觉演示：并行投影不是完整注意力，计算量和功能也不同；真实非零输入会让 tanh 饱和，谱半径大于一不保证梯度爆炸。")


def part10(exp):
    exp.prepare()
    torch.manual_seed(exp.args.seed)
    model = MLPLM(len(exp.tok), 3, 64, 256)
    x, y = window_dataset(exp.train[:100], 3, exp.bos, exp.eos)
    keep = (x[:, -1] != exp.bos).nonzero().flatten()[:256]
    x, y = x[keep], y[keep]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    print("  故障：固定小批次，错误目标 Y=X 的最后一个 token；当前输入直接泄漏答案。")
    for step in range(1, 101):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits(model, x, exp.bos), x[:, -1])
        loss.backward()
        opt.step()
        if step in (1, 20, 100):
            print(f"  错位故障训练步={step}；复制当前 token 的训练 loss={loss.item():.4f}")
    correct = evaluate(model, window_dataset(exp.val, 3, exp.bos, exp.eos), exp.bos)
    print(f"  错位故障模型；正确下一个 token 任务的全量验证 nat/token={correct:.4f}")
    for i, seq in enumerate(sample_mlp(model, exp.bos, exp.eos, n=5, seed=exp.args.seed, max_tokens=32), 1):
        print(f"  错位故障采样编号={i}；采样上限=32；公式={exp.tok.decode(seq)!r}")
    for add_eos in (True, False):
        toy = BigramLM(4, 2, 3, k=0).fit([[0, 1]*10]*20, add_eos=add_eos)
        samples = toy.sample(200, exp.args.seed, max_tokens=128)
        print(f"  终止符故障玩具对照：训练添加 EOS={add_eos}；平滑参数 k={toy.k:.4f}；"
              f"采样条数=200；最大长度=128；采到最大长度比例={np.mean([len(s)==128 for s in samples]):.4f}")
    for merges in (0, 2000):
        if merges:
            exp.counts(2)
            loss = exp.count_results[2]["val"]
            k, nt = exp.count_results[2]["k"], exp.n_tokens
        else:
            raw = BPETokenizer([], specials=("<|bos|>", "<|eos|>"))
            train = [raw.encode(s, allow_special=False) for s in exp.train_text]
            val = [raw.encode(s, allow_special=False) for s in exp.val_text]
            raw_model = BigramLM(len(raw), raw.special_ids["<|bos|>"], raw.special_ids["<|eos|>"]).fit(train)
            stats, candidates = raw_model.score_counts(val), []
            for k in KS:
                raw_model.k = k
                candidates.append((raw_model.nll_from_counts(stats), k))
            loss, k = min(candidates)
            nt = sum(len(s)+1 for s in val)
        print(f"  不同分词器 bigram：请求合并次数={merges}；最佳平滑参数 k={k:.4f}；"
              f"验证 nat/token={loss:.4f}；验证 bits/byte={bits_per_byte(loss,nt,exp.n_bytes):.4f}")
    model = exp.counts(2)
    previous = model.k
    model.k = 0.0
    print(f"  无平滑故障：平滑参数 k={model.k:.4f}；全量验证 nat/token={model.nll_per_token(exp.val):.4f}")
    model.k = previous


PARTS = {
    1: ("链式法则", part1), 2: ("错开一位", part2), 3: ("bigram 手算", part3),
    4: ("bigram 在 LaTeX 上", part4), 5: ("n-gram 的稀疏", part5),
    6: ("固定窗口 MLP", part6), 7: ("远处的上下文", part7), 8: ("换算", part8),
    9: ("RNN 的直觉", part9), 10: ("常见错误", part10),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--contexts", nargs="+", type=int, default=[3, 8])
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.n_train, args.steps, args.embed_dim, args.hidden_dim, args.threads, *args.contexts) < 1:
        parser.error("样本量、步数、维度、线程数和上下文长度必须为正")
    if any(p not in PARTS for p in args.parts):
        parser.error("部分号应在 1 至 10 之间")
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    exp = Experiment(args)
    started = time.perf_counter()
    print(f"第 7 章；实际计算设备=CPU；线程数={args.threads}；随机种子={args.seed}；数据访问=仅本地；浮点输出保留四位小数。")
    print("采样协议：温度=1.0000；不截断候选；默认最多内容 token 数=128；遇 EOS 停止；花括号配平不是语法正确率。")
    for p in args.parts or PARTS:
        title, fn = PARTS[p]
        print("\n" + "="*70 + f"\n第 {p} 部分：{title}（正文 7.{p}）\n" + "="*70)
        start = time.perf_counter()
        fn(exp)
        print(f"第 {p} 部分用时秒={time.perf_counter()-start:.4f}")
    report = dict(config=vars(args), counts=exp.count_results, mlp=exp.results,
                  histories=exp.histories, total_seconds=time.perf_counter()-started)
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n第 7 章所选实验完成；tour 总用时秒={time.perf_counter()-started:.4f}")


if __name__ == "__main__":
    main()
