"""第 9 章：Transformer 块。所有实验只在 CPU 上运行，不保存权重。

    python -m book.part2_lm.ch09_block.tour [部分号 ...]
    python -m book.part2_lm.ch09_block.tour 2 --layers 2 4 8 16
    python -m book.part2_lm.ch09_block.tour 4 --lr 0.003 0.02
    python -m book.part2_lm.ch09_block.tour 5 --ff-ratio 2

辅助模型只用于结构对照；RoPE 与完整 GPT 的初始化留给第 10、11 章。
"""
import argparse
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ch09-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from miniocr.data.latex_lm import FormulaWindows, evaluate_lm, load_latex_lm
from miniocr.models.attention_lm import AttentionLM
from miniocr.models.gpt import Block, GPTConfig, RMSNorm, SwiGLU
from miniocr.nn.attention import MultiHeadAttention, causal_mask

OUT = Path("outputs/book/ch09")


def section(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


def fmt(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().tolist()
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(fmt(v) for v in x) + "]"
    return f"{x:.4f}"


def rms(x):
    return x.float().square().mean().sqrt().item()


class NoResidualBlock(Block):
    """故意删掉两条残差，只用于对照。"""

    def forward(self, x, cos, sin, cache=None):
        x = self.drop(self.attn(self.n1(x), cos, sin, cache))
        return self.drop(self.mlp(self.n2(x)))


class PostNormBlock(Block):
    """沿用 RMSNorm 与相同参数，唯一改变归一化的位置。"""

    def forward(self, x, cos, sin, cache=None):
        x = self.n1(x + self.drop(self.attn(x, cos, sin, cache)))
        return self.n2(x + self.drop(self.mlp(x)))


class GELUMLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class BlockLM(nn.Module):
    """教学用小 LM：学习式位置、Block 堆叠、final norm、独立输出层。

    cos=1、sin=0 关闭 RoPE；右补齐位于真实位置之后，因果注意力不会
    读取它们。last_only 避免对无用位置做词表投影。初始化沿用部件默认值，
    嵌入与第 8 章一致；这里不使用第 11 章的 GPT 初始化或权重共享。
    """

    def __init__(self, vocab, bos, context=16, layers=1, d=64, mode="pre", ff=0, gelu=False):
        super().__init__()
        self.context, self.bos_id = context, bos
        self.token = nn.Embedding(vocab, d)
        self.position = nn.Embedding(context, d)
        nn.init.normal_(self.token.weight, std=0.1)
        nn.init.normal_(self.position.weight, std=0.02)
        cfg = GPTConfig(d_model=d, n_head=1, d_ff=ff, dropout=0.0)
        cls = {"pre": Block, "post": PostNormBlock, "none": NoResidualBlock}[mode]
        self.blocks = nn.ModuleList(cls(cfg) for _ in range(layers))
        self.norm = RMSNorm(d)
        self.output = nn.Linear(d, vocab)
        self.register_buffer("cos", torch.ones(context, d))
        self.register_buffer("sin", torch.zeros(context, d))

        if gelu:
            for block in self.blocks:
                block.mlp = GELUMLP(d, 3 * cfg.ff_dim() // 2)

    def forward(self, idx, lengths=None, last_only=False):
        B, T = idx.shape
        x = self.token(idx) + self.position(torch.arange(T))
        for block in self.blocks:
            x = block(x, self.cos[:T], self.sin[:T])
        if last_only:
            lengths = torch.full((B,), T, dtype=torch.long) if lengths is None else lengths
            x = x[torch.arange(B), lengths - 1]
        logits = self.output(self.norm(x))
        logits[..., self.bos_id] = -torch.inf
        return logits


def last_logits(model, x, lengths):
    return model(x, lengths, last_only=True)


class Experiment:
    def __init__(self, args):
        self.args, self.data = args, None

    def prepare(self):
        if self.data is not None:
            return
        start = time.perf_counter()
        self.data = load_latex_lm()
        tok = self.data.tokenizer
        self.bos, self.eos = (tok.special_ids[s] for s in ("<|bos|>", "<|eos|>"))
        self.train = FormulaWindows([s[1:-1] for s in self.data.train], self.bos, self.eos, self.args.context)
        self.val = FormulaWindows([s[1:-1] for s in self.data.val], self.bos, self.eos, self.args.context)
        print(f"  标准划分=latex-v1；BPE 训练条数={len(self.data.train_text)}；请求合并次数=2000；实际合并次数={len(tok.merges)}；词表大小={len(tok)}")
        print(f"  训练公式数={len(self.data.train)}；验证公式数={len(self.data.val)}；验证目标数={len(self.val)}；原文字节数={self.data.n_bytes}")
        print("  每条公式重置；BOS 只作条件且屏蔽输出；EOS 计入；长公式逐目标滑窗、不丢目标；位置从零开始；测试集不参与。")
        print(f"  数据准备用时秒={time.perf_counter()-start:.4f}")

    def make(self, layers=1, mode="pre", gelu=False, attention=False):
        self.prepare()
        torch.manual_seed(self.args.seed)
        if attention:
            return AttentionLM(len(self.data.tokenizer), self.bos, context=self.args.context)
        ff = 0 if self.args.ff_ratio is None else 64 * math.ceil(self.args.ff_ratio * 64 / 64)
        return BlockLM(len(self.data.tokenizer), self.bos, self.args.context, layers,
                       mode=mode, ff=ff, gelu=gelu)

    def fit(self, model, label, layers, lr):
        args = self.args
        print(f"  训练模型={label}；设备=CPU；种子={args.seed}；层数={layers}；维度=64；头数=1；步数={args.steps}；学习率={lr:.4f}；批量={args.batch}；上下文={args.context}；参数量={sum(p.numel() for p in model.parameters())}")
        print("  优化器=AdamW；权重衰减=0.0100；无预热、无梯度裁剪；精度=fp32；均匀抽取目标。")
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        g = torch.Generator().manual_seed(args.seed)
        history, running = [], []
        start = time.perf_counter()
        for step in range(1, args.steps + 1):
            x, lengths, y = self.train.batch(torch.randint(len(self.train), (args.batch,), generator=g))
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(last_logits(model, x, lengths), y)
            loss.backward()
            opt.step()
            history.append(loss.item())
            running.append(loss.item())
            if step % max(1, args.steps // 4) == 0 or step == args.steps:
                print(f"  模型={label}；训练步={step}；区间训练 nat/token={sum(running)/len(running):.4f}")
                running.clear()
        train_sec = time.perf_counter() - start
        print(f"  模型={label}；实际训练用时秒={train_sec:.4f}")
        start = time.perf_counter()
        result = evaluate_lm(model, self.val, self.data.n_bytes, window_logits=last_logits)
        print(f"  模型={label}；全量验证 nat/token={result['nat_per_token']:.4f}；bits/byte={result['bits_per_byte']:.4f}；评测用时秒={time.perf_counter()-start:.4f}")
        return history, result


def plot(histories, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, history in histories:
        width = max(1, len(history) // 20)
        means = [sum(history[i:i+width]) / len(history[i:i+width]) for i in range(0, len(history), width)]
        ax.plot(range(width, len(history)+width, width), means, label=label)
    ax.set(xlabel="Training step", ylabel="Train NLL (nat/token)")
    ax.legend()
    fig.tight_layout()
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  训练曲线图片={path}")


def part1(exp):
    section("1. 一层注意力不够（9.1）：短训只看趋势")
    print("  基线是第 8 章 AttentionLM；块同时增加残差、RMSNorm、SwiGLU 与 final norm，且投影无偏置；不是仅增加前馈的单变量消融。")
    print("  两者均用学习式位置嵌入，关闭 RoPE；相同种子、目标批次、步数、学习率与上下文。")
    scores = []
    for label, attn in (("单层注意力", True), ("单层 Decoder 块", False)):
        _, result = exp.fit(exp.make(attention=attn), label, 1, 0.003)
        scores.append(result['nat_per_token'])
    print(f"  块减去单层注意力的验证差值={scores[1]-scores[0]:.4f}；单次短训不能证明架构的普遍优劣。")


def stack_diagnostic(depth, mode, seed=0):
    torch.manual_seed(seed)
    cfg = GPTConfig(d_model=64, n_head=1)
    cls = {"pre": Block, "none": NoResidualBlock, "post": PostNormBlock}[mode]
    blocks = nn.ModuleList(cls(cfg) for _ in range(depth))
    # 隔离网络深度：各配置用同一输入及同一线性读出方向。
    g = torch.Generator().manual_seed(123)
    x = (torch.randn(2, 8, 64, generator=g) * 0.1).requires_grad_()
    direction = torch.randn(x.shape, generator=g) / math.sqrt(x.numel())
    cos, sin = torch.ones(8, 64), torch.zeros(8, 64)
    h, scales = x, []
    for block in blocks:
        h = block(h, cos, sin)
        scales.append(rms(h))
    (h * direction).sum().backward()
    grads = [math.sqrt(sum(p.grad.square().sum().item() for p in b.parameters() if p.grad is not None)) for b in blocks]
    return x.grad.norm().item(), scales, grads, rms(RMSNorm(64)(h))


def part2(exp):
    section("2. 残差（9.2）：梯度通路与短训")
    print("  初始化诊断：部件默认初始化；输入 RMS 约 0.1000；固定随机线性读出，无 final norm；小梯度另打印 log10。")
    for depth in (2, 4, 8, 16, 32):
        for mode, label in (("pre", "有残差"), ("none", "无残差")):
            grad, scales, _, _ = stack_diagnostic(depth, mode, exp.args.seed)
            print(f"  深度={depth}；结构={label}；输入梯度范数={grad:.4f}；梯度 log10={math.log10(max(grad, 1e-300)):.4f}；各层输出 RMS={fmt(scales)}")
    x = torch.tensor(2.0, requires_grad=True)
    y = x + x.square()
    y.backward()
    print(f"  手算：x={x.item():.4f}；f(x)=x²；y={y.item():.4f}；f'(x)={2*x.item():.4f}；dy/dx=1+f'(x)={x.grad.item():.4f}")
    for depth in exp.args.layers:
        histories = []
        for mode, label in (("pre", "有残差"), ("none", "无残差")):
            h, _ = exp.fit(exp.make(depth, mode), f"{depth}层{label}", depth, 0.003)
            histories.append((mode, h))
        plot(histories, f"residual_L{depth}.png")
    print("  归一化会重新放大无残差支路；梯度不保证单调消失。应结合梯度方向、训练曲线与验证指标判断。")


def part3(exp):
    section("3. LayerNorm 与 RMSNorm（9.3）")
    x = torch.tensor([1., 2., 3., 4.])
    mean, var = x.mean(), x.var(unbiased=False)
    ln, rn = (x-mean)/torch.sqrt(var+1e-6), x/torch.sqrt(x.square().mean()+1e-6)
    print(f"  输入={fmt(x)}；均值={mean:.4f}；总体方差={var:.4f}；均方={x.square().mean():.4f}")
    print(f"  手算 LayerNorm={fmt(ln)}；手算 RMSNorm={fmt(rn)}；epsilon=0.000001（此常量保留原值）")
    print(f"  nn.LayerNorm 最大误差={(nn.LayerNorm(4, eps=1e-6)(x)-ln).abs().max():.4f}；自写 RMSNorm 最大误差={(RMSNorm(4)(x)-rn).abs().max():.4f}")
    if hasattr(nn, "RMSNorm"):
        print(f"  nn.RMSNorm 最大误差={(nn.RMSNorm(4, eps=1e-6)(x)-rn).abs().max():.4f}")
    print(f"  LayerNorm 参数量={sum(p.numel() for p in nn.LayerNorm(4).parameters())}；RMSNorm 参数量={sum(p.numel() for p in RMSNorm(4).parameters())}")
    xb = (torch.arange(4096).float() % 71 + 10000).bfloat16()
    ref = xb.double().square().mean().sqrt()
    direct = xb.square().mean().sqrt()
    promoted = xb.float().square().mean().sqrt()
    print(f"  大偏移 bf16 输入；fp64 参考 RMS={ref:.4f}；bf16 直接 RMS={direct:.4f}；转 fp32 RMS={promoted:.4f}")
    print(f"  bf16 直接 RMS 绝对误差={abs(direct.double()-ref):.4f}；转 fp32 RMS 绝对误差={abs(promoted.double()-ref):.4f}")
    print("  参考值从同一份已量化 bf16 输入计算；转 fp32 改善运算舍入，无法找回输入量化时丢失的信息。")


def unigram_baseline(train, val, vocab):
    """与第 7 章 tour 第 4 部分同口径：训练目标计数，加 k 平滑，验证选 k。"""
    counts = torch.bincount(train.ids[train.ends], minlength=vocab).double()
    frequency = torch.bincount(val.ids[val.ends], minlength=vocab).double()
    # ends 只含内容与 EOS；BOS 不计数，也不占平滑分母的类别数。
    trials = []
    for k in (1.0, 0.1, 0.01, 0.001):
        logp = ((counts + k) / (counts.sum() + k * (vocab - 1))).log()
        trials.append((-(logp * frequency).sum().item() / len(val), k))
    return min(trials)


def part4(exp):
    section("4. Pre-Norm 与 Post-Norm（9.4）")
    for depth in (4, 8, 16, 32):
        for mode in ("pre", "post"):
            _, scales, grads, _ = stack_diagnostic(depth, mode, exp.args.seed)
            print(f"  深度={depth}；归一化位置={mode}；各层残差流 RMS={fmt(scales)}")
            print(f"  深度={depth}；归一化位置={mode}；各层参数梯度范数={fmt(grads)}；底层/顶层梯度比={grads[0]/grads[-1]:.4f}")
    exp.prepare()
    unigram, k = unigram_baseline(exp.train, exp.val, len(exp.data.tokenizer))
    print(f"  本次 unigram；最佳平滑参数 k={k:.4f}；全量验证 nat/token={unigram:.4f}；bits/byte={unigram*len(exp.val)/(math.log(2)*exp.data.n_bytes):.4f}。")
    print("  平滑候选 k=[1.0000, 0.1000, 0.0100, 0.0010]，按验证 nat/token 选最小；同口径参考 5.4230 nat/token 来自第 7 章 tour 第 4 部分。")
    print("  若与参考不同，请核对本地数据快照或样例降级；以下比较使用本次重算值。")
    threshold = 0.1
    for lr in exp.args.lr:
        histories, scores = [], []
        for mode in ("pre", "post"):
            h, result = exp.fit(exp.make(8, mode), f"8层{mode} lr={lr:g}", 8, lr)
            histories.append((mode, h))
            scores.append(result['nat_per_token'])
        plot(histories, f"pre_post_lr{lr:g}.png")
        delta = scores[1] - scores[0]
        print(f"  Post 减 Pre 的验证差值={delta:.4f}；相同学习率={lr:.4f}。")
        if not all(math.isfinite(score) for score in scores):
            conclusion = "出现非有限验证值，无法判断差距"
        elif abs(delta) <= threshold:
            conclusion = "本次差异不明显，不据此宣布两者等价"
        else:
            conclusion = "本次 Post-Norm 更吃力" if delta > 0 else "本次 Pre-Norm 更吃力"
        print(f"  学习率={lr:.4f}；明显差距阈值=|Post-Pre|>{threshold:.4f} nat/token（非统计显著性检验）；{conclusion}；仅限本配置短训。")
        above = ["无法判断" if not math.isfinite(score) else str(score > unigram) for score in scores]
        both = "无法判断" if "无法判断" in above else str(all(score > unigram for score in scores))
        print(f"  学习率={lr:.4f}；验证高于 unigram {unigram:.4f} nat/token：Pre={above[0]}；Post={above[1]}；两者都高于={both}。")


def toy_swiglu():
    m = SwiGLU(GPTConfig(d_model=2, d_ff=2))
    with torch.no_grad():
        m.gate.weight.copy_(torch.tensor([[1., -2.], [-1., 1.]]))
        m.up.weight.copy_(torch.tensor([[2., 1.], [1., -1.]]))
        m.down.weight.copy_(torch.tensor([[1., 2.], [-1., 1.]]))
    return m, torch.tensor([1., 2.])


def part5(exp):
    section("5. SwiGLU（9.5）：门控与参数预算")
    m, x = toy_swiglu()
    gate, up = m.gate(x), m.up(x)
    print(f"  输入={fmt(x)}；gate 权重={fmt(m.gate.weight)}；up 权重={fmt(m.up.weight)}；down 权重={fmt(m.down.weight)}")
    print(f"  gate={fmt(gate)}；silu(gate)={fmt(F.silu(gate))}；up={fmt(up)}；逐元素乘积={fmt(F.silu(gate)*up)}；down 输出={fmt(m(x))}")
    for d in (64, 192, 256):
        f = GPTConfig(d_model=d).ff_dim()
        print(f"  维度={d}；理想中间维 8d/3={8*d/3:.4f}；ff_dim 向上取整到64倍数={f}；4d GELU 参数量={8*d*d}；取整后 SwiGLU 参数量={3*d*f}；比值={3*d*f/(8*d*d):.4f}")
    histories = []
    for gelu, label in ((True, "GELU-MLP"), (False, "SwiGLU")):
        model = exp.make(gelu=gelu)
        mlp = model.blocks[0].mlp
        print(f"  前馈类型={label}；中间维={mlp.up.out_features}；前馈参数量={sum(p.numel() for p in mlp.parameters())}；无偏置，按 2dh=3df 精确对齐。")
        h, _ = exp.fit(model, label, 1, 0.003)
        histories.append((label, h))
    plot(histories, "gelu_swiglu.png")
    print("  中间维取整对小模型影响较大；训练比较把 GELU 中间维设为 3f/2，不能同时声称它严格等于 4d。")


def part6(exp):
    section("6. 组装 Decoder 块（9.6）")
    torch.manual_seed(exp.args.seed)
    cfg = GPTConfig(d_model=64, n_head=4)
    block = Block(cfg).eval()
    hooks = []
    for name, module in block.named_modules():
        if name:
            hooks.append(module.register_forward_hook(
                lambda m, args, out, name=name: print(f"  子模块={name}；输入形状={tuple(args[0].shape)}；输出形状={tuple(out.shape)}；参数量={sum(p.numel() for p in m.parameters())}")))
    x = torch.randn(2, 6, 64)
    cos, sin = torch.ones(6, 16), torch.zeros(6, 16)
    with torch.no_grad():
        before = block(x, cos, sin)
    for hook in hooks:
        hook.remove()
    d, f = cfg.d_model, cfg.ff_dim()
    print(f"  注意力公式参数量 4d²={4*d*d}；前馈公式参数量 3df={3*d*f}；归一化公式参数量 2d={2*d}")
    print(f"  Block 公式总参数量={4*d*d+3*d*f+2*d}；实际总参数量={sum(p.numel() for p in block.parameters())}")
    changed = x.clone()
    changed[:, 4:] += 10
    with torch.no_grad():
        after = block(changed, cos, sin)
    print(f"  改未来 token 后；过去输出精确相等={torch.equal(before[:, :4], after[:, :4])}；过去最大差={(before[:,:4]-after[:,:4]).abs().max():.4f}")


def part7(exp):
    section("7. 三种结构（9.7）")
    torch.manual_seed(exp.args.seed)
    encoder, decoder = torch.randn(2, 5, 8), torch.randn(2, 3, 8)
    for name, x, causal in (("Encoder 双向", encoder, False), ("Decoder 因果", decoder, True)):
        mask = causal_mask(x.shape[1]) if causal else torch.ones(x.shape[1], x.shape[1], dtype=torch.bool)
        m = MultiHeadAttention(8, 2, causal=causal)
        y = m(x)
        print(f"  结构={name}；可见 mask（1=可以看）={mask.int().tolist()}；输出形状={tuple(y.shape)}；权重形状={tuple(m.last_weights.shape)}")
    cross = MultiHeadAttention(8, 2)
    y = cross(decoder, kv=encoder)
    print(f"  Encoder-Decoder 交叉注意力：q 来自解码器形状={tuple(decoder.shape)}；k/v 来自编码器形状={tuple(encoder.shape)}")
    print(f"  交叉可见 mask={torch.ones(3,5,dtype=torch.int).tolist()}；交叉权重形状={tuple(cross.last_weights.shape)}；输出形状={tuple(y.shape)}")
    print("  完整 Encoder-Decoder 的解码器先因果自注意力，再对编码器输出做交叉注意力；源序列可全见。这里展示注意力部件，不训练完整模型。")


def part8(exp):
    section("8. 常见错误（9.8）")
    torch.manual_seed(exp.args.seed)
    x = torch.randn(2, 6, 4)
    changed = x.clone()
    changed[:, 3:] += 20
    for dim, label in ((-1, "特征维（正确）"), (1, "时间维（错误）"), ((0, 1), "batch与时间维（错误）"), (0, "只沿batch维（错误）")):
        def normalize(z):
            return (z-z.mean(dim, keepdim=True)) / torch.sqrt(z.var(dim, unbiased=False, keepdim=True)+1e-6)
        delta = (normalize(x)[:, :3]-normalize(changed)[:, :3]).abs().max()
        print(f"  归一化维度={label}；改未来后过去是否变化={delta.item()!=0}；过去最大差={delta:.4f}")
    other = x.clone()
    other[1] += 20
    batch_norm = lambda z: (z-z.mean(0, keepdim=True))/torch.sqrt(z.var(0, unbiased=False, keepdim=True)+1e-6)
    print(f"  只沿 batch 维不会混入未来，但会串样本；改另一条样本后本样本最大差={(batch_norm(x)[0]-batch_norm(other)[0]).abs().max():.4f}")
    print(f"  零分支 f=0：正确 x+f(norm(x)) 与输入差={rms(x+torch.zeros_like(x)-x):.4f}；错误 f(norm(x)) 与输入差={rms(torch.zeros_like(x)-x):.4f}")
    half = torch.tensor([300., 400., 500., 600.], dtype=torch.float16)
    bad = half * torch.rsqrt(half.square().mean()+1e-6)
    good = RMSNorm(4)(half)
    print(f"  fp16 直接均方={half.square().mean().item():.4f}；错误输出={fmt(bad)}；fp32 内部计算输出={fmt(good)}")
    for depth in (2, 4, 8, 16, 32):
        _, scales, _, normalized = stack_diagnostic(depth, "pre", exp.args.seed)
        print(f"  深度={depth}；遗漏 final norm 的输出 RMS={scales[-1]:.4f}；补 final norm 后 RMS={normalized:.4f}")
    print("  RMS 随层数增长是当前未缩放初始化的观察；第 11 章会讲残差输出初始化缩放。")


PARTS = {1: part1, 2: part2, 3: part3, 4: part4, 5: part5, 6: part6, 7: part7, 8: part8}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", nargs="+", type=int, default=[8], help="第 2 部分的短训层数")
    parser.add_argument("--lr", nargs="+", type=float, default=[0.003, 0.02], help="第 4 部分逐对比较的共同学习率")
    parser.add_argument("--ff-ratio", type=float, default=None, help="SwiGLU 中间维/d，向上取整到 64 倍数")
    args = parser.parse_args()
    if min(args.steps, args.batch, args.context, *args.layers) < 1 or any(not math.isfinite(lr) or lr <= 0 for lr in args.lr) or (args.ff_ratio is not None and args.ff_ratio <= 0) or any(k not in PARTS for k in args.parts):
        parser.error("步数、维度、学习率须为正，部分号须为 1–8")
    torch.set_num_threads(2)
    print(f"第 9 章；实际设备=CPU；线程数=2；随机种子={args.seed}；仅本地数据；浮点输出四位小数。")
    started = time.perf_counter()
    exp = Experiment(args)
    for k in args.parts or PARTS:
        start = time.perf_counter()
        PARTS[k](exp)
        print(f"第 {k} 部分用时秒={time.perf_counter()-start:.4f}")
    print(f"第 9 章所选实验完成；tour 总用时秒={time.perf_counter()-started:.4f}")


if __name__ == "__main__":
    main()
