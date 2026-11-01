"""第 10 章：位置编码与 RoPE；仅 CPU、本地数据，不保存权重。

python -m book.part2_lm.ch10_rope.tour [部分号 ...]
python -m book.part2_lm.ch10_rope.tour 9 --context 32 --eval-contexts 32 64 128
辅助实现用于数值对照，不是挖空练习；练习 10.1 在 miniocr.models.gpt。
"""
import argparse
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ch10-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from miniocr.data.latex_lm import FormulaWindows, evaluate_lm, load_latex_lm
from miniocr.models.gpt import GPT, GPTConfig, apply_rope, rope_cache, rotate_half

OUT = Path("outputs/book/ch10")


def fmt(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().tolist()
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(fmt(v) for v in x) + "]"
    return f"{x:.4f}"


def error(label, a, b):
    delta = (a - b).abs().max().item()
    print(f"  {label}最大差={delta:.4f}；差值乘一百万={delta*1e6:.4f}")


def rotate_at(x, position, base=10000.0):
    cos, sin = rope_cache(position + 1, x.shape[-1], base)
    return apply_rope(x, cos[position], sin[position])


def adjacent_rope(x, positions, base=10000.0):
    """相邻两维的独立参考公式，支持 (..., T, dh)。"""
    dh = x.shape[-1]
    if dh <= 0 or dh % 2:
        raise ValueError("RoPE 的 dh 必须是正偶数")
    angle = torch.as_tensor(positions, dtype=torch.float32)[..., None] * base ** (
        -torch.arange(0, dh, 2).float() / dh)
    c, s = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a*c-b*s, a*s+b*c), -1).flatten(-2)


def half_to_adjacent(dh):
    return torch.stack((torch.arange(dh//2), torch.arange(dh//2, dh)), -1).flatten()


def sinusoidal(positions, d):
    angle = torch.as_tensor(positions).float()[:, None] * 10000 ** (-torch.arange(0, d, 2).float()/d)
    return torch.stack((angle.sin(), angle.cos()), -1).flatten(-2)


def part1(args):
    torch.manual_seed(args.seed)
    text = "a^{2}"
    perm = torch.tensor([3, 1, 2, 0, 4])
    x = torch.randn(5, 8)
    wq, wk, wv = (torch.randn(8, 8) / math.sqrt(8) for _ in range(3))
    def attention(z):
        return ((z@wq) @ (z@wk).T / math.sqrt(8)).softmax(-1) @ (z@wv)
    y, yp = attention(x), attention(x[perm])
    print(f"  字符级序列甲={text}；序列乙={''.join(text[i] for i in perm)}；维度=8；无位置、无 mask。")
    error("置换等变 Y(PX) 与 P Y(X)", yp, y[perm])
    error("逆置换对齐后的输出集合", yp[perm.argsort()], y)
    print("  这是置换等变，不是逐位置输出相等；因果 mask 会提供隐式顺序信息，不能把此结论直接套到 NoPE GPT。")


def part2(args):
    print("  正弦式：PE(t,2i)=sin(t·10000^(−2i/d))；PE(t,2i+1)=cos(t·10000^(−2i/d))。")
    for p, row in enumerate(sinusoidal(torch.arange(4), 8)):
        print(f"  位置={p}；八维位置编码={fmt(row)}")
    table = nn.Embedding(4, 8)
    try:
        table(torch.tensor([4]))
    except IndexError as exc:
        print(f"  学习式表长=4；请求位置=4；异常={type(exc).__name__}: {exc}")
    for m, n in ((2, 5), (10, 13)):
        pe = sinusoidal([m, n], 8)
        print(f"  正弦点积；位置甲={m}；位置乙={n}；距离={n-m}；点积={pe[0]@pe[1]:.4f}")
    torch.manual_seed(args.seed)
    a, b = torch.randn(2, 8)
    wq, wk = torch.randn(8, 8), torch.randn(8, 8)
    pm, pn = sinusoidal([2, 5], 8)
    terms = [(a@wq)@(b@wk), (a@wq)@(pn@wk), (pm@wq)@(b@wk), (pm@wq)@(pn@wk)]
    print(f"  内容×内容={terms[0]:.4f}；内容×位置={terms[1]:.4f}；位置×内容={terms[2]:.4f}；位置×位置={terms[3]:.4f}")
    print(f"  展开四项之和={sum(terms):.4f}；直接 q·k={((a+pm)@wq)@((b+pn)@wk):.4f}")
    print("  PE 本身的点积只依赖距离；任意投影后的位置项也未必保留这一性质，且交叉项混合内容与绝对位置。")


def part3(args):
    q, k = torch.tensor([1., 0.], dtype=torch.float64), torch.tensor([0.6, 0.8], dtype=torch.float64)
    def rotate(x, angle):
        c, s = math.cos(angle), math.sin(angle)
        return torch.tensor([x[0]*c-x[1]*s, x[0]*s+x[1]*c])
    for m, n in ((2, 5), (10, 13)):
        rq, rk = rotate(q, m*0.5), rotate(k, n*0.5)
        print(f"  位置甲={m}；位置乙={n}；频率θ=0.5000；旋转q={fmt(rq)}；旋转k={fmt(rk)}；点积={rq@rk:.4f}")
    print(f"  手算 qᵀR((n−m)θ)k = 0.6cos(1.5)−0.8sin(1.5)={0.6*math.cos(1.5)-0.8*math.sin(1.5):.4f}")


def part4(args):
    for i in range(4):
        theta = 10000 ** (-2*i/8)
        print(f"  头维度=8；base=10000；维度对=({i},{i+4})；频率θ={theta:.4f}；波长={2*math.pi/theta:.4f}")
    torch.manual_seed(args.seed)
    q, k = torch.randn(2, 8)
    score = rotate_at(q, 2) @ rotate_at(k, 5)
    for shift in (1, 100, 1000):
        moved = rotate_at(q, 2+shift) @ rotate_at(k, 5+shift)
        print(f"  偏移={shift}；原点积={score:.4f}；平移点积={moved:.4f}")
        error("相对位置性质", score, moved)


def part5(args):
    q = torch.arange(1., 9.)
    k = torch.tensor([3., -1., 2., 0., 1., -2., 4., 1.])
    p = half_to_adjacent(8)
    half, adjacent = rotate_at(q, 3), adjacent_rope(q, 3)
    print(f"  同一输入q={fmt(q)}；位置=3；前后两半输出={fmt(half)}；相邻两维输出={fmt(adjacent)}")
    print(f"  前后两半转相邻的索引置换={p.tolist()}；迁移权重时相应重排 q/k 投影输出通道。")
    error("置换后的两种实现", half[p], adjacent_rope(q[p], 3))
    for label, rotate in (("前后两半", rotate_at), ("相邻两维", adjacent_rope)):
        a = rotate(q, 2) @ rotate(k, 5)
        b = rotate(q, 102) @ rotate(k, 105)
        error(f"{label}平移", a, b)
    a = rotate_at(q, 2) @ rotate_at(k, 5)
    b = adjacent_rope(q, 2) @ adjacent_rope(k, 5)
    print(f"  权重通道未重排；训练约定分数={a:.4f}；混用约定分数={b:.4f}；绝对差={abs(a-b):.4f}")


def value_shift_demo(seed):
    torch.manual_seed(seed)
    q, k, v = torch.randn(3, 1, 1, 6, 8)
    def output(start, rotate_v):
        cos, sin = rope_cache(start+6, 8)
        c, s = cos[start:], sin[start:]
        qr, kr = apply_rope(q, c, s), apply_rope(k, c, s)
        vr = apply_rope(v, c, s) if rotate_v else v
        return F.scaled_dot_product_attention(qr, kr, vr, is_causal=True)
    return output(0, False), output(50, False), output(0, True), output(50, True)


def part6(args):
    a, b, c, d = value_shift_demo(args.seed)
    print("  同一段内容，因果 mask，相对布局相同；起始绝对位置分别为0、50。")
    error("只转q/k的注意力输出", a, b)
    error("也转v的注意力输出", c, d)
    print("  只转 q/k 改变注意力权重；转 v 还会旋转被聚合的内容，绝对平移后的输出一般不再相等。")


def part7(args):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for base in (10000, 1000000):
        cos, sin = rope_cache(1025, 64, base)
        scores = apply_rope(torch.ones(1025, 64), cos, sin).sum(-1)
        ax.plot(range(1025), scores, label=f"base={base}")
        for distance in (0, 1, 16, 64, 128, 512, 1024):
            print(f"  全1向量q=k；头维度=64；base={base}；距离={distance}；点积={scores[distance]:.4f}")
    ax.set(xlabel="Relative distance", ylabel="Dot product (q=k=ones)", title="Oscillatory kernel; not monotonic decay")
    ax.legend()
    fig.tight_layout()
    path = OUT / "distance_base.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  曲线图片={path}")
    print('  “平均意义上的上界式衰减”是远程衰减的直觉描述；本图是全1向量的振荡核，不是任意q/k的严格单调上界。')
    print("  具体q/k不一定单调；范数给出的通用上界不随距离下降。base增大让低频更慢，不保证外推效果更好。")


def part8(args):
    length, scale, dh = 64, 4, 64
    theta = 10000 ** (-(dh-2)/dh)
    print("  角度用连续位置边界L、4L比较；实际长度L的最后编号是L−1。以下只做数值、不训练。")
    for label, base, divisor in (("原始", 10000, 1), ("位置插值", 10000, scale), ("增大base", 1000000, 1)):
        low = base ** (-(dh-2)/dh)
        print(f"  方法={label}；base={base}；位置除数={divisor}；最低频在训练边界L={length}的角度={length*low/divisor:.4f}；在4L={4*length}的角度={4*length*low/divisor:.4f}")
    print(f"  原训练最低频角度区间边界=[0.0000,{length*theta:.4f}]。超出指未取模相位/距离超出训练覆盖，不是cos/sin无定义。")
    print("  插值压缩所有频率，也压缩相邻位置差；改base改变各频率的比例，最高频仍为1。二者均不保证无需适配就能保持质量。")


class PositionGPT(GPT):
    """仅本章对照：沿用 GPT 参数与默认前向，可选学习式位置、partial RoPE。

    partial 保留完整 dh 的频率网格，只转前若干个前后配对平面；不是按
    缩小后的维度重算频率。last_logits 仅跳过无用位置的词表投影。
    """
    def __init__(self, cfg, mode="rope", partial=1.0):
        if mode not in ("rope", "learned", "none"):
            raise ValueError("未知位置模式")
        if not 0 < partial <= 1:
            raise ValueError("partial 必须在 (0,1] 内")
        cfg.rope = mode == "rope"
        super().__init__(cfg)
        self.mode, self.partial = mode, partial
        self.position = nn.Embedding(cfg.block_size, cfg.d_model) if mode == "learned" else None
        if self.position is not None:
            self._init(self.position)
        self.set_context(cfg.block_size)

    def set_context(self, length):
        if self.position is not None and length > self.position.num_embeddings:
            raise ValueError(f"学习式位置无定义：请求长度 {length} 超过训练表长 {self.position.num_embeddings}")
        dh = self.cfg.d_model // self.cfg.n_head
        cos, sin = rope_cache(length, dh, self.cfg.rope_base)
        pairs = max(1, int(dh//2*self.partial)) if self.mode == "rope" else 0
        keep = torch.arange(dh) % (dh//2) < pairs
        self.cos, self.sin = torch.where(keep, cos, 1.), torch.where(keep, sin, 0.)
        self.cfg.block_size = length

    def embeddings(self, idx, pos0=0):
        x = self.tok(idx)
        if self.position is not None:
            x = x + self.position(torch.arange(pos0, pos0+idx.shape[1], device=idx.device))
        return x

    def forward(self, idx, **kwargs):
        return super().forward(inputs_embeds=self.embeddings(idx, kwargs.get("pos0", 0)), **kwargs)

    def last_logits(self, idx, lengths):
        x = self.embeddings(idx)
        t = idx.shape[1]
        if t > self.cfg.block_size:
            raise ValueError("先调用 set_context 扩展数值位置表")
        for block in self.blocks:
            x = block(x, self.cos[:t], self.sin[:t])
        return self.head(self.norm(x[torch.arange(len(x)), lengths-1]))


def last_logits(model, x, lengths):
    return model.last_logits(x, lengths)


def part9(args):
    data = load_latex_lm()
    tok = data.tokenizer
    bos, eos = (tok.special_ids[s] for s in ("<|bos|>", "<|eos|>"))
    train = FormulaWindows([s[1:-1] for s in data.train], bos, eos, args.context)
    contexts = list(dict.fromkeys([args.context] + (args.eval_contexts or [2*args.context])))
    vals = {c: FormulaWindows([s[1:-1] for s in data.val], bos, eos, c) for c in contexts}
    print(f"  标准划分=latex-v1；训练公式={len(data.train)}；验证公式={len(data.val)}；BPE合并={len(tok.merges)}；词表={len(tok)}")
    print(f"  验证目标数={len(vals[args.context])}；原文字节数={data.n_bytes}；每个内容token与EOS计一次；BOS屏蔽输出；按目标加权。")
    print("  逐目标窗口，不跨公式；长公式滑窗不丢目标；窗口位置从零开始；只投影最后真实位置；测试集不参与。")
    scores = {}
    for mode in args.modes:
        torch.manual_seed(args.seed)
        model = PositionGPT(GPTConfig(vocab_size=len(tok), block_size=args.context,
                            d_model=64, n_head=2, n_layer=1, rope_base=args.base), mode, args.partial)
        print(f"  模型={mode}；设备=CPU；层数=1；维度=64；头数=2；上下文={args.context}；步数={args.steps}；批量={args.batch}；学习率={args.lr:.4f}；种子={args.seed}；base={args.base:.4f}；旋转比例={args.partial:.4f}；参数量={model.n_params()}")
        print("  AdamW；权重衰减=0.0100；无预热、无梯度裁剪；fp32；共同参数初值与抽样目标相同，学习式另有位置参数。")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        g = torch.Generator().manual_seed(args.seed)
        start, losses = time.perf_counter(), []
        for step in range(1, args.steps+1):
            x, lengths, y = train.batch(torch.randint(len(train), (args.batch,), generator=g))
            logits = model.last_logits(x, lengths)
            logits[:, bos] = -torch.inf
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if step % max(1, args.steps//4) == 0 or step == args.steps:
                print(f"  模型={mode}；训练步={step}；区间训练nat/token={sum(losses)/len(losses):.4f}")
                losses.clear()
        print(f"  模型={mode}；实际训练用时秒={time.perf_counter()-start:.4f}")
        for context, val in vals.items():
            try:
                model.set_context(context)
            except ValueError as exc:
                print(f"  模型={mode}；评测上下文={context}；无法评测={exc}")
                continue
            start = time.perf_counter()
            result = evaluate_lm(model, val, data.n_bytes, batch=128, window_logits=last_logits)
            scores[mode, context] = result["nat_per_token"]
            print(f"  模型={mode}；评测上下文={context}；val nat/token={result['nat_per_token']:.4f}；bits/byte={result['bits_per_byte']:.4f}；评测用时秒={time.perf_counter()-start:.4f}")
        if (mode, args.context) in scores:
            for c in contexts:
                if c > args.context and (mode, c) in scores:
                    print(f"  模型={mode}；外推上下文={c}；外推减训练窗口的验证差值={scores[mode,c]-scores[mode,args.context]:.4f}")
    print("  RoPE/无位置通过重建数值表解除实现的block_size限制；学习式没有额外位置权重。能运行不代表质量不变，可能改善也可能变差。")
    print("  验证目标及字节数不变，但窗口变长也增加了可见内容；短训、单个种子不能证明任何位置方案普遍更好。")


@torch.no_grad()
def cache_diagnostic(seed):
    torch.manual_seed(seed)
    model = GPT(GPTConfig(vocab_size=32, block_size=32, n_layer=2, n_head=2, d_model=32)).eval()
    ids = torch.tensor([[1, 8, 3, 12, 5, 7, 2, 9]])
    full = model(ids)[0][:, -1]
    good, bad = model.new_caches(), model.new_caches()
    model(ids[:, :-1], caches=good)
    model(ids[:, :-1], caches=bad)
    right = model(ids[:, -1:], caches=good, pos0=ids.shape[1]-1)[0][:, -1]
    wrong = model(ids[:, -1:], caches=bad, pos0=0)[0][:, -1]
    return full, right, wrong


def part10(args):
    full, good, bad = cache_diagnostic(args.seed)
    error("正确KV cache与整段logits", good, full)
    error("新token错误pos0=0与正确logits", bad, good)
    inv = 10000 ** (-torch.arange(0, 64, 2).float()/64)
    for pos in (4096, 8192):
        fp = pos*inv
        bf = torch.tensor(pos, dtype=torch.bfloat16)*inv.bfloat16()
        print(f"  位置={pos}；fp32与bf16角度最大误差={(fp-bf.float()).abs().max():.4f}；cos最大误差={(fp.cos()-bf.cos().float()).abs().max():.4f}；sin最大误差={(fp.sin()-bf.sin().float()).abs().max():.4f}")
    print("  这里先将fp32频率量化为bf16，再用bf16乘位置与计算三角函数；正确缓存用fp32计算角度/cos/sin，最后才转激活dtype。")
    a, b, c, d = value_shift_demo(args.seed)
    error("把RoPE加在v上：绝对平移输出", c, d)
    try:
        rope_cache(4, 7)
    except ValueError as exc:
        print(f"  头维度=7；异常={type(exc).__name__}: {exc}")
    print("  KV cache的位置必须接续历史；第15章将完整实现缓存解码。")


PARTS = {1: (part1, "没有位置信息会怎样"), 2: (part2, "绝对位置编码"),
         3: (part3, "二维旋转"), 4: (part4, "推广到d维"), 5: (part5, "两种约定"),
         6: (part6, "为什么只转q和k"), 7: (part7, "远程衰减与base"),
         8: (part8, "长上下文扩展"), 9: (part9, "在LaTeX上对比"), 10: (part10, "常见错误")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--eval-contexts", nargs="+", type=int)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--base", type=float, default=10000)
    parser.add_argument("--partial", type=float, choices=(0.5, 1.0), default=1.0)
    parser.add_argument("--modes", nargs="+", choices=("rope", "learned", "none"), default=["rope", "learned", "none"])
    args = parser.parse_args()
    if min(args.steps, args.batch, args.context, *(args.eval_contexts or [1])) < 1 or not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.base) or args.base <= 0 or any(p not in PARTS for p in args.parts):
        parser.error("步数、批量、上下文、学习率、base须为正且有限；部分号须为1–10")
    torch.set_num_threads(2)
    print(f"第10章；设备=CPU；线程数=2；种子={args.seed}；浮点输出四位小数；微小误差另打印乘一百万。")
    start = time.perf_counter()
    for p in args.parts or PARTS:
        fn, title = PARTS[p]
        print("\n" + "="*70 + f"\n{p}. {title}（10.{p}）\n" + "="*70)
        begun = time.perf_counter()
        fn(args)
        print(f"第 {p} 部分用时秒={time.perf_counter()-begun:.4f}")
    print(f"第10章所选实验完成；tour总用时秒={time.perf_counter()-start:.4f}")


if __name__ == "__main__":
    main()
