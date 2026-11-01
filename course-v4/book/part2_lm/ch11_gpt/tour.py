"""第 11 章：搭建 GPT 与初始化；CPU、本地语料，不下载或保存权重。

python -m book.part2_lm.ch11_gpt.tour [部分号 ...]
辅助对照不是挖空练习；练习 11.1、11.2 在 miniocr.models.gpt。
"""
import argparse
from contextlib import contextmanager
import copy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time
from types import MethodType

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from torch import nn
from torch.nn import functional as F

from miniocr.data.latex_lm import FormulaWindows, evaluate_lm, load_latex_lm
from miniocr.models.gpt import GPT, GPTConfig, apply_rope


def parameter_terms(vocab, d, layers, ff, *, tied=True, gelu=False,
                    bias=False, layer_norm=False, learned_positions=0):
    """按矩阵、偏置、归一化逐项计数；仅普通 MHA，不含 GQA/混合层。"""
    return {
        "词嵌入": vocab*d,
        "位置嵌入": learned_positions*d,
        "每块注意力": 4*d*d + (4*d if bias else 0),
        "每块前馈": (2 if gelu else 3)*d*ff + ((ff+d if gelu else 2*ff+d) if bias else 0),
        "每块两个归一化": 2*d*(2 if layer_norm else 1),
        "最后归一化": d*(2 if layer_norm else 1),
        "独立输出层": 0 if tied else vocab*d,
    }


def parameter_count(cfg):
    if cfg.kv_heads() != cfg.n_head or cfg.qk_norm:
        raise ValueError("本章公式只覆盖普通 MHA、无 Q/K norm")
    terms = parameter_terms(cfg.vocab_size, cfg.d_model, cfg.n_layer,
                            cfg.ff_dim(), tied=cfg.tie_weights)
    return total_params(terms, cfg.n_layer)


def total_params(terms, layers):
    return sum(value*(layers if name.startswith("每块") else 1) for name, value in terms.items())


def fmt(values):
    return "[" + ", ".join(f"{v:.4f}" for v in values) + "]"


def rms(x):
    return x.float().square().mean().sqrt().item()


def last_logits(model, idx, lengths):
    """与完整 GPT 前向等价，只投影每个窗口最后一个真实位置。"""
    x = model.tok(idx)
    t = idx.shape[1]
    assert t <= model.cfg.block_size
    for block in model.blocks:
        x = block(x, model.cos[:t], model.sin[:t])
    return model.head(model.norm(x[torch.arange(len(x)), lengths-1]))


def reinitialize(model, std, scaled=True):
    """按唯一参数遍历，避免共享矩阵被初始化两次。"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim == 2:
                scale = math.sqrt(2*model.cfg.n_layer) if scaled and name.endswith(
                    ("attn.o.weight", "mlp.down.weight")) else 1
                nn.init.normal_(param, std=std/scale)


def full_attention(self, x, cos, sin, cache=None):
    """故障注入：普通 MHA 的正确投影/RoPE，但所有位置都可见。"""
    assert cache is None and self.h == self.hk and self.q_norm is None
    b, t, _ = x.shape
    q, k, v = [proj(x).view(b, t, self.h, self.dh).transpose(1, 2)
               for proj in (self.q, self.k, self.v)]
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    mask = torch.ones(t, t, dtype=torch.bool, device=x.device)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return self.o(y.transpose(1, 2).reshape(b, t, -1))


@contextmanager
def visible_future(model):
    originals = [block.attn.forward for block in model.blocks]
    try:
        for block in model.blocks:
            block.attn.forward = MethodType(full_attention, block.attn)
        yield
    finally:
        for block, forward in zip(model.blocks, originals):
            block.attn.forward = forward


@torch.no_grad()
def future_difference(model, x):
    changed = x.clone()
    cut = x.shape[1]//2
    changed[:, cut:] = (changed[:, cut:]+17) % model.cfg.vocab_size
    return (model(x)[0][:, :cut]-model(changed)[0][:, :cut]).abs().max().item()


class Experiment:
    def __init__(self, args):
        self.args, self.data = args, None

    def prepare(self):
        if self.data is not None:
            return
        self.data = load_latex_lm()
        self.vocab = len(self.data.tokenizer)
        self.bos = self.data.tokenizer.special_ids["<|bos|>"]
        self.stream = torch.tensor([i for seq in self.data.train for i in seq])
        print(f"  标准划分=latex-v1；训练公式={len(self.data.train)}；验证公式={len(self.data.val)}；词表={self.vocab}；BPE合并={len(self.data.tokenizer.merges)}")
        print("  训练取BOS…EOS拼接流，忽略BOS目标；评测逐公式，BOS屏蔽输出、EOS计入、按目标加权；长公式逐目标滑窗且位置重置。")

    def model(self, book=False, **overrides):
        self.prepare()
        cfg = GPTConfig(vocab_size=self.vocab, block_size=128) if book else GPTConfig(
            vocab_size=self.vocab, block_size=self.args.context, n_layer=self.args.layers,
            n_head=4, d_model=self.args.dim, tie_weights=not self.args.untied)
        torch.manual_seed(self.args.seed)
        model = GPT(replace(cfg, **overrides))
        if not book and self.args.init_std is not None:
            reinitialize(model, self.args.init_std)
        return model

    def batch(self, generator, batch, context, stream=None):
        stream = self.stream if stream is None else stream
        starts = torch.randint(len(stream)-context, (batch,), generator=generator)
        rows = stream[starts[:, None]+torch.arange(context+1)]
        return rows[:, :-1], rows[:, 1:]

    def evaluate(self, model, label):
        start = time.perf_counter()
        eos = self.data.tokenizer.special_ids["<|eos|>"]
        val = FormulaWindows([s[1:-1] for s in self.data.val], self.bos, eos, model.cfg.block_size)
        result = evaluate_lm(model, val, self.data.n_bytes, batch=128, window_logits=last_logits)
        print(f"  模型={label}；正确因果val nat/token={result['nat_per_token']:.4f}；bits/byte={result['bits_per_byte']:.4f}；验证目标={len(val)}；原文字节数={self.data.n_bytes}；评测用时秒={time.perf_counter()-start:.4f}")
        return result

    def train(self, model, label, steps, *, fixed=None, unshifted=False, seconds=45):
        args, cfg = self.args, model.cfg
        batch = args.batch if fixed is None else len(fixed[0])
        print(f"  模型={label}；设备=CPU；层数={cfg.n_layer}；维度={cfg.d_model}；头数={cfg.n_head}；中间维={cfg.ff_dim()}；批量={batch}；上下文={cfg.block_size}；计划步数={steps}；学习率={args.lr:.4f}；种子={args.seed}；共享={cfg.tie_weights}；参数量={model.n_params()}")
        print(f"  初始化基础std={args.init_std if args.init_std is not None else 0.02:.4f}；残差输出缩放=1/√(2L)。")
        print("  AdamW；权重衰减=0.0100；fp32；无预热、无梯度裁剪；训练交叉熵包含BOS输出类别，BOS目标忽略；验证再屏蔽BOS输出。")
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        generator = torch.Generator().manual_seed(args.seed)
        history, start = [], time.perf_counter()
        for step in range(1, steps+1):
            x, y = fixed if fixed is not None else self.batch(generator, batch, cfg.block_size)
            target = x if unshifted else y
            _, loss = model(x, target, loss_mask=target != self.bos)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            history.append(loss.item())
            if step in {1, 10, 50, 100, 200, steps}:
                print(f"  模型={label}；训练步={step}；当前批次loss={loss.item():.4f}")
            if fixed is None and time.perf_counter()-start > seconds:
                print(f"  达到训练时间预算；实际步数={step}；跨机器比较须核对实际步数。")
                break
        print(f"  模型={label}；实际训练步数={len(history)}；训练用时秒={time.perf_counter()-start:.4f}")
        return history


@torch.no_grad()
def part1(exp):
    model = exp.model(book=True).eval()
    print(f"  本书配置：词表={model.cfg.vocab_size}；最大上下文=128；层数=4；头数=4；维度=256；其余默认；演示批量=2；序列长=16")
    modules = [("词嵌入", model.tok)] + [(f"块{i+1}", b) for i, b in enumerate(model.blocks)] + [("最后归一化", model.norm), ("输出层", model.head)]
    handles = [module.register_forward_hook(lambda m, inp, out, label=label:
               print(f"  模块={label}；输出形状={tuple(out.shape)}")) for label, module in modules]
    try:
        logits, _ = model(torch.arange(32).reshape(2, 16) % exp.vocab)
        print(f"  最终logits形状={tuple(logits.shape)}")
    finally:
        for handle in handles:
            handle.remove()


def part2(exp):
    model = exp.model(book=True)
    cfg = model.cfg
    terms = parameter_terms(exp.vocab, 256, 4, cfg.ff_dim())
    print("  本书公式：V·d + L·(4d² + 3df + 2d) + d；共享head新增参数为0。")
    for name, value in terms.items():
        print(f"  项目={name}；参数量={value}")
    print(f"  中间维={cfg.ff_dim()}；公式总数={parameter_count(cfg)}；n_params实测={model.n_params()}；不共享增加={exp.vocab*256}")
    untied = GPT(replace(cfg, tie_weights=False))
    print(f"  不共享实测={untied.n_params()}；实测增加={untied.n_params()-model.n_params()}")
    terms = parameter_terms(50257, 768, 12, 3072, gelu=True, bias=True,
                            layer_norm=True, learned_positions=1024)
    print("  GPT-2 small：词表=50257；上下文=1024；层数=12；维度=768；GELU中间维=3072；LayerNorm含β；线性层含偏置；共享无偏置head。")
    for name, value in terms.items():
        print(f"  GPT-2项目={name}；参数量={value}")
    print(f"  GPT-2精确总参数={total_params(terms, 12)}")
    root = Path.home()/".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B"
    paths = sorted(root.glob("snapshots/*/config.json"))
    if not paths:
        print("  本地Qwen3.5-0.8B配置未缓存；跳过，不联网。")
        return
    path = paths[-1]
    config = json.loads(path.read_text())
    print(f"  只读本地配置={path}；未加载权重。")
    for key, value in config.items():
        print(f"  Qwen配置字段={key}；值={json.dumps(value, ensure_ascii=False)}")
    print("  此配置含线性注意力、门控、视觉分支等本章未讲部件：第 32 章再讲；不套普通MHA公式硬算参数。")


def part3(exp):
    shared = exp.model(n_layer=1, d_model=32)
    untied = GPT(replace(shared.cfg, tie_weights=False))
    untied.load_state_dict(shared.state_dict())
    x, y = exp.batch(torch.Generator().manual_seed(3), 2, 8)
    shared(x, y)[1].backward()
    untied(x, y)[1].backward()
    a, b, both = untied.tok.weight.grad, untied.head.weight.grad, shared.tok.weight.grad
    delta = (a+b-both).abs().max().item()
    print(f"  head.weight is tok.weight={shared.head.weight is shared.tok.weight}；data_ptr相同={shared.head.weight.data_ptr() == shared.tok.weight.data_ptr()}；节省参数={untied.n_params()-shared.n_params()}")
    print(f"  嵌入路径梯度范数={a.norm():.4f}；输出路径梯度范数={b.norm():.4f}；共享梯度范数={both.norm():.4f}；两路和最大差={delta:.4f}；差值乘一百万={delta*1e6:.4f}；两路之和等于共享梯度={torch.allclose(a+b, both, atol=1e-6, rtol=1e-5)}")
    print("  两路贡献来自相同数值、拆成两个独立参数的对照模型；不是把同一.grad打印两遍。")


@torch.no_grad()
def part4(exp):
    model = exp.model(book=True).eval()
    seq = torch.tensor([i for s in exp.data.val for i in s])
    context = min(64, len(seq)-1)
    x, y = exp.batch(torch.Generator().manual_seed(0), 8, context, seq)
    print(f"  初始loss检查：验证批量=8；实际序列长={context}；样例降级时按可用长度缩短。")
    print(f"  可预测类别（屏蔽BOS）={exp.vocab-1}；ln(V−1)={math.log(exp.vocab-1):.4f}；GPT原始类别={exp.vocab}；lnV={math.log(exp.vocab):.4f}")
    for std in (None, 1.0, 0.2):
        candidate = copy.deepcopy(model)
        if std is not None:
            torch.manual_seed(exp.args.seed)
            reinitialize(candidate, std, scaled=False)
        logits, loss = candidate(x, y, loss_mask=y != exp.bos)
        masked = logits.clone()
        masked[..., exp.bos] = -torch.inf
        adjusted = F.cross_entropy(masked[y != exp.bos], y[y != exp.bos])
        label = "本书0.0200且残差缩放" if std is None else f"全部矩阵std={std:.4f}且不缩放"
        print(f"  初始化={label}；原始loss={loss:.4f}；屏蔽BOS后loss={adjusted:.4f}；logits标准差={logits.std():.4f}")
    zero = GPT(replace(model.cfg, tie_weights=False))
    nn.init.zeros_(zero.head.weight)
    print(f"  独立head零初始化；loss={zero(x, y, loss_mask=y != exp.bos)[1]:.4f}；lnV={math.log(exp.vocab):.4f}；logits标准差={zero(x)[0].std():.4f}")
    print("  GPT.forward不屏蔽BOS输出，两种口径均打印；零head使用不共享版本，避免同时清空嵌入。")


@torch.no_grad()
def part5(exp):
    for depth in (4, 8, 16, 32):
        scaled = exp.model(book=True, n_layer=depth, d_model=64)
        plain = copy.deepcopy(scaled)
        for name, p in plain.named_parameters():
            if name.endswith(("attn.o.weight", "mlp.down.weight")):
                p.mul_(math.sqrt(2*depth))
        x = torch.arange(64).reshape(2, 32) % exp.vocab
        for label, model in (("缩放", scaled), ("不缩放", plain)):
            h = model.tok(x)
            values = [rms(h)]
            for block in model.blocks:
                h = block(h, model.cos[:32], model.sin[:32])
                values.append(rms(h))
            print(f"  层数={depth}；维度=64；初始化={label}；嵌入及逐层残差RMS={fmt(values)}")
    print("  对照共用同一随机方向；不缩放版把输出矩阵乘回√(2L)，标准差恢复0.0200；RMS不保证逐层单调。")


def part6(exp):
    model = exp.model(n_layer=2, d_model=128, block_size=64)
    batch = exp.batch(torch.Generator().manual_seed(exp.args.seed), 8, 64)
    exp.train(model, "固定一个批次", 200, fixed=batch)
    with torch.no_grad():
        loss = model(*batch, loss_mask=batch[1] != exp.bos)[1]
    print(f"  第200步更新后的固定批次loss={loss:.4f}")
    print("  固定一个批次能过拟合到接近0，是检查代码能否学习的冒烟测试，不是坏信号；不代表泛化能力。")


def part7(exp):
    for label in ("正常", "标签未错位", "全可见mask"):
        model = exp.model(n_layer=2, d_model=96, block_size=16)
        if label == "全可见mask":
            with visible_future(model):
                exp.train(model, label, exp.args.fault_steps, seconds=25)
                x, _ = exp.batch(torch.Generator().manual_seed(9), 2, 16)
                delta = future_difference(model, x)
                print(f"  全可见mask：改未来token后过去logits是否变化={delta > 1e-6}；最大差={delta:.4f}")
            print(f"  恢复因果mask后过去logits最大差={future_difference(model, x):.4f}")
        else:
            exp.train(model, label, exp.args.fault_steps, unshifted=label == "标签未错位", seconds=25)
        exp.evaluate(model, label)
    print("  三者初值、配置、样本随机种子、计划步数相同；全部用正确错位标签和因果mask评测。全可见训练能偷看下一token，但低到什么程度取决于短训优化。")


def part8(exp):
    model = exp.model()
    exp.train(model, "LaTeX小GPT", exp.args.steps, seconds=45)
    exp.evaluate(model, "LaTeX小GPT")
    print("  第7章MLP上文8基线：val nat/token=3.8124；bits/byte=2.7493。")
    print("  这是第13章正式训练前的冒烟测试；训练预算、上下文和参数量不同，不能据此宣称架构胜负；样例降级数据时不可对比历史基线。")


@torch.no_grad()
def part9(exp):
    model = exp.model(book=True)
    x = torch.arange(16).reshape(1, 16) % exp.vocab
    h = model.tok(x)
    for block in model.blocks:
        h = block(h, model.cos[:16], model.sin[:16])
    print(f"  忘final norm：残差RMS={rms(h):.4f}；归一化后RMS={rms(model.norm(h)):.4f}；logits最大差={(model.head(h)-model(x)[0]).abs().max():.4f}")
    before = model.tok.weight.clone()
    nn.init.normal_(model.head.weight, std=0.2)
    print(f"  共享后单独初始化head：嵌入是否改变={not torch.equal(before, model.tok.weight)}；嵌入最大差={(before-model.tok.weight).abs().max():.4f}")
    unique = sum(p.numel() for p in model.parameters())
    state = sum(p.numel() for p in model.state_dict().values())
    print(f"  parameters去重计数={unique}；state_dict重复计数={state}；多算={state-unique}；恰为V·d={exp.vocab*256}")
    try:
        model(torch.zeros(1, 129, dtype=torch.long))
    except AssertionError as exc:
        print(f"  超过block_size异常={type(exc).__name__}: {exc}")


PARTS = {1: (part1, "组装与形状（11.1–11.2）"), 2: (part2, "参数量与真实配置（11.3、11.9）"),
         3: (part3, "共享权重（11.4）"), 4: (part4, "初始loss（11.5）"),
         5: (part5, "残差缩放（11.6）"), 6: (part6, "过拟合一个批次（11.7）"),
         7: (part7, "两类可疑低loss（11.8）"), 8: (part8, "LaTeX短训（11.7）"),
         9: (part9, "常见错误（11.10）")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--fault-steps", type=int, default=350)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--init-std", type=float)
    parser.add_argument("--untied", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.fault_steps, args.batch, args.context, args.layers, args.dim) < 1 or args.dim % 8 or any(p not in PARTS for p in args.parts) or not math.isfinite(args.lr) or args.lr <= 0 or (args.init_std is not None and (not math.isfinite(args.init_std) or args.init_std <= 0)):
        parser.error("部分号须为1–9；数值须为正且有限；维度须为8的倍数")
    torch.set_num_threads(2)
    print(f"第11章；设备=CPU；线程数=2；种子={args.seed}；浮点数四位小数。")
    start = time.perf_counter()
    exp = Experiment(args)
    for part in args.parts or PARTS:
        fn, title = PARTS[part]
        print("\n" + "="*70 + f"\n{part}. {title}\n" + "="*70)
        begun = time.perf_counter()
        fn(exp)
        print(f"第 {part} 部分用时秒={time.perf_counter()-begun:.4f}")
    print(f"第11章所选实验完成；tour总用时秒={time.perf_counter()-start:.4f}")


if __name__ == "__main__":
    main()
