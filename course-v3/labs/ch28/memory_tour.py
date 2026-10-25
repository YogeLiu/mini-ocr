"""第 28 章：训练时的内存从哪里来；梯度检查点；LoRA；冻结哪一部分。先估算，再在 CPU 上精确测量（不需要 GPU）。

    python -m labs.ch28.memory_tour 1 2 3 4 5   # 估算与测量：不训练，几分钟
    python -m labs.ch28.memory_tour 6           # 全参 / 冻结 / LoRA 的训练对照（需要 outputs/ch22/vlm_mlp_stage1.pt）
    python -m labs.ch28.memory_tour 7 --gpu     # 有空闲 GPU 时：用 CUDA 的显存统计核对第 2 部分的测量
"""
import os
import sys
import time

if "--gpu" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from labs.ch22.vlm_tour import need
from labs.ch26.sft_tour import Pool, forward_loss, to_tensors
from miniocr.models.gpt import GPT, GPTConfig
from miniocr.models.lora import LoRALinear, apply_lora, count_params, merge_lora
from miniocr.models.vit import ViT, ViTConfig
from miniocr.models.vlm import VLM, MergeProjector, MLPProjector, set_trainable
from miniocr.train.memory import SavedTensors, checkpoint_blocks, fmt_bytes, static_memory

DEV = "cuda" if "--gpu" in sys.argv and torch.cuda.is_available() else "cpu"
torch.set_num_threads(min(16, os.cpu_count() or 1))
OUT = "outputs/ch28"
os.makedirs(OUT, exist_ok=True)
LM_TARGETS = ("qkv", "o", "gate", "up", "down")          # 我们的 GPT 里的线性层（第 16 章）


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def our_vlm(connector="mlp", seed=0):
    """与第 22 章结构相同的 VLM（随机权重：测内存不需要训练好的权重）。"""
    from miniocr.data.corpus import text_vocab
    V = text_vocab()
    torch.manual_seed(seed)
    vit = ViT(ViTConfig(img_size=64, patch=8, d_model=128, n_layer=4, n_head=4, pos="learned", pool="mean"))
    cfg = torch.load(need("outputs/ch16/gpt_char.pt"), map_location="cpu")["cfg"]
    lm = GPT(GPTConfig(**cfg))
    conn = MLPProjector(128, lm.cfg.d_model) if connector == "mlp" else MergeProjector(128, lm.cfg.d_model)
    return VLM(vit, conn, lm, V.stoi["<img>"]).to(DEV)


def batch(n=8, seed=0, task=None, nv=64):
    from labs.ch26.sft_tour import make_example
    rng = np.random.default_rng(seed)
    return [make_example(rng, task, multi_turn=0.0, nv=nv) for _ in range(n)]


def lora_setup(vlm, r):
    """冻结 ViT 和语言模型，语言模型的线性层装上 LoRA，连接器全部训练。返回可训练参数个数。"""
    set_trainable(vlm, vision=False, connector=True, lm=False)
    apply_lora(vlm.lm, LM_TARGETS, r=r)
    return count_params(vlm)[1]


# ---------------------------------------------------------------------- 1
def lora_params_of(model, targets, r):
    return sum(r * (m.in_features + m.out_features) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in targets)


def part1_static():
    section("1. 参数、梯度、优化器状态：只和参数量有关（估算）")
    rows = []
    m = our_vlm()
    total, _ = count_params(m)
    rows.append(("我们的 VLM", total, lora_params_of(m.lm, LM_TARGETS, 16), m.lm))
    from transformers import AutoConfig, AutoModelForImageTextToText
    real = {"Qwen3.5-0.8B": "Qwen/Qwen3.5-0.8B", "Qwen2.5-VL-3B": "Qwen/Qwen2.5-VL-3B-Instruct", "LLaVA-1.5-7B": "llava-hf/llava-1.5-7b-hf"}
    hf_targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    for name, repo in real.items():
        cfg = AutoConfig.from_pretrained(repo)
        with torch.device("meta"):
            mm = AutoModelForImageTextToText.from_config(cfg)
        lm = mm.model.language_model if hasattr(mm.model, "language_model") else mm.language_model
        rows.append((name, sum(p.numel() for p in mm.parameters()), lora_params_of(lm, hf_targets, 16), None))
        del mm
    print("  LoRA：r = 16，装在语言模型的所有注意力和 MLP 线性层上（我们的 GPT：qkv、o、gate、up、down）")
    print(f"  {'模型':<14}{'参数':>9}{'LoRA 参数':>12}   {'全参 fp32 + AdamW':>18}{'全参 bf16 混合精度':>20}{'LoRA（bf16 底座）':>18}")
    for name, n, nl, _ in rows:
        a = static_memory(n, n, "fp32")["total"]
        b = static_memory(n, n, "bf16", master_fp32=True)["total"]
        c = static_memory(n + nl, nl, "bf16", master_fp32=True)["total"]   # LoRA 的 A、B 也是参数
        print(f"  {name:<14}{n / 1e6:8.1f}M{nl / 1e6:10.2f}M ({nl / n:5.2%})  {fmt_bytes(a):>14}{fmt_bytes(b):>20}{fmt_bytes(c):>18}")
    print("  全参 fp32 + AdamW：每个参数 4（权重）+ 4（梯度）+ 8（m、v）= 16 字节")
    print("  bf16 混合精度：2（bf16 权重）+ 2（梯度）+ 8（m、v）+ 4（fp32 主权重）= 16 字节——并不省，省的是激活值和计算")
    print("  LoRA：底座 2 字节/参数，只有 LoRA 参数要梯度和优化器状态（16 字节/参数）")


# ---------------------------------------------------------------------- 2
def measure(m, items, mode=None, steps=0):
    """一次前向保存的激活值（字节）；steps > 0 时另外测平均每步（前向 + 反向）的秒数。"""
    imgs, p = to_tensors(items)
    m.train()
    with SavedTensors(m) as s:
        loss = forward_loss(m, imgs, p)
    loss.backward() if loss.requires_grad else None
    m.zero_grad(set_to_none=True)
    sec = None
    if steps:
        t0 = time.time()
        for _ in range(steps):
            loss = forward_loss(m, imgs, p)
            if loss.requires_grad:
                loss.backward()
            m.zero_grad(set_to_none=True)
        sec = (time.time() - t0) / steps
    return s.bytes, s.count, sec


def part2_activations():
    section("2. 激活值：在 CPU 上精确数出前向保存了多少（批 8 条，描述任务）")
    items = batch(8, task="caption")
    L = max(len(it[1]) for it in items)
    print(f"  每条约 {L} 个 token，其中 64 个是视觉 token；语言模型 {our_vlm().lm.cfg.n_layer} 层 d=256，ViT 4 层 d=128")
    configs = [("全部训练", dict()), ("冻结 ViT", dict(vision=False)), ("冻结语言模型（只训 ViT + 连接器）", dict(lm=False)),
               ("只训连接器（第 22 章阶段 1）", dict(vision=False, lm=False)), ("只训 ViT", dict(connector=False, lm=False))]
    print(f"    {'设置':<28}{'可训练参数':>12}{'保存的激活值':>14}{'张量数':>8}{'每步（前向+反向）':>18}")
    for name, kw in configs:
        m = our_vlm()
        n = set_trainable(m, **kw)
        b, c, sec = measure(m, items, steps=3)
        print(f"    {name:<28}{n:>12,}{fmt_bytes(b):>14}{c:>8}{sec * 1000:>14.0f} 毫秒")
    m = our_vlm()
    lora_setup(m, 16)
    n = count_params(m)[1]
    b, c, sec = measure(m, items, steps=3)
    print(f"    {'LoRA r=16（语言模型）+ 连接器':<28}{n:>12,}{fmt_bytes(b):>14}{c:>8}{sec * 1000:>14.0f} 毫秒")
    m = our_vlm()
    set_trainable(m, False, False, False)
    b, c, sec = measure(m, items, steps=3)
    print(f"    {'全部冻结（只做前向）':<28}{0:>12,}{fmt_bytes(b):>14}{c:>8}{sec * 1000:>14.0f} 毫秒")
    print("  对照参数本身：", fmt_bytes(sum(p.numel() for p in m.parameters()) * 4), "（fp32）")


# ---------------------------------------------------------------------- 3
def part3_scaling():
    section("3. 激活值怎样随序列长度、视觉 token 数增长")
    m = our_vlm()
    set_trainable(m)
    print("  固定 1 条样本，改变序列长度（在描述的回答后面补文字）：")
    xs, ys = [], []
    base = batch(1, task="caption")[0]
    for extra in (0, 64, 128, 256, 320):
        img, ids, roles, t = base
        it = (img, ids + ids[-1:] * extra, roles + roles[-1:] * extra, t)
        b, _, _ = measure(m, [it])
        xs.append(len(it[1]))
        ys.append(b)
        print(f"    序列 {len(it[1]):4d} 个 token：{fmt_bytes(b)}")
    k, c0 = np.polyfit(xs, ys, 1)
    lm = m.lm.cfg
    print(f"  直线拟合：每多 1 个 token 多 {k / 1024:.1f} KB = {k / lm.n_layer / lm.d_model / 4:.1f} 个 d 维 fp32 向量 / 层；"
          f"固定部分 {fmt_bytes(c0)}（ViT 与连接器的激活值、输出层等）")
    print("  每个 token 的这部分激活值来自：每层的 RMSNorm 输入输出、qkv、注意力输出、SwiGLU 的 gate/up/中间结果……"
          "（第 28.3 节逐项数）")
    print("\n  同一批 8 条描述，MLP 连接器（64 个视觉 token） vs 2×2 合并（16 个）：")
    for conn, nv in (("mlp", 64), ("merge", 16)):
        mm = our_vlm(conn)
        set_trainable(mm)
        b, _, sec = measure(mm, batch(8, task="caption", nv=nv), steps=3)
        print(f"    {conn:<6} 视觉 token {nv:3d}：激活值 {fmt_bytes(b):>10}，每步 {sec * 1000:.0f} 毫秒")


# ---------------------------------------------------------------------- 4
def part4_checkpoint():
    section("4. 梯度检查点：用计算换内存")
    items = batch(8, task="caption")
    for name, ck_lm, ck_vit in (("不用", False, False), ("语言模型的块", True, False), ("语言模型 + ViT 的块", True, True)):
        m = our_vlm()
        set_trainable(m)
        if ck_lm:
            checkpoint_blocks(m.lm.blocks)
        if ck_vit:
            checkpoint_blocks(m.vision.blocks)
        b, c, sec = measure(m, items, steps=3)
        print(f"    检查点：{name:<14} 激活值 {fmt_bytes(b):>10}（{c:3d} 个张量）  每步 {sec * 1000:.0f} 毫秒")
    m1, m2 = our_vlm(), our_vlm()
    checkpoint_blocks(m2.lm.blocks)
    imgs, p = to_tensors(items)
    l1 = forward_loss(m1.train(), imgs, p)
    l1.backward()
    l2 = forward_loss(m2.train(), imgs, p)
    l2.backward()
    d = max((a.grad - b.grad).abs().max().item() for a, b in zip(m1.parameters(), m2.parameters()) if a.grad is not None)
    print(f"  两种方式的损失 {l1.item():.6f} vs {l2.item():.6f}，梯度最大差 {d:.1e}：检查点不改变数学，只改变什么时候算")


# ---------------------------------------------------------------------- 5
def part5_lora_checks():
    section("5. LoRA：参数量、初始等价、合并")
    m = our_vlm()
    total = count_params(m)[0]
    print(f"  整个 VLM {total:,} 个参数；语言模型的线性层：")
    for n, mod in m.lm.named_modules():
        if isinstance(mod, torch.nn.Linear) and n.startswith("blocks.0."):
            print(f"    {n:<20} {mod.in_features:4d} → {mod.out_features:4d}   {mod.weight.numel():>8,} 个参数；r = 16 时 LoRA 加 "
                  f"{16 * (mod.in_features + mod.out_features):>6,} 个")
    for r in (1, 4, 16, 64):
        mm = our_vlm()
        set_trainable(mm, False, False, False)
        names = apply_lora(mm.lm, LM_TARGETS, r=r)
        n = count_params(mm)[1]
        print(f"    r = {r:3d}：替换 {len(names)} 层，LoRA 参数 {n:>9,}（占整个 VLM 的 {n / total:6.2%}）")
    items = batch(4)
    imgs, p = to_tensors(items)
    m = our_vlm().eval()
    with torch.no_grad():
        a = forward_loss(m, imgs, p).item()
        apply_lora(m.lm, LM_TARGETS, r=16)
        b = forward_loss(m, imgs, p).item()
        for mod in m.modules():
            if isinstance(mod, LoRALinear):
                mod.B.normal_(0, 0.02)
        c = forward_loss(m, imgs, p).item()
        merge_lora(m)
        d = forward_loss(m, imgs, p).item()
    print(f"  损失：原模型 {a:.6f}；刚装上 LoRA（B = 0）{b:.6f}；B 随机后 {c:.6f}；合并之后 {d:.6f}")


# ---------------------------------------------------------------------- 6
def part6_train_compare():
    section("6. 训练对照：全参、冻结、LoRA（从第 22 章阶段 1 出发，与第 26 章相同的 2000 步）")
    from labs.ch26 import sft_tour as c26
    path = "outputs/ch22/vlm_mlp_stage1.pt"
    if not os.path.exists(path):
        print(f"  缺少 {path}：先运行 python -m labs.ch22.vlm_tour 2")
        return
    settings = [("全参", None), ("冻结 ViT", dict(vision=False)), ("冻结语言模型", dict(lm=False)),
                ("LoRA r=4 + 连接器", 4), ("LoRA r=16 + 连接器", 16), ("LoRA r=64 + 连接器", 64)]
    for name, how in settings:
        m = c26.load_stage1()
        if isinstance(how, int):
            lr = 1e-3
            orig = c26.set_trainable
            c26.set_trainable = lambda vlm, _r=how, **kw: lora_setup(vlm, _r)
        elif how is not None:
            lr = 3e-4
            orig = c26.set_trainable
            c26.set_trainable = lambda vlm, _kw=how, **kw: set_trainable(vlm, **_kw)
        else:
            lr, orig = 3e-4, None
        try:
            sec, _, _ = c26.train(m, c26.STEPS, Pool(None, 80), lr=lr)
        finally:
            if orig is not None:
                c26.set_trainable = orig
        n = count_params(m)[1]
        print(f"    {name:<18} 可训练 {n:>10,}  学习率 {lr:g}  {sec:.0f} 秒：{c26.fmt(c26.quick_eval(m))}")


# ---------------------------------------------------------------------- 7
def part7_gpu():
    section("7. 用 CUDA 的显存统计核对（需要 --gpu 和空闲的 GPU）")
    if DEV != "cuda":
        print("  没有 --gpu 或没有可用的 GPU，跳过")
        return
    items = batch(8, task="caption")
    for name, kw in (("全部训练", dict()), ("只训连接器", dict(vision=False, lm=False))):
        m = our_vlm()
        set_trainable(m, **kw)
        imgs, p = to_tensors(items)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        with SavedTensors(m) as s:
            loss = forward_loss(m.train(), imgs, p)
        after = torch.cuda.memory_allocated()
        print(f"    {name:<10} SavedTensors {fmt_bytes(s.bytes):>10}；前向前后显存增加 {fmt_bytes(after - before):>10}"
              f"（多出来的是 logits、损失等还没释放的结果）")
        loss.backward()


PARTS = {1: part1_static, 2: part2_activations, 3: part3_scaling, 4: part4_checkpoint, 5: part5_lora_checks,
         6: part6_train_compare, 7: part7_gpu}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS)
    for k in want:
        PARTS[k]()
    if want == list(PARTS):
        print("\n第 28 章的演示全部完成。")
