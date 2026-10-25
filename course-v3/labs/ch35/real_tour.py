"""第 35 章：搬到真实模型。transformers / peft / trl 与我们自己的代码的对应；聊天模板与回答的 mask；
SmolVLM-256M 与 Qwen3.5-0.8B 的 LoRA（阻断项检查）；显存估算；用我们自己的训练循环做真实模型的 SFT 与 GRPO。

    python -m labs.ch35.real_tour 1 2 5     # 不加载权重：库的对应与默认值、聊天模板、显存估算（约 10 秒）
    python -m labs.ch35.real_tour 3 4       # 加载权重，CPU：SmolVLM 与 Qwen3.5 的 LoRA 检查（作者的 CPU 上约 20 秒，峰值内存约 8 GB）
    python -m labs.ch35.real_tour 6         # 你自己跑：SmolVLM-256M 的 LoRA SFT（文字行 OCR），训练前后的 CER
    python -m labs.ch35.real_tour 7         # 你自己跑：SmolVLM-256M 的 GRPO（我们自己的循环，1 − NED 奖励）
    python -m labs.ch35.real_tour 8         # 你自己跑：Qwen3.5-0.8B 的 LoRA SFT；Xiaomi-OCR-0 的零样本对照（需要先下载）
    加 --gpu 用 GPU（先用 nvidia-smi 确认显存够用：GPU 可能被别的程序占着）
"""
import math
import os
import resource
import sys
import time

if "--gpu" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from miniocr.hf_train import answer_mask_by_prefix, hf_answer_logp, lora_target_names, trainable_params, trl_reward
from miniocr.hf_vlm import QWEN, SMOL
from miniocr.metrics import cer, exact_match

torch.set_num_threads(8)
DEV = torch.device("cuda" if "--gpu" in sys.argv and torch.cuda.is_available() else "cpu")
OUT = "outputs/ch35"
os.makedirs(OUT, exist_ok=True)
QUESTION = "Read the text in the image."
QWEN_ALL = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj")


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ---------------------------------------------------------------------- 1
def part1_libraries():
    section("1. 三个库与我们自己的代码：对应关系与默认值（transformers / peft / trl）")
    import dataclasses
    import peft
    import transformers
    import trl
    print(f"  transformers {transformers.__version__}   peft {peft.__version__}   trl {trl.__version__}   torch {torch.__version__}")
    rows = [("模型的前向、generate", "miniocr.models.vlm、generate.py（第 17、22 章）", "AutoModelForImageTextToText"),
            ("图像预处理、聊天模板", "hf_vlm.qwen_preprocess、expand_image_tokens（第 24 章）", "AutoProcessor、apply_chat_template"),
            ("回答的 mask", "train.sft.chat_roles（第 26 章）、hf_train.answer_mask_by_prefix", "SFTConfig(assistant_only_loss / completion_only_loss)"),
            ("LoRA", "models.lora（第 28 章）、hf_train.lora_target_names", "peft.LoraConfig、get_peft_model、merge_and_unload"),
            ("回答的对数概率", "rl.logprobs（第 27 章）、hf_train.hf_answer_logp", "trainer 内部（融合的 LM head）"),
            ("DPO", "rl.dpo.dpo_loss（第 30 章）", "DPOTrainer、DPOConfig"),
            ("GRPO / DAPO", "rl.grpo.grpo_loss、rl.pg.group_advantages（第 29、31 章）", "GRPOTrainer、GRPOConfig"),
            ("奖励", "rl.rewards.task_reward（第 31 章）、hf_train.trl_reward", "GRPOTrainer(reward_funcs=[...])")]
    for what, ours, theirs in rows:
        print(f"    {what:<14} 我们：{ours}\n    {'':<14} 库里：{theirs}")
    from trl import DPOConfig, GRPOConfig, SFTConfig
    show = {GRPOConfig: ["learning_rate", "beta", "epsilon", "epsilon_high", "loss_type", "scale_rewards", "num_generations",
                         "num_iterations", "temperature", "max_completion_length", "mask_truncated_completions"],
            DPOConfig: ["learning_rate", "beta", "loss_type"],
            SFTConfig: ["learning_rate", "assistant_only_loss", "completion_only_loss", "packing", "max_length"]}
    print("  trl 的默认值（直接从配置类读出来）：")
    for C, names in show.items():
        inst = C(output_dir=os.path.join(OUT, "tmp"), report_to=[], use_cpu=True)          # 没有 GPU 时必须 use_cpu
        print(f"    {C.__name__}: " + ", ".join(f"{k}={getattr(inst, k)!r}" for k in names))
    from trl.trainer.utils import _supports_trl_loss_kernel
    x = torch.zeros(2, 3, 5)
    print(f"  trl 的融合 LM head（Triton kernel）支持 CPU 张量吗：{_supports_trl_loss_kernel(x, torch.zeros(2, 3, dtype=torch.long))}"
          "（SFTTrainer、DPOTrainer、GRPOTrainer 都无条件地装上它，所以这三个 trainer 在这个版本里只能在 GPU 上跑）")


# ---------------------------------------------------------------------- 2
def part2_templates():
    section("2. 聊天模板与回答的 mask：用\"提示是完整序列的前缀\"来定出回答")
    from transformers import AutoProcessor, AutoTokenizer
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
    ans = "Total 42.50"
    tok = AutoTokenizer.from_pretrained(QWEN)
    pr = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    full_t = tok.apply_chat_template(msgs + [{"role": "assistant", "content": [{"type": "text", "text": ans}]}], tokenize=False)
    print(f"  Qwen3.5 生成提示的结尾：{pr[-40:]!r}")
    print(f"  Qwen3.5 训练用的完整对话的结尾：{full_t[-50:]!r}")
    a, b = tok(pr)["input_ids"], tok(pr + ans + "<|im_end|>")["input_ids"]
    m = answer_mask_by_prefix(a, b)
    print(f"  提示 {len(a)} 个 token，完整 {len(b)} 个；算损失的 token：{tok.convert_ids_to_tokens([t for t, k in zip(b, m) if k])}")
    p = AutoProcessor.from_pretrained(SMOL)
    pr2 = p.apply_chat_template(msgs, add_generation_prompt=True)
    print(f"  SmolVLM 生成提示的结尾：{pr2[-30:]!r}")
    t = p.tokenizer
    for sep in (" ", ""):
        a, b = t(pr2)["input_ids"], t(pr2 + sep + ans + "<end_of_utterance>")["input_ids"]
        try:
            m = answer_mask_by_prefix(a, b)
            print(f"    回答前加 {sep!r:<4}：前缀成立；算损失的 token：{t.convert_ids_to_tokens([x for x, k in zip(b, m) if k])}")
        except ValueError as e:
            print(f"    回答前加 {sep!r:<4}：{e}")
    for tail in ("Assistant: 42", "Assistant:42"):
        print(f"    {tail!r:<16} 分词 → {t.convert_ids_to_tokens(t(tail, add_special_tokens=False)['input_ids'])}")


# ---------------------------------------------------------------------- 数据与模型
def line_data(seed, n):
    """第 13、24 章的文字行（噪声 0.15、形变 1.0），转成白底黑字的 RGB PIL 图。测试集用种子 2（和第 24 章相同）。"""
    from PIL import Image
    from miniocr.data.text import LineDataset
    ds = LineDataset(seed, n, noise=0.15, distort=1.0)
    imgs = [Image.fromarray(((1 - ds.x[i, 0].numpy()) * 255).astype(np.uint8)).convert("RGB") for i in range(n)]
    return imgs, ds.texts


def load_smol():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    p = AutoProcessor.from_pretrained(SMOL)
    m = AutoModelForImageTextToText.from_pretrained(SMOL, dtype=torch.float32).to(DEV)
    return p, m


def smol_prompt(p):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
    return p.apply_chat_template(msgs, add_generation_prompt=True)


def smol_example(p, img, text, longest=512):
    """一条训练样本：processor 处理"提示 + 空格 + 回答 + <end_of_utterance>"，mask 用前缀定出。"""
    pr = smol_prompt(p)
    kw = {"size": {"longest_edge": longest}}
    a = p(text=[pr], images=[[img]], return_tensors="pt", **kw)
    b = p(text=[pr + " " + text + "<end_of_utterance>"], images=[[img]], return_tensors="pt", **kw)
    mask = torch.tensor([answer_mask_by_prefix(a["input_ids"][0].tolist(), b["input_ids"][0].tolist())])
    return {k: v.to(DEV) for k, v in b.items()}, mask.to(DEV)


@torch.no_grad()
def smol_read(p, m, imgs, longest=512, max_new=32):
    m.eval()
    out = []
    for img in imgs:
        inp = p(text=[smol_prompt(p)], images=[[img]], return_tensors="pt", size={"longest_edge": longest}).to(DEV)
        g = m.generate(**inp, max_new_tokens=max_new, do_sample=False)
        out.append(p.tokenizer.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    return out


# ---------------------------------------------------------------------- 3
def part3_smol_lora():
    section("3. SmolVLM-256M + peft 的 LoRA（CPU）：加在哪、恒等、能学、能合并")
    from peft import LoraConfig, get_peft_model
    p, m = load_smol()
    names = lora_target_names(m)
    allq = [n for n, _ in m.named_modules() if n.split(".")[-1] == "q_proj"]
    print(f"  名字以 q_proj 结尾的层共 {len(allq)} 个，其中视觉塔里 {sum('vision' in n for n in allq)} 个；"
          f"lora_target_names 选出 {len(names)} 个（只在语言模型里）：{names[0]} … {names[-1]}")
    imgs, texts = line_data(0, 4)
    batch = [smol_example(p, i, t) for i, t in zip(imgs, texts)]
    b0, k0 = batch[0]
    print(f"  一条样本：{b0['input_ids'].shape[1]} 个 token，pixel_values {tuple(b0['pixel_values'].shape)}，"
          f"算损失的 {int(k0.sum())} 个：{p.tokenizer.decode(b0['input_ids'][0][k0[0].bool()])!r}")
    m.eval()
    with torch.no_grad():
        base = m(**b0).logits
    pm = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=names))
    tr, tot = trainable_params(pm)
    print(f"  LoRA r = 16：可训练 {tr:,d} / {tot:,d}（{tr / tot:.2%}）")
    with torch.no_grad():
        print(f"  刚加上 LoRA（B = 0）时 logits 的最大差：{(pm(**b0).logits - base).abs().max().item():.2e}")
    opt = torch.optim.AdamW([q for q in pm.parameters() if q.requires_grad], lr=1e-3)
    pm.train()
    print("  同样 4 条样本重复训练 5 步（过拟合检查：只是确认梯度能让损失下降，不是训练结果）：")
    for step in range(5):
        t0, tot_l = time.time(), 0.0
        for b, k in batch:
            loss = -hf_answer_logp(pm(**b).logits, b["input_ids"], k, "mean").mean() / len(batch)
            loss.backward()
            tot_l += loss.item()
        opt.step()
        opt.zero_grad()
        print(f"    第 {step + 1} 步  每个回答 token 的平均损失 {tot_l:.3f}   {time.time() - t0:.1f} 秒（4 条）")
    pm.eval()
    with torch.no_grad():
        before = pm(**b0).logits
        merged = pm.merge_and_unload()
        after = merged(**b0).logits
    print(f"  merge_and_unload 前后 logits 的最大差：{(before - after).abs().max().item():.2e}；"
          f"和训练前的最大差：{(after - base).abs().max().item():.2f}；峰值内存 {peak_gb():.1f} GB")


# ---------------------------------------------------------------------- 4
def part4_qwen_check():
    section("4. Qwen3.5-0.8B 的阻断项检查（CPU）：kernel、层的种类、LoRA 的目标、前向反向")
    import importlib.util
    for mod, what, fallback in (("fla", "flash-linear-attention（线性注意力的 chunk kernel）", "PyTorch 的参考实现（正确但慢）"),
                                ("causal_conv1d", "causal_conv1d（短卷积）", "PyTorch 的参考实现（正确但慢）"),
                                ("flash_attn", "flash-attn（全注意力层）", "PyTorch 的 SDPA")):
        print(f"  {what:<44} {'已安装' if importlib.util.find_spec(mod) else '没有安装 → ' + fallback}")
    from labs.ch24.hf_tour import qwen, qwen_ask, qwen_inputs
    from peft import LoraConfig, get_peft_model
    tok, m = qwen()
    kinds = {}
    for n, mod in m.named_modules():
        if isinstance(mod, torch.nn.Linear) and "visual" not in n:
            kinds[n.split(".")[-1]] = kinds.get(n.split(".")[-1], 0) + 1
    print("  语言模型里的线性层（按名字的最后一段）：" + "，".join(f"{k} {v}" for k, v in kinds.items()))
    n_layers = m.config.text_config.num_hidden_layers
    for label, suf in (("只写 q/k/v/o_proj", ("q_proj", "k_proj", "v_proj", "o_proj")), ("加上线性注意力的投影", QWEN_ALL)):
        names = lora_target_names(m, suf)
        layers = {int(n.split("layers.")[1].split(".")[0]) for n in names}
        print(f"    LoRA 目标 {label:<18} {len(names):3d} 个层，覆盖 {len(layers):2d} / {n_layers} 个 decoder 层")
    imgs, texts = line_data(0, 2)
    inp, n_ans = qwen_inputs(imgs[0], QUESTION, answer=texts[0], scale=2)
    T = inp["input_ids"].shape[1]
    mask = torch.zeros(1, T)
    mask[0, T - n_ans:] = 1
    m.eval()
    with torch.no_grad():
        base = m(**inp).logits
    before = qwen_ask(imgs[1], QUESTION, max_new=24, scale=2)
    targets = lora_target_names(m, QWEN_ALL)                         # 要在 get_peft_model 之前取：之后这些层被 LoRA 包起来了
    pm = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=targets))
    tr, tot = trainable_params(pm)
    with torch.no_grad():
        d = (pm(**inp).logits - base).abs().max().item()
    print(f"  LoRA r = 16（全部 {len(targets)} 个目标）：可训练 {tr:,d} / {tot:,d}（{tr / tot:.2%}）；"
          f"B = 0 时 logits 的最大差 {d:.2e}")
    pm.train()
    for step in range(2):
        t0 = time.time()
        loss = -hf_answer_logp(pm(**inp).logits, inp["input_ids"], mask, "mean").mean()
        loss.backward()
        pm.zero_grad()
        print(f"    前向 + 反向（{T} 个 token，其中 {n_ans} 个回答 token）：{time.time() - t0:.1f} 秒")
    pm.eval()
    with torch.no_grad():
        after = qwen_ask(imgs[1], QUESTION, max_new=24, scale=2)
    print(f"  加上 LoRA（未更新）前后，贪心读第 2 行的结果相同：{before == after}（{after!r}，真值 {texts[1]!r}）")
    print(f"  峰值内存 {peak_gb():.1f} GB")


# ---------------------------------------------------------------------- 5
def part5_memory():
    section("5. 显存估算：参数 + 梯度 + 优化器状态（第 28 章的 static_memory），不含激活值")
    from miniocr.train.memory import fmt_bytes, static_memory
    rows = [("SmolVLM-256M", 256_484_928, 1_843_200), ("Qwen3.5-0.8B", 852_985_920, 4_915_200)]
    print(f"    {'模型':<14} {'全参数 fp32':>12} {'全参数 bf16 混合':>16} {'LoRA（bf16 底座）':>18}")
    for name, n, lora in rows:
        a = static_memory(n, n)["total"]
        b = static_memory(n, n, "bf16", master_fp32=True)["total"]
        c = static_memory(n + lora, lora, "bf16", master_fp32=True)["total"]
        print(f"    {name:<14} {fmt_bytes(a):>12} {fmt_bytes(b):>16} {fmt_bytes(c):>18}")
    print("  （参数量：SmolVLM 来自第 3 部分的 LoRA 统计减去 LoRA 参数；Qwen3.5 同理来自第 4 部分）")


# ---------------------------------------------------------------------- 6
def part6_smol_sft(n_train=2000, epochs=1, accum=8, lr=1e-4, n_test=50, longest=512):
    section(f"6. SmolVLM-256M 的 LoRA SFT：{n_train} 行文字、{epochs} 轮、每 {accum} 条更新一次（你自己跑）")
    from peft import LoraConfig, get_peft_model
    p, m = load_smol()
    te_imgs, te_texts = line_data(2, n_test)
    t0 = time.time()
    pred0 = smol_read(p, m, te_imgs, longest)
    print(f"  训练前：测试 {n_test} 行  CER {cer(pred0, te_texts):7.2%}  整行全对 {exact_match(pred0, te_texts):6.1%}  "
          f"（{time.time() - t0:.0f} 秒）")
    for i in range(min(3, n_test)):
        print(f"    真值 {te_texts[i]!r:<30} 预测 {pred0[i]!r}")
    pm = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=lora_target_names(m)))
    params = [q for q in pm.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    imgs, texts = line_data(0, n_train)
    steps = epochs * math.ceil(n_train / accum)
    rng = np.random.default_rng(0)
    t0, step, run = time.time(), 0, 0.0
    pm.train()
    for ep in range(epochs):
        order = rng.permutation(n_train)
        for j in range(0, n_train, accum):
            for i in order[j:j + accum]:
                b, k = smol_example(p, imgs[i], texts[i], longest)
                loss = -hf_answer_logp(pm(**b).logits, b["input_ids"], k, "mean").mean() / accum
                loss.backward()
                run += loss.item()
            for g in opt.param_groups:                                   # 线性预热 5%，然后余弦衰减到 0
                w = max(1, steps // 20)
                g["lr"] = lr * min(1.0, (step + 1) / w) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / steps)))
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 25 == 0 or step == steps:
                print(f"    第 {step:4d} / {steps} 步  损失 {run / min(25, step % 25 or 25):.4f}  {time.time() - t0:.0f} 秒")
                run = 0.0
    pred1 = smol_read(p, pm, te_imgs, longest)
    print(f"  训练后：测试 {n_test} 行  CER {cer(pred1, te_texts):7.2%}  整行全对 {exact_match(pred1, te_texts):6.1%}")
    for i in range(min(3, n_test)):
        print(f"    真值 {te_texts[i]!r:<30} 预测 {pred1[i]!r}")
    pm.save_pretrained(os.path.join(OUT, "smol_lora"))
    print(f"  LoRA 权重存在 {OUT}/smol_lora（只有 LoRA 的部分，几 MB）")


# ---------------------------------------------------------------------- 7
def smol_rollout(p, m, img, G, max_new=24, temperature=1.0, longest=512):
    """同一张图采 G 个回答。返回 (G 条完整序列的输入 dict, 回答的 mask (G, T), 回答的文字列表)。"""
    inp = p(text=[smol_prompt(p)], images=[[img]], return_tensors="pt", size={"longest_edge": longest}).to(DEV)
    P = inp["input_ids"].shape[1]
    with torch.no_grad():
        g = m.generate(**inp, max_new_tokens=max_new, do_sample=True, temperature=temperature, top_k=0,
                       num_return_sequences=G)
    eos = p.tokenizer.convert_tokens_to_ids("<end_of_utterance>")
    resp = g[:, P:]
    mask = torch.zeros_like(resp, dtype=torch.float)
    texts = []
    for r in range(G):
        row = resp[r].tolist()
        n = row.index(eos) + 1 if eos in row else len(row)                 # 回答到第一个结束符为止（含）
        mask[r, :n] = 1
        texts.append(p.tokenizer.decode(row[:n], skip_special_tokens=True).strip())
    full = {"input_ids": g, "attention_mask": torch.cat([inp["attention_mask"].repeat(G, 1), mask.long()], 1),
            "pixel_values": inp["pixel_values"].repeat(G, *[1] * (inp["pixel_values"].dim() - 1))}
    if "pixel_attention_mask" in inp:
        full["pixel_attention_mask"] = inp["pixel_attention_mask"].repeat(G, *[1] * (inp["pixel_attention_mask"].dim() - 1))
    full_mask = torch.cat([torch.zeros(G, P, device=g.device), mask], 1)
    return full, full_mask, texts


def part7_smol_grpo(steps=100, n_prompt=4, G=8, lr=1e-5, n_test=50, longest=512, adapter=None):
    section(f"7. SmolVLM-256M 的 GRPO（我们自己的循环）：{steps} 步，每步 {n_prompt} 题 × {G} 个回答（你自己跑）")
    from peft import LoraConfig, PeftModel, get_peft_model
    from miniocr.rl.grpo import grpo_loss
    from miniocr.rl.logprobs import token_logprobs
    from miniocr.rl.pg import group_advantages, zero_variance_fraction
    p, m = load_smol()
    adapter = adapter or os.path.join(OUT, "smol_lora")
    if os.path.exists(adapter):
        pm = PeftModel.from_pretrained(m, adapter, is_trainable=True)
        print(f"  从第 6 部分的 LoRA（{adapter}）出发")
    else:
        pm = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=lora_target_names(m)))
        print("  没有找到第 6 部分的 LoRA：从原始的 SmolVLM 出发（它几乎读不出这些行，组内奖励大多相同，学得会很慢）")
    reward = trl_reward("text")
    te_imgs, te_texts = line_data(2, n_test)
    pred0 = smol_read(p, pm, te_imgs, longest)
    print(f"  训练前：CER {cer(pred0, te_texts):7.2%}  整行全对 {exact_match(pred0, te_texts):6.1%}")
    imgs, texts = line_data(5, steps * n_prompt)
    params = [q for q in pm.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    t0 = time.time()
    for step in range(steps):
        rs, zf, losses = [], [], []
        pm.eval()
        groups = [smol_rollout(p, pm, imgs[step * n_prompt + j], G, longest=longest) for j in range(n_prompt)]
        pm.train()
        for j, (full, mask, outs) in enumerate(groups):
            r = torch.tensor(reward(outs, [texts[step * n_prompt + j]] * G))
            adv = group_advantages(r, G).to(DEV)
            lp = token_logprobs(pm(**full).logits, full["input_ids"])         # (G, T − 1)
            k = mask[:, 1:]
            loss, st = grpo_loss(lp, lp.detach(), adv, k, mode="token")      # μ = 1：比值恒为 1，不裁剪
            (loss / n_prompt).backward()
            rs.append(r.mean().item())
            zf.append(zero_variance_fraction(r, G))
            losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()
        if (step + 1) % 10 == 0 or step == 0:
            print(f"    第 {step + 1:4d} 步  平均奖励 {np.mean(rs):.3f}  没有梯度的组 {np.mean(zf):4.0%}  {time.time() - t0:.0f} 秒")
    pred1 = smol_read(p, pm, te_imgs, longest)
    print(f"  训练后：CER {cer(pred1, te_texts):7.2%}  整行全对 {exact_match(pred1, te_texts):6.1%}")


# ---------------------------------------------------------------------- 8
def part8_qwen_sft(n_train=500, accum=8, lr=1e-4, n_test=50):
    section(f"8. Qwen3.5-0.8B 的 LoRA SFT（{n_train} 行）与 Xiaomi-OCR-0 的零样本对照（你自己跑）")
    from labs.ch24.hf_tour import qwen, qwen_ask, qwen_inputs
    from peft import LoraConfig, get_peft_model
    tok, m = qwen()
    m.to(DEV)
    te_imgs, te_texts = line_data(2, n_test)
    pred0 = [qwen_ask(i, QUESTION, max_new=32, scale=2) for i in te_imgs]
    print(f"  训练前：CER {cer(pred0, te_texts):7.2%}  整行全对 {exact_match(pred0, te_texts):6.1%}（第 24 章用的是另一个提示词）")
    pm = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=lora_target_names(m, QWEN_ALL)))
    params = [q for q in pm.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    imgs, texts = line_data(0, n_train)
    t0 = time.time()
    pm.train()
    for j in range(0, n_train, accum):
        for i in range(j, min(j + accum, n_train)):
            inp, n_ans = qwen_inputs(imgs[i], QUESTION, answer=texts[i], scale=2)
            inp = {k: v.to(DEV) for k, v in inp.items()}
            T = inp["input_ids"].shape[1]
            mask = torch.zeros(1, T, device=DEV)
            mask[0, T - n_ans:] = 1
            (-hf_answer_logp(pm(**inp).logits, inp["input_ids"], mask, "mean").mean() / accum).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()
        if (j // accum + 1) % 10 == 0:
            print(f"    第 {j // accum + 1} 步  {time.time() - t0:.0f} 秒")
    pm.eval()
    pred1 = [qwen_ask(i, QUESTION, max_new=32, scale=2) for i in te_imgs]
    print(f"  训练后：CER {cer(pred1, te_texts):7.2%}  整行全对 {exact_match(pred1, te_texts):6.1%}")
    from huggingface_hub import try_to_load_from_cache
    if try_to_load_from_cache("SeerRay-Lab/Xiaomi-OCR-0", "config.json") is None:
        print("  本机没有 SeerRay-Lab/Xiaomi-OCR-0：要做零样本对照，先看它的模型卡和许可证（Apache-2.0），"
              "下载后把 labs/ch24/hf_tour.py 里的 QWEN 换成它的名字，用同一个 qwen_ask 评测（第 35.8 节）")


PARTS = {1: part1_libraries, 2: part2_templates, 3: part3_smol_lora, 4: part4_qwen_check, 5: part5_memory,
         6: part6_smol_sft, 7: part7_smol_grpo, 8: part8_qwen_sft}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:] if a.isdigit()] or [1, 2, 3, 4, 5]
    for k in want:
        PARTS[k]()
