"""第 24 章：解剖真实的 VLM。配置对照 → processor 的每一步 → 模型内部 → M-RoPE → 在我们的玩具数据上零样本测试。
只用 CPU（不影响 GPU 上的其他程序）。需要 requirements-hf.txt 和下载好的 Qwen3.5-0.8B、SmolVLM-256M-Instruct
（第 1 部分另外读取 Qwen2.5-VL-3B 和 LLaVA-1.5-7B 的配置文件，不需要它们的权重）。

    python -m labs.ch24.hf_tour             # 全部 7 个部分（CPU 8 线程约 45 分钟）
    python -m labs.ch24.hf_tour 1 2         # 只看配置和 processor（不加载权重，几秒）
"""
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image

from miniocr.data.text import LineDataset
from miniocr.hf_vlm import (QWEN, SMOL, count_params, expand_image_tokens, idefics3_layout, mrope_axes,
                            mrope_positions, n_qwen_tokens, qwen_preprocess, smart_resize)
from miniocr.metrics import cer

torch.set_num_threads(8)
OUT = "outputs/ch24"
os.makedirs(OUT, exist_ok=True)
REPOS = {"Qwen3.5-0.8B": QWEN, "Qwen2.5-VL-3B": "Qwen/Qwen2.5-VL-3B-Instruct", "SmolVLM-256M": SMOL,
         "LLaVA-1.5-7B": "llava-hf/llava-1.5-7b-hf"}
N_TEST = int(os.environ.get("N_TEST", 50))     # 每个任务的样本数（冒烟测试时可调小）
_M = {}


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------- 模型
def qwen():
    """Qwen3.5-0.8B：分词器、我们自己的图像预处理、模型（float32，CPU）。"""
    if "qwen" not in _M:
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(QWEN)
        m = AutoModelForImageTextToText.from_pretrained(QWEN, dtype=torch.float32).eval()
        _M["qwen"] = (tok, m)
    return _M["qwen"]


def smol():
    if "smol" not in _M:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        p = AutoProcessor.from_pretrained(SMOL)
        m = AutoModelForImageTextToText.from_pretrained(SMOL, dtype=torch.float32).eval()
        _M["smol"] = (p, m)
    return _M["smol"]


def qwen_inputs(img, question, answer=None, scale=None, **resize_kw):
    """组装 Qwen3.5 的输入：聊天模板 → 分词 → 展开占位符 → 图像预处理。不用 AutoProcessor
    （它的视频部分依赖 torchvision），每一步都用本章自己写的函数。answer 不为 None 时把答案接在后面（用于计算损失）。"""
    tok, m = qwen()
    if scale:
        img = img.resize((img.width * scale, img.height * scale), Image.BICUBIC)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text)["input_ids"]
    n_ans = None
    if answer is not None:
        ans = tok(answer + "<|im_end|>")["input_ids"]
        ids, n_ans = ids + ans, len(ans)
    pv, grid = qwen_preprocess(img, **resize_kw)
    ids = expand_image_tokens(ids, grid, m.config.image_token_id)
    ids = torch.tensor([ids])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pv, image_grid_thw=grid,
                mm_token_type_ids=(ids == m.config.image_token_id).int()), n_ans


@torch.no_grad()
def qwen_ask(img, question, max_new=40, **kw):
    tok, m = qwen()
    inp, _ = qwen_inputs(img, question, **kw)
    out = m.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def smol_ask(img, question, max_new=40, longest=None):
    p, m = smol()
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    text = p.apply_chat_template(msgs, add_generation_prompt=True)
    kw = {} if longest is None else {"size": {"longest_edge": longest}}
    inp = p(text=[text], images=[[img]], return_tensors="pt", **kw)
    out = m.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return p.tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------- 数据
def toy_scene():
    """一张 64×64 的 ShapesWorld 场景图（第 22 章测试集的第 0 张）。"""
    from labs.ch22.vlm_tour import test_set
    X, pairs = test_set("caption")
    return Image.fromarray(X[0].permute(1, 2, 0).numpy()), pairs[0][1]


def lines(n=N_TEST):
    """第 13、22 章的文字行测试集（种子 2，噪声 0.15、形变 1.0）的前 n 行，转成白底黑字的 RGB 图。"""
    ds = LineDataset(2, n, noise=0.15, distort=1.0)
    imgs = [Image.fromarray(((1 - ds.x[i, 0].numpy()) * 255).astype(np.uint8)).convert("RGB") for i in range(n)]
    return imgs, ds.texts


# ---------------------------------------------------------------------- 1
def part1():
    section("1. 四个真实 VLM 的配置对照（只读 config.json，在 meta 设备上数参数）")
    from transformers import AutoConfig
    print(f"  {'模型':<15}{'视觉塔':<26}{'连接器':<24}{'语言模型':<26}{'总参数':>8}")
    for name, repo in REPOS.items():
        c = AutoConfig.from_pretrained(repo)
        v, t = c.vision_config, c.text_config
        depth = getattr(v, "depth", None) or v.num_hidden_layers
        p = count_params(c)
        tot = sum(p.values())
        vis = f"{depth} 层 d={v.hidden_size} p={v.patch_size}"
        if c.model_type == "qwen3_5":
            conn, lm = "2×2 合并 + MLP", f"{t.num_hidden_layers} 层 d={t.hidden_size}（18 线性 + 6 全注意力）"
        elif c.model_type == "qwen2_5_vl":
            conn, lm = "2×2 合并 + MLP", f"{t.num_hidden_layers} 层 d={t.hidden_size}"
        elif c.model_type == "idefics3":
            conn, lm = f"pixel shuffle {c.scale_factor}×{c.scale_factor} + 线性", f"{t.num_hidden_layers} 层 d={t.hidden_size}"
        else:
            conn, lm = "两层 MLP", f"{t.num_hidden_layers} 层 d={t.hidden_size}"
        print(f"  {name:<15}{vis:<26}{conn:<24}{lm:<26}{tot / 1e6:>7.0f}M")
        print(f"  {'':<15}{p['vision'] / 1e6:>6.1f}M（{p['vision'] / tot:.0%}）{'':<13}{p['connector'] / 1e6:>6.1f}M（{p['connector'] / tot:.1%}）"
              f"{'':<11}{p['lm'] / 1e6:>7.1f}M（词表 {t.vocab_size:,}，嵌入 {t.vocab_size * t.hidden_size / 1e6:.0f}M）")
    print("\n  每张图的视觉 token 数（同一张图在四个模型里）：")
    print(f"  {'原图':<12}{'Qwen3.5':>10}{'Qwen2.5-VL':>12}{'SmolVLM':>10}{'SmolVLM(512)':>14}{'LLaVA-1.5':>11}")
    for h, w in [(64, 64), (32, 256), (448, 448), (384, 576), (1024, 1024), (3000, 4000)]:
        q3 = n_qwen_tokens(h, w, patch=16)
        q25 = n_qwen_tokens(h, w, patch=14, min_pixels=3136, max_pixels=12845056)
        s = idefics3_layout(h, w)[2]
        s512 = idefics3_layout(h, w, longest=512)[2]
        print(f"  {f'{h}×{w}':<12}{q3:>10,}{q25:>12,}{s:>10,}{s512:>14,}{576:>11}")
    print("  Qwen2.5-VL：min_pixels 3136、max_pixels 12845056（preprocessor_config.json），28×28 像素一个 token；")
    print("  LLaVA-1.5：短边缩放到 336 再从中间裁出 336×336——不是正方形的图，两边会被裁掉（32×256 的文字行只剩中间 1/8）；")
    print("  SmolVLM：默认把长边放大/缩小到 2048 再切成 512×512 的块（加一张全局图），每块 64 个 token；")
    print("          括号里是把长边设成 512 时（只有一块）。四个模型的 mean/std：Qwen3.5、SmolVLM 用 0.5/0.5，")
    print("          Qwen2.5-VL、LLaVA 用 CLIP 的 (0.481, 0.458, 0.408) / (0.269, 0.261, 0.276)")


# ---------------------------------------------------------------------- 2
def part2():
    section("2. Qwen3.5 的 processor：缩放 → 归一化 → 切块（与 transformers 对照）")
    print("  smart_resize（factor = patch 16 × merge 2 = 32，像素数限制在 [256², 4096²]）：")
    for h, w in [(64, 64), (32, 256), (192, 256), (500, 333), (1080, 1920), (3000, 4000), (32, 4000)]:
        rh, rw = smart_resize(h, w)
        n = (rh // 16) * (rw // 16) // 4
        note = "放大" if rh * rw > h * w * 1.05 else ("缩小" if rh * rw < h * w * 0.95 else "")
        print(f"    {h:>5}×{w:<5} → {rh:>5}×{rw:<5}  网格 {rh // 16:>3}×{rw // 16:<3}  视觉 token {n:>6,}  "
              f"宽高比 {w / h:6.2f} → {rw / rh:6.2f}  {note}")
    img, cap = toy_scene()
    img.resize((256, 256), Image.NEAREST).save(f"{OUT}/toy_scene.png")
    pv, grid = qwen_preprocess(img)
    print(f"\n  玩具场景图 64×64（{cap}）：")
    print(f"    pixel_values {tuple(pv.shape)} = (网格 {grid[0, 1]}×{grid[0, 2]} 个 patch, 3 通道 × 2 帧 × 16 × 16)")
    print(f"    image_grid_thw {grid.tolist()}；进入语言模型的视觉 token = 1·{grid[0, 1]}·{grid[0, 2]} / 4 = {grid.prod() // 4}")
    print(f"    数值范围 [{pv.min():.2f}, {pv.max():.2f}]（(x/255 − 0.5)/0.5）；两帧相同："
          f"{torch.equal(pv.view(-1, 3, 2, 256)[:, :, 0], pv.view(-1, 3, 2, 256)[:, :, 1])}")
    print("    前 8 行对应的 patch（行, 列）：", end=" ")
    gh, gw = int(grid[0, 1]), int(grid[0, 2])
    order = [(wr * 2 + r, wc * 2 + c) for wr in range(gh // 2) for wc in range(gw // 2) for r in range(2) for c in range(2)]
    print(order[:8], "← 先 2×2 窗口，再下一个窗口")
    try:
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
        ip = Qwen2VLImageProcessorPil.from_pretrained(QWEN)
        for name, im in [("玩具场景 64×64", img), ("文字行 32×256", lines(1)[0][0])]:
            ref = ip(images=[im], return_tensors="pt")
            mine, g = qwen_preprocess(im)
            print(f"    与 transformers 对照（{name}）：grid {g.tolist()} vs {ref['image_grid_thw'].tolist()}，"
                  f"pixel_values 最大差 {(mine - ref['pixel_values']).abs().max():.1e}")
    except Exception as e:      # noqa: BLE001
        print("    （跳过 transformers 对照：", e, "）")


# ---------------------------------------------------------------------- 3
@torch.no_grad()
def part3():
    section("3. 模型内部：占位符 → 视觉塔 → merger → 替换嵌入")
    tok, m = qwen()
    img, cap = toy_scene()
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the image."}]}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    print("  聊天模板：", repr(text))
    ids = tok(text)["input_ids"]
    print(f"  分词后 {len(ids)} 个 token：", [tok.decode([i]) for i in ids])
    inp, _ = qwen_inputs(img, "Describe the image.")
    L = inp["input_ids"].shape[1]
    print(f"  展开占位符后 {L} 个 token（{(inp['input_ids'] == m.config.image_token_id).sum()} 个 <|image_pad|>）")
    s = tok.decode(inp["input_ids"][0])
    s2 = tok(text.replace("<|image_pad|>", "<|image_pad|>" * 64))["input_ids"]
    print(f"  与「先在字符串里复制 64 个占位符再分词」的结果相同：{inp['input_ids'][0].tolist() == s2}")

    vis = m.model.visual
    pv, grid = inp["pixel_values"], inp["image_grid_thw"]
    rec = {}
    hs = [vis.patch_embed.register_forward_hook(lambda mod, i, o: rec.__setitem__("patch_embed", o)),
          vis.blocks[-1].register_forward_hook(lambda mod, i, o: rec.__setitem__("blocks", o)),
          vis.merger.register_forward_hook(lambda mod, i, o: rec.__setitem__("merger", o))]
    out = vis(pv, grid_thw=grid)
    for h in hs:
        h.remove()
    print(f"\n  视觉塔（{len(vis.blocks)} 个 block，d={vis.config.hidden_size}）：")
    print(f"    pixel_values {tuple(pv.shape)} → Conv3d（核 2×16×16）{tuple(rec['patch_embed'].shape)}"
          f" → + 插值的可学习位置表（{vis.config.num_position_embeddings} = 48×48 格）+ 每层的二维 RoPE")
    print(f"    → {len(vis.blocks)} 个 block {tuple(rec['blocks'].shape)} → merger：每 4 行拼成 {4 * vis.config.hidden_size} 维 → "
          f"MLP → {tuple(rec['merger'].shape)}（= 语言模型的 d）")

    # 语言模型收到的 inputs_embeds：占位符的位置应该正好是 merger 的输出
    got = {}
    h = m.model.language_model.register_forward_pre_hook(lambda mod, a, kw: got.__setitem__("e", kw["inputs_embeds"]),
                                                         with_kwargs=True)
    m(**inp)
    h.remove()
    e = got["e"][0]
    mask = inp["input_ids"][0] == m.config.image_token_id
    print(f"  语言模型的输入里，占位符位置 = merger 输出：最大差 {(e[mask] - out.pooler_output).abs().max():.1e}；"
          f"文字位置 = 词嵌入：最大差 {(e[~mask] - m.get_input_embeddings()(inp['input_ids'][0][~mask])).abs().max():.1e}")
    tn = e[~mask].norm(dim=-1).mean()
    vn = e[mask].norm(dim=-1).mean()
    print(f"  向量长度：文字嵌入平均 {tn:.2f}，视觉 token 平均 {vn:.2f}（{vn / tn:.1f} 倍）")

    t = m.config.text_config
    lin = sum(x == "linear_attention" for x in t.layer_types)
    print(f"\n  语言模型 {t.num_hidden_layers} 层：{lin} 层线性注意力（Gated DeltaNet）+ {t.num_hidden_layers - lin} 层全注意力"
          f"（每 {t.full_attention_interval} 层一个）")
    kv_full = 2 * t.num_hidden_layers * t.num_key_value_heads * t.head_dim * 2
    kv_mix = 2 * (t.num_hidden_layers - lin) * t.num_key_value_heads * t.head_dim * 2
    state = lin * t.linear_num_value_heads * t.linear_key_head_dim * t.linear_value_head_dim * 4
    print(f"    KV cache（bf16）：若 24 层都是全注意力，每个 token {kv_full / 1024:.0f} KB；实际只有 6 层存 KV，"
          f"每个 token {kv_mix / 1024:.0f} KB；18 层线性注意力各存一个固定大小的状态，共 {state / 2**20:.1f} MB（与长度无关）")
    for L_ in (1_000, 10_000, 100_000):
        print(f"    序列 {L_:>7,}：全注意力版 {kv_full * L_ / 2**30:6.2f} GB，混合版 {(kv_mix * L_ + state) / 2**30:6.2f} GB")


# ---------------------------------------------------------------------- 4
@torch.no_grad()
def part4():
    section("4. M-RoPE：三维位置编号与频率分配")
    tok, m = qwen()
    img, _ = toy_scene()
    wide = img.resize((128, 64), Image.BICUBIC)
    inp, _ = qwen_inputs(wide, "Describe.", min_pixels=128 * 64)
    ids = inp["input_ids"][0].tolist()
    mine = mrope_positions(ids, inp["image_grid_thw"], m.config.image_token_id)
    ref, delta = m.model.get_rope_index(inp["input_ids"], mm_token_type_ids=inp["mm_token_type_ids"],
                                        image_grid_thw=inp["image_grid_thw"])
    print(f"  图 64×128 → 网格 {inp['image_grid_thw'][0, 1:].tolist()} → 合并后 2×4 = 8 个视觉 token；序列长 {len(ids)}")
    print(f"  我们的 mrope_positions 与 model.get_rope_index 相同：{torch.equal(mine, ref[:, 0])}")
    s = ids.index(m.config.image_token_id)
    print("  位置    token              时间  行  列")
    for i in range(max(0, s - 2), min(len(ids), s + 11)):
        print(f"  {i:>4}    {tok.decode([ids[i]])!r:<18}{mine[0, i]:>4}{mine[1, i]:>4}{mine[2, i]:>4}")
    print(f"  序列长 {len(ids)}，最大位置编号 {mine.max()}：rope_deltas = {delta.item()}（生成新 token 时用 1D 位置 + 这个差值）")

    rot = m.model.language_model.rotary_emb
    t = m.config.text_config
    dim = int(t.head_dim * t.rope_parameters["partial_rotary_factor"])
    print(f"\n  头维度 {t.head_dim}，只有前 {dim} 维旋转（partial_rotary_factor 0.25）→ {dim // 2} 个频率，"
          f"θ = {t.rope_parameters['rope_theta']:.0e}")
    # 探针：第 a 个位置只在第 a 轴上有偏移，看哪个频率"转动"了。低频转得极慢，所以同时用一个小偏移和一个大偏移
    moved = 0
    for off in (7, 999_983):
        probe = torch.zeros(3, 1, 3, dtype=torch.long)
        for ax in range(3):
            probe[ax, 0, ax] = off
        cos, _ = rot(torch.zeros(1, dtype=torch.float64), probe)
        moved = moved + (1 - cos[0, :, :dim // 2].double()).abs()        # (3 个位置, 32 个频率)
    found = ["THW"[int(moved[:, f].argmax())] for f in range(dim // 2)]
    expect = mrope_axes(dim // 2, t.rope_parameters["mrope_section"], interleaved=True)
    print("  实测每个频率属于哪个轴：", "".join(found))
    print("  mrope_axes 的预测：      ", "".join(expect), " 相同：", found == expect)
    old = mrope_axes(64, (16, 24, 24), interleaved=False)
    print("  Qwen2.5-VL（分段式）：   ", "".join(old))
    inv = rot.inv_freq
    print(f"  频率 ω_i 从 {inv[0]:.3f} 到 {inv[-1]:.2e}（周期 {2 * np.pi / inv[0]:.1f} 到 {2 * np.pi / inv[-1]:.1e} 个位置）；"
          f"分段式里列方向只分到最低频的 24 个，交错式里三个轴都覆盖高、中、低频")


# ---------------------------------------------------------------------- 5
@torch.no_grad()
def part5():
    section(f"5. 零样本：两个真实模型做我们的玩具任务（每个任务 {N_TEST} 个样本）")
    from labs.ch22.vlm_tour import TASKS, test_set
    prompts = {"caption": "List every shape in the image from left to right, with size, color and shape. "
                          "Answer like: a large red circle, a small blue square",
               "count": "How many shapes are in the image? Answer with a single digit.",
               "left": "What color is the leftmost shape? Answer with one word: red, green, blue or yellow.",
               "exists": None}
    res = {}
    for task in TASKS:
        X, pairs = test_set(task)
        for name, ask in [("Qwen3.5-0.8B", qwen_ask), ("SmolVLM-256M", smol_ask)]:
            t0 = time.time()
            ok, errs, preds, refs = 0, [], [], []
            for i in range(N_TEST):
                img = Image.fromarray(X[i].permute(1, 2, 0).numpy())
                q, a = pairs[i]
                q = prompts[task] or (q + " Answer yes or no.")
                kw = {"longest": 512} if name.startswith("Smol") else {}
                r = ask(img, q, max_new=60 if task == "caption" else 5, **kw)
                r_ = r.lower().strip(" .")
                if task == "caption":
                    preds.append(r_)
                    refs.append(a)
                    good = r_ == a
                else:
                    good = r_.split()[0].strip(".,") == a if r_ else False
                ok += good
                if not good and len(errs) < 2:
                    errs.append((q[:40], a, r[:60]))
            res[(name, task)] = ok / N_TEST
            extra = f"  CER {cer(preds, refs):.1%}" if refs else ""
            print(f"  {name:<14}{task:<9}准确率 {ok / N_TEST:6.1%}{extra}  （{time.time() - t0:.0f} 秒）")
            for q, a, r in errs:
                print(f"      真值 {a!r:<40} 回答 {r!r}")
    return res


# ---------------------------------------------------------------------- 6
@torch.no_grad()
def part6():
    section(f"6. 零样本 OCR：文字行（第 13 章测试集的前 {N_TEST} 行）与视觉 token 数")
    imgs, texts = lines()
    q = "Read the text in the image. Output only the text, exactly as written."
    for scale in (1, 2, 3, 4):
        t0 = time.time()
        outs = [qwen_ask(im, q, max_new=32, scale=scale, min_pixels=32 * 32) for im in imgs]
        n = n_qwen_tokens(32 * scale, 256 * scale, min_pixels=32 * 32)
        c = cer(outs, texts)
        acc = np.mean([o == t for o, t in zip(outs, texts)])
        print(f"  Qwen3.5  放大 {scale}×（{32 * scale}×{256 * scale}，{n:>3} 个视觉 token）：CER {c:6.2%}  整行正确 {acc:6.1%}"
              f"  （{(time.time() - t0) / N_TEST:.1f} 秒/行）")
        if scale == 2:
            for o, t in list(zip(outs, texts))[:4]:
                print(f"      真值 {t!r:<32} 读出 {o!r}")
    t0 = time.time()
    outs = [qwen_ask(im, q, max_new=32) for im in imgs]
    c = cer(outs, texts)
    print(f"  Qwen3.5  默认（smart_resize 放大到 ≥256²：96×736，{n_qwen_tokens(32, 256)} 个 token）：CER {c:6.2%}"
          f"  整行正确 {np.mean([o == t for o, t in zip(outs, texts)]):6.1%}")
    for longest in (512, 1024):
        t0 = time.time()
        outs = [smol_ask(im, q, max_new=32, longest=longest) for im in imgs]
        c = cer(outs, texts)
        print(f"  SmolVLM  长边 {longest}（{idefics3_layout(32, 256, longest=longest)[2]} 个 token）：CER {c:6.2%}  "
              f"整行正确 {np.mean([o == t for o, t in zip(outs, texts)]):6.1%}  （{(time.time() - t0) / N_TEST:.1f} 秒/行）")
        if longest == 512:
            for o, t in list(zip(outs, texts))[:3]:
                print(f"      真值 {t!r:<32} 读出 {o!r}")
    print("  对照：第 13 章 CRNN（80 万参数、专门训练）在同一测试集上 CER 2.28%")


# ---------------------------------------------------------------------- 7
def _variants(ids, grid, image_id):
    """四种位置编号：原样的 M-RoPE、普通一维位置、图内所有 token 同一个位置、图内行列互换；再加一个对照：所有 token 位置都是 0。"""
    p3 = mrope_positions(ids, grid, image_id)
    img = torch.tensor(ids) == image_id
    swap = p3.clone()
    swap[1, img], swap[2, img] = p3[2, img], p3[1, img]
    return {"M-RoPE（原样）": p3, "一维位置": torch.arange(len(ids)).expand(3, -1).clone(),
            "图内全同位置": torch.where(img, p3[0], p3), "图内行列互换": swap,
            "全部位置 = 0（对照）": torch.zeros_like(p3)}


@torch.no_grad()
def _score(samples):
    """samples: [(图, 问题, 答案)]。每种位置编号下，答案 token 的平均损失和逐 token 正确率（teacher forcing）。"""
    tok, m = qwen()
    nll, acc = {}, {}
    for img, q, a in samples:
        inp, n_ans = qwen_inputs(img, q, answer=a, min_pixels=32 * 32)
        ids = inp["input_ids"][0].tolist()
        y = inp["input_ids"][0, -n_ans:]
        for v, pos in _variants(ids, inp["image_grid_thw"], m.config.image_token_id).items():
            logits = m(**inp, position_ids=pos.unsqueeze(1)).logits[0, -n_ans - 1:-1]
            nll.setdefault(v, []).append(torch.nn.functional.cross_entropy(logits, y).item())
            acc.setdefault(v, []).append((logits.argmax(-1) == y).float().mean().item())
    for v in nll:
        print(f"    {v:<18}答案 token 的平均损失 {np.mean(nll[v]):6.3f}   逐 token 预测正确 {np.mean(acc[v]):6.1%}")


def part7():
    section("7. 位置编号有多重要：换掉 M-RoPE，看答案的损失（teacher forcing，不重新训练）")
    n = min(30, N_TEST)
    imgs, texts = lines(3 * n)
    blocks = []
    for k in range(n):                                    # 三行叠成一块 96×256，放大 2 倍：网格 12×32 → 6×16 个 token
        im = Image.new("RGB", (256, 96), "white")
        for r in range(3):
            im.paste(imgs[3 * k + r], (0, 32 * r))
        blocks.append((im.resize((512, 192), Image.BICUBIC), "Read the text in the image. Output only the text.",
                       "\n".join(texts[3 * k:3 * k + 3])))
    print(f"  三行文字（{n} 块）：")
    _score(blocks)
    from labs.ch22.vlm_tour import test_set
    X, pairs = test_set("left")
    scenes = [(Image.fromarray(X[i].permute(1, 2, 0).numpy()).resize((256, 256), Image.NEAREST),
               "What color is the leftmost shape? Answer with one word: red, green, blue or yellow.", pairs[i][1])
              for i in range(n)]
    print(f"  最左边物体的颜色（{n} 张场景图）：")
    _score(scenes)

    print("  纯文字（同一段英文，两个语言模型各自的困惑度）：")
    text = ("The history of optical character recognition begins in the early twentieth century, when engineers "
            "built machines that could read printed characters for telegraphy and for helping blind people read. ") * 2
    tq, mq = qwen()
    ps, ms = smol()
    for name, tk, lm, head, three in [("Qwen3.5-0.8B", tq, mq.model.language_model, mq.lm_head, True),
                                      ("SmolVLM-256M", ps.tokenizer, ms.model.text_model, ms.lm_head, False)]:
        ids = torch.tensor([tk(text)["input_ids"]])
        L = ids.shape[1]
        g = torch.Generator().manual_seed(0)
        row = []
        for v, pos in [("正常", torch.arange(L)), ("全 0", torch.zeros(L, dtype=torch.long)),
                       ("随机打乱", torch.randperm(L, generator=g))]:
            pos = pos.view(1, 1, -1).expand(3, 1, -1) if three else pos.view(1, -1)
            with torch.no_grad():
                logits = head(lm(input_ids=ids, position_ids=pos).last_hidden_state)
            row.append(f"{v} {torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:]).item():.2f}")
        print(f"    {name:<14}每个 token 的损失：" + "，".join(row))


def main():
    parts = [int(a) for a in sys.argv[1:] if a.isdigit()] or [1, 2, 3, 4, 5, 6, 7]
    print(f"第 24 章：解剖真实的 VLM（CPU，{torch.get_num_threads()} 线程）")
    for p in parts:
        globals()[f"part{p}"]()


if __name__ == "__main__":
    main()
