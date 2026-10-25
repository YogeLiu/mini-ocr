"""第 27 章：回答的对数概率。逐 token 的对数概率、错位、熵、总和与平均、用似然做选择题、置信度与校准。
只用 CPU。第 1、2 部分用第 16 章训练好的字符级 GPT；第 3、4 部分用第 24 章的 Qwen3.5-0.8B（只推理）；
第 5 部分用第 26 章 SFT 之后的 VLM（需要你先在第 26 章保存 outputs/ch26/sft.pt）。

    python -m labs.ch27.logprob_tour 1 2      # 字符级 GPT：几秒
    python -m labs.ch27.logprob_tour 3 4      # Qwen3.5：CPU 上约 8 分钟（N_TEST=10 先试跑）
    python -m labs.ch27.logprob_tour 5        # 我们自己的 VLM
"""
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image

from miniocr.data.corpus import mixed_corpus, text_vocab
from miniocr.eval.calib import auroc, ece, reliability
from miniocr.metrics import cer, ned
from miniocr.models.gpt import GPT, GPTConfig
from miniocr.rl.logprobs import seq_logprob, token_entropy, token_logprobs

torch.set_num_threads(8)
OUT = "outputs/ch27"
os.makedirs(OUT, exist_ok=True)
V = text_vocab()
N_TEST = int(os.environ.get("N_TEST", 50))


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def char_gpt():
    from labs.ch22.vlm_tour import need
    ck = torch.load(need("outputs/ch16/gpt_char.pt"), map_location="cpu")
    m = GPT(GPTConfig(**ck["cfg"]))
    m.load_state_dict(ck["model"])
    return m.eval()


def show(s):
    return s.replace("\n", "⏎")


# ---------------------------------------------------------------------- 1
@torch.no_grad()
def part1_char_gpt():
    section("1. 逐 token 的对数概率与熵（第 16 章的字符级 GPT，验证集的一段文字）")
    m = char_gpt()
    text = mixed_corpus(1, 3)[:300]
    ids = torch.tensor([[V.stoi["<bos>"]] + V.encode(text)])
    logits = m(ids)[0]
    lp = token_logprobs(logits, ids)[0]
    h = token_entropy(logits)[0]
    tg = torch.cat([ids[:, 1:], torch.full((1, 1), -100)], 1)
    loss = m(ids, targets=tg)[1].item()
    print(f"  {len(text)} 个字符；−平均对数概率 {-lp.mean().item():.4f}，模型自己算的损失 {loss:.4f}（应该相同）")
    print(f"  平均每个字符 {-lp.mean().item() / np.log(2):.2f} 比特；困惑度 {np.exp(-lp.mean().item()):.2f}（字符表 {len(V)} 个）")
    first = text.split("\n")[0]
    print(f"\n  第一行 '{first}' 每个字符的对数概率（括号里是熵）：")
    row = []
    for i, ch in enumerate(first[:40]):
        row.append(f"{ch!r}:{lp[i].item():.2f}({h[i].item():.1f})")
    for k in range(0, len(row), 8):
        print("    " + "  ".join(row[k:k + 8]))
    # 按字符的上下文分类：词的第一个字母、词内、空格、标点、数字、公式（$$ 之间）
    cls = []
    in_formula = False
    for i, ch in enumerate(text):
        if text[i:i + 2] == "$$" or (i > 0 and text[i - 1:i + 1] == "$$"):
            in_formula = not in_formula if text[i:i + 2] == "$$" else in_formula
            cls.append("公式")
            continue
        prev = text[i - 1] if i else "\n"
        if in_formula:
            cls.append("公式")
        elif ch.isdigit():
            cls.append("数字")
        elif ch.isalpha():
            cls.append("词首字母" if not prev.isalpha() else "词内字母")
        elif ch in " \n":
            cls.append("空格换行")
        else:
            cls.append("标点")
    print("\n  按字符的类别：平均对数概率、平均熵（下一个字符是这一类时）")
    for c in ("词内字母", "空格换行", "标点", "公式", "词首字母", "数字"):
        sel = [i for i, x in enumerate(cls) if x == c]
        if sel:
            print(f"    {c:<6} {len(sel):4d} 个  对数概率 {lp[sel].mean().item():6.2f}（概率 {np.exp(lp[sel].mean().item()):5.1%}）"
                  f"  熵 {h[sel].mean().item():.2f}")
    hi = h.argsort(descending=True)[:5].tolist()
    print("  熵最大的 5 个位置（模型最拿不准下一个字符的地方）：")
    for i in hi:
        print(f"    前文 …'{show(text[max(0, i - 15):i])}' → 下一个是 '{show(text[i])}'  熵 {h[i].item():.2f}  对数概率 {lp[i].item():.2f}")

    print("\n  右侧补齐 8 个 <pad>：前面位置的对数概率最大差 ", end="")
    pad = torch.cat([ids, torch.full((1, 8), V.stoi["<pad>"])], 1)
    print(f"{(token_logprobs(m(pad)[0], pad)[0, :lp.shape[0]] - lp).abs().max().item():.1e}")
    mb = char_gpt().to(torch.bfloat16)
    lpb = token_logprobs(mb(ids)[0], ids)[0]
    print(f"  模型改用 bf16：每个字符的对数概率平均差 {(lpb - lp).abs().mean().item():.4f}，最大差 {(lpb - lp).abs().max().item():.3f}；"
          f"整段的总和 {lp.sum().item():.2f}（fp32） vs {lpb.sum().item():.2f}（bf16）")


# ---------------------------------------------------------------------- 2
@torch.no_grad()
def part2_offset_and_length():
    section("2. 错位一格会怎样；总和与平均（字符级 GPT）")
    m = char_gpt()
    text = mixed_corpus(1, 3)[:300]
    ids = torch.tensor([[V.stoi["<bos>"]] + V.encode(text)])
    logits = m(ids)[0]
    right = token_logprobs(logits, ids)[0]
    lp_all = torch.log_softmax(logits.float(), -1)[0]
    same = lp_all[torch.arange(ids.shape[1]), ids[0]]                     # 错：位置 t 的输出去取 ids[t]
    two = lp_all[torch.arange(ids.shape[1] - 2), ids[0, 2:]]              # 错：位置 t 的输出去取 ids[t + 2]
    print(f"  正确（位置 t 预测 ids[t + 1]）：平均对数概率 {right.mean().item():6.2f}")
    print(f"  错成 ids[t]（没错位）：        平均对数概率 {same.mean().item():6.2f}")
    print(f"  错成 ids[t + 2]（多错一位）：  平均对数概率 {two.mean().item():6.2f}")
    print(f"  均匀猜测：                     平均对数概率 {-np.log(len(V)):6.2f}")
    print("  错位不会报错，只会得到一个'看起来合理'的负数；和均匀猜测比一比，就能发现")

    print("\n  同一个开头，几种接法（回答 = 接上的字符 + 换行）：总和 vs 平均")
    prompt = "Every that fire word just thought "
    cands = ["it.", "state.", "a.", "state state state state.", "thought thought thought."]
    p_ids = [V.stoi["<bos>"]] + V.encode(prompt)
    print(f"    开头 '{prompt}'")
    rows = []
    for c in cands:
        full = torch.tensor([p_ids + V.encode(c + "\n")])
        lp = token_logprobs(m(full)[0], full)
        mask = torch.zeros_like(lp)
        mask[0, len(p_ids) - 1:] = 1
        rows.append((c, int(mask.sum()), seq_logprob(lp, mask).item(), seq_logprob(lp, mask, "mean").item()))
    for c, n, s, a in rows:
        print(f"    {c!r:<34} {n:3d} 个字符  总和 {s:7.2f}  平均 {a:6.2f}")
    print(f"    按总和最好：{max(rows, key=lambda r: r[2])[0]!r}；按平均最好：{max(rows, key=lambda r: r[3])[0]!r}")


# ---------------------------------------------------------------------- 3、4：Qwen3.5
def qwen_score_ids(img, question, ans_ids, **kw):
    """Qwen3.5 对"图 + 问题 → ans_ids（一串 token 编号）"的逐 token 对数概率。直接用编号，不经过文字。"""
    from labs.ch24.hf_tour import qwen, qwen_inputs
    tok, m = qwen()
    inp, _ = qwen_inputs(img, question, **kw)
    extra = torch.tensor([ans_ids])
    ids = torch.cat([inp["input_ids"], extra], 1)
    inp = dict(inp, input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=(ids == m.config.image_token_id).int())
    with torch.no_grad():
        logits = m(**inp).logits
    return token_logprobs(logits, ids)[0, -len(ans_ids):]


def qwen_text_ids(text):
    """文字 + <|im_end|> 的 token 编号（"重新分词"：和模型生成时用的编号不一定相同）。"""
    from labs.ch24.hf_tour import qwen
    tok, _ = qwen()
    return tok(text + "<|im_end|>")["input_ids"]


@torch.no_grad()
def qwen_generate_ids(img, question, max_new=32, **kw):
    """贪心生成，返回生成的 token 编号（到 <|im_end|> 为止，包括它）和解码后的文字。"""
    from labs.ch24.hf_tour import qwen, qwen_inputs
    tok, m = qwen()
    inp, _ = qwen_inputs(img, question, **kw)
    out = m.generate(**inp, max_new_tokens=max_new, do_sample=False)[0, inp["input_ids"].shape[1]:].tolist()
    end = tok.convert_tokens_to_ids("<|im_end|>")
    if end in out:
        out = out[:out.index(end) + 1]
    return out, tok.decode(out, skip_special_tokens=True).strip()


def part3_choice():
    section(f"3. 用似然做选择题：Qwen3.5-0.8B 在玩具任务上（每个任务 {N_TEST} 个样本）")
    from labs.ch22.vlm_tour import test_set
    from labs.ch24.hf_tour import qwen
    tok, _ = qwen()
    prompts = {"count": ("How many shapes are in the image? Answer with a single digit.", ["1", "2", "3"]),
               "left": ("What color is the leftmost shape? Answer with one word: red, green, blue or yellow.",
                        ["red", "green", "blue", "yellow"]),
               "exists": (None, ["yes", "no"])}
    gen_acc = {"count": "98%", "left": "86%", "exists": "98%"}           # 第 24 章第 5 部分：生成后比较
    confs, oks = [], []
    for task, (q0, cands) in prompts.items():
        X, pairs = test_set(task)
        t0 = time.time()
        acc = {"无空格": 0, "前面加空格": 0}
        mass = {"无空格": [], "前面加空格": []}
        for i in range(N_TEST):
            img = Image.fromarray(X[i].permute(1, 2, 0).numpy())
            q, a = pairs[i]
            q = q0 or (q + " Answer yes or no.")
            for form, pre in (("无空格", ""), ("前面加空格", " ")):
                s = np.array([qwen_score_ids(img, q, qwen_text_ids(pre + c)).sum().item() for c in cands])
                acc[form] += cands[int(s.argmax())] == a
                mass[form].append(np.exp(s).sum())                      # 所有候选的概率加起来：模型把多少概率给了这些写法
                if form == "前面加空格":
                    p = np.exp(s - s.max())
                    p /= p.sum()
                    confs.append(p.max())
                    oks.append(cands[int(p.argmax())] == a)
        toks = {c: tok.convert_ids_to_tokens(qwen_text_ids(" " + c)) for c in cands[:2]}
        print(f"  {task:<7} 候选 {cands}；例：' {cands[0]}' 的 token {toks[cands[0]]}")
        for form in acc:
            print(f"          {form:<6} 按似然选 {acc[form] / N_TEST:6.1%}   候选的概率之和（中位数）{np.median(mass[form]):7.1%}")
        print(f"          第 24 章生成后比较：{gen_acc[task]}   （{time.time() - t0:.0f} 秒）")
    print(f"\n  置信度 = 候选之间 softmax 的最大值（加空格的写法）。{len(confs)} 道题的校准：ECE {ece(confs, oks):.3f}")
    print("    置信度段      平均置信度  准确率  题数")
    for lo, hi, c, acc_, n in reliability(confs, oks):
        print(f"    [{lo:.1f}, {hi:.1f})    {c:9.1%}  {acc_:6.1%}  {n:4d}")


def part4_ocr_confidence():
    section(f"4. OCR 的置信度：Qwen3.5 读文字行（放大 2×，{N_TEST} 行），对数概率能不能发现读错的行")
    from labs.ch24.hf_tour import lines, qwen
    tok, _ = qwen()
    imgs, texts = lines(N_TEST)
    q = "Read the text in the image. Output only the text, exactly as written."
    kw = dict(scale=2, min_pixels=32 * 32)
    rows = []
    t0 = time.time()
    for im, t in zip(imgs, texts):
        ids, out = qwen_generate_ids(im, q, **kw)
        lp = qwen_score_ids(im, q, ids, **kw)                          # 生成时的编号
        lp_re = qwen_score_ids(im, q, qwen_text_ids(out), **kw)        # 把文字重新分词后的编号
        rows.append(dict(ref=t, out=out, mean=lp.mean().item(), min=lp.min().item(), sum=lp.sum().item(), n=len(ids),
                         sum_re=lp_re.sum().item(), toks=tok.convert_ids_to_tokens(ids), ned=ned(out, t), exact=out == t))
    print(f"  用时 {time.time() - t0:.0f} 秒；CER {cer([r['out'] for r in rows], texts):.2%}，整行正确 {np.mean([r['exact'] for r in rows]):.1%}")
    same = np.mean([r["toks"][:-1] == tok.convert_ids_to_tokens(qwen_text_ids(r["out"]))[:-1] for r in rows])
    print(f"  重新分词后与生成时的 token 完全相同的行：{same:.0%}；例：生成 {rows[0]['toks']}，"
          f"重新分词 {tok.convert_ids_to_tokens(qwen_text_ids(rows[0]['out']))}")
    print(f"  整行的对数概率（中位数）：用生成时的编号 {np.median([r['sum'] for r in rows]):.2f}，用重新分词的编号 "
          f"{np.median([r['sum_re'] for r in rows]):.2f}")
    bad = np.array([r["ned"] > 0.5 for r in rows])
    wrong = np.array([not r["exact"] for r in rows])
    print(f"  '严重读错'（NED > 0.5）{bad.sum()} 行，'不完全对' {wrong.sum()} 行")
    print("  用哪个分数挑出它们（AUROC：1 = 完美区分，0.5 = 瞎猜）：")
    for name, s in (("−平均对数概率", -np.array([r["mean"] for r in rows])), ("−最低的对数概率", -np.array([r["min"] for r in rows])),
                    ("−总对数概率", -np.array([r["sum"] for r in rows])), ("−总对数概率（重新分词）", -np.array([r["sum_re"] for r in rows])),
                    ("输出的 token 数（对照：只看长度）", np.array([r["n"] for r in rows], float))):
        print(f"    {name:<34} 严重读错 {auroc(s, bad):.3f}   不完全对 {auroc(s, wrong):.3f}")
    conf = [np.exp(r["sum"]) for r in rows]
    print(f"  置信度 = exp(总对数概率) = 整行的概率；对'整行正确'的 ECE {ece(conf, [r['exact'] for r in rows]):.3f}")
    print("    置信度段      平均置信度  准确率  行数")
    for lo, hi, c, acc, n in reliability(conf, [r["exact"] for r in rows]):
        print(f"    [{lo:.1f}, {hi:.1f})    {c:9.1%}  {acc:6.1%}  {n:4d}")
    print("  NED 最大的 6 行：")
    for r in sorted(rows, key=lambda r: -r["ned"])[:6]:
        print(f"    真值 {r['ref']!r:<26} 读出 {r['out']!r:<30} NED {r['ned']:.2f}  总和 {r['sum']:7.2f}  最低 {r['min']:6.2f}")
    print("  对照：完全读对的行里总对数概率最低的 3 行：")
    for r in sorted([r for r in rows if r["exact"]], key=lambda r: r["sum"])[:3]:
        print(f"    {r['ref']!r:<26} 总和 {r['sum']:7.2f}  最低 {r['min']:6.2f}  token {r['toks']}")


# ---------------------------------------------------------------------- 5
def part5_our_vlm():
    section("5. 我们自己的 VLM（第 26 章 SFT 之后）：选择题的似然与校准")
    path = "outputs/ch26/sft.pt"
    if not os.path.exists(path):
        print(f"  缺少 {path}：先跑第 26 章第 2 部分（python -m labs.ch26.sft_tour 2），它会自动保存这个模型")
        return
    from labs.ch22.vlm_tour import build, img_float
    from labs.ch23.instruct_tour import prompt, test_examples
    m = build("mlp")
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    cands = {"count": ["1", "2", "3"], "count_color": ["0", "1", "2", "3"], "left": ["red", "green", "blue", "yellow"],
             "exists": ["yes", "no"]}
    for task, cs in cands.items():
        X, qs, refs = test_examples(task, n=200)
        confs, oks, margins = [], [], []
        for i in range(len(qs)):
            p = prompt(qs[i])
            scores = []
            for c in cs:
                ids = torch.tensor([p + V.encode(c) + [V.stoi["<eos>"]]])
                with torch.no_grad():
                    logits = m(img_float(X[i:i + 1]), ids)[0]
                lp = token_logprobs(logits, ids)[0, len(p) - 1:]
                scores.append(lp.sum().item())
            s = np.array(scores)
            pr = np.exp(s - s.max())
            pr /= pr.sum()
            k = int(pr.argmax())
            confs.append(pr[k])
            oks.append(cs[k] == refs[i])
            ok_idx = cs.index(refs[i])
            margins.append(s[ok_idx] - max(np.delete(s, ok_idx)))
        print(f"  {task:<12} 按似然选 {np.mean(oks):6.1%}  ECE {ece(confs, oks):.3f}  正确答案比最强的错误答案高 "
              f"{np.median(margins):.2f}（中位数，单位 nat）")


PARTS = {1: part1_char_gpt, 2: part2_offset_and_length, 3: part3_choice, 4: part4_ocr_confidence, 5: part5_our_vlm}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS)
    for k in want:
        PARTS[k]()
    if want == list(PARTS):
        print("\n第 27 章的演示全部完成。")
