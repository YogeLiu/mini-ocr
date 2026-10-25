"""第 17 章：生成与 KV cache。用第 16 章预训练好的 outputs/ch16/gpt_char.pt，只用 CPU。

    python -m labs.ch17.gen_tour             # 全部 6 个部分
    python -m labs.ch17.gen_tour 3           # 只运行第 3 部分
"""
import math
import os
import sys
import time
from collections import Counter

if "--gpu" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from miniocr.data.corpus import text_vocab
from miniocr.data.text import WORDS
from miniocr.generate import beam_search, generate, process_logits
from miniocr.models.gpt import GPT, GPTConfig

torch.set_num_threads(min(16, os.cpu_count() or 1))
V = text_vocab()
BANNED = list(range(len(V.specials)))                 # 不生成特殊符号
WORDSET = {w.lower() for w in WORDS}
_M = {}


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def load():
    if "m" not in _M:
        path = "outputs/ch16/gpt_char.pt"
        if not os.path.exists(path):
            sys.exit("先运行 python -m labs.ch16.gpt_tour 3 预训练模型")
        ck = torch.load(path, map_location="cpu")
        m = GPT(GPTConfig(**ck["cfg"]))
        m.load_state_dict(ck["model"]); m.eval()
        _M["m"] = m
    return _M["m"]


def enc(s):
    return torch.tensor([V.encode(s)])


def word_ok_rate(text):
    """生成文字里的英文单词有多少比例是真实单词（语料的单词表里有）。粗略的"质量"指标。"""
    ws = [w.strip(".,!?:;()'$%").lower() for w in text.split()]
    ws = [w for w in ws if w.isalpha()]
    return sum(w in WORDSET for w in ws) / max(1, len(ws)), len(ws)


def distinct(text, n=4):
    """不同的 n 字符片段占全部 n 字符片段的比例：越低越重复。"""
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


# ---------------------------------------------------------------------- 1
def part1_distribution():
    section("1. 下一个字符的分布：温度、top-k、top-p 怎样改变它")
    m = load()
    prompt = "The cat "
    with torch.no_grad():
        logits = m(enc(prompt))[0][:, -1].float()
    logits[:, BANNED] = float("-inf")
    p = logits.softmax(-1)[0]
    top = p.topk(8)
    print(f"  提示 {prompt!r}，模型给出的下一个字符（前 8 个）：" +
          "  ".join(f"{V.itos[i]!r}:{v:.3f}" for v, i in zip(top.values.tolist(), top.indices.tolist())))
    ent = -(p * p.clamp_min(1e-30).log()).sum()
    print(f"  熵 {ent:.3f} nat（相当于在 e^熵 = {ent.exp():.1f} 个字符里均匀地选）")
    print(f"  {'设置':22s} {'最大概率':>8s} {'熵':>6s} {'可选字符数':>10s}")
    for name, kw in (("温度 0.5", dict(temperature=0.5)), ("温度 1.0（原分布）", dict()), ("温度 1.5", dict(temperature=1.5)),
                     ("top-k 5", dict(top_k=5)), ("top-p 0.9", dict(top_p=0.9)), ("top-p 0.5", dict(top_p=0.5))):
        q = process_logits(logits, **kw).softmax(-1)[0]
        e = -(q * q.clamp_min(1e-30).log()).sum()
        print(f"  {name:22s} {q.max():8.3f} {e:6.3f} {int((q > 0).sum()):10d}")
    print("  温度 T：p ∝ exp(logit / T)。T < 1 让大的更大（更确定），T > 1 让分布更平；T → 0 就是贪心")
    print("  top-k 只留最大的 k 个；top-p 留累计概率达到 p 的最少几个——分布尖时留得少，平时留得多")


# ---------------------------------------------------------------------- 2
def part2_tradeoff():
    section("2. 多样性与质量的权衡：同一个提示生成 20 次")
    m = load()
    prompt = "The "
    print(f"  {'设置':16s} {'真实单词比例':>12s} {'不同的 4 字符片段':>16s} {'20 次里不同的开头':>16s}")
    for name, kw in (("贪心", dict(greedy=True)), ("温度 0.5", dict(temperature=0.5)), ("温度 0.8", dict(temperature=0.8)),
                     ("温度 1.0", dict()), ("温度 1.5", dict(temperature=1.5)), ("top-k 10", dict(top_k=10)),
                     ("top-p 0.9", dict(top_p=0.9))):
        g = torch.Generator().manual_seed(0)
        outs = [V.decode(generate(m, enc(prompt), 120, generator=g, banned=BANNED, **kw)[0].tolist()) for _ in range(20)]
        ok = sum(word_ok_rate(o)[0] for o in outs) / 20
        dist = sum(distinct(o) for o in outs) / 20
        starts = len({o[:16] for o in outs})
        print(f"  {name:16s} {ok:12.1%} {dist:16.1%} {starts:16d}")
        _M[name] = outs
    for name in ("贪心", "温度 0.5", "温度 1.0", "温度 1.5"):
        print(f"  {name}：{_M[name][0][:110]!r}")
    print("  贪心每次都一样（20 次只有 1 种）；温度越高越多样，但错字越多")


# ---------------------------------------------------------------------- 3
def part3_repetition():
    section("3. 贪心会陷入循环；重复惩罚")
    m = load()
    for prompt in ("- ", "| Name | Price |\n|---|---|\n"):
        for name, kw in (("贪心", dict(greedy=True)), ("贪心 + 重复惩罚 1.3", dict(greedy=True, repetition_penalty=1.3))):
            g = torch.Generator().manual_seed(0)
            out = V.decode(generate(m, enc(prompt), 200, generator=g, banned=BANNED, **kw)[0].tolist())[len(prompt):]
            lines = [l for l in out.split("\n") if l]
            rep = 1 - len(set(lines)) / max(1, len(lines))
            print(f"  提示 {prompt!r:28s} {name:20s} 重复的行 {rep:5.0%}  不同的 4 字符片段 {distinct(out):5.1%}")
            print("    " + repr(out[:150]))
    print("  贪心只看当前最可能的字符；一旦生成了一段在训练语料里常见的文字，下一次遇到同样的上下文就会再写一遍，")
    print("  形成循环。重复惩罚把已经出现过的字符的 logit 压低（> 0 的除以 1.3）。")
    print("  注意它对字符级模型很粗暴：空格、e 这些字符本来就该反复出现。真实模型的惩罚一般只有 1.05 左右（Qwen-VL 的默认值）")


# ---------------------------------------------------------------------- 4
def part4_cache():
    section("4. KV cache：结果完全一样，速度快很多")
    m = load()
    cfg = m.cfg
    prompt = enc("# Report\n")
    a = generate(m, prompt, 150, greedy=True, use_cache=True)
    b = generate(m, prompt, 150, greedy=True, use_cache=False)
    print(f"  贪心生成 150 个字符：用 cache 和不用 cache 的结果{'完全相同' if torch.equal(a, b) else '不同！'}")
    with torch.no_grad():
        full = m(a[:, :-1])[0]
        caches = [dict() for _ in m.blocks]
        step = [m(a[:, :prompt.shape[1]], caches=caches)[0]]
        for t in range(prompt.shape[1], a.shape[1] - 1):
            step.append(m(a[:, t:t + 1], caches=caches, pos0=t)[0])
        diff = (torch.cat(step, 1) - full).abs().max().item()
    print(f"  逐个位置用 cache 算出的 logits 与一次算完整序列的 logits：最大差 {diff:.1e}（浮点误差）")
    print(f"  {'生成长度':>8s} {'不用 cache（秒）':>16s} {'用 cache（秒）':>14s} {'加速':>6s}")
    for n in (100, 200, 400):
        t = time.perf_counter(); generate(m, prompt, n, greedy=True, use_cache=False); t0 = time.perf_counter() - t
        t = time.perf_counter(); generate(m, prompt, n, greedy=True, use_cache=True); t1 = time.perf_counter() - t
        print(f"  {n:8d} {t0:16.2f} {t1:14.2f} {t0 / t1:6.1f}×")
    print("  不用 cache：第 t 步把 t 个 token 全部重算，总计算量 ∝ 1 + 2 + … + n ≈ n²/2；用 cache：每步只算 1 个新 token")
    L, d = cfg.n_layer, cfg.d_model
    per_tok = 2 * L * d * 4
    print(f"  cache 的大小：每个 token 每层存 k、v 各 d 维 → 2 × {L} 层 × {d} × 4 字节 = {per_tok:,} 字节/token；"
          f"256 个 token 共 {per_tok * 256 / 1024:.0f} KB")
    q = dict(L=28, kv=8, dh=128)
    pt = 2 * q["L"] * q["kv"] * q["dh"] * 2
    print(f"  Qwen3-0.6B（bf16，GQA 8 组键值头 × 128 维）：2 × 28 × 8 × 128 × 2 = {pt:,} 字节/token；"
          f"一个 8000 token 的文档 {pt * 8000 / 2**20:.0f} MB；如果不用 GQA（16 组）要翻倍")


# ---------------------------------------------------------------------- 5
def part5_prefill():
    section("5. 预填充与解码：两个阶段，两种瓶颈")
    m = load()
    text = (open(__file__).read() * 2)[:250]
    ids = torch.tensor([[V.stoi.get(c, V.stoi[" "]) for c in text]])
    with torch.no_grad():
        m(ids[:, :8])                                              # 预热
        t = time.perf_counter()
        for _ in range(5):
            m(ids, caches=[dict() for _ in m.blocks])
        t_pre = (time.perf_counter() - t) / 5
        caches = [dict() for _ in m.blocks]
        m(ids[:, :1], caches=caches)
        t = time.perf_counter()
        for i in range(1, ids.shape[1]):
            m(ids[:, i:i + 1], caches=caches, pos0=i)
        t_dec = time.perf_counter() - t
    n = ids.shape[1]
    print(f"  一次送入 {n} 个 token（预填充）：{t_pre * 1e3:.1f} ms，{n / t_pre:,.0f} token/秒")
    print(f"  一个一个送入同样 {n} 个 token（解码）：{t_dec * 1e3:.1f} ms，{n / t_dec:,.0f} token/秒（慢 {t_dec / t_pre:.0f} 倍）")
    print("  预填充时 T 个 token 一起做矩阵乘法，算力被充分利用；解码时每步只有 1 个 token，")
    print("  大部分时间花在把权重从内存读进来上（每生成一个 token 都要把全部参数读一遍）。")
    print("  所以 VLM 处理一张图（几千个视觉 token 的预填充）往往比写出几百个字的答案（逐个解码）还快。")
    print("  服务端把多个请求拼成一批一起解码（批处理），就是为了让每次读权重能服务更多 token")


# ---------------------------------------------------------------------- 6
def part6_beam():
    section("6. 束搜索：概率更高，不一定更好")
    m = load()

    def logp(seq, T0):
        with torch.no_grad():
            lp = torch.log_softmax(m(seq[:, :-1])[0].float(), -1)
        return lp[0, torch.arange(T0 - 1, seq.shape[1] - 1), seq[0, T0:]].sum().item()

    for prompt in ("$$\\frac{", "The ", "| Name | "):
        x = enc(prompt)
        g = generate(m, x, 40, greedy=True, banned=BANNED)
        rows = [("贪心", g)]
        for k in (4, 16):
            rows.append((f"束搜索 {k}", beam_search(m, x, 40, beam=k)))
        print(f"  提示 {prompt!r}：")
        for name, s in rows:
            print(f"    {name:10s} 累计 log 概率 {logp(s, x.shape[1]):8.2f}  {V.decode(s[0].tolist())[len(prompt):]!r}")
    print("  束搜索找到的序列总概率更高，但开放式写作时更容易重复、偏向最\"保险\"的内容。")
    print("  OCR 不一样：答案是唯一的（图上写了什么），概率最高的答案通常就是对的。")
    print("  实际中 VLM 做 OCR 大多用贪心（快、确定），加一点重复惩罚防止循环；第 25 章会比较解码方式对 CER 的影响")


PARTS = {1: part1_distribution, 2: part2_tradeoff, 3: part3_repetition, 4: part4_cache, 5: part5_prefill, 6: part6_beam}

if __name__ == "__main__":
    for k in [int(a) for a in sys.argv[1:] if a.isdigit()] or list(PARTS):
        PARTS[k]()
    print("\n第 17 章的演示全部完成。")
