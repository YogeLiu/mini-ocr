"""（维护者用）生成 A2 测试的快照：python -m assignments.solutions.a2.make_fixtures

期望输出来自参考实现 miniocr。生成前做两项交叉核对，不一致就报错：
  - BPE：增量版 train_bpe 与朴素版 train_bpe_naive 逐条相同；
  - GPT：miniocr 的 logits 与本文件里一份独立写法（复数形式的 RoPE + F.scaled_dot_product_attention）相同。
语料由 SAMPLE_FORMULAS（课程自写公式）随机拼成，不含任何外部数据集的文本，快照可以随课程分发。
"""
import os
import random

import torch
import torch.nn.functional as F

from miniocr.data.latex_corpus import SAMPLE_FORMULAS
from miniocr.models.gpt import GPT, GPTConfig
from miniocr.tokenizer import BPETokenizer, train_bpe, train_bpe_naive

from .adapters import run_train_bpe

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "a2", "tests", "fixtures")
TOY = ["low low low low low lower lower newest newest newest newest newest newest widest widest widest"]
SPECIAL = "<|endoftext|>"


def corpus(n_lines=400, seed=0):
    rng = random.Random(seed)
    sents = SAMPLE_FORMULAS
    lines = []
    for _ in range(n_lines):
        k = rng.randint(1, 3)
        line = " ".join(rng.choice(sents) for _ in range(k))
        if rng.random() < 0.3:
            line += f" {rng.randint(0, 2030)}"
        lines.append(line)
    return lines


def rope_complex(x, base, pos0):
    dh = x.shape[-1]
    half = dh // 2
    z = torch.complex(x[..., :half].double(), x[..., half:].double())
    theta = base ** (-torch.arange(half, dtype=torch.float64) * 2 / dh)
    t = torch.arange(pos0, pos0 + x.shape[-2], dtype=torch.float64)
    z = z * torch.polar(torch.ones(len(t), half, dtype=torch.float64), t[:, None] * theta[None])
    return torch.cat([z.real, z.imag], -1).to(x.dtype)


def independent_forward(sd, cfg, idx):
    """不经过 miniocr 的另一种写法，只用来核对快照。"""
    L, h, d = cfg["n_layer"], cfg["n_head"], cfg["d_model"]
    rms = lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    x = sd["tok.weight"][idx]
    B, T, _ = x.shape
    for i in range(L):
        p = f"blocks.{i}."
        a = rms(x, sd[p + "n1.weight"])
        q, k, v = (a @ sd[p + f"attn.{n}.weight"].T for n in "qkv")
        q, k, v = (t.view(B, T, h, d // h).transpose(1, 2) for t in (q, k, v))
        q, k = rope_complex(q, cfg["rope_base"], 0), rope_complex(k, cfg["rope_base"], 0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, d)
        x = x + y @ sd[p + "attn.o.weight"].T
        m = rms(x, sd[p + "n2.weight"])
        x = x + (F.silu(m @ sd[p + "mlp.gate.weight"].T) * (m @ sd[p + "mlp.up.weight"].T)) @ sd[p + "mlp.down.weight"].T
    return rms(x, sd["norm.weight"]) @ sd["head.weight"].T


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = corpus()
    toy_vocab, toy_merges = run_train_bpe(TOY, 6, [])
    assert toy_merges == train_bpe_naive(TOY, 6)
    n_merges = 80
    train_text = [l + (SPECIAL if i % 7 == 0 else "") for i, l in enumerate(lines)]   # 一部分行带特殊符号：训练时必须切掉
    vocab, merges = run_train_bpe(train_text, n_merges, [SPECIAL])
    assert merges == train_bpe_naive(lines, n_merges), "增量版与朴素版不一致"
    assert len(merges) == n_merges, f"语料只能学出 {len(merges)} 条合并"
    tok = BPETokenizer(merges, [SPECIAL])
    sample = "\n".join(lines[:60]) + SPECIAL + "龘靐齉 emoji 🙂" + SPECIAL + SPECIAL + "\n"
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample
    torch.save({"toy": TOY, "toy_merges": toy_merges, "lines": lines, "train_text": train_text, "n_merges": n_merges,
                "special": SPECIAL, "vocab": vocab, "merges": merges, "sample": sample, "sample_ids": ids},
               os.path.join(OUT, "bpe.pt"))

    cfg = dict(vocab_size=64, block_size=32, n_layer=2, n_head=2, d_model=32, d_ff=96, rope_base=10000.0, tie_weights=True)
    torch.manual_seed(0)
    model = GPT(GPTConfig(**{k: v for k, v in cfg.items()}, attn_impl="manual")).eval()
    with torch.no_grad():                    # 默认初始化太小，各层都接近恒等；换成更大的随机值，测试才有区分度
        for n, p in model.named_parameters():
            is_norm = n.endswith(("n1.weight", "n2.weight")) or n == "norm.weight"
            p.copy_(torch.randn_like(p) * (0.3 if p.dim() > 1 else 0.2) + (1.0 if is_norm else 0.0))
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    idx = torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        logits = model(idx)[0]
        torch.testing.assert_close(independent_forward(sd, cfg, idx), logits, atol=1e-5, rtol=1e-4)
    torch.save({"cfg": cfg, "state_dict": sd, "idx": idx, "logits": logits, "n_params": model.n_params()},
               os.path.join(OUT, "gpt.pt"))
    print(f"写入 {OUT}：bpe.pt（{len(merges)} 条合并，样例 {len(ids)} 个 token），gpt.pt（{model.n_params():,} 个参数）")


if __name__ == "__main__":
    main()
