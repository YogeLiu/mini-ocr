"""（维护者用）生成 A2 测试的快照：python -m assignments.solutions.a2_sequences.make_fixtures

快照里的"期望输出"来自参考实现；生成前先和书里的实现（miniocr，已与 PyTorch 对拍过）交叉核对，不一致就报错。
"""
import os

import torch

from miniocr.data.corpus import mixed_corpus
from miniocr.models.gpt import GPT as BookGPT
from miniocr.models.gpt import GPTConfig as BookCfg
from miniocr.tokenize import bpe_train

from .a2.model import GPT, GPTConfig
from .a2.tokenizer import Tokenizer, train_bpe

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "a2_sequences", "tests", "fixtures")
TOY = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
SPECIAL = "<|end|>"


def main():
    os.makedirs(OUT, exist_ok=True)
    corpus = mixed_corpus(7, 300)
    _, toy_merges = train_bpe(TOY, 96 + 6)
    vocab, merges = train_bpe(corpus, 96 + 1 + 300, [SPECIAL])
    assert merges == bpe_train(corpus, 300), "与第 14 章的 bpe_train 不一致"
    assert toy_merges == bpe_train(TOY, 6)
    tok = Tokenizer(vocab, merges, [SPECIAL])
    sample = corpus[:3000] + SPECIAL + corpus[3000:4000] + SPECIAL + SPECIAL + "\n"
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample
    torch.save({"toy": TOY, "toy_merges": toy_merges, "corpus": corpus, "vocab_size": 96 + 1 + 300,
                "special": SPECIAL, "vocab": vocab, "merges": merges, "sample": sample, "sample_ids": ids},
               os.path.join(OUT, "bpe.pt"))

    cfg = dict(vocab_size=64, block_size=32, n_layer=2, n_head=2, d_model=32, d_ff=0, rope_base=10000.0, tie_weights=True)
    torch.manual_seed(0)
    model = GPT(GPTConfig(**cfg)).eval()
    with torch.no_grad():                                      # 初始化太小，各层都接近恒等；换成更大的随机值，测试才有区分度
        for n, p in model.named_parameters():
            p.copy_(torch.randn_like(p) * (0.3 if p.dim() > 1 else 0.2) + (1.0 if n.endswith("n1.weight") or n.endswith("n2.weight") or n == "norm.weight" else 0.0))
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    idx = torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(1))
    logits = model(idx)[0].detach()
    book = BookGPT(BookCfg(vocab_size=64, block_size=32, n_layer=2, n_head=2, d_model=32)).eval()
    book.load_state_dict(sd)
    torch.testing.assert_close(book(idx)[0], logits, atol=1e-5, rtol=1e-4)
    torch.save({"cfg": cfg, "state_dict": sd, "idx": idx, "logits": logits}, os.path.join(OUT, "gpt.pt"))
    print(f"写入 {OUT}：bpe.pt（{len(merges)} 条合并），gpt.pt（{sum(v.numel() for k, v in sd.items() if k != 'head.weight'):,} 个参数）")


if __name__ == "__main__":
    main()
