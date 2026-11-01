"""独立复数 RoPE + PyTorch SDPA 的测试 oracle。"""
import torch
import torch.nn.functional as F

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

