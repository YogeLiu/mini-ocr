"""第 15 章：缓存的形状、故障、数学预算与计时；不加载权重、不联网。"""
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
import time
from types import MethodType

import torch
import torch.nn.functional as F

from .device import synchronize
from .models.gpt import apply_rope, repeat_kv
from .nn.attention import attention


# >>> 练习 15.2：分段预填充的 mask
def cached_mask(T_new, T_past, device=None):
    """新块全看见旧缓存，块内因果；返回 (T_new, T_past+T_new) 的 bool 张量。"""
    if T_new < 1 or T_past < 0:
        raise ValueError("T_new 须为正，T_past 须非负")
    rows = T_past + torch.arange(T_new, device=device)[:, None]
    return torch.arange(T_past + T_new, device=device)[None, :] <= rows
# <<<


# >>> 练习 15.3：按切法预填充
@torch.no_grad()
def prefill_chunks(model, idx, sizes):
    """按正整数 sizes 完整切分提示；pos0 为已处理长度，返回全部 logits、逐层 caches。"""
    sizes = list(sizes)
    if not sizes or any(not isinstance(n, int) or n < 1 for n in sizes) or sum(sizes) != idx.shape[1]:
        raise ValueError("sizes 必须是正整数，且总和等于提示长度")
    caches, pieces, pos = model.new_caches(), [], 0
    for size in sizes:
        pieces.append(model(idx[:, pos:pos+size], caches=caches, pos0=pos)[0])
        pos += size
    return torch.cat(pieces, 1), caches
# <<<


# >>> 练习 15.4：显存公式与对拍
def cache_nbytes(caches):
    """逐层 k、v 的逻辑字节数（numel × element_size），不计容器及分配器开销。"""
    return sum(c[key].numel() * c[key].element_size()
               for c in caches for key in ("k", "v") if key in c)


def kv_cache_bytes(n_layer, kv_heads, head_dim, T, B=1, bytes_per=2):
    """数学结论：K/V 各存一份，共 2·L·h_kv·dh·T·B·bytes 字节。"""
    return 2 * n_layer * kv_heads * head_dim * T * B * bytes_per


def _head_dim(cfg):
    return cfg.head_dim if isinstance(cfg, LocalModelConfig) else cfg.d_model // cfg.n_head


def kv_cache_bytes_for(cfg, T, B=1, bytes_per=2):
    """从 GPTConfig 或本地骨架配置计算 KV 字节；混合模型仅计 full_attention 层。"""
    layers = cfg.full_layers if isinstance(cfg, LocalModelConfig) else cfg.n_layer
    return kv_cache_bytes(layers, cfg.kv_heads(), _head_dim(cfg), T, B, bytes_per)
# <<<


BUGS = ("mask_offset", "pre_rope_k", "repeat_before_cache", "no_cache_update")


def _bug_forward(self, x, cos, sin, cache, bug):
    # 教学用副本：只在带缓存的 eval 前向中改变一处语义。
    B, T, _ = x.shape
    q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
    k = self.k(x).view(B, T, self.hk, self.dh).transpose(1, 2)
    v = self.v(x).view(B, T, self.hk, self.dh).transpose(1, 2)
    if self.q_norm is not None:
        q, k = self.q_norm(q), self.k_norm(k)
    raw_k = k
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    if bug == "repeat_before_cache":
        k, v = repeat_kv(k, self.h // self.hk), repeat_kv(v, self.h // self.hk)
    past = cache["k"].shape[2] if "k" in cache else 0
    stored_k = raw_k if bug == "pre_rope_k" else k
    if past:
        stored_k = torch.cat([cache["k"], stored_k], 2)
        k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
    if bug != "no_cache_update" or not past:
        cache["k"], cache["v"] = stored_k, v
    if bug != "repeat_before_cache":
        k, v = repeat_kv(k, self.h // self.hk), repeat_kv(v, self.h // self.hk)
    mask = cached_mask(T, past, x.device)
    if bug == "mask_offset" and past and T > 1:
        mask = torch.ones_like(mask).tril()  # 单 token 路径全可见；仅多 token 路径忘了偏移。
    if self.impl == "manual":
        y = attention(q, k, v, mask)[0]
    else:
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return self.o(y.transpose(1, 2).reshape(B, T, self.h * self.dh))


@contextmanager
def inject_bug(model, bug):
    """临时替换各层 attention.forward；仅供 eval 实验，异常退出也恢复，支持嵌套。

    mask_offset：多 token 且有历史时误用 tril()，单 token 路径仍全可见。
    pre_rope_k：过早保存 norm 后、RoPE 前的 k；本块用旋转后的 k，读旧缓存时忘了旋转。
    repeat_before_cache：先复制头再缓存，结果正确但 GQA 缓存膨胀。
    no_cache_update：只写第一次预填充，之后拼接计算却忘了写回。
    with 内创建的错误缓存不可在退出后继续使用；权重与原始模块不变。
    """
    if bug not in BUGS:
        raise ValueError(f"未知故障：{bug}")
    if model.training:
        raise ValueError("故障注入仅用于 model.eval()")
    originals = []
    try:
        for block in model.blocks:
            attn = block.attn
            originals.append((attn, "forward" in attn.__dict__, attn.forward))

            def forward(self, x, cos, sin, cache=None, original=attn.forward):
                if cache is None:
                    return original(x, cos, sin, cache)
                return _bug_forward(self, x, cos, sin, cache, bug)

            attn.forward = MethodType(forward, attn)
        yield model
    finally:
        for attn, had_override, original in originals:
            if had_override:
                attn.forward = original
            else:
                del attn.forward


@dataclass(frozen=True)
class LocalModelConfig:
    """只用于数学预算的本地骨架；显式头维，不将 Qwen 的 h·dh 强制设为 hidden_size。"""
    name: str
    n_layer: int
    full_layers: int
    n_head: int
    n_kv_head: int
    head_dim: int
    d_model: int
    d_ff: int
    vocab_size: int
    source: str
    note: str = ""

    def kv_heads(self):
        return self.n_kv_head

    def ff_dim(self):
        return self.d_ff


def local_model_configs(hub=None):
    """只 glob/读取本地 config.json，绝不下载；缺文件时用内置待验证值。

    多 snapshot 时选路径字典序最后一个，并返回确切来源；坏 JSON/缺字段直接报错。
    Qwen3 直接读取 Qwen3-0.6B 配置；Qwen3.5 的 linear_attention 固定状态不计入 KV。
    回退的 hidden_size/intermediate_size/vocab_size 同样是待验证骨架假设。
    """
    hub = Path(hub) if hub is not None else Path.home()/".cache/huggingface/hub"
    specs = [
        ("Qwen3-0.6B", "Qwen3-0.6B", dict(num_hidden_layers=28,
         num_attention_heads=16, num_key_value_heads=8, head_dim=128, hidden_size=1024,
         intermediate_size=3072, vocab_size=151936)),
        ("Qwen3.5-0.8B", "Qwen3.5-0.8B", dict(num_hidden_layers=24,
         num_attention_heads=8, num_key_value_heads=2, head_dim=256, hidden_size=1024,
         intermediate_size=3584, vocab_size=248320,
         layer_types=["linear_attention"]*18 + ["full_attention"]*6)),
    ]
    result = []
    for name, repo, fallback in specs:
        paths = sorted((hub/f"models--Qwen--{repo}"/"snapshots").glob("*/config.json"))
        source, data = "内置数值（待验证）", fallback
        if paths:
            data = json.loads(paths[-1].read_text(encoding="utf-8"))
            data = data.get("text_config", data)
            source = f"本地 config {paths[-1]}"
        layers = data["num_hidden_layers"]
        full = data["layer_types"].count("full_attention") if "layer_types" in data else layers
        note = (("Qwen3-0.6B 本地配置" if paths else "Qwen3-0.6B 内置回退配置（待验证）")
                if name == "Qwen3-0.6B" else "仅计 full_attention KV；不含 linear_attention 固定状态")
        result.append(LocalModelConfig(name, layers, full, data["num_attention_heads"],
                      data["num_key_value_heads"], data["head_dim"], data["hidden_size"],
                      data["intermediate_size"], data["vocab_size"], source, note))
    return result


# >>> 练习 15.5：计算量与访存粗算
def step_cost(cfg, T_new, T_past, B=1, bytes_per=2):
    """数学粗算：仅矩阵乘，乘加=2 FLOPs，线性权重每步读一次（含输出层，嵌入查表不算）。

    N = L·(2d·h·dh + 2d·h_kv·dh + 3d·d_ff) + d·vocab；不含 norm 参数。
    线性 FLOPs=2NBT_new；QKᵀ+AV=4L(h·dh)BT_new(T_past+T_new)。
    GPT 的 h·dh=d；真实 Qwen 显式头维可能使 h·dh≠d。按整个矩形计算注意力，
    包含因果遮蔽位置，是上界近似。忽略 softmax、norm、RoPE、激活读写等。
    字节=权重 N·bytes + KV(历史与新块各读一次、新块写一次)。假设先写新 KV，
    再为注意力读取整个可见 KV；不假设新块一直驻留在寄存器/片上缓存。
    忽略 torch.cat 重分配/复制、硬件多级缓存与权重驻留；不是实际 DRAM 流量预测。
    混合线性注意力不适用此 FLOPs 公式，显式拒绝以免把全部层当成 full attention。
    """
    if T_new < 1 or T_past < 0 or B < 1 or bytes_per < 1:
        raise ValueError("长度/批量/字节数非法")
    if isinstance(cfg, LocalModelConfig) and cfg.full_layers != cfg.n_layer:
        raise ValueError("step_cost 不支持混合线性注意力")
    d, L, dh = cfg.d_model, cfg.n_layer, _head_dim(cfg)
    N = L*(2*d*cfg.n_head*dh + 2*d*cfg.kv_heads()*dh + 3*d*cfg.ff_dim()) + d*cfg.vocab_size
    linear = 2*N*B*T_new
    attn = 4*L*cfg.n_head*dh*B*T_new*(T_past+T_new)
    weights = N*bytes_per
    kv_read = kv_cache_bytes_for(cfg, T_past+T_new, B, bytes_per)
    kv_write = kv_cache_bytes_for(cfg, T_new, B, bytes_per)
    total_bytes = weights + kv_read + kv_write
    return dict(matmul_params=N, linear_flops=linear, attention_flops=attn,
                flops=linear+attn, weight_bytes=weights, kv_read_bytes=kv_read,
                kv_write_bytes=kv_write, bytes=total_bytes, intensity=(linear+attn)/total_bytes)
# <<<


def intensity_sweep(cfg, lengths=(1, 16, 128, 512, 2048), pasts=(512, 4096),
                    batches=(1, 8, 32, 128), bytes_per=2):
    """返回预填充长度扫描与解码批量扫描，单位 FLOPs/byte。"""
    rows = [dict(phase="预填充", T_new=T, T_past=0, B=1,
                 **step_cost(cfg, T, 0, bytes_per=bytes_per)) for T in lengths]
    rows += [dict(phase="解码", T_new=1, T_past=T, B=B,
                  **step_cost(cfg, 1, T, B, bytes_per)) for T in pasts for B in batches]
    return rows


@torch.no_grad()
def time_decode(model, prompt_len, n_new=1, use_cache=True, repeats=7):
    """预填充 prompt_len 个 token 后，计 n_new 次单 token 前向+argmax 的中位 ms/token。

    每次重复重建缓存，预填充不计时；先预热一次。首个待处理 token 固定为 0，
    后续用贪心 token；无缓存时重算相同前缀。长度上限约束 prompt_len+n_new。
    包含 Python 调度与动态 cat，计时前后同步；不包含 EOS、分词器和网络。
    """
    if min(prompt_len, n_new, repeats) < 1 or prompt_len+n_new > model.cfg.block_size:
        raise ValueError("计时长度/重复次数非法或超出 block_size")
    model.eval()
    device = next(model.parameters()).device
    prompt = torch.zeros((1, prompt_len), dtype=torch.long, device=device)
    times = []
    for run in range(repeats+1):
        caches = model.new_caches() if use_cache else None
        if use_cache:
            model(prompt, caches=caches)
        out = torch.cat([prompt, prompt[:, :1]], 1)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(n_new):
            logits = (model(out[:, -1:], caches=caches, pos0=out.shape[1]-1)[0]
                      if use_cache else model(out)[0])
            out = torch.cat([out, logits[:, -1].argmax(-1, keepdim=True)], 1)
        synchronize(device)
        if run:
            times.append((time.perf_counter()-start)*1000/n_new)
    return median(times)


@torch.no_grad()
def time_prefill(model, prompt_len, repeats=7):
    """含缓存写入的预填充+首 token argmax 中位毫秒；一次预热、每次同步，近似 TTFT。"""
    if min(prompt_len, repeats) < 1 or prompt_len > model.cfg.block_size:
        raise ValueError("计时长度/重复次数非法或超出 block_size")
    model.eval()
    device = next(model.parameters()).device
    idx = torch.zeros((1, prompt_len), dtype=torch.long, device=device)
    times = []
    for run in range(repeats+1):
        caches = model.new_caches()
        synchronize(device)
        start = time.perf_counter()
        model(idx, caches=caches)[0][:, -1].argmax(-1)
        synchronize(device)
        if run:
            times.append((time.perf_counter()-start)*1000)
    return median(times)
