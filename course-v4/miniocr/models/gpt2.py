"""第 17 章：从零实现 GPT-2；仅本地读取权重，不依赖 transformers 建模代码。"""
from dataclasses import dataclass, fields
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPT2Config:
    """GPT-2 small 的骨架；省略 dropout，适用于 eval 权重对拍。"""
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    layer_norm_epsilon: float = 1e-5
    activation: str = "gelu_new"

    def __post_init__(self):
        if min(self.vocab_size, self.n_positions, self.n_embd, self.n_layer, self.n_head) < 1:
            raise ValueError("词表、位置、宽度、层数和头数必须为正")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd 必须能被 n_head 整除")
        if self.activation not in ("gelu_new", "gelu"):
            raise ValueError("activation 只支持 gelu_new / gelu")

    @classmethod
    def from_hf_dict(cls, d):
        """从 HF config 字典构造；不支持改变注意力语义或 MLP 宽度的变体。"""
        if d.get("n_inner") not in (None, 4 * d.get("n_embd", 768)):
            raise ValueError("本章只支持 n_inner=4*n_embd")
        for key, default in (("scale_attn_weights", True), ("scale_attn_by_inverse_layer_idx", False),
                             ("reorder_and_upcast_attn", False), ("add_cross_attention", False),
                             ("tie_word_embeddings", True)):
            if d.get(key, default) != default:
                raise ValueError(f"本章不支持 {key}={d[key]}")
        values = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        values["activation"] = d.get("activation_function", d.get("activation", "gelu_new"))
        return cls(**values)


def gelu_new(x):
    """GPT-2 的 tanh 近似 GELU；输入与输出同形状，显式保留运算顺序。"""
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))


class GPT2Attention(nn.Module):
    """带偏置的 MHA，q/k/v 各为独立 Linear(d,d)。"""
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.n_embd // cfg.n_head
        self.q = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.k = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.v = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    # >>> 练习 17.3：因果注意力的分头与合头
    def forward(self, x):
        """x (B,T,d) → q/k/v (B,h,T,dh) → 因果注意力 → (B,T,d)。"""
        B, T, d = x.shape
        q, k, v = [layer(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                   for layer in (self.q, self.k, self.v)]
        scores = (q @ k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        y = scores.softmax(dim=-1) @ v
        return self.proj(y.transpose(1, 2).reshape(B, T, d))
    # <<<


class GPT2MLP(nn.Module):
    """带偏置的 d→4d→d 前馈层。"""
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.activation = cfg.activation

    def forward(self, x):
        """输入输出均为 (B,T,d)，中间激活为 (B,T,4d)。"""
        x = self.fc(x)
        return self.proj(gelu_new(x) if self.activation == "gelu_new" else F.gelu(x))


class GPT2Block(nn.Module):
    """两个含 γ/β 的 LayerNorm，以及注意力和 MLP 残差支路。"""
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.attn = GPT2Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.mlp = GPT2MLP(cfg)

    # >>> 练习 17.4：Pre-LN 块的残差流
    def forward(self, x):
        """(B,T,d) → (B,T,d)；第二个 LayerNorm 读取注意力更新后的残差流。"""
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))
    # <<<


class GPT2(nn.Module):
    """学习式位置嵌入 + Pre-LN GPT-2；head 与 wte 为同一个 Parameter。"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.blocks = nn.ModuleList(GPT2Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight

    def forward(self, idx, return_hidden=False):
        """idx (B,T) → logits (B,T,V)；可返回 (logits, tuple(hidden))。

        hidden 与 HF output_hidden_states 对齐：各块输入共 L 项，再加 ln_f
        后的最终输出，共 L+1 个 (B,T,d)；最后一项不是未经 ln_f 的残差。
        """
        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError("idx 必须是非空序列 (批量, 序列长度)")
        T = idx.shape[1]
        if T > self.cfg.n_positions:
            raise ValueError(f"序列长度 {T} 超过最大位置数 n_positions={self.cfg.n_positions}；请缩短输入")
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        hidden = []
        for block in self.blocks:
            if return_hidden:
                hidden.append(x)
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return (logits, tuple(hidden + [x])) if return_hidden else logits


@torch.no_grad()
def generate_greedy(model, idx, n_new):
    """(B,T) → (B,T+n_new)，固定步数、不提前停 EOS、不用 KV cache。

    采用越界报错策略：输出总长也不超过 n_positions，不滑窗重置位置编号。
    调用者应先 eval()；与 HF 对比时禁用 EOS 提前停止。
    """
    if not isinstance(n_new, int) or n_new < 0:
        raise ValueError("n_new 必须为非负整数")
    if idx.shape[1] + n_new > model.cfg.n_positions:
        raise ValueError("输入加续写长度超过 n_positions；请缩短输入或续写")
    for _ in range(n_new):
        idx = torch.cat((idx, model(idx)[:, -1].argmax(-1, keepdim=True)), dim=1)
    return idx


# >>> 练习 17.1：共享权重的参数量公式
def n_params_formula(cfg):
    """返回互不重叠的逐项参数数目及 total；Vd+Pd+L(12d²+13d)+2d。"""
    d, L = cfg.n_embd, cfg.n_layer
    counts = dict(token_embedding=cfg.vocab_size*d, position_embedding=cfg.n_positions*d,
                  attention_weights=L*4*d*d, attention_biases=L*4*d,
                  mlp_weights=L*8*d*d, mlp_biases=L*5*d,
                  block_layernorms=L*4*d, final_layernorm=2*d)
    return dict(counts, total=sum(counts.values()))
# <<<


# >>> 练习 17.2：HF Conv1D 权重的转换
def convert_hf_gpt2_state_dict(hf_sd, cfg, bug=None):
    """HF GPT2Model/LMHeadModel → 本章 state_dict，张量仍在原设备。

    Conv1D (in,out) 转为 Linear (out,in)；c_attn (d,3d) 沿输出维按
    q/k/v 切分，然后各自转置。忽略因果 mask buffer；head 复用 wte。
    no_transpose 只跳过原始完整正方形 attn.c_proj (d,d) 的转置，其他
    非方阵（包括 c_attn）仍正确切分/转置，以展示 strict 加载也抓不到的错误。
    qkv_order 把切分顺序误当作 k/q/v（权重和偏置一起交换）。
    """
    if bug not in (None, "no_transpose", "qkv_order"):
        raise ValueError(f"未知转换故障：{bug}")
    src = {k.removeprefix("transformer."): v for k, v in hf_sd.items()}
    out = {k: src[k] for k in ("wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias")}
    if "lm_head.weight" in src and not torch.equal(src["lm_head.weight"], src["wte.weight"]):
        raise ValueError("lm_head.weight 与 wte.weight 不一致，无法使用共享权重")
    out["head.weight"] = out["wte.weight"]
    for i in range(cfg.n_layer):
        s, t = f"h.{i}.", f"blocks.{i}."
        for name in ("ln_1", "ln_2"):
            for suffix in ("weight", "bias"):
                out[t+name+"."+suffix] = src[s+name+"."+suffix]
        order = ("k", "q", "v") if bug == "qkv_order" else ("q", "k", "v")
        for suffix in ("weight", "bias"):
            chunks = src[s+"attn.c_attn."+suffix].split(cfg.n_embd, dim=-1)
            if len(chunks) != 3:
                raise ValueError("c_attn 输出维必须为 3*n_embd")
            for name, chunk in zip(order, chunks):
                out[t+"attn."+name+"."+suffix] = chunk.T if suffix == "weight" else chunk
        for hf_name, own in (("attn.c_proj", "attn.proj"), ("mlp.c_fc", "mlp.fc"),
                             ("mlp.c_proj", "mlp.proj")):
            w = src[s+hf_name+".weight"]
            out[t+own+".weight"] = w if bug == "no_transpose" and w.shape[0] == w.shape[1] else w.T
            out[t+own+".bias"] = src[s+hf_name+".bias"]
    return out
# <<<


def _hub_root(hub):
    """优先显式 hub，其次环境变量，最后默认 HF 缓存。"""
    return Path(hub if hub is not None else os.environ.get("HF_HUB_CACHE", Path(os.environ.get(
        "HF_HOME", Path.home()/".cache/huggingface"))/"hub")).expanduser()


def _configs(repo, hub=None):
    """本地快照按路径排序；选择顺序可复现，不声称是最新版本。"""
    return sorted((_hub_root(hub)/repo/"snapshots").glob("*/config.json"))


def find_local_gpt2(hub=None):
    """只 glob 本地 GPT-2 快照；返回同时含 config 与 model.safetensors 的目录或 None。"""
    paths = _configs("models--openai-community--gpt2", hub) + _configs("models--gpt2", hub)
    return next((p.parent for p in reversed(sorted(paths))
                 if (p.parent/"model.safetensors").is_file()), None)


def load_gpt2_pretrained(path=None, device="cpu"):
    """离线 safetensors → strict 加载，返回 (eval 模型, cfg)，默认 fp32 CPU。"""
    from safetensors.torch import load_file
    path = Path(path) if path is not None else find_local_gpt2()
    if path is None:
        raise FileNotFoundError("未找到本地 GPT-2 权重；本章不会自动下载")
    cfg = GPT2Config.from_hf_dict(json.loads((path/"config.json").read_text(encoding="utf-8")))
    model = GPT2(cfg)
    model.load_state_dict(convert_hf_gpt2_state_dict(load_file(str(path/"model.safetensors")), cfg), strict=True)
    return model.to(device).eval(), cfg


def config_field_notes():
    """返回官方 GPT-2 config 字段的一句话中文解释（值由调用方读取本地 JSON）。"""
    return {
        "activation_function": "MLP 激活；gelu_new 是 tanh 近似 GELU，gelu 是精确式",
        "architectures": "保存时声明的模型类，不是参数张量",
        "attn_pdrop": "注意力概率的 dropout 比率；eval 时关闭",
        "bos_token_id": "序列起始 token；官方 GPT-2 与 EOS 都是 50256 (<|endoftext|>)",
        "eos_token_id": "序列结束 token；官方 GPT-2 与 BOS 共用 50256",
        "embd_pdrop": "嵌入相加后的 dropout 比率；eval 时关闭",
        "initializer_range": "随机初始化正态分布的标准差，加载权重后不再决定数值",
        "layer_norm_epsilon": "LayerNorm 方差分母中的稳定常数 epsilon",
        "model_type": "模型族标识 gpt2，用于自动配置分派",
        "n_ctx": "历史上下文字段；官方值与 n_positions 相同，位置表实际由 n_positions 决定",
        "n_embd": "残差流和 token/位置嵌入的宽度 d",
        "n_head": "MHA 注意力头数，q/k/v 头数相同",
        "n_layer": "Transformer 块数 L",
        "n_positions": "学习式位置表行数 P，也是本章最大序列长度",
        "resid_pdrop": "残差支路 dropout 比率；eval 时关闭",
        "summary_activation": "序列分类/摘要头投影后的激活，普通 LM head 不使用",
        "summary_first_dropout": "序列分类/摘要头前的 dropout，普通 LM head 不使用",
        "summary_proj_to_labels": "序列分类/摘要头是否投影到标签维度，普通 LM head 不使用",
        "summary_type": "序列分类/摘要头如何选取或汇聚位置，例如 cls_index",
        "summary_use_proj": "序列分类/摘要头是否使用投影层，普通 LM head 不使用",
        "task_specific_params": "按任务保存的生成默认建议，本章显式指定贪心参数覆盖采样建议",
        "vocab_size": "token 词表大小 V，也是共享输出矩阵的行数",
        "n_inner": "MLP 中间宽度；null 表示 4*n_embd",
        "tie_word_embeddings": "输出 head 与输入 token 嵌入是否共享参数",
        "use_cache": "HF 是否返回或使用 KV cache，本章手写模型不用缓存",
        "torch_dtype": "保存的精度元数据；本章显式使用 fp32",
        "dtype": "保存的精度元数据；本章显式使用 fp32",
        "transformers_version": "保存配置所用的 transformers 版本",
        "scale_attn_weights": "是否按 head_dim 的平方根缩放注意力分数",
        "scale_attn_by_inverse_layer_idx": "是否额外按层编号缩放，官方 small 不启用",
        "reorder_and_upcast_attn": "是否重排并提高注意力计算精度，官方 small 不启用",
        "add_cross_attention": "是否添加跨注意力，本章不支持该变体",
    }


def architecture_diff_rows(hub=None):
    """差异表：每格是 value/source；只读配置和 safetensors keys，不加载 Qwen 张量。

    Qwen3 直接读取生成模型 Qwen3-0.6B；无证据的项不猜测。
    GPT-2 的结构项来自本文件，数值来自本地 config；本书 GPT 来自默认配置/实现。
    """
    from safetensors import safe_open
    from miniocr.models.gpt import GPTConfig, RMSNorm
    b = GPTConfig()
    gp = _configs("models--openai-community--gpt2", hub) or _configs("models--gpt2", hub)
    qp = _configs("models--Qwen--Qwen3-0.6B", hub)
    g = json.loads(gp[-1].read_text()) if gp else {}
    q = json.loads(qp[-1].read_text()) if qp else {}
    gs = str(gp[-1]) if gp else "待验证：无本地 GPT-2 config"
    qs = str(qp[-1]) if qp else "待验证：无本地 Qwen3-0.6B config"
    keys, keyfiles = set(), sorted(qp[-1].parent.glob("*.safetensors")) if qp else []
    for path in keyfiles:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys.update(f.keys())
    ks = "本地 safetensors 只读 keys: " + ",".join(str(p) for p in keyfiles)
    def cell(value, source):
        return dict(value=value, source=source)
    def gc(key):
        return cell(g.get(key, "待验证"), gs+":"+key)
    def qc(key):
        return cell(q.get(key, f"待验证：本地配置未提供 {key}"), qs+":"+key)
    def bc(value, key):
        return cell(value, "miniocr.models.gpt.GPTConfig 默认/实现:"+key)
    def structure(value):
        return cell(value, "miniocr.models.gpt2 实现；官方权重对拍见部分4")
    def qkeys(value):
        return cell(value if keyfiles else "待验证", ks if keyfiles else "待验证：无本地权重键")
    def qheads():
        h, kv = q.get("num_attention_heads"), q.get("num_key_value_heads")
        return cell(f"{'MHA' if h == kv else 'GQA'} q={h},kv={kv}" if h and kv else "待验证",
                    qs+":num_attention_heads,num_key_value_heads")
    rows = [
        ("位置编码", structure("学习式绝对位置"), bc(b.pos_emb, "pos_emb/rope"),
         cell("RoPE" if "rope_theta" in q else "待验证", qs+":rope_theta,rope_scaling")),
        ("归一化", structure("LayerNorm（γ/β），Pre-LN"), bc(b.norm, "norm"),
         cell("RMSNorm" if "rms_norm_eps" in q else "待验证", qs+":rms_norm_eps")),
        ("MLP/激活", cell(f"非门控/{g.get('activation_function', '待验证')},中间宽度="
                         + str(g.get('n_inner') or (4*g['n_embd'] if 'n_embd' in g else '待验证')),
                         gs+":activation_function,n_inner（缺省4*n_embd）;miniocr.models.gpt2.GPT2MLP"),
         bc(f"SwiGLU/SiLU,中间宽度={b.ff_dim()}", "ff_dim()/SwiGLU"),
         cell(f"激活={q.get('hidden_act', '待验证')},中间宽度={q.get('intermediate_size', '待验证')};"
              + ("gate/up/down 门控" if all(any(f'.{n}_proj.weight' in k for k in keys)
                   for n in ('gate', 'up', 'down')) else "门控结构待验证"), qs+":hidden_act,intermediate_size;"+ks)),
        ("注意力头", cell(f"MHA q={g['n_head']},kv={g['n_head']}" if 'n_head' in g else "待验证", gs+":n_head"),
         bc(f"{'MHA' if b.n_head == b.kv_heads() else 'GQA'} q={b.n_head},kv={b.kv_heads()}", "n_head/kv_heads()"), qheads()),
        ("Q/K norm", structure(False), bc(b.qk_norm, "qk_norm"),
         qkeys(all(any(f'.{n}_norm.weight' in k for k in keys) for n in ('q', 'k')))),
        ("线性层偏置", structure("q/k/v/proj/fc 有，head 无"), bc(False, "Linear(bias=False)"),
         cell(f"注意力={q.get('attention_bias', '待验证')};MLP=" +
              (str(any('.mlp.' in k and k.endswith('.bias') for k in keys)) if keyfiles else "待验证"),
              qs+":attention_bias;"+ks)),
        ("权重共享", structure(True), bc(b.tie_weights, "tie_weights"), qc("tie_word_embeddings")),
        ("词表大小", gc("vocab_size"), bc(b.vocab_size, "vocab_size"), qc("vocab_size")),
        ("最大位置", gc("n_positions"), bc(b.block_size, "block_size"), qc("max_position_embeddings")),
        ("层数", gc("n_layer"), bc(b.n_layer, "n_layer"), qc("num_hidden_layers")),
        ("宽度", gc("n_embd"), bc(b.d_model, "d_model"), qc("hidden_size")),
        ("head_dim", cell(g['n_embd']//g['n_head'] if 'n_embd' in g and 'n_head' in g else "待验证", gs+":n_embd/n_head"),
         bc(b.d_model//b.n_head, "d_model/n_head"), qc("head_dim")),
        ("norm eps", gc("layer_norm_epsilon"), bc(RMSNorm(b.d_model).eps, "RMSNorm 默认 eps"), qc("rms_norm_eps")),
        ("rope_theta", structure("不适用"), bc(b.rope_base, "rope_base"), qc("rope_theta")),
    ]
    return [dict(field=name, gpt2=a, book=b, qwen3=c) for name, a, b, c in rows]
