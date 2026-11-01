"""第 16 章：训练资源核算；张量存储量不等于设备分配器峰值。

    measure_static_bytes       按 storage 去重测参数、梯度、主权重和优化器状态
    activation_bytes_estimate  本项目 manual/MHA 路径保存张量的逐项公式
    measure_activations        用已有 SavedTensors 测一次前向保存量
    qwen_like_params           按本地 Qwen3 骨架计数，保留共享输出层的不确定性
    training_flops             6ND、完整注意力矩阵项与 Kaplan 半矩阵口径
    count_flops                FlopCounterMode 实测前向或前向加反向
    resource_report            汇总静态、激活、计算量、MFU 和 token 预算时间
"""
import json
import math
from pathlib import Path

import torch

from .memory import SavedTensors, static_memory, train_flops
from .scaling import parameter_counts


def measure_static_bytes(model, optimizer, master_model=None):
    """统计实际 storage.nbytes，按 (device, storage.data_ptr) 全局去重。

    params/grads 属于计算模型；master/master_grads 属于可选 fp32 主模型。
    optim 是优化器张量状态（递归遍历，含 m/v，排除 step）；step 单列。
    total 是以上六项之和，optimizer_breakdown 仅为 optim 的细分。
    不含 buffer、Python 对象、分配器缓存。AdamW 必须先 step 才有状态。
    混合精度示例复制 bf16 梯度到主模型、step 后释放 master.grad；若不释放，
    master_grads 仍如实计数，不把多出来的副本藏进 16 字节/参数口径。
    """
    seen = set()

    def size(t):
        st = t.untyped_storage()
        key = (str(t.device), st.data_ptr())
        if key in seen:
            return 0
        seen.add(key)
        return st.nbytes()

    out = dict(params=0, grads=0, master=0, master_grads=0, optim=0, step=0)
    for module, pk, gk in ((model, "params", "grads"), (master_model, "master", "master_grads")):
        if module is not None:
            for p in module.parameters():
                out[pk] += size(p)
                if p.grad is not None:
                    out[gk] += size(p.grad)
    breakdown = {}

    def visit(value, name):
        if isinstance(value, torch.Tensor):
            n = size(value)
            if name == "step":
                out["step"] += n
            else:
                out["optim"] += n
                breakdown[name] = breakdown.get(name, 0)+n
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, str(key))
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, name)

    for state in optimizer.state.values():
        visit(state, "state")
    out["total"] = sum(out.values())
    out["optimizer_breakdown"] = breakdown
    return out


def activation_bytes_estimate(cfg, B, T, bytes_per=4, attn="manual"):
    """预测 SavedTensors 的去重保存量；不是峰值，单位 byte。

    精确域：全参训练、统一权重/buffer dtype、manual、MHA、RMSNorm、dropout=0、无 checkpoint、
    无 learned position；输入 idx 与 targets 是独立连续 int64 (B,T) storage，
    GPT(idx, targets) 的普通交叉熵。RoPE buffer 与参数 storage 被排除。
    记 L=层数、d=宽度、f=SwiGLU 宽度、h=头数、V=词表。fp32 时 s=4：

    每层 RMSNorm 输入及乘 gamma 前结果：2·2·B·T·d·s；
    两次 rsqrt：2·B·T·1·s；q/k/v 共用的线性输入与 gate/up 共用输入：2·B·T·d·s；
    QKᵀ/AV 反向保存 q,k,v（可能为 reshape 副本）：3·B·T·d·s；
    o 投影输入：B·T·d·s；SwiGLU gate、up、silu、乘积：4·B·T·f·s；
    softmax 权重：B·h·T·T·s（scores 本身未保存）；mask：T·T·1。
    末层 norm 输入/乘 gamma 前结果/head 输入：3·B·T·d·s；rsqrt：B·T·s；
    交叉熵 log-softmax：B·T·V·4（logits 本身不在 saved hooks 账内）；
    idx、targets：2·B·T·8；交叉熵 total_weight 标量：4。

    总量=aT+bT²+c，a=L·sB(10d+4f+2)+sB(3d+1+V)+16B，
    b=L(sBh+1)，c=4。返回 logits_bytes 提醒调用者返回的 logits 还活着，
    但不把它加到 saved total。不同算子实现可能改变保存策略，测试用于检查边界。
    bytes_per=2 表示 model.to(bfloat16/float16)，包括 RoPE buffer，非 autocast。
    RMSNorm 内部始终 fp32：上式 norm 输入/归一化结果/rsqrt 保持每元素 4 字节；
    线性输入、q/k/v、SwiGLU、权重用 s 字节；CE 保持 4 字节，索引保持 8 字节。
    一般式 a=L·B[(16+6s)d+4sf+8]+B[(8+s)d+4+4V+16]，b=L(sBh+1)。
    普通 AMP 的 fp32 参数与 autocast 转换缓存不在此精确域，不能套用低精度副本账。
    """
    if not isinstance(B, int) or not isinstance(T, int) or min(B, T) < 1 or T > cfg.block_size:
        raise ValueError("B/T 必须为正整数，T 不超过 block_size")
    if (bytes_per not in (2, 4) or attn != "manual" or cfg.attn_impl != "manual" or
            cfg.kv_heads() != cfg.n_head or cfg.qk_norm or cfg.norm != "rmsnorm" or
            cfg.dropout != 0 or (cfg.rope and cfg.pos_emb == "learned")):
        raise ValueError("待验证：精确估算仅支持统一2/4字节dtype/manual/MHA/RMSNorm/无dropout与learned position")
    d, f, h, L, V, s = cfg.d_model, cfg.ff_dim(), cfg.n_head, cfg.n_layer, cfg.vocab_size, bytes_per
    rows = []

    def add(name, shape, n):
        rows.append(dict(name=name, formula=shape, bytes=n))

    add("块内norm输入与归一化结果", "L·4·B·T·d·4", L*4*B*T*d*4)
    add("块内rsqrt", "L·2·B·T·1·4", L*2*B*T*4)
    add("qkv及gate/up共享线性输入", "L·2·B·T·d·s", L*2*B*T*d*s)
    add("注意力q/k/v", "L·3·B·T·d·s", L*3*B*T*d*s)
    add("o投影输入", "L·B·T·d·s", L*B*T*d*s)
    add("SwiGLU四个中间量", "L·4·B·T·f·s", L*4*B*T*f*s)
    add("softmax权重", "L·B·h·T²·s", L*B*h*T*T*s)
    add("布尔mask", "L·T²·1", L*T*T)
    add("最终norm与head输入", "B·T·d·(2·4+s)", B*T*d*(8+s))
    add("最终rsqrt", "B·T·4", B*T*4)
    add("交叉熵log_softmax", "B·T·V·4", B*T*V*4)
    add("独立idx与targets", "2·B·T·8", 2*B*T*8)
    add("交叉熵total_weight", "1·4", 4)
    total = sum(row["bytes"] for row in rows)
    for row in rows:
        row["fraction"] = row["bytes"]/total
    return dict(total=total, breakdown=rows, logits_bytes=B*T*V*s,
                linear_coefficient=L*B*((16+6*s)*d+4*s*f+8)+B*((8+s)*d+4+4*V+16),
                quadratic_coefficient=L*(s*B*h+1), constant=4,
                block_linear_coefficient=B*((16+6*s)*d+4*s*f+8),
                block_quadratic_coefficient=s*B*h+1,
                status="数学结论：统一dtype的manual全参保存量；不含返回logits与峰值工作区；非autocast")


def measure_activations(model, B, T):
    """独立 idx/targets，一次前向的 SavedTensors.bytes；不做 backward/更新。

    排除模型参数和 buffer，按 storage 去重；不含只被 Python 持有的 logits。
    """
    device = next(model.parameters()).device
    x = torch.zeros((B, T), dtype=torch.long, device=device)
    with SavedTensors(model) as saved:
        logits, loss = model(x, targets=x.clone())
    return saved.bytes


def qwen_like_params(cfg, tie_word_embeddings=None):
    """Qwen3 标准骨架：Vd·(1 若 tied 否则 2) + L·[2d·h·dh +
    2d·hk·dh + 2dh(q/k norm) + 3df + 2d(norm)] + d(最终 norm)。

    cfg 为 local_model_configs 的全注意力配置。未传 tie 时只重读 cfg.source
    已选定的本地 JSON；字段缺失则假设 True 并标待验证，同时给出两种结果。
    不加载权重，不下载；本地配置或待验证的内置回退由 cfg.note 标注。
    """
    if cfg.full_layers != cfg.n_layer:
        raise ValueError("仅支持 Qwen3 全注意力骨架")
    origin = "调用者指定"
    if tie_word_embeddings is None:
        origin = "待验证：缺 tie_word_embeddings，暂按共享输出层"
        if cfg.source.startswith("本地 config "):
            data = json.loads(Path(cfg.source.removeprefix("本地 config ")).read_text())
            text_cfg = data.get("text_config", data)
            tie_word_embeddings = text_cfg.get("tie_word_embeddings", data.get("tie_word_embeddings"))
            if tie_word_embeddings is not None:
                origin = cfg.source+" 的 tie_word_embeddings"
        if tie_word_embeddings is None:
            tie_word_embeddings = True
    if not isinstance(tie_word_embeddings, bool):
        raise ValueError("tie_word_embeddings 必须为 bool")
    d, dh, V = cfg.d_model, cfg.head_dim, cfg.vocab_size
    body = cfg.n_layer*(2*d*cfg.n_head*dh+2*d*cfg.kv_heads()*dh+2*dh+
                       3*d*cfg.ff_dim()+2*d)+d
    return dict(total=body+V*d*(1 if tie_word_embeddings else 2),
                tied=body+V*d, untied=body+2*V*d,
                tie_word_embeddings=tie_word_embeddings, tie_source=origin,
                source=cfg.source, note=cfg.note)


def training_flops(cfg, batch, context, accum=1):
    """D=B·T·accum；6N_matmul·D 沿用旧参数口径（含少量 norm）。

    manual 完整 QKᵀ 和 AV 每层前向各 2BT²d，反向约 2 倍，
    训练额外 12LBT²d·accum。Kaplan 有效因果半矩阵近似额外 6LBT²d·accum，
    即每层每 token 前向 2Td、训练乘 3。FlopCounterMode 对 manual 计完整矩阵。
    均忽略逐元素算子，不是硬件指令计数；冻结部分参数也不能按可训练比例缩放。
    """
    D = batch*context*accum
    n = parameter_counts(cfg)["matmul"]
    base = train_flops(n, D)
    extra = 12*cfg.n_layer*context*cfg.d_model*D
    return dict(tokens=D, matmul_params=n, six_nd=base, attention=extra,
                with_attention=base+extra, kaplan_half_attention=base+extra//2,
                forward_2nd=base//3, forward_with_attention=(base+extra)//3)


def count_flops(model, x, targets, backward=True):
    """FlopCounterMode 的算子 FLOPs；默认 forward+backward，不含 optimizer。

    清空旧梯度，保留本轮梯度供对拍；乘加=2，manual 注意力包含被 mask 的位置。
    """
    from torch.utils.flop_counter import FlopCounterMode
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as counter:
        loss = model(x, targets=targets)[1]
        if backward:
            loss.backward()
    return counter.get_total_flops()


def resource_report(model_cfg, batch, context, accum=1, precision="fp32",
                    trainable_fraction=1.0, peak_tflops=None, tokens_per_sec=None,
                    token_budget=None):
    """资源预算 dict；batch 为微批，tokens/step=B·T·accum，激活只算一个微批。

    静态调用 static_memory，trainable=round(N·fraction)。mixed 表示 bf16
    权重/梯度 + fp32 主权重/m/v，计算模型整体 .to(bfloat16)，非 autocast；
    不假定其激活等于 fp32 的一半。manual 精确域返回 activation.total；
    冻结时仍用全参保存量作预算参考并标待验证（比例不能表达具体冻结位置）；
    SDPA 等不支持路径返回 None。total_estimate=static.total+activation.total 不是峰值。
    FLOPs 保留完整前后向口径，冻结模型时为上界近似。
    achieved_flops=6N·tokens/s；MFU=achieved/(peak_tflops·10^12)；
    预算耗时=6N·token_budget/(peak·MFU)=token_budget/tokens_per_sec。
    MFU 不截断到 1，便于发现峰值精度/稀疏口径填错。
    """
    if any(not isinstance(v, int) or v < 1 for v in (batch, context, accum)):
        raise ValueError("batch/context/accum 必须为正整数")
    if context > model_cfg.block_size or precision not in ("fp32", "mixed"):
        raise ValueError("context 超界或 precision 非法")
    if not math.isfinite(trainable_fraction) or not 0 <= trainable_fraction <= 1:
        raise ValueError("trainable_fraction 必须在 [0,1]")
    for v in (peak_tflops, tokens_per_sec, token_budget):
        if v is not None and (not math.isfinite(v) or v <= 0):
            raise ValueError("峰值/吞吐/token预算必须为有限正数")
    counts = parameter_counts(model_cfg)
    nt = round(counts["total"]*trainable_fraction)
    static = static_memory(counts["total"], nt, "bf16" if precision == "mixed" else "fp32",
                           master_fp32=precision == "mixed")
    try:
        activation = activation_bytes_estimate(model_cfg, batch, context,
                                               bytes_per=2 if precision == "mixed" else 4,
                                               attn=model_cfg.attn_impl)
        if trainable_fraction != 1:
            activation["status"] = "待验证：采用全参保存量作预算参考，实际冻结位置未知"
    except ValueError as exc:
        activation = dict(total=None, status=str(exc))
    flops = training_flops(model_cfg, batch, context, accum)
    achieved = 6*counts["matmul"]*tokens_per_sec if tokens_per_sec is not None else None
    mfu = achieved/(peak_tflops*1e12) if achieved is not None and peak_tflops is not None else None
    seconds = token_budget/tokens_per_sec if token_budget is not None and tokens_per_sec is not None else None
    return dict(parameters=counts, trainable_params=nt, static=static, activation=activation,
                total_estimate=static["total"]+activation["total"] if activation["total"] is not None else None,
                tokens_per_step=flops["tokens"], flops=flops, achieved_flops=achieved,
                mfu=mfu, estimated_seconds=seconds, token_budget=token_budget,
                peak_tflops=peak_tflops, tokens_per_sec=tokens_per_sec,
                scope="静态+保存张量，不是峰值；FLOPs为完整前后向，冻结时待验证")
