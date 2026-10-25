"""第 18 章：训练工程。

    train_step              一步训练，支持梯度累积：几个小批的梯度加起来，等价于一个大批
    save_checkpoint / load_checkpoint   保存与恢复"训练的全部状态"，续训后与不中断完全一致
    flops_per_token         每个训练 token 的浮点运算量：≈ 6N + 注意力的部分
    memory_breakdown        参数、梯度、Adam 状态各占多少字节
    saved_activation_bytes  反向传播要保存的激活值有多少字节（用 saved_tensors_hooks 实测，CPU 上也能量）
"""
import torch


# >>> 练习 18.1：梯度累积
def train_step(model, opt, micro_batches, clip=1.0):
    """micro_batches 是 [(x, y), ...]，共 n 个小批（每个小批的样本数相同）。
    对每个小批：loss = model(x, targets=y)[1] / n，loss.backward()（梯度在 .grad 里自动累加）；
    全部做完后裁剪梯度、opt.step()、opt.zero_grad(set_to_none=True)。返回 n 个小批的平均损失（Python float）。
    除以 n 是因为每个小批的 loss 已经是"小批内的平均"，n 个相加后再除以 n 才是"大批的平均"。"""
    n = len(micro_batches)
    total = 0.0
    for x, y in micro_batches:
        loss = model(x, targets=y)[1] / n
        loss.backward()
        total += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return total
# <<<


# >>> 练习 18.2：检查点
def save_checkpoint(path, model, opt, step, generator, extra=None):
    """保存续训需要的一切：模型参数、优化器状态（Adam 的 m、v 和步数）、当前步数、数据生成器的随机状态。
    少了任何一样，续训的结果都会和不中断时不同。"""
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "rng": generator.get_state(), "extra": extra}, path)


def load_checkpoint(path, model, opt, generator):
    """恢复 save_checkpoint 保存的状态，返回 (step, extra)。"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    generator.set_state(ck["rng"])
    return ck["step"], ck["extra"]
# <<<


# >>> 练习 18.3：计算量
def flops_per_token(n_params_non_emb, n_layer, d_model, T):
    """训练一个 token 的浮点运算次数（乘法和加法各算一次）：
        矩阵乘法部分：前向 2N（每个参数一次乘、一次加），反向 4N（对输入、对权重各一次）→ 6N
        注意力分数部分：每层 QKᵀ 和 权重·V 各 2·T·d（前向），乘 3 算上反向 → 每层 12·T·d；因果 mask 只需一半，这里不打折
    N 用不含嵌入的参数量（嵌入是查表，不做乘法；共享权重的输出层另算在 N 外时要加回 2·V·d，这里忽略）。"""
    return 6 * n_params_non_emb + 12 * n_layer * T * d_model
# <<<


def memory_breakdown(model, param_bytes=4):
    """训练时与参数量成正比的内存：参数、梯度、Adam 的 m 和 v（都与参数同样大小）。混合精度时情况不同（第 11 章）。"""
    n = model.n_params() if hasattr(model, "n_params") else sum(p.numel() for p in model.parameters())
    return {"参数": n * param_bytes, "梯度": n * param_bytes, "Adam m": n * 4, "Adam v": n * 4}


def saved_activation_bytes(fn):
    """运行 fn()（一次前向并返回损失），统计 autograd 为反向传播保存下来的张量一共多少字节。
    同一块内存被保存多次只算一次（按数据指针去重）。"""
    seen, total = set(), [0]

    def pack(t):
        key = (t.data_ptr(), t.numel(), t.dtype)
        if key not in seen:
            seen.add(key)
            total[0] += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = fn()
    return total[0], loss
