"""第 4 章　检查点：把"训练进行到哪一步"完整地存下来，中断以后能接着训，结果和不中断时一样。

    rng_state          取出 Python、NumPy、PyTorch（CPU 与 CUDA）以及若干 torch.Generator 的随机数状态   练习 4.5
    set_rng_state      把上面取出的状态放回去                                                    练习 4.5
    save_checkpoint    保存模型、优化器、步数、随机数状态和任意附加信息（先写临时文件再改名）            练习 4.6
    load_checkpoint    恢复上面这些东西，返回 (步数, 附加信息)                                     练习 4.6

要续训和不中断"逐位一致"，下面五样东西缺一不可（第 4 章 4.10 节）：
    1. 模型参数（model.state_dict()）
    2. 优化器状态：动量、Adam 的 m、v 和步数 t。只存模型、不存优化器，续训第一步就会不一样
    3. 当前步数：学习率调度要靠它算出下一步的学习率
    4. 随机数状态：Dropout 的掩码、数据打乱的顺序都来自随机数生成器
    5. 数据读到了哪里：本模块不管数据加载器；只要打乱用的 torch.Generator 也交给 rng_state，
       并且每一步取数据的方式由"步数 + 生成器"决定，恢复以后就会取到同一批数据

优化器既可以是 torch.optim 里的（有 state_dict / load_state_dict），也可以是 miniocr.train.optim 里
自己写的（状态在 opt.state 里，配置是 lr、betas、eps、weight_decay 等属性，两者都会保存和恢复）。两种都支持。

注意：torch.load(weights_only=False) 会执行文件里的 pickle 代码，只用来加载你自己保存的文件。
"""
import copy
import os
import random

import numpy as np
import torch


# >>> 练习 4.5：随机数状态的保存与恢复
def rng_state(generators=None):
    """返回一个 dict，包含：
        "python"：random.getstate()
        "numpy"： np.random.get_state()
        "torch"： torch.get_rng_state()（CPU 上的全局生成器，一个 uint8 张量）
        "cuda"：  torch.cuda.get_rng_state_all()（没有 CUDA 时为 None）
        "generators"：{名字: g.get_state()}，generators 是 {名字: torch.Generator}，可以为 None
    返回的张量都要 clone，之后再抽随机数也不会改动这份记录。"""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
        "generators": {k: g.get_state().clone() for k, g in (generators or {}).items()},
    }


def set_rng_state(state, generators=None):
    """把 rng_state 的结果放回去。generators 里的每个生成器按名字恢复；名字在 state 里找不到就报 KeyError。"""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    for k, g in (generators or {}).items():
        g.set_state(state["generators"][k])
# <<<


def _optimizer_state(opt):
    if hasattr(opt, "state_dict"):                         # torch.optim.*（state_dict 里本来就有超参数）
        return {"kind": "torch", "state": opt.state_dict()}
    # 自己写的优化器：除了参数列表和 state，其余属性都是配置（lr、betas、eps、weight_decay、decoupled、momentum……），
    # 一起存下来。否则用非默认配置训练、加载时用默认配置重建，续训会悄悄换成另一个算法。
    config = {k: copy.deepcopy(v) for k, v in vars(opt).items() if k not in ("params", "state")}
    return {"kind": "miniocr", "lr": opt.lr, "config": config, "state": copy.deepcopy(opt.state)}


def _load_optimizer_state(opt, saved):
    if saved["kind"] == "torch":
        opt.load_state_dict(saved["state"])
    else:
        for k, v in saved.get("config", {}).items():
            setattr(opt, k, copy.deepcopy(v))
        opt.lr = saved["lr"]
        opt.state = copy.deepcopy(saved["state"])


# >>> 练习 4.6：保存与恢复检查点
def save_checkpoint(path, model, optimizer, step, generators=None, extra=None):
    """把训练状态写进 path。内容是一个 dict：
        {"model": model.state_dict(), "optimizer": ..., "step": step, "rng": rng_state(generators), "extra": extra}
    optimizer 可以为 None。先写到 path + ".tmp"，写完再 os.replace 改名：
    写到一半断电，旧的检查点还在，不会留下一个损坏的文件。"""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else _optimizer_state(optimizer),
        "step": step,
        "rng": rng_state(generators),
        "extra": extra,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, generators=None, map_location="cpu"):
    """从 path 恢复模型参数、优化器状态（optimizer 不为 None 时）和随机数状态。
    返回 (step, extra)。step 是保存时传入的"已完成步数"；步数从 0 开始编号，
    所以下一次更新的编号正好是 step，续训写 range(step, total)，不要写 step + 1。"""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt["optimizer"] is not None:
        _load_optimizer_state(optimizer, ckpt["optimizer"])
    set_rng_state(ckpt["rng"], generators)
    return ckpt["step"], ckpt["extra"]
# <<<
