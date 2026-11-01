"""第 16 章：浮点精度与 loss scaling；全部实验可在 CPU 上运行。

    float_format       从指数/尾数位推导 IEEE 二进制格式的范围与间隔
    format_table       返回 fp32、fp16、bf16 的格式表
    spacing            正规数所在 binade 内的浮点间隔
    quantize_gradients 模拟放大、fp16 保存、fp32 除回后的梯度误差
    SimpleGradScaler   教学版动态缩放器，支持稠密 fp32 梯度与普通优化器
"""
import math

import torch


_FORMATS = {"fp32": (8, 23, torch.float32), "fp16": (5, 10, torch.float16),
            "bf16": (8, 7, torch.bfloat16)}


def float_format(name):
    """bias=2^(e−1)−1；max=(2−2^−m)2^bias，tiny=2^(1−bias)，
    subnormal=tiny·2^−m，eps=2^−m。尾数位 m 不含隐藏的首位 1。
    """
    e, m, dtype = _FORMATS[name]
    bias = 2**(e-1)-1
    return dict(name=name, dtype=dtype, sign_bits=1, exponent_bits=e, mantissa_bits=m,
                max=(2-2.0**-m)*2.0**bias, tiny=2.0**(1-bias),
                subnormal=2.0**(1-bias-m), eps=2.0**-m)


def format_table():
    """按 fp32、fp16、bf16 顺序返回 float_format 的公式表。"""
    return [float_format(name) for name in _FORMATS]


def spacing(x, dtype):
    """正规有限数的 binade 间隔 = 2^floor(log2|x|)·eps。

    dtype 接受 fp32/fp16/bf16 或对应 torch.dtype。拒绝零、非正规数、溢出。
    在 2 的幂边界返回绝对值增大方向的间隔；max 处不表示到 inf 的距离。
    """
    if not isinstance(dtype, str):
        dtype = next(name for name, (_, _, dt) in _FORMATS.items() if dt == dtype)
    f = float_format(dtype)
    if not math.isfinite(x) or not f["tiny"] <= abs(x) <= f["max"]:
        raise ValueError("x 必须在该格式的有限正规数范围内")
    return 2.0**(math.frexp(abs(x))[1]-1)*f["eps"]


def quantize_gradients(grad, scale=1.0):
    """q=fp16(g·S)，恢复值=fp32(q)/S；相对误差为 ||恢复−g||₂/||g||₂。

    返回零/inf 比例、相对误差及恢复张量；这是存储舍入模拟，不是 CPU AMP。
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale 必须为有限正数")
    q = (grad.float()*scale).half()
    restored = q.float()/scale
    error = (restored-grad).norm()/grad.norm().clamp_min(torch.finfo(torch.float32).tiny)
    return dict(zero_fraction=(q == 0).float().mean().item(),
                inf_fraction=torch.isinf(q).float().mean().item(),
                relative_error=error.item(), restored=restored)


class SimpleGradScaler:
    """教学版 scale → backward → unscale_ → clip → step → update 状态机。

    每个 optimizer 每步只能 unscale/step 一次；step 可自动 unscale。
    任意非有限梯度使该 optimizer 跳步；update 汇总所有 optimizer，遇非有限
    值 S←S·backoff、计数归零；否则每 growth_interval 步 S←S·growth。
    与 torch.amp.GradScaler 相同的状态字典键；只支持普通优化器的稠密 fp32
    梯度，不实现 AMP-aware/fused 优化器、稀疏梯度或分布式归约。step 返回
    是否实际执行，区别于 torch 版返回 optimizer.step 的返回值。
    """

    def __init__(self, init_scale=2**16, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000):
        if not math.isfinite(init_scale) or not 0 < init_scale <= torch.finfo(torch.float32).max:
            raise ValueError("init_scale 必须为 fp32 可表示的有限正数")
        if not math.isfinite(growth_factor) or growth_factor <= 1 or not 0 < backoff_factor < 1:
            raise ValueError("growth_factor > 1，0 < backoff_factor < 1")
        if not isinstance(growth_interval, int) or growth_interval < 1:
            raise ValueError("growth_interval 必须为正整数")
        self._scale = float(torch.tensor(init_scale, dtype=torch.float32))
        self._growth_factor, self._backoff_factor = growth_factor, backoff_factor
        self._growth_interval, self._growth_tracker = growth_interval, 0
        self._states = {}
        self.found_inf = False

    def scale(self, loss):
        """返回 loss·S；不修改原 loss。"""
        return loss*self._scale

    def unscale_(self, optimizer):
        """原地 g←g/S 并记录 inf/nan；同一步重复调用报 RuntimeError。"""
        key = id(optimizer)
        if key in self._states:
            raise RuntimeError("本步已 unscale_ 或 step；须先 update")
        grads = [p.grad for group in optimizer.param_groups for p in group["params"]
                 if p.grad is not None]
        if not grads:
            raise RuntimeError("没有梯度可检查")
        if any(g.is_sparse or g.dtype != torch.float32 for g in grads):
            raise ValueError("教学版仅支持稠密 fp32 梯度")
        bad = False
        with torch.no_grad():
            for g in grads:
                bad |= not bool(torch.isfinite(g).all())
                g.mul_(1.0/self._scale)
        self._states[key] = dict(stage="unscaled", found_inf=bad)
        self.found_inf |= bad

    def step(self, optimizer):
        """自动 unscale（若尚未做）；非有限梯度跳过更新，返回是否执行。"""
        key = id(optimizer)
        if key not in self._states:
            self.unscale_(optimizer)
        state = self._states[key]
        if state["stage"] == "stepped":
            raise RuntimeError("本步已 step；须先 update")
        executed = not state["found_inf"]
        if executed:
            optimizer.step()
        state["stage"] = "stepped"
        return executed

    def update(self):
        """依据本步 inf/nan 更新 S 和连续成功计数，再清空逐步状态。"""
        if not self._states:
            raise RuntimeError("update 前须检查梯度")
        if self.found_inf:
            self._scale = float(torch.tensor(self._scale*self._backoff_factor, dtype=torch.float32))
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker == self._growth_interval:
                candidate = torch.tensor(self._scale*self._growth_factor, dtype=torch.float32)
                if torch.isfinite(candidate):
                    self._scale = float(candidate)
                self._growth_tracker = 0
        self._states.clear()
        self.found_inf = False

    def get_scale(self):
        """当前缩放倍数 S（Python float）。"""
        return self._scale

    def state_dict(self):
        """返回可在 update 后保存的状态；键名兼容 torch GradScaler。"""
        return dict(scale=self._scale, growth_factor=self._growth_factor,
                    backoff_factor=self._backoff_factor, growth_interval=self._growth_interval,
                    _growth_tracker=self._growth_tracker)

    def load_state_dict(self, state):
        """恢复 update 边界处的状态；逐 optimizer 临时状态不序列化。"""
        self.__init__(state["scale"], state["growth_factor"], state["backoff_factor"],
                      state["growth_interval"])
        self._growth_tracker = state["_growth_tracker"]
