"""设备工具：同一份代码在 CPU、Apple 芯片（MPS）和 NVIDIA GPU（CUDA）上都能跑。

    pick_device        选设备：显式指定 > 环境变量 MINIOCR_DEVICE > cuda > mps > cpu
    device_from_argv   从命令行取 --device cpu|mps|cuda（tour 脚本统一用它）
    move_batch         把 dict / tuple / list 里的张量都搬到同一个设备
    assert_same_device 模型和输入不在同一设备时立刻报错，而不是等到矩阵乘法里报一个难懂的错
    synchronize        计时前等设备把排队的计算做完（GPU 和 MPS 都是异步执行的）
    autocast           混合精度上下文；CPU 和 MPS 上默认关闭（见第 2.5 单元）

v3 第 35 章踩过的坑：模型在 GPU、输入留在 CPU。v4 的约定是：构造输入的函数都接收 device 参数，
默认取 model 所在的设备；拿不准时调用 assert_same_device。
"""
import contextlib
import os
import sys

import torch


def pick_device(name=None):
    """name 为 "cpu" / "mps" / "cuda" 时直接用它（不可用就报错）；为 None 时先看环境变量 MINIOCR_DEVICE，
    再按 cuda → mps → cpu 的顺序选第一个可用的。返回 torch.device。"""
    name = name or os.environ.get("MINIOCR_DEVICE")
    if name:
        dev = torch.device(name)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定了 cuda，但 torch.cuda.is_available() 为 False（5080 被占用时先停掉 vLLM，再用 nvidia-smi 确认）")
        if dev.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("指定了 mps，但这台机器不支持 MPS（需要 Apple 芯片和对应版本的 PyTorch）")
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_from_argv(argv=None, default=None):
    """tour 脚本的约定：python -m ... --device mps。没写时按 pick_device 的顺序选。"""
    argv = sys.argv if argv is None else argv
    name = default
    if "--device" in argv:
        i = argv.index("--device")
        name = argv[i + 1] if i + 1 < len(argv) else None
    return pick_device(name)


def move_batch(batch, device):
    """递归地把张量搬到 device；非张量原样返回。"""
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_batch(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_batch(v, device) for v in batch)
    return batch


def model_device(model):
    return next(model.parameters()).device


def assert_same_device(model, *tensors):
    """所有张量都必须和模型的第一个参数在同一设备上（cuda:0 和 cuda 视为相同）。"""
    dev = model_device(model)
    for t in tensors:
        if torch.is_tensor(t) and t.device.type != dev.type:
            raise RuntimeError(f"模型在 {dev}，输入在 {t.device}：先用 move_batch(..., {dev.type!r}) 把输入搬过去")


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def autocast(device, dtype=None):
    """dtype 为 None 或 torch.float32 时不做任何事。CUDA 上 bf16/fp16 都可以；CPU 上只支持 bf16；
    MPS 上的 autocast 支持取决于 PyTorch 版本，课程的 MPS 配置默认用 fp32（第 2.5 单元）。"""
    if dtype is None or dtype == torch.float32:
        return contextlib.nullcontext()
    device = torch.device(device)
    return torch.autocast(device_type=device.type, dtype=dtype)
