"""只读本地 Qwen3-0.6B（chat 权重，并非未下载的 Qwen3-0.6B-Base）。"""
from pathlib import Path

import torch

from miniocr.models.gpt2 import _hub_root


def find_local_qwen3(hub=None):
    """只 glob 本地快照，返回含 config.json/model.safetensors 的目录或 None。

    支持显式 hub、HF_HUB_CACHE、HF_HOME；多个快照按路径逆序选择，不查询网络。
    """
    paths = sorted((_hub_root(hub)/"models--Qwen--Qwen3-0.6B"/"snapshots").glob("*/config.json"))
    return next((p.parent for p in reversed(paths) if (p.parent/"model.safetensors").is_file()), None)


def load_qwen3(path=None, device="cpu", dtype=torch.float32):
    """从本地目录加载 eval 模型与 tokenizer，返回 (model, tokenizer)，默认 CPU fp32。

    path 只接受已存在的本地目录，绝不将字符串当作 Hub repo ID，绝不下载。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = Path(path).expanduser() if path is not None else find_local_qwen3()
    if path is None or not path.is_dir():
        raise FileNotFoundError("未找到本地 Qwen3-0.6B 权重；不会自动下载")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype, local_files_only=True)
    return model.to(device).eval(), tokenizer
