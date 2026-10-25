"""所有作业共用的限制检查：作业包里不能用现成的层、损失、优化器。用语法树检查，不靠字符串匹配。
每个作业的说明书第 1 节是规则的正式版本；这里查不出所有绕过的办法。"""
import ast
import os

ALLOWED = {"torch.nn.Module", "torch.nn.Parameter", "torch.nn.ModuleList", "torch.nn.ModuleDict",
           "torch.nn.Sequential", "torch.optim.Optimizer"}
BANNED_PREFIX = ("torch.nn.functional", "torch.optim", "torch.nn.utils.clip_grad", "torch.nn.")
BANNED_TORCH = {"torch.conv2d", "torch.max_pool2d", "torch.batch_norm", "torch.log_softmax", "torch.softmax",
                "torch.cross_entropy", "torch.nn.init", "torch.ctc_loss", "torch.lstm", "torch.gru", "torch.rnn_tanh",
                "torch.embedding", "torch.layer_norm", "torch.rms_norm",
                "torch.gelu", "torch.upsample_bilinear2d", "torch.nms"}
BANNED_METHOD = {"unfold", "log_softmax", "softmax", "conv2d", "max_pool2d", "batch_norm", "cross_entropy", "ctc_loss",
                 "scaled_dot_product_attention", "silu", "embedding", "layer_norm", "rms_norm", "lstm",
                 "gelu", "interpolate", "upsample_bilinear2d", "logsigmoid"}


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def violations(src, fname):
    tree = ast.parse(src)
    alias, local = {}, set()                     # alias：nn → torch.nn 之类；local：从你自己的模块导入的名字
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:                    # import torch.nn as nn → nn = torch.nn；import torch.nn → torch = torch
                if a.asname:
                    alias[a.asname] = a.name
                else:
                    alias[a.name.split(".")[0]] = a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                name = a.asname or a.name
                if n.level > 0:
                    local.add(name)
                elif n.module:
                    alias[name] = f"{n.module}.{a.name}"
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            bad += [f"{fname}:{n.lineno} import {a.name}" for a in n.names if _banned(a.name)]
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            bad += [f"{fname}:{n.lineno} from {n.module} import {a.name}" for a in n.names
                    if _banned(n.module) or _banned(f"{n.module}.{a.name}")]
    inner = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for n in ast.walk(tree):
        if id(n) in inner:                       # 只看最外层：torch.optim.Optimizer 不会因为 torch.optim 被误报
            continue
        full = None
        if isinstance(n, ast.Attribute):
            d = _dotted(n)
            if d:
                head, *rest = d.split(".")
                if head in local or head == "self":
                    continue
                full = ".".join([alias.get(head, head)] + rest)
            elif n.attr in BANNED_METHOD:
                bad.append(f"{fname}:{n.lineno} .{n.attr}(...)")
                continue
        elif isinstance(n, ast.Name) and n.id in alias:
            full = alias[n.id]
        if not full:
            continue
        if _banned(full) or full.split(".")[-1] in BANNED_METHOD and not full.startswith("torch.nn.Module"):
            bad.append(f"{fname}:{getattr(n, 'lineno', '?')} {full}")
    return bad


def _banned(full):
    if any(full == a or full.startswith(a + ".") for a in ALLOWED):
        return False
    return full in BANNED_TORCH or full.startswith(BANNED_PREFIX) or full in ("torch.optim", "torch.nn.functional")


def check_dir(code_dir):
    """返回 code_dir 下所有 .py 文件里的违规（字符串列表，已去重排序）。"""
    bad = []
    for f in sorted(os.listdir(code_dir)):
        if f.endswith(".py"):
            bad += violations(open(os.path.join(code_dir, f), encoding="utf-8").read(), f)
    return sorted(set(bad))
