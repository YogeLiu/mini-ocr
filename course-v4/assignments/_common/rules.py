"""所有作业共用的限制检查：机制实现部分不能用现成的层、损失、优化器、注意力。用语法树检查，不靠字符串匹配。
每个作业的说明书第 1 节是规则的正式版本；这里查不出所有绕过的办法。哪些目录受检查由作业的 rules.toml 决定。"""
import ast
import os

ALLOWED = {"torch.nn.Module", "torch.nn.Parameter", "torch.nn.ModuleList", "torch.nn.ModuleDict",
           "torch.nn.Sequential", "torch.optim.Optimizer"}
MINIOCR_ALLOWED = ("miniocr.data", "miniocr.device")
# 数据包中混有机制答案、包装函数和从其他 miniocr 模块导入的实现。
# 只限制这些入口；standard_split、SFTExample 等数据契约仍可使用。
DATA_REFERENCE = {
    "latex_lm": {"train_tokenizer", "load_latex_lm", "BPETokenizer", "train_bpe",
                 "bits_per_byte", "evaluate_nll", "evaluate_lm", "evaluate_latex"},
    "sft": {"encode_sft", "collate_sft", "render_chat", "zero_shot"},
    "vlm_sft": {"answer_labels", "FormulaCollator", "audit_samples"},
    "latex_corpus": {"tokenize_latex", "apply_token_rules", "parse_latex", "serialize"},
    "image_tensor_corpus": {"render"},
}
# normalization_key 在多个数据模块中再导出；用于划分的内部调用不受影响。
DATA_NORMALIZERS = {"latex_corpus", "postcorrect", "dedup", "formula_images",
                    "contrastive_pairs", "image_tensor_corpus", "latex_images_ch32",
                    "vlm_sft", "distill_images"}
DATA_TORCH = {"symbols", "latex_lm", "sft", "vlm_sft", "image_tensors", "contrastive_pairs"}
DATA_FUNCTIONAL = {"symbols", "latex_lm", "image_tensors"}
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
                    if _banned(f"{n.module}.{a.name}")]          # 看完整的符号：from torch.optim import Optimizer 放行
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
    if full.startswith("miniocr.data."):
        parts = full.split(".")
        if parts[-1] == "*":                     # 星号导入会把参考实现/框架别名一起带入
            return True
        if len(parts) >= 4:
            module, symbol = parts[2:4]
            if (symbol in DATA_REFERENCE.get(module, ()) or
                    module in DATA_NORMALIZERS and symbol == "normalization_key"):
                return True
            # 按原始框架路径检查再导出的别名，既挡 Linear/CE，也保留 zeros/Module。
            if module in DATA_TORCH and symbol == "torch":
                return _banned(".".join(["torch"] + parts[4:]))
            if module in DATA_FUNCTIONAL and symbol == "F":
                return _banned(".".join(["torch", "nn", "functional"] + parts[4:]))
    if any(full == a or full.startswith(a + ".") for a in ALLOWED):
        return False
    if full == "miniocr" or full.startswith("miniocr."):          # 参考答案：只放行数据与设备工具
        return not any(full == a or full.startswith(a + ".") for a in MINIOCR_ALLOWED)
    if full.startswith("assignments.") and not full.startswith("assignments._common"):
        return True                                                # 不能 import 其他作业（可以复制自己的代码）
    return full in BANNED_TORCH or full.startswith(BANNED_PREFIX) or full in ("torch.optim", "torch.nn.functional")


def check_dir(code_dir):
    """返回 code_dir 下（含子目录）所有 .py 文件里的违规（字符串列表，已去重排序）。"""
    bad = []
    for root, dirs, files in os.walk(code_dir):                    # 递归：子包里的代码同样检查
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for f in sorted(files):
            if f.endswith(".py"):
                path = os.path.join(root, f)
                rel = os.path.relpath(path, code_dir)
                bad += violations(open(path, encoding="utf-8").read(), rel)
    return sorted(set(bad))


def mechanism_dirs(assignment_dir):
    """读作业目录下的 rules.toml，返回要受禁用检查的目录（绝对路径）。v4 的约定：

        mechanism = ["mechanism"]        # 机制实现目录：禁止现成的层、损失、优化器、注意力
        # 其他目录（框架实验部分）不检查，可以 import transformers、peft、trl 等

    没有 rules.toml 时，整个作业目录都当作机制实现。"""
    import tomllib
    path = os.path.join(assignment_dir, "rules.toml")
    if not os.path.exists(path):
        return [assignment_dir]
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    return [os.path.join(assignment_dir, d) for d in cfg.get("mechanism", [])]


def check_assignment(assignment_dir):
    """对 rules.toml 里列出的每个机制实现目录运行 check_dir，返回全部违规。"""
    bad = []
    for d in mechanism_dirs(assignment_dir):
        bad += [f"{os.path.basename(d)}/{b}" for b in check_dir(d)]
    return bad
