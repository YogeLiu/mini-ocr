"""第 17 章：解剖 GPT-2。离线、本地权重、fp32；默认 CPU 两线程，不训练模型。

python -m book.part3_post_training.ch17_gpt2.tour [部分号 1–11 ...]
"""
import argparse
from contextlib import contextmanager
import json
import math
import os
import re
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import torch
from torch.nn import functional as F

from miniocr.models.gpt2 import (
    GPT2Config, architecture_diff_rows, config_field_notes,
    convert_hf_gpt2_state_dict, find_local_gpt2, gelu_new, generate_greedy,
    load_gpt2_pretrained, n_params_formula,
)

TEXTS = ("The quick brown fox jumps over the lazy dog.",
         r"The formula is \frac{a+b}{\sqrt{x^2+1}} and \alpha+\beta=\gamma.")
_DEVICE = "cpu"


def emit(*, 来源="本项目观察", **values):
    """打印可摘录的 key=value；观察标记设备与线程，来源不混为数学结论。"""
    if 来源 == "本项目观察":
        values = dict(设备=_DEVICE, 线程=torch.get_num_threads(), **values)
    print("；".join(f"{k}={v:.8g}" if isinstance(v, float) else f"{k}={v}"
                   for k, v in dict(来源=来源, **values).items()), flush=True)


class Experiment:
    """按需加载并复用模型；单独运行任意部分也不依赖前面的实验。"""
    def __init__(self, args):
        self.args, self.path = args, find_local_gpt2()
        self._hf = self._own = self._tok = None

    def available(self):
        if self.path is None:
            emit(来源="待验证", 状态="跳过：未找到本地 GPT-2 权重；本章不会自动下载")
            return False
        return True

    def hf(self):
        if self._hf is None:
            from transformers import GPT2LMHeadModel
            self._hf = GPT2LMHeadModel.from_pretrained(
                str(self.path), local_files_only=True, attn_implementation="eager",
                dtype=torch.float32).to(self.args.device).eval()
            emit(HF注意力实现="eager", dtype="float32", 权重路径=self.path)
        return self._hf

    def own(self):
        if self._own is None:
            self._own, _ = load_gpt2_pretrained(self.path, self.args.device)
        return self._own

    def tok(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
        return self._tok

    def ids(self, text=TEXTS[0]):
        return self.tok()(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.args.device)

    @contextmanager
    def bug(self, name):
        """复用模型存储；故障演示退出后恢复权重及激活，支持乱序运行。"""
        model, hf = self.own(), self.hf()
        try:
            if name == "gelu":
                for block in model.blocks:
                    block.mlp.activation = "gelu"
            else:
                model.load_state_dict(convert_hf_gpt2_state_dict(hf.state_dict(), model.cfg, name), strict=True)
            yield model
        finally:
            for block in model.blocks:
                block.mlp.activation = model.cfg.activation
            if name != "gelu":
                model.load_state_dict(convert_hf_gpt2_state_dict(hf.state_dict(), model.cfg), strict=True)


def hidden_label(index, layers):
    """HF 边界编号：0 是嵌入；1..L-1 是上一块输出；L 是最终归一化后。"""
    return "嵌入相加" if index == 0 else (f"块{index-1}后（ln_f后）" if index == layers else f"块{index-1}后")


def compare(model, hf, idx, label, tolerance=1e-4):
    """对比全部 logits 和 HF 同口径 hidden；返回两边 logits。"""
    actual, hidden = model(idx, return_hidden=True)
    ref = hf(idx, output_hidden_states=True, use_cache=False)
    error = (actual-ref.logits).abs().max().item()
    scale = ref.logits.abs().max().item()
    emit(实验=label, logits最大绝对误差=error, 最大绝对logit=scale,
         相对logits量级=error/max(scale, 1e-30),
         下一token_top1一致=bool(torch.equal(actual[:, -1].argmax(-1), ref.logits[:, -1].argmax(-1))),
         全位置top1一致=bool(torch.equal(actual.argmax(-1), ref.logits.argmax(-1))))
    first = None
    for i, (a, b) in enumerate(zip(hidden, ref.hidden_states)):
        diff = (a-b).abs()
        emit(实验=label, hidden边界=i, 边界含义=hidden_label(i, model.cfg.n_layer), 最大绝对误差=diff.max().item())
        locations = (diff > tolerance).nonzero()
        if first is None and locations.numel():
            first = (i, *locations[0].tolist())
    emit(实验=label, hidden绝对容差=tolerance, 首个超差边界及批量位置通道=first,
         首个超差边界含义=hidden_label(first[0], model.cfg.n_layer) if first else "无")
    return actual, ref.logits


def hf_greedy(hf, idx, n_new):
    """固定步数且禁用 EOS 提前停止，与 generate_greedy 采用相同口径。"""
    return hf.generate(idx, attention_mask=torch.ones_like(idx), do_sample=False,
                       max_new_tokens=n_new, eos_token_id=None, pad_token_id=hf.config.eos_token_id,
                       use_cache=False)


def part1(exp):
    formula = n_params_formula(GPT2Config())
    for key, value in formula.items():
        emit(来源="数学结论", 默认GPT2参数项=key, 参数量=value)
    if not exp.available():
        return
    hf, own = exp.hf(), exp.own()
    # 等价于 print(model)，逐行加 key/value 和来源，方便把结构摘录进实验记录。
    for line in str(hf).splitlines():
        emit(HF结构=line)
    n = sum(p.numel() for p in hf.parameters())
    sd_n = sum(p.numel() for p in hf.state_dict().values())
    actual_formula = n_params_formula(own.cfg)["total"]
    own_n = sum(p.numel() for p in own.parameters())
    emit(HF_parameters去重=n, HF_state_dict未去重=sd_n, 差值=sd_n-n,
         重复项="lm_head.weight / transformer.wte.weight",
         state_dict中非参数元素=sum(t.numel() for k, t in hf.state_dict().items()
                                   if k not in dict(hf.named_parameters(remove_duplicate=False))),
         本章代码统计=own_n, 当前配置公式=actual_formula,
         三者一致=(n == own_n == actual_formula),
         head_wte同一存储=own.head.weight.data_ptr() == own.wte.weight.data_ptr())


def part2(exp):
    if not exp.available():
        return
    data = json.loads((exp.path/"config.json").read_text())
    notes = config_field_notes()
    for key, value in data.items():
        emit(字段=key, 值=json.dumps(value, ensure_ascii=False), 解释=notes.get(key, "待验证：未收录字段"),
             配置来源=exp.path/"config.json")


def safetensors_accounting(path):
    """只读 8 字节长度和 JSON 头；按形状数元素、按偏移数存储字节，不加载张量。"""
    file_bytes = path.stat().st_size
    with path.open("rb") as f:
        length = f.read(8)
        if len(length) != 8:
            raise ValueError("safetensors 长度字段不足 8 字节")
        header_bytes = int.from_bytes(length, "little")
        if header_bytes > file_bytes-8:
            raise ValueError("safetensors JSON 头长度超过文件大小")
        header = json.loads(f.read(header_bytes))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    buffers = {k: v for k, v in tensors.items()
               if re.fullmatch(r"(?:transformer\.)?h\.\d+\.attn\.(?:bias|masked_bias)", k)}
    params = {k: v for k, v in tensors.items() if k not in buffers}
    def storage_bytes(items):
        return sum(v["data_offsets"][1]-v["data_offsets"][0] for v in items.values())
    parameter_elements = sum(math.prod(v["shape"]) for v in params.values())
    parameter_bytes, buffer_bytes = storage_bytes(params), storage_bytes(buffers)
    return dict(
        tensor_count=len(tensors),
        has_lm_head=any(k.removeprefix("transformer.").startswith("lm_head.") for k in tensors),
        has_transformer_prefix=any(k.startswith("transformer.") for k in tensors),
        dtypes=sorted({v["dtype"] for v in tensors.values()}),
        attention_bias_shapes={k: tuple(v["shape"]) for k, v in buffers.items() if k.endswith(".attn.bias")},
        parameter_elements=parameter_elements, parameter_fp32_bytes=parameter_elements*4,
        parameter_bytes=parameter_bytes, buffer_bytes=buffer_bytes,
        header_bytes=header_bytes, file_bytes=file_bytes,
        file_equation_holds=(file_bytes == 8+header_bytes+parameter_bytes+buffer_bytes),
    )


def part3(exp):
    if not exp.available():
        return
    sd, cfg = exp.hf().state_dict(), exp.own().cfg
    converted = convert_hf_gpt2_state_dict(sd, cfg)
    mappings = [("transformer."+k, k, "照搬") for k in ("wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias")]
    mappings.append(("lm_head.weight", "head.weight", "与 wte 共享；不新增参数"))
    for suffix in ("weight", "bias"):
        for norm in ("ln_1", "ln_2"):
            mappings.append((f"transformer.h.0.{norm}.{suffix}", f"blocks.0.{norm}.{suffix}", "照搬"))
        for i, name in enumerate(("q", "k", "v")):
            mappings.append((f"transformer.h.0.attn.c_attn.{suffix}", f"blocks.0.attn.{name}.{suffix}",
                             f"沿输出维切分第{i}片"+("，再转置" if suffix == "weight" else "")))
        for source, target in (("attn.c_proj", "attn.proj"), ("mlp.c_fc", "mlp.fc"), ("mlp.c_proj", "mlp.proj")):
            mappings.append((f"transformer.h.0.{source}.{suffix}", f"blocks.0.{target}.{suffix}",
                             "转置 (in,out)→(out,in)" if suffix == "weight" else "照搬"))
    for source, target, op in mappings:
        emit(HF键=source, HF形状=tuple(sd[source].shape), 本章键=target,
             本章形状=tuple(converted[target].shape), 转换=op)
    emit(来源="数学结论", 说明="仅列第0层、嵌入、ln_f、head；其他层同理；Conv1D 权重是 (in,out)")
    path = exp.path/"model.safetensors"
    account = safetensors_accounting(path)
    emit(原始文件=path, 读取方式="仅 8 字节长度 + JSON 头，不加载张量",
         张量键总数=account["tensor_count"], 预期张量键总数=160,
         含lm_head键=account["has_lm_head"], 带transformer前缀=account["has_transformer_prefix"],
         dtype集合=account["dtypes"])
    emit(attn_bias缓冲区个数=len(account["attention_bias_shapes"]),
         attn_bias缓冲区形状=account["attention_bias_shapes"])
    emit(来源="数学结论", 说明="fp32 每元素 4 字节；元素数取自本地文件头形状（本项目观察）",
         参数元素总数乘4字节=f'{account["parameter_elements"]} × 4 = {account["parameter_fp32_bytes"]}')
    emit(参数字节数=account["parameter_bytes"], 缓冲区字节数=account["buffer_bytes"],
         JSON头长度=account["header_bytes"], 文件字节数=account["file_bytes"],
         参数存储等于fp32计算值=account["parameter_bytes"] == account["parameter_fp32_bytes"],
         等式="文件字节 = 8 + 头长度 + 参数字节 + 缓冲区字节",
         代入=f'{account["file_bytes"]} = 8 + {account["header_bytes"]} + '
              f'{account["parameter_bytes"]} + {account["buffer_bytes"]}',
         等式成立=account["file_equation_holds"])


def part4(exp):
    if exp.available():
        for text in TEXTS:
            emit(输入文本=repr(text))
            compare(exp.own(), exp.hf(), exp.ids(text), "正确转换")


def part5(exp):
    if not exp.available():
        return
    idx, own, hf = exp.ids(), exp.own(), exp.hf()
    a = own(idx)[:, -1].softmax(-1)
    b = hf(idx, use_cache=False).logits[:, -1].softmax(-1)
    for name, probabilities in (("本章GPT2", a), ("HF", b)):
        probs, ids = probabilities[0].topk(5)
        for rank, (p, i) in enumerate(zip(probs.tolist(), ids.tolist()), 1):
            emit(模型=name, 排名=rank, token_id=i, token=repr(exp.tok().decode([i])), 概率=p)
    emit(top5编号一致=torch.equal(a.topk(5).indices, b.topk(5).indices), 概率最大误差=(a-b).abs().max().item())
    x, y = generate_greedy(own, idx, 20), hf_greedy(hf, idx, 20)
    emit(续写token数=20, 贪心token完全一致=torch.equal(x, y), EOS策略="固定步数，不提前停止", KVcache=False)
    emit(模型="本章GPT2", 续写文本=repr(exp.tok().decode(x[0, idx.shape[1]:])))
    emit(模型="HF", 续写文本=repr(exp.tok().decode(y[0, idx.shape[1]:])))


def byte_decoder():
    """GPT-2 byte-to-Unicode 显示映射的逆；显示字符数不等于 UTF-8 原文字节数。"""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, extra = bs.copy(), 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256+extra)
            extra += 1
    return dict(zip(map(chr, cs), bs))


def part6(exp):
    from miniocr.data.latex_corpus import SAMPLE_FORMULAS, corpus_available, standard_split
    from miniocr.tokenizer import BPETokenizer, show_tokens, train_bpe
    local = corpus_available()
    start = time.perf_counter()
    split = (standard_split(n_train=256) if local else
             standard_split(n_train=256, n_val=4, n_test=4, formulas=SAMPLE_FORMULAS))
    train = split["train"]
    tok = BPETokenizer(train_bpe(train, 256), specials=())
    emit(BPE语料="本地语料 standard_split 训练集前缀" if local else "SAMPLE_FORMULAS 流程样例",
         训练条数=len(train), 实际合并次数=len(tok.merges), 词表大小=len(tok),
         划分种子=0, 读取划分及BPE秒=time.perf_counter()-start,
         说明="仅训练小分词器，不训练语言模型；示例切分不能代表泛化压缩率")
    ref = exp.tok() if exp.available() else None
    decoder = byte_decoder()
    for text in ("模型读入图像，写出文字。", "模", "hello world",
                 r"\frac{a+b}{\sqrt{x^2+1}}", r"\alpha+\beta=\gamma", r"\text{面积}=ab"):
        pieces = [tok.vocab[i] for i in tok.encode(text)]
        emit(分词器="第6章BPE", 文本=repr(text), 切分=show_tokens(pieces), token数=len(pieces),
             每token字节数=[len(p) for p in pieces], 总UTF8字节数=len(text.encode()))
        if ref is not None:
            ids = ref.encode(text, add_special_tokens=False)
            tokens = ref.convert_ids_to_tokens(ids)
            raw = [bytes(decoder[c] for c in t) for t in tokens]
            emit(分词器="GPT-2", 文本=repr(text), 词表记号=tokens, token数=len(ids),
                 每token字节数=[len(p) for p in raw], 原始字节hex=[p.hex() for p in raw],
                 字节往返一致=b"".join(raw) == text.encode())
    emit(来源="数学结论", 记号="Ġ", 对应原始字节=decoder["Ġ"], 含义="空格；不是原文中的拉丁字母",
         UTF8说明="一个汉字的 UTF-8 字节可能跨 token；不可用逐 token 解码后的替换字符计字节")


def part7(exp):
    emit(Qwen3说明="直接读取 Qwen3-0.6B 生成模型；只读本地配置和权重 keys，不加载张量")
    for row in architecture_diff_rows():
        emit(项目=row["field"], **{name: f"{row[name]['value']} [来源:{row[name]['source']}]"
                                for name in ("gpt2", "book", "qwen3")})


def part8(exp):
    if not exp.available():
        return
    sd, model = exp.hf().state_dict(), exp.own()
    # 先实测非方阵漏转置的加载失败；只实例化一个 Linear，避免再复制整个模型。
    d = model.cfg.n_embd
    for key, in_features, out_features in (("attn.c_attn", d, 3*d),
                                           ("mlp.c_fc", d, 4*d),
                                           ("mlp.c_proj", 4*d, d)):
        w = sd[f"transformer.h.0.{key}.weight"]
        # 此处先保留 c_attn 的合并形式，单独展示 Conv1D→Linear 的转置必要性。
        probe = torch.nn.Linear(in_features, out_features, bias=False)
        try:
            probe.load_state_dict({"weight": w}, strict=True)
        except RuntimeError as exc:
            emit(故障="原始非方阵未转置", HF键=key, 原始形状=tuple(w.shape),
                 目标形状=tuple(probe.weight.shape), 异常=type(exc).__name__, 摘要=" ".join(str(exc).split())[:360])
    emit(故障="no_transpose", 静默层="所有 attn.c_proj", 形状=tuple(sd["transformer.h.0.attn.c_proj.weight"].shape),
         说明="原始完整正方阵不转置仍能 strict 加载；c_attn/c_fc/mlp.c_proj 保持正确转换")
    with exp.bug("no_transpose") as bad:
        compare(bad, exp.hf(), exp.ids(), "no_transpose")


def part9(exp):
    if exp.available():
        with exp.bug("qkv_order") as bad:
            compare(bad, exp.hf(), exp.ids(), "qkv_order")


def part10(exp):
    x = torch.linspace(-5, 5, 10001, dtype=torch.float64)
    delta = (gelu_new(x)-F.gelu(x)).abs()
    emit(来源="数学结论", 精确式="x/2*(1+erf(x/sqrt(2)))", 近似式="x/2*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))")
    emit(扫描设备="cpu", 扫描dtype="float64", 网格点数=x.numel(), 网格范围="[-5,5]", 网格最大差=delta.max().item(),
         最大差处x=x[delta.argmax()].item(), 说明="有限网格数值验证，不是连续域最大值证明")
    if not exp.available():
        return
    idx, hf = exp.ids(), exp.hf()
    reference = hf_greedy(hf, idx, 20)
    with exp.bug("gelu") as bad:
        compare(bad, hf, idx, "精确GELU")
        generated = generate_greedy(bad, idx, 20)
        emit(实验="精确GELU", 贪心token完全一致=torch.equal(generated, reference), 续写token数=20,
             续写文本=repr(exp.tok().decode(generated[0, idx.shape[1]:])),
             HF续写文本=repr(exp.tok().decode(reference[0, idx.shape[1]:])))


def part11(exp):
    if not exp.available():
        return
    model, hf = exp.own(), exp.hf()
    P = model.cfg.n_positions
    idx = torch.zeros((1, P), dtype=torch.long, device=exp.args.device)
    out = model(idx)
    emit(模型="本章GPT2", 序列长度=P, logits形状=tuple(out.shape), 全部有限=bool(out.isfinite().all()))
    del out
    idx = torch.zeros((1, P+1), dtype=torch.long, device=exp.args.device)
    for name, fn in (("本章GPT2", lambda: model(idx)), ("HF", lambda: hf(idx, use_cache=False))):
        try:
            out = fn()
            emit(模型=name, 序列长度=P+1, 结果="未报错", 说明="本地版本的实际行为")
            del out
        except Exception as exc:
            emit(模型=name, 序列长度=P+1, 异常类型=type(exc).__name__, 摘要=" ".join(str(exc).split())[:300])


PARTS = {
    1: ("模型结构与参数账", part1), 2: ("配置逐字段解读", part2),
    3: ("权重映射", part3), 4: ("logits 与残差流对拍", part4),
    5: ("下一 token 与贪心续写", part5), 6: ("字节级 BPE 与 LaTeX", part6),
    7: ("三种骨架的差异", part7), 8: ("故障 A：漏转置", part8),
    9: ("故障 B：qkv 顺序", part9), 10: ("故障 C：GELU 版本", part10),
    11: ("故障 D：位置越界", part11),
}
SECTIONS = {
    1: "17.1–17.2", 2: "17.3", 3: "17.6", 4: "17.7", 5: "17.7",
    6: "17.9", 7: "17.10", 8: "17.8.1", 9: "17.8.2", 10: "17.8.3", 11: "17.8.4",
}


def main():
    global _DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if any(p not in PARTS for p in args.parts):
        parser.error("部分号须在 1 至 11 之间")
    # 正文与验收以 CPU fp32 为准；其他设备只在显式选择且可用时运行。
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 设备不可用：请检查 PyTorch CUDA 支持、驱动及 CUDA_VISIBLE_DEVICES，或使用 --device cpu")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS 设备不可用：需要支持 MPS 的 PyTorch 和 macOS 设备，或使用 --device cpu")
    if args.device != "cpu":
        print("待验证：非 CPU 设备的误差量级未在本章标定", flush=True)
    _DEVICE = args.device
    torch.set_num_threads(2)
    start = time.perf_counter()
    exp = Experiment(args)
    import transformers
    emit(torch版本=torch.__version__, transformers版本=transformers.__version__, seed=args.seed,
         离线=True, HF注意力="eager", 本章注意力="显式矩阵乘+因果mask", dtype="float32")
    with torch.inference_mode():
        for p in args.parts or PARTS:
            torch.manual_seed(args.seed)
            title, fn = PARTS[p]
            emit(部分=p, 对应正文=SECTIONS[p], 标题=title)
            t = time.perf_counter()
            fn(exp)
            emit(部分=p, 用时秒=time.perf_counter()-t)
    emit(tour总用时秒=time.perf_counter()-start)


if __name__ == "__main__":
    main()
