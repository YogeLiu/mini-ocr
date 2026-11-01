"""第 15 章：KV cache 与推理效率。离线、不下载、不训练，默认 CPU 两线程。

python -m book.part2_lm.ch15_kvcache.tour [部分号 1–9 ...] [--seed 0] [--ckpt PATH]
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import os
from pathlib import Path
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import torch

from miniocr.device import pick_device, synchronize
from miniocr.generate import generate
from miniocr.kvcache import (
    BUGS, cached_mask, prefill_chunks, cache_nbytes, kv_cache_bytes,
    kv_cache_bytes_for, inject_bug, local_model_configs, intensity_sweep,
    time_decode, time_prefill,
)
from miniocr.models.gpt import GPT, GPTConfig, apply_rope
from miniocr.train.memory import fmt_bytes
from miniocr.train.scaling import ladder_config


def _format(value):
    if isinstance(value, float):
        return f"{value:.6g}"  # 保留微小 logits 差，避免打印成 0.0000。
    return str(value)


def emit(**values):
    """统一键=值；差值用有效数字，计时单位显式标注。"""
    print("；".join(f"{key}={_format(value)}" for key, value in values.items()), flush=True)


def difference(a, b):
    return (a-b).abs().max().item()


class Experiment:
    def __init__(self, args):
        self.args, self.model = args, None

    def small(self, **kwargs):
        values = dict(vocab_size=128, d_model=64, n_head=4, n_kv_head=2,
                      n_layer=2, block_size=64, attn_impl="sdpa")
        values.update(kwargs)
        torch.manual_seed(self.args.seed)
        return GPT(GPTConfig(**values)).to(self.args.device).eval()

    def prompt(self, model, T=10, B=1):
        return torch.randint(model.cfg.vocab_size, (B, T), device=self.args.device)

    def prepare(self):
        """只恢复检查点结构、权重和 BOS/EOS；不加载语料或分词器，不训练。"""
        if self.model is not None:
            return
        torch.manual_seed(self.args.seed)
        if self.args.ckpt:
            saved = torch.load(self.args.ckpt, map_location="cpu", weights_only=False)
            identity = saved["identity"]
            self.model = GPT(GPTConfig(**identity["model"]))
            self.model.load_state_dict(saved["model"])
            self.bos, self.eos = identity["bos"], identity["eos"]
            if not all(isinstance(x, int) and 0 <= x < self.model.cfg.vocab_size
                       for x in (self.bos, self.eos)):
                raise ValueError("检查点 BOS/EOS 必须是词表内的整数编号")
            emit(模型来源=str(self.args.ckpt), BOS=self.bos, EOS=self.eos)
        else:
            # 第 13 章 tiny 的层数/宽度/头数；演示词表，不依赖分词器。
            self.model = GPT(ladder_config("tiny", 256, 256))
            self.bos, self.eos = 0, None
            emit(模型来源="第13章 tiny 骨架，演示词表256、上下文256；随机初始化",
                 说明="未定义真实分词器；0 充当 BOS，禁用 EOS 以观察完整长度")
        self.model = self.model.to(self.args.device).eval()
        emit(模型配置=asdict(self.model.cfg))


@torch.no_grad()
def attention_trace(model, idx):
    """hook 捕获每层输入，重新算 norm 后、RoPE 后的 K 和未旋转的 V，并保留注意力输出。"""
    rows, handles = [], []

    def hook(attn, args, output):
        x, cos, sin = args[:3]
        B, T, _ = x.shape
        k = attn.k(x).view(B, T, attn.hk, attn.dh).transpose(1, 2)
        if attn.k_norm is not None:
            k = attn.k_norm(k)
        k = apply_rope(k, cos, sin)
        v = attn.v(x).view(B, T, attn.hk, attn.dh).transpose(1, 2)
        rows.append(dict(k=k, v=v, output=output))

    try:
        handles = [b.attn.register_forward_hook(hook) for b in model.blocks]
        logits = model(idx)[0]
    finally:
        for handle in handles:
            handle.remove()
    return logits, rows


@torch.no_grad()
def chunk_logits(model, idx, sizes, wrong_pos=False):
    """故意 pos0=0 的对照；正确路径使用库函数。"""
    if not wrong_pos:
        return prefill_chunks(model, idx, sizes)
    caches, parts, pos = model.new_caches(), [], 0
    for size in sizes:
        parts.append(model(idx[:, pos:pos+size], caches=caches, pos0=0)[0])
        pos += size
    return torch.cat(parts, 1), caches


@torch.no_grad()
def decode_difference(model, prompt, steps=20, wrong_pos=False):
    """用无 cache 贪心路径作共同输入；每次增量前向与同一前缀整段重算比较。

    先完整预填充，再处理 steps 个 token；避免序列分叉后把输入差异混进缓存误差。
    """
    reference = generate(model, prompt, steps, greedy=True, use_cache=False)
    caches = model.new_caches()
    initial = model(prompt, caches=caches)[0]
    error = difference(initial, model(prompt)[0])
    for pos in range(prompt.shape[1], reference.shape[1]):
        actual = model(reference[:, pos:pos+1], caches=caches,
                       pos0=0 if wrong_pos else pos)[0][:, -1]
        full = model(reference[:, :pos+1])[0][:, -1]
        error = max(error, difference(actual, full))
    return error


@torch.no_grad()
def cached_greedy(model, prompt, steps=20, wrong_pos=False):
    """独立贪心轨迹，用于记录首次分叉步；位置错误由调用处直接传 pos0=0。"""
    caches, out = model.new_caches(), prompt
    for step in range(steps):
        logits = (model(out, caches=caches)[0] if step == 0 else
                  model(out[:, -1:], caches=caches, pos0=0 if wrong_pos else out.shape[1]-1)[0])
        out = torch.cat([out, logits[:, -1].argmax(-1, keepdim=True)], 1)
    return out


def part1(exp):
    model = exp.small(vocab_size=32, d_model=8, n_head=2, n_kv_head=1,
                      block_size=16, attn_impl="manual")
    idx = exp.prompt(model, 5)
    a, old = attention_trace(model, idx[:, :4])
    b, new = attention_trace(model, idx)
    for layer, (before, after) in enumerate(zip(old, new), 1):
        emit(层=layer, 旧K形状=tuple(before["k"].shape), 新K形状=tuple(after["k"].shape),
             旧V形状=tuple(before["v"].shape), 新V形状=tuple(after["v"].shape),
             前4位置K最大差=difference(before["k"], after["k"][:, :, :4]),
             前4位置V最大差=difference(before["v"], after["v"][:, :, :4]),
             前4位置注意力输出最大差=difference(before["output"], after["output"][:, :4]))
    emit(旧logits形状=tuple(a.shape), 新logits形状=tuple(b.shape),
         前4位置logits最大差=difference(a, b[:, :4]),
         说明="因果性使新 token 不改变旧位置 K/V 与输出；第5位置只用自己的 Q，旧 Q 无需缓存")


def part2(exp):
    model = exp.small(qk_norm=True)
    idx = exp.prompt(model)
    _, trace = attention_trace(model, idx)
    _, caches = prefill_chunks(model, idx, [10])
    for layer, (cache, row) in enumerate(zip(caches, trace), 1):
        emit(阶段="预填充", 层=layer, K形状=tuple(cache["k"].shape), V形状=tuple(cache["v"].shape),
             与投影归一化RoPE后K最大差=difference(cache["k"], row["k"]),
             kv_heads=model.cfg.kv_heads(), n_head=model.cfg.n_head)
    model(exp.prompt(model, 1), caches=caches, pos0=10)
    for layer, cache in enumerate(caches, 1):
        emit(阶段="解码1步", 层=layer, K形状=tuple(cache["k"].shape), V形状=tuple(cache["v"].shape))
    emit(说明="缓存 K 已加 RoPE，V 不旋转；两者都在 repeat_kv 之前存储")


def part3(exp):
    for impl in ("manual", "sdpa"):
        for name, hk, norm in (("MHA", 4, False), ("GQA", 2, False), ("GQA+qk_norm", 2, True)):
            model = exp.small(n_kv_head=hk, qk_norm=norm, attn_impl=impl)
            prompt = exp.prompt(model, B=2)
            a = generate(model, prompt, 30, greedy=True, use_cache=True)
            b = generate(model, prompt, 30, greedy=True, use_cache=False)
            emit(配置=name, attention=impl, B=2, 新token数=30, 贪心完全相同=torch.equal(a, b),
                 逐步logits最大差=decode_difference(model, prompt, 30))


def part4(exp):
    emit(T_past=3, T_new=2, mask_01=cached_mask(2, 3).int().tolist())
    for impl in ("manual", "sdpa"):
        model = exp.small(attn_impl=impl)
        idx = exp.prompt(model)
        full = model(idx)[0]
        for sizes in ([10], [6, 4], [3, 3, 4], [4, 1, 5], [1]*10):
            actual, _ = prefill_chunks(model, idx, sizes)
            emit(attention=impl, 切法=sizes, 全部logits最大差=difference(actual, full))
    emit(说明="mask 决定看谁，pos0 决定 RoPE 位置；分块时二者都要延续历史")


def part5(exp):
    model = exp.small(qk_norm=True, tie_weights=False)
    prompt = exp.prompt(model)
    full = model(prompt)[0]
    good_bytes = cache_nbytes(prefill_chunks(model, prompt, [6, 4])[1])
    reference = generate(model, prompt, 20, greedy=True, use_cache=False)
    for bug in ("正确", "pos0=0", *BUGS):
        with inject_bug(model, bug) if bug in BUGS else nullcontext():
            logits, caches = chunk_logits(model, prompt, [6, 4], wrong_pos=bug == "pos0=0")
            error = decode_difference(model, prompt, 20, wrong_pos=bug == "pos0=0")
            generated = cached_greedy(model, prompt, 20, wrong_pos=bug == "pos0=0")
            different = (generated[:, 10:] != reference[:, 10:]).any(0).nonzero()
            first = int(different[0, 0])+1 if len(different) else "无"
            emit(实现=bug, 分段预填充logits最大差=difference(logits, full),
                 逐token解码logits最大差=error, 贪心首次分叉步=first,
                 分段后缓存字节=cache_nbytes(caches), 与正确缓存字节比=cache_nbytes(caches)/good_bytes)
    emit(说明="随机初始化 logits 很小，贪心序列可能碰巧相同，所以要看 logits 差；"
         "解码差在同一参考前缀上测，分叉另走独立贪心轨迹",
         故障边界="mask_offset 只影响分段预填充；no_cache_update 第二块输出仍正确，但第三次前向开始丢历史")


def part6(exp):
    tiny = exp.small(**asdict(ladder_config("tiny", 256, 32)))
    if exp.args.ckpt:
        exp.prepare()
        tiny = exp.model
    for label, model in (("项目GQA小模型", exp.small()), ("第13章检查点" if exp.args.ckpt else "第13章tiny", tiny)):
        T, B = min(10, model.cfg.block_size), 2
        _, caches = prefill_chunks(model, exp.prompt(model, T, B), [T])
        expected = kv_cache_bytes_for(model.cfg, T, B, 4)
        actual = cache_nbytes(caches)
        emit(结论="数学结论与张量对拍", 模型=label, dtype="fp32", T=T, B=B,
             公式字节=expected, 实际字节=actual, 相等=expected == actual)
    for cfg in local_model_configs():
        emit(结论="数学结论", 模型=cfg.name, 来源=cfg.source, 说明=cfg.note,
             总层数=cfg.n_layer, full_attention层数=cfg.full_layers,
             n_head=cfg.n_head, kv_heads=cfg.kv_heads(), head_dim=cfg.head_dim)
        for label, hk in (("原配置GQA", cfg.kv_heads()), ("若不用GQA", cfg.n_head), ("若MQA", 1)):
            layer_cases = [("仅full_attention", cfg.full_layers)]
            if cfg.full_layers != cfg.n_layer:
                layer_cases.append(("假如全部层都是full_attention", cfg.n_layer))
            for scope, layers in layer_cases:
                emit(结论="数学结论", 模型=cfg.name, 对照=label, 层口径=scope, L=layers,
                     dtype="bf16", 每token字节=kv_cache_bytes(layers, hk, cfg.head_dim, 1))
                for T, B in ((4096, 1), (4096, 32), (32768, 1)):
                    size = kv_cache_bytes(layers, hk, cfg.head_dim, T, B)
                    emit(结论="数学结论", 模型=cfg.name, 对照=label, 层口径=scope,
                         T=T, B=B, KV字节=size, KV容量=fmt_bytes(size))


def part7(exp):
    qwen = local_model_configs()[0]
    emit(模型=qwen.name, 来源=qwen.source, 说明=qwen.note,
         hidden_size=qwen.d_model, 注意力宽度=qwen.n_head*qwen.head_dim,
         公式说明="真实 Qwen 用 h·dh 计算注意力宽度；小 GPT 中 h·dh=d",
         口径="权重每步读一次，历史与新KV各读一次、新KV写一次；忽略激活/softmax/cat复制，见step_cost docstring")
    for name, cfg in (("本项目GQA小模型", exp.small().cfg), (qwen.name, qwen)):
        for row in intensity_sweep(cfg):
            emit(结论="数学结论（粗算，假设见 docstring）", 模型=name, dtype="bf16",
                 阶段=row["phase"], T_new=row["T_new"], T_past=row["T_past"], B=row["B"],
                 FLOPs=row["flops"], 访存字节=row["bytes"], 算术强度FLOPs每字节=row["intensity"])


def cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "未知"


def part8(exp):
    if exp.args.ckpt:
        exp.prepare()
        model = exp.model
    else:
        model = exp.small(d_model=256, n_head=4, n_kv_head=2, n_layer=4, block_size=1024)
    emit(结论="本项目观察", 设备=exp.args.device, CPU型号=cpu_name(), 线程=2,
         torch版本=torch.__version__, 模型配置=asdict(model.cfg), 重复次数=7, 预热次数=1,
         计时口径="单步不含预填充；每轮重建缓存；FP32；B=1；输入为重复token 0")
    limit = model.cfg.block_size
    lengths = [n for n in (64, 128, 256, 512, 1024) if n <= limit]
    if not lengths and limit >= 2:
        lengths = [limit]
    for total in lengths:
        for cache in (False, True):
            ms = time_decode(model, total-1, use_cache=cache)
            emit(结论="本项目观察", 阶段="再生成1个token", 当前总长度=total, 已缓存历史=total-1,
                 use_cache=cache, 毫秒每token=ms, 设备=exp.args.device, 线程=2)
    for length in sorted({min(n, limit) for n in (64, 256, 1024)}):
        ms = time_prefill(model, length)
        emit(结论="本项目观察", 阶段="预填充TTFT近似", 提示长度=length,
             毫秒=ms, 毫秒每token=ms/length, 设备=exp.args.device, 线程=2)
    emit(解释="只代表本机所选设备与实现；CPU 调度、权重驻留及动态缓存复制都会影响曲线，不推广到GPU")


def part9(exp):
    exp.prepare()
    model = exp.model
    prompt = torch.tensor([[exp.bos]], device=exp.args.device)
    outputs = []
    max_new = model.cfg.block_size - 1
    emit(生成上限=max_new, 起点="BOS", EOS=exp.eos,
         计时口径="端到端：含预填充和所有解码步骤；每条路径预热短前缀后完整测一次")
    for cache in (False, True):
        generate(model, prompt, min(4, max_new), greedy=True, eos=exp.eos, use_cache=cache)
        synchronize(exp.args.device)
        start = time.perf_counter()
        out = generate(model, prompt, max_new, greedy=True, eos=exp.eos, use_cache=cache)
        synchronize(exp.args.device)
        elapsed = time.perf_counter()-start
        count = out.shape[1]-1
        outputs.append(out)
        emit(use_cache=cache, 生成token数=count, 总耗时秒=elapsed,
             毫秒每token=1000*elapsed/count if count else "无生成token",
             设备=exp.args.device, 线程=2)
    emit(输出完全相同=torch.equal(*outputs),
         说明="仅比较 token 与耗时；无检查点时不能将随机 token 解释为有效公式")


PARTS = {1: (part1, "重复计算在哪里（15.1）"),
         2: (part2, "缓存里有什么（15.2）"),
         3: (part3, "对拍：带 cache 与不带 cache（15.3）"),
         4: (part4, "分段预填充：mask 与 pos0（15.4）"),
         5: (part5, "故障演练（15.9）"),
         6: (part6, "显存估算（15.5、15.6）"),
         7: (part7, "预填充与解码的算术强度（15.7）"),
         8: (part8, "CPU 计时：每 token 时间随长度变化（15.7）"),
         9: (part9, "长公式生成速度（15.8）")}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=Path)
    args = parser.parse_args(argv)
    if any(part not in PARTS for part in args.parts):
        parser.error("部分号须为1–9")
    if args.seed < 0:
        parser.error("seed 须非负")
    if args.ckpt is not None and not args.ckpt.is_file():
        parser.error(f"检查点不存在：{args.ckpt}")
    try:
        pick_device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))
    return args


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    emit(章节=15, 设备=args.device, 线程=2, 种子=args.seed, 默认行为="CPU；离线；不下载；不训练")
    emit(小模型说明="随机初始化，只验证数值一致性，不代表质量；计时只观察当前实现趋势")
    start = time.perf_counter()
    exp = Experiment(args)
    for part in args.parts or PARTS:
        fn, title = PARTS[part]
        emit(部分=part, 内容=title)
        synchronize(args.device)
        t0 = time.perf_counter()
        fn(exp)
        synchronize(args.device)
        emit(部分=part, 用时秒=time.perf_counter()-t0, 设备=args.device, 线程=2)
    emit(总用时秒=time.perf_counter()-start, 设备=args.device, 线程=2)


if __name__ == "__main__":
    main()
