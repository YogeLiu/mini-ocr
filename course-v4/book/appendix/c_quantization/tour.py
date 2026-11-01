"""附录 C：量化。自写线性量化，算误差；GPTQ、AWQ 只用小矩阵演示思路。

python -m book.appendix.c_quantization.tour [部分号 1–7 ...]
"""
import argparse
from dataclasses import replace
import os

os.environ["HF_HUB_OFFLINE"] = "1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import torch

from book.appendix import emit
from miniocr.models.gpt import GPT
from miniocr.train.scaling import ladder_config


def qrange(bits):
    return -2**(bits-1), 2**(bits-1)-1


def quant_params(x, bits, symmetric=False):
    """返回 (scale, zero_point)。非对称：min、max 映到整数两端，并保证 0 在范围内。对称：zero_point=0。"""
    qmin, qmax = qrange(bits)
    if symmetric:
        scale = x.abs().max()/qmax
        return (scale if scale > 0 else torch.ones_like(scale)), torch.zeros_like(scale)
    lo, hi = x.min().clamp_max(0), x.max().clamp_min(0)
    scale = (hi-lo)/(qmax-qmin)
    if scale == 0:
        scale = torch.ones_like(scale)
    return scale, (qmin-lo/scale).round()+0.  # +0. 把 −0 变成 0


def quantize(x, scale, zero_point, bits):
    """q = round(x/scale + zero_point)，截断到整数范围。返回整数值（仍用浮点张量保存）。"""
    qmin, qmax = qrange(bits)
    return (x/scale+zero_point).round().clamp(qmin, qmax)


def dequantize(q, scale, zero_point):
    return (q-zero_point)*scale


def roundtrip(x, bits, symmetric=False):
    """量化再还原。返回 (还原值, scale, zero_point)。"""
    scale, zp = quant_params(x, bits, symmetric)
    return dequantize(quantize(x, scale, zp, bits), scale, zp), scale, zp


def error_stats(x, bits, symmetric=False):
    restored, scale, _ = roundtrip(x, bits, symmetric)
    e = restored-x
    return dict(scale=scale.item(), 上界=scale.item()/2, 最大误差=e.abs().max().item(),
                最大误差比上界=(e.abs().max()/(scale/2)).item(), MSE=e.square().mean().item(),
                均匀噪声MSE=scale.item()**2/12, 相对误差=(e.norm()/x.norm()).item())


def error_histogram(x, bits, bins=10):
    """误差除以 scale/2 落到 [−1, 1]，统计每格占比。"""
    restored, scale, _ = roundtrip(x, bits)
    return torch.histc((restored-x)/(scale/2), bins=bins, min=-1, max=1)/x.numel()


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return GPT(replace(ladder_config("tiny", 512, 64), attn_impl="manual")).eval()


@torch.no_grad()
def quantize_model(model, bits, symmetric=False):
    """就地替换矩阵权重（含嵌入），一维的 norm 不量化。"""
    for p in model.parameters():
        if p.ndim >= 2:
            p.copy_(roundtrip(p, bits, symmetric)[0])
    return model


def per_row_roundtrip(w, bits):
    """每行一个对称 scale。"""
    return torch.stack([roundtrip(row, bits, True)[0] for row in w])


def round_to_grid(w, step, qmax=None):
    q = (w/step).round()
    if qmax is not None:
        q = q.clamp(-qmax, qmax)
    return q*step


def output_error(error, H):
    """‖E X‖²= tr(E H Eᵀ)，H = X Xᵀ。E 的每一行对应一个输出。"""
    return torch.trace(error @ H @ error.T).item()


def gptq_step(w, H, col, step, qmax=None):
    """GPTQ 的一步：量化第 col 列，把它的误差传给后面还没量化的列。

    δ_F = −e · H[col, F] · H[F, F]⁻¹，使 ‖e x_col + δ_F X_F‖² 最小。返回新矩阵。
    """
    out = w.clone()
    out[:, col] = round_to_grid(w[:, col], step, qmax)
    rest = list(range(col+1, w.shape[1]))
    if rest:
        e = (out[:, col]-w[:, col]).unsqueeze(1)
        update = torch.linalg.solve(H[rest][:, rest], H[col, rest])
        out[:, rest] = w[:, rest]-e*update
    return out


def gptq(w, H, step, qmax=None):
    """逐列做完。这是思路的最小实现：每步解一次线性方程，没有批处理和 Cholesky 加速。"""
    for col in range(w.shape[1]):
        w = gptq_step(w, H, col, step, qmax)
    return w


def awq_scales(x, alpha):
    """X 形状 (通道, 样本)。每个输入通道的放大倍数 s = (平均|激活|)^α。"""
    return x.abs().mean(1).pow(alpha)


def awq_roundtrip(w, x, alpha, bits):
    """先放大重要通道的权重，再量化，再缩回去：Ŵ = Q(W·diag(s))·diag(s)⁻¹。"""
    s = awq_scales(x, alpha)
    return per_row_roundtrip(w*s, bits)/s


def awq_search(w, x, bits, alphas):
    """网格搜 α。返回 [(α, 输出误差)]。"""
    H = x @ x.T
    return [(alpha, output_error(awq_roundtrip(w, x, alpha, bits)-w, H)) for alpha in alphas]


def part1(_):
    x = torch.tensor([-1., 0., .5, 1.], dtype=torch.float64)
    for bits in (8, 4):
        for symmetric in (False, True):
            restored, scale, zp = roundtrip(x, bits, symmetric)
            emit(来源="数学结论", bits=bits, 对称=symmetric, 输入=x.tolist(), scale=scale.item(), zero_point=zp.item(),
                 整数=quantize(x, scale, zp, bits).tolist(), 还原=[round(v, 6) for v in restored.tolist()],
                 最大误差=(restored-x).abs().max().item(), 上界=scale.item()/2)
    emit(来源="数学结论", 整数范围=dict(int8=qrange(8), int4=qrange(4)), 级数=dict(int8=256, int4=16),
         每参数字节=dict(fp32=4, bf16=2, int8=1, int4=.5))


def part2(args):
    model = tiny_model(args.seed)
    emit(模型="tiny", V=512, 参数量=model.n_params(), 权重="随机初始化", 设备="cpu")
    for name, p in model.named_parameters():
        if name in ("tok.weight", "blocks.0.attn.q.weight", "blocks.0.mlp.gate.weight", "blocks.1.attn.o.weight"):
            for bits in (8, 4):
                emit(张量=name, bits=bits, 形状=list(p.shape), **error_stats(p.detach().double(), bits))
    for bits in (8, 4):
        ratio = max(error_stats(p.detach().double(), bits)["最大误差比上界"] for p in model.parameters() if p.ndim >= 2)
        emit(bits=bits, 全部矩阵最大误差比上界=ratio, 说明="不超过1即没有违反scale/2")


def part3(args):
    w = tiny_model(args.seed).tok.weight.detach().double()
    for bits in (8, 4):
        hist = error_histogram(w, bits)
        emit(张量="tok.weight", bits=bits, 误差分布10格=[round(v, 3) for v in hist.tolist()], 合计=hist.sum().item())
    emit(张量="tok.weight", 权重标准差=w.std().item(), 权重最大绝对值=w.abs().max().item(),
         最大绝对值比标准差=(w.abs().max()/w.std()).item())


def part4(args):
    w = tiny_model(args.seed).blocks[0].mlp.gate.weight.detach().double()
    for bits in (8, 4):
        emit(张量="mlp.gate", 场景="原样，张量一个scale", bits=bits, MSE=error_stats(w, bits, True)["MSE"])
    spiked = w.clone()
    spiked[0, 0] = w.abs().max()*10
    for bits in (8, 4):
        restored = per_row_roundtrip(spiked, bits)
        emit(张量="mlp.gate", 场景="一个权重放大到最大值10倍", bits=bits,
             整张量一个scale的MSE=error_stats(spiked, bits, True)["MSE"],
             每行一个scale的MSE=(restored-spiked).square().mean().item())


@torch.no_grad()
def part5(args):
    x = torch.randint(512, (2, 32), generator=torch.Generator().manual_seed(args.seed))
    base = tiny_model(args.seed)(x)[0]
    for bits in (8, 4):
        logits = quantize_model(tiny_model(args.seed), bits)(x)[0]
        emit(模型="tiny随机初始化", bits=bits, B=2, T=32, logits相对差=((logits-base).norm()/base.norm()).item(),
             argmax一致比例=(logits.argmax(-1) == base.argmax(-1)).double().mean().item(),
             说明="随机模型logits很小，只看量级趋势，不代表训练好的模型")


def part6(args):
    w = torch.tensor([[.6, .6]], dtype=torch.float64)
    H = torch.tensor([[1., .5], [.5, 1.]], dtype=torch.float64)
    plain = round_to_grid(w, 1.)
    one = gptq_step(w, H, 0, 1.)
    full = gptq(w, H, 1.)
    emit(来源="数学结论", W=w.tolist(), H=H.tolist(), 网格步长=1, 逐个取整=plain.tolist(), 逐个取整输出误差=output_error(plain-w, H),
         第一步之后=one.tolist(), GPTQ结果=full.tolist(), GPTQ输出误差=output_error(full-w, H))
    g = torch.Generator().manual_seed(args.seed)
    plain_sum, gptq_sum = 0., 0.
    for trial in range(5):
        latent = torch.randn(3, 64, generator=g, dtype=torch.float64)
        x = torch.randn(8, 3, generator=g, dtype=torch.float64) @ latent+.1*torch.randn(8, 64, generator=g, dtype=torch.float64)
        w = torch.randn(4, 8, generator=g, dtype=torch.float64)
        H = x @ x.T
        step = w.abs().max().item()/7
        plain_e = output_error(round_to_grid(w, step, 7)-w, H)
        gptq_e = output_error(gptq(w, H, step, 7)-w, H)
        plain_sum, gptq_sum = plain_sum+plain_e, gptq_sum+gptq_e
        emit(设备="cpu", 试验=trial, W形状=[4, 8], X形状=[8, 64], 级数="int4（±7）", 逐个取整=plain_e, GPTQ=gptq_e, 比=gptq_e/plain_e)
    emit(设备="cpu", 五次合计比=gptq_sum/plain_sum)


def part7(args):
    w = torch.tensor([[.1, 1.]], dtype=torch.float64)
    x = torch.tensor([[10., 0.], [0., .1]], dtype=torch.float64)
    for alpha in (0., 1.):
        restored = awq_roundtrip(w, x, alpha, 4)
        emit(来源="数学结论", W=w.tolist(), 通道平均绝对激活=x.abs().mean(1).tolist(), α=alpha, s=awq_scales(x, alpha).tolist(),
             还原=[[round(v, 6) for v in restored[0].tolist()]], 权重误差=[round(v, 6) for v in (restored-w)[0].tolist()],
             输出误差=output_error(restored-w, x @ x.T))
    g = torch.Generator().manual_seed(args.seed)
    w = torch.randn(4, 8, generator=g, dtype=torch.float64)
    magnitude = torch.tensor([20., 10., 5., 2., 1., .5, .2, .1], dtype=torch.float64)
    x = magnitude[:, None]*torch.randn(8, 64, generator=g, dtype=torch.float64)
    rows = awq_search(w, x, 4, [i/10 for i in range(11)])
    for alpha, error in rows:
        emit(设备="cpu", W形状=[4, 8], X形状=[8, 64], bits=4, α=alpha, 输出误差=error, 比α0=error/rows[0][1])
    emit(设备="cpu", 最优α=min(rows, key=lambda r: r[1])[0], 说明="α=0即普通每行量化；真实AWQ按模块搜索")


PARTS = {1: (part1, "手算量化（C.4.1）"),
         2: (part2, "tiny模型权重的误差和上界（C.4.2）"),
         3: (part3, "误差分布（C.4.2）"),
         4: (part4, "离群值和scale粒度（C.4.3）"),
         5: (part5, "量化后的logits（C.4.3）"),
         6: (part6, "GPTQ（C.4.4）"),
         7: (part7, "AWQ（C.4.5）")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if any(part not in PARTS for part in args.parts):
        parser.error("部分号须为1–7")
    torch.set_num_threads(2)
    emit(来源="环境", 附录="C", 种子=args.seed, torch版本=torch.__version__, 默认行为="CPU；不下载；不写文件；QLoRA无实验")
    for part in args.parts or PARTS:
        torch.manual_seed(args.seed)
        fn, title = PARTS[part]
        emit(来源="环境", 部分=part, 内容=title)
        fn(args)


if __name__ == "__main__":
    main()
