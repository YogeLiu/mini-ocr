"""第 14 章：解码策略；不含 KV cache（第 15 章）。

过滤器输入均为 (B, V)，以 −inf 屏蔽 token，不修改调用者的 logits。
这里独立实现教学流水线，旧 miniocr.generate 的接口与行为保持不变。
"""
from dataclasses import dataclass
import math

import torch


PROCESS_ORDER = ("重复惩罚", "温度", "top-k", "top-p", "min-p")


def apply_repetition_penalty(logits, prev, penalty):
    """CTRL（Keskar et al., 2019）/ HF 的正除负乘约定。

    penalty>1 时已出现 logit 不增；0<penalty<1 则鼓励重复。
    prev 是 (B,t)，按集合处理，同一编号出现多次也只变换一次。
    """
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty 须为正有限数")
    out = logits.clone()
    if penalty == 1 or prev is None or prev.numel() == 0:
        return out
    seen = logits.gather(1, prev)  # 总是读取原值；重复索引写回的是同一个变换值。
    changed = torch.where(seen > 0, seen / penalty, seen * penalty)
    return out.scatter(1, prev, changed)


def apply_temperature(logits, temperature):
    """温度缩放不改变最大值位置；要贪心请用 greedy。"""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature 须为正有限数；要贪心请用 greedy")
    return logits / temperature


def top_k_filter(logits, k):
    """保留第 k 大值及全部并列；k<=0 或 k>=V 不过滤。"""
    if not isinstance(k, int):
        raise ValueError("k 须为整数")
    if k <= 0 or k >= logits.shape[-1]:
        return logits.clone()
    cutoff = logits.topk(k, dim=-1).values[:, -1:]
    return logits.masked_fill(logits < cutoff, -torch.inf)


def top_p_filter(logits, p, min_keep=1):
    """保留累计概率首次达到 p 的最小前缀，至少 min_keep 个。"""
    if not 0 < p <= 1 or not isinstance(min_keep, int) or min_keep < 1:
        raise ValueError("p 须在 (0,1]，min_keep 须为正整数")
    if p == 1:
        return logits.clone()
    ordered, indices = logits.sort(dim=-1, descending=True)
    prob = ordered.softmax(-1)
    cum_before = prob.cumsum(-1) - prob
    # 浮点舍入可能在恰好相等的边界多/少留一个；不人为添加 epsilon。
    keep = cum_before < p
    keep[:, :min_keep] = True
    keep = torch.zeros_like(keep).scatter(1, indices, keep)  # 回到原 token 编号！
    return logits.masked_fill(~keep, -torch.inf)


def min_p_filter(logits, min_p):
    """选做：Nguyen et al. (2024) min-p，保留 prob>=min_p*max(prob)。"""
    if not 0 <= min_p <= 1:
        raise ValueError("min_p 须在 [0,1]")
    if min_p == 0:
        return logits.clone()
    prob = logits.softmax(-1)
    return logits.masked_fill(prob < min_p * prob.amax(-1, keepdim=True), -torch.inf)


@dataclass
class DecodeConfig:
    greedy: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0

    def __post_init__(self):
        if not isinstance(self.greedy, bool) or not isinstance(self.top_k, int):
            raise ValueError("greedy 须为 bool，top_k 须为整数（<=0 不过滤）")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature 须为正有限数；要贪心请用 greedy")
        if not 0 < self.top_p <= 1 or not 0 <= self.min_p <= 1:
            raise ValueError("top_p 须在 (0,1]，min_p 须在 [0,1]")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty 须为正有限数")


def process_logits(logits, cfg, prev=None, trace=None):
    """固定顺序：重复惩罚 → 温度 → top-k → top-p → min-p。

    重复惩罚最先，因为它改的是原始 logit 的大小。温度在截断之前，
    使 top-p 的集合由温度调整后的分布决定。trace 列表保存逐阶段副本。
    greedy 只做重复惩罚；温度与截断不改变 argmax，故跳过。
    并列最大值仍由 argmax 取原词表中第一个位置。
    """
    out = logits.clone()
    if isinstance(trace, list):
        trace.append(("原始", out.clone()))
    stages = [lambda x: apply_repetition_penalty(x, prev, cfg.repetition_penalty)]
    if not cfg.greedy:
        stages += [lambda x: apply_temperature(x, cfg.temperature),
                   lambda x: top_k_filter(x, cfg.top_k),
                   lambda x: top_p_filter(x, cfg.top_p),
                   lambda x: min_p_filter(x, cfg.min_p)]
    for name, transform in zip(PROCESS_ORDER, stages):
        out = transform(out)
        if isinstance(trace, list):
            trace.append((name, out.clone()))
    return out


def select_next(logits, cfg, prev=None, generator=None):
    """返回 (B,)；独立 generator 让采样可复现。每行至少须有一个可选 token。"""
    out = process_logits(logits, cfg, prev)
    if cfg.greedy:
        return out.argmax(-1)
    return torch.multinomial(out.softmax(-1), 1, generator=generator)[:, 0]


@torch.no_grad()
def sample_sequences(model, idx, max_new, cfg, eos=None, banned=None, generator=None):
    """每步整段重算，不用 cache；返回含提示的 (B,T) 序列。

    超过 block_size 时只送最后一个窗口，RoPE 位置从 0 重新编号。
    重复惩罚使用全部已有序列（含提示），与 HF 处理 input_ids 的约定一致。
    EOS 行以后补 EOS，全部结束提前停；banned 是禁止输出的编号列表。
    """
    if max_new < 0 or idx.ndim != 2 or idx.shape[1] == 0:
        raise ValueError("需要非空二维提示和非负 max_new")
    model.eval()
    out = idx.clone()
    done = torch.zeros(len(idx), dtype=torch.bool, device=idx.device)
    for _ in range(max_new):
        logits = model(out[:, -model.cfg.block_size:])[0][:, -1].float().clone()
        if banned:
            logits[:, banned] = -torch.inf
        nxt = select_next(logits, cfg, prev=out, generator=generator)
        if eos is not None:
            nxt = torch.where(done, eos, nxt)
            done |= nxt == eos
        out = torch.cat([out, nxt[:, None]], dim=1)
        if eos is not None and done.all():
            break
    return out


def gpt_step_fn(model):
    """包装为 step_fn：同一步前缀等长，直接堆叠；超长窗口位置重置为 0。"""
    @torch.no_grad()
    def step_fn(prefixes):
        model.eval()
        device = next(model.parameters()).device
        idx = torch.tensor([s[-model.cfg.block_size:] for s in prefixes], device=device)
        return model(idx)[0][:, -1].float().log_softmax(-1)
    return step_fn


def gnmt_length_penalty(length, alpha):
    """Wu et al. (2016) GNMT 式；长度为生成 token 数，含 EOS、不含提示。"""
    if length < 0 or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("length 与 alpha 须非负，alpha 须有限")
    return ((5 + length) / 6) ** alpha


@dataclass
class Hypothesis:
    tokens: list[int]  # 不含提示；含 EOS（若有）。
    logprob: float
    length: int
    score: float
    finished: bool
    truncated: bool


def _hypothesis(tokens, logprob, eos, alpha):
    finished = bool(tokens) and eos is not None and tokens[-1] == eos
    return Hypothesis(tokens, logprob, len(tokens),
                      logprob / gnmt_length_penalty(len(tokens), alpha), finished, not finished)


def _extensions(step_fn, prompt, alive):
    """扩展所有正概率边；零概率边不可能发生，不作为候选。"""
    rows = step_fn([prompt + tokens for tokens, _ in alive]).detach().cpu().tolist()
    return [(tokens + [token], total + lp)
            for (tokens, total), row in zip(alive, rows)
            for token, lp in enumerate(row) if math.isfinite(lp)]


def beam_search(step_fn, prompt, max_new, beam, eos, alpha=0.0, return_all=False):
    """与模型无关的束搜索；step_fn(list[list[int]]) 返回 (n,V) log 概率。

    每步全部扩展后按累计 log 概率取前 beam 个；EOS 占本步名额，随后
    移入 finished，绝不再次扩展。并列时沿用前缀/token 的枚举顺序。
    没有存活候选或达到 max_new 就停；到上限的存活候选标记 truncated。
    长度惩罚仅用于最终排序，束内剪枝仍按累计 log 概率。这是本章简化，
    不声称复现 GNMT/HF 的剪枝与停止策略。beam=1 因而等于贪心。
    return_all 返回所有保留下来的完成/截断候选，不包括已被剪掉的路径。
    """
    if not isinstance(beam, int) or beam < 1 or max_new < 0:
        raise ValueError("beam 须为正整数，max_new 须非负")
    gnmt_length_penalty(0, alpha)
    alive, finished = [([], 0.0)], []
    for _ in range(max_new):
        candidates = sorted(_extensions(step_fn, prompt, alive), key=lambda c: -c[1])[:beam]
        alive = []
        for tokens, total in candidates:
            if eos is not None and tokens[-1] == eos:
                finished.append(_hypothesis(tokens, total, eos, alpha))
            else:
                alive.append((tokens, total))
        if not alive:
            break
    pool = finished + [_hypothesis(tokens, lp, eos, alpha) for tokens, lp in alive]
    if not pool:
        raise ValueError("step_fn 没有提供任何正概率候选")
    pool.sort(key=lambda h: -h.score)
    return pool if return_all else pool[0]


def brute_force_search(step_fn, prompt, max_new, eos, alpha=0.0):
    """只用于小玩具：不剪枝，枚举全部 EOS 路径及到上限的未结束路径。"""
    if max_new < 0:
        raise ValueError("max_new 须非负")
    gnmt_length_penalty(0, alpha)
    alive, finished = [([], 0.0)], []
    for _ in range(max_new):
        candidates = _extensions(step_fn, prompt, alive)
        alive = []
        for tokens, lp in candidates:
            if eos is not None and tokens[-1] == eos:
                finished.append(_hypothesis(tokens, lp, eos, alpha))
            else:
                alive.append((tokens, lp))
        if not alive:
            break
    pool = finished + [_hypothesis(tokens, lp, eos, alpha) for tokens, lp in alive]
    if not pool:
        raise ValueError("step_fn 没有提供任何正概率候选")
    return max(pool, key=lambda h: h.score)


class BigramToyLM:
    """转移表的键顺序即词表顺序；省略的边概率为零。prefix 是 token ID 列表。"""
    def __init__(self, transitions: dict[str, dict[str, float]]):
        self.vocab = list(transitions)
        self.stoi = {word: i for i, word in enumerate(self.vocab)}
        probs = torch.zeros(len(self.vocab), len(self.vocab), dtype=torch.float64)
        for word, row in transitions.items():
            for nxt, prob in row.items():
                probs[self.stoi[word], self.stoi[nxt]] = prob
        if (not len(probs) or not torch.isfinite(probs).all() or (probs < 0).any()
                or not torch.allclose(probs.sum(-1), torch.ones(len(probs), dtype=probs.dtype))):
            raise ValueError("转移概率须非负有限且每行和为 1")
        self._logprobs = probs.log()

    def logits(self, prefix):
        return self._logprobs[prefix[-1]].clone()

    def step_fn(self, prefixes):
        return torch.stack([self.logits(prefix) for prefix in prefixes])


SHORT_BIAS_TRANSITIONS = {
    "<bos>": {"<eos>": 0.30, "x": 0.70},
    "<eos>": {"<eos>": 1.0},
    "x": {"^": 0.60, "<eos>": 0.40},
    "^": {"2": 0.60, "x": 0.40},
    "2": {"<eos>": 0.90, "^": 0.10},
}


def short_bias_toy():
    """固定手算分布；词表顺序为 <bos>, <eos>, x, ^, 2。"""
    return BigramToyLM(SHORT_BIAS_TRANSITIONS)


def distinct_n(token_lists, n):
    """Li et al. (2016) distinct-n：跨样本汇总，但不跨样本边界连接 n-gram。"""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n 须为正整数")
    grams = [tuple(seq[i:i+n]) for seq in token_lists for i in range(len(seq)-n+1)]
    return len(set(grams)) / len(grams) if grams else None


def unique_ratio(texts):
    """不同完整字符串数/样本数；空样本无定义。"""
    return len(set(texts)) / len(texts) if texts else None


def decoding_report(texts, token_lists, ended):
    """沿用 parse_report 的 count/successes/rate/interval；ended 为逐条 bool。

    token_lists 应去掉 BOS/EOS；平均长度与 distinct-n 使用同一内容口径。
    texts 可含提示内容，调用者应明确口径；截断样本同样纳入解析率分母。
    """
    from miniocr.train.pretrain import parse_report
    if not len(texts) == len(token_lists) == len(ended):
        raise ValueError("texts、token_lists、ended 须逐条对应")
    result = parse_report(texts)  # 内部复用 wilson_interval，不另抄公式。
    result.pop("examples")
    return dict(result, distinct_1=distinct_n(token_lists, 1), distinct_2=distinct_n(token_lists, 2),
                unique_ratio=unique_ratio(texts),
                mean_length=sum(map(len, token_lists))/len(texts) if texts else None,
                ended_with_eos=sum(ended))


def speculative_accept_reject(p, q, generator):
    """单 token 正确性演示；p/q 是同设备、已归一化的 1 维概率向量。"""
    x = int(torch.multinomial(q, 1, generator=generator))
    # q[x]>0，因为 x 从 q 抽取；p=q 时总接受，不会归一化零残差。
    accept = torch.minimum(torch.ones_like(p[x]), p[x] / q[x])
    if torch.rand((), device=p.device, generator=generator) < accept:
        return x, True
    residual = (p - q).clamp_min(0)
    return int(torch.multinomial(residual / residual.sum(), 1, generator=generator)), False


def speculative_output_distribution(p, q):
    """解析输出分布与接受率 β；用 min(p,q) 避免 q=0 时出现 0/0。"""
    accepted = torch.minimum(p, q)
    beta = accepted.sum()
    residual = (p - q).clamp_min(0)
    mass = residual.sum()
    correction = residual / mass if mass > 0 else torch.zeros_like(p)
    return accepted + (1 - beta) * correction, beta
