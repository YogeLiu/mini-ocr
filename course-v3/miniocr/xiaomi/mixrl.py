"""第 34 章：Mix-RL 的奖励与多因素挖掘（报告 2.3、3.3 节）。

    kie_reward        关键信息抽取：先检查 JSON 合法，再按字段比较；缺字段得 0
    parse_page        整页输出的约定格式 "[类型] 内容"（每个区块一段，空行隔开）→ [(类型, 内容)]
    mix_reward        按任务路由：区块（文字 1 − NED、公式 CDM、表格 TEDS）、整页（第 33 章的 page_score）、KIE、VQA（ANLS）；
                      解析不了、格式不对、超长的输出一律 0
    sampling_weights  式 10：簇内按"专家一致度高、模型自洽度低"加权抽样
"""
import json

import numpy as np

from ..metrics import anls, ned

PAGE_TYPES = ("title", "text", "formula", "table")


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


# >>> 练习 34.7：KIE 奖励
def kie_reward(pred, ref):
    """pred：模型输出的字符串（可能包在 ```json … ``` 里，先去掉这层围栏）；ref：真值字典（可以嵌套）。
    pred 不能被 json.loads 解析、或者解析出来不是字典：返回 0。
    两边都展平成 "a.b.c" 形式的键；对 ref 的每个字段：pred 里有这个键时得 max(0, 1 − NED(str(预测值), str(真值)))，没有得 0。
    返回 ref 所有字段得分的平均（ref 为空字典时，pred 也为空得 1，否则 0）。"""
    s = pred.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        s = s.rsplit("```", 1)[0]
    try:
        p = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return 0.0
    if not isinstance(p, dict):
        return 0.0
    fp, fr = _flatten(p), _flatten(ref)
    if not fr:
        return float(not fp)
    return float(np.mean([max(0.0, 1 - ned(str(fp[k]), str(v))) if k in fp else 0.0 for k, v in fr.items()]))
# <<<


def parse_page(text):
    """"[title] Hello\\n\\n[text] world" → [("title", "Hello"), ("text", "world")]。任何一段不是 "[已知类型] 内容" 时返回 None。"""
    out = []
    for para in [p.strip() for p in text.strip().split("\n\n") if p.strip()]:
        if not para.startswith("[") or "]" not in para:
            return None
        kind, content = para[1:].split("]", 1)
        if kind not in PAGE_TYPES:
            return None
        out.append((kind, content.strip()))
    return out


# >>> 练习 34.8：按任务路由的奖励
def mix_reward(task, pred, ref, max_chars=2000):
    """task ∈ {"text", "formula", "table", "page", "kie", "vqa"}。先查格式与长度：len(pred) > max_chars 时 0。
        text / formula / table：miniocr.rl.rewards.doc_reward(task, pred, ref)
        page：ref 是 [(类型, 内容)]；parse_page(pred) 为 None 时 0，否则 miniocr.eval.page.page_score(解析结果, ref) 的总分
        kie：kie_reward(pred, ref)（ref 是字典）
        vqa：ref 是正确答案的列表，anls([pred], [ref])
    其他 task 抛出 ValueError。"""
    if len(pred) > max_chars:
        return 0.0
    if task in ("text", "formula", "table"):
        from ..rl.rewards import doc_reward
        return float(doc_reward(task, pred, ref))
    if task == "page":
        from ..eval.page import page_score
        blocks = parse_page(pred)
        return 0.0 if blocks is None else float(page_score(blocks, ref)[0])
    if task == "kie":
        return kie_reward(pred, ref)
    if task == "vqa":
        return float(anls([pred], [ref]))
    raise ValueError(task)
# <<<


# >>> 练习 34.9：簇内抽样权重（式 10）
def sampling_weights(a, u, labels, f_a=1.0, f_u=1.0, tau=1.0, eps=1e-3):
    """a：专家一致度（越高标签越可信），u：模型自洽度（越低模型越没把握），labels：簇编号；都是长度 N 的数组。
    未归一化的权重 w_i = (a_i + eps)^(f_a/τ) · (1 − u_i + eps)^(f_u/τ)；
    在每个簇内归一化：P(i | k) = w_i / Σ_{j 在簇 k} w_j。返回 numpy 数组 (N,)，每个簇内加起来是 1。"""
    a, u, labels = np.asarray(a, float), np.asarray(u, float), np.asarray(labels)
    w = (a + eps) ** (f_a / tau) * (1 - u + eps) ** (f_u / tau)
    out = np.zeros_like(w)
    for k in np.unique(labels):
        m = labels == k
        out[m] = w[m] / w[m].sum()
    return out
# <<<
