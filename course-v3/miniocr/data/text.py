"""第 13 章起：文字行的内容。全部是自己生成的英文单词、数字和少量标点，不依赖任何外部语料。

    WORDS            约 450 个常见英文单词（小写）
    LINE_CHARSET     文字行里可能出现的全部字符
    sample_line_text 随机生成一行文字：单词、数字、价格、日期、百分比的组合
    LineDataset      渲染好的文字行图像 + 标签字符串（PyTorch Dataset）
"""
import string

import numpy as np
import torch
from torch.utils.data import Dataset

from .imgops import to_float
from .render import font, render_line

WORDS = """the of and to in is it you that he was for on are with as his they be at one have this from or had by
word but what some we can out other were all there when up use your how said an each she which do their time if will
way about many then them write would like so these her long make thing see him two has look more day could go come did
number sound no most people my over know water than call first who may down side been now find any new work part take
get place made live where after back little only round man year came show every good me give our under name very through
just form sentence great think say help low line differ turn cause much mean before move right boy old too same tell does
set three want air well also play small end put home read hand port large spell add even land here must big high such
follow act why ask men change went light kind off need house picture try us again animal point mother world near build
self earth father head stand own page should country found answer school grow study still learn plant cover food sun four
between state keep eye never last let thought city tree cross farm hard start might story saw far sea draw left late run
while press close night real life few north open seem together next white children begin got walk example ease paper group
always music those both mark often letter until mile river car feet care second book carry took science eat room friend
began idea fish mountain stop once base hear horse cut sure watch color face wood main enough plain girl usual young ready
above ever red list though feel talk bird soon body dog family direct pose leave song measure door product black short
numeral class wind question happen complete ship area half rock order fire south problem piece told knew pass since top
whole king space heard best hour better true during hundred five remember step early hold west ground interest reach fast
verb sing listen six table travel less morning ten simple several vowel toward war lay against pattern slow center love
person money serve appear road map rain rule govern pull cold notice voice unit power town fine certain fly fall lead cry
dark machine note wait plan figure star box noun field rest correct able pound done beauty drive stood contain front teach
week final gave green oh quick develop ocean warm free minute strong special mind behind clear tail produce fact street""".split()

LINE_CHARSET = string.ascii_letters + string.digits + " .,:-'%$()"


def _token(rng):
    """一行里的一个"词"：大多是单词（随机大小写形式），也有数字、价格、日期、百分比。"""
    r = rng.random()
    if r < 0.70:
        w = WORDS[int(rng.integers(len(WORDS)))]
        s = rng.random()
        if s < 0.15:
            w = w.capitalize()
        elif s < 0.20:
            w = w.upper()
        if rng.random() < 0.08:
            w += str(rng.choice(list(".,:")))
        return w
    if r < 0.80:
        return str(int(rng.integers(0, 10 ** int(rng.integers(1, 5)))))
    if r < 0.86:
        return f"${int(rng.integers(1, 1000))}.{int(rng.integers(0, 100)):02d}"
    if r < 0.92:
        return f"{int(rng.integers(1, 32)):02d}-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1990, 2030))}"
    if r < 0.96:
        return f"{int(rng.integers(0, 101))}%"
    return f"({WORDS[int(rng.integers(len(WORDS)))]})"


def sample_line_text(rng, min_words=1, max_words=6, max_chars=28):
    """随机拼一行文字，长度不超过 max_chars 个字符。"""
    n = int(rng.integers(min_words, max_words + 1))
    toks = []
    for _ in range(n):
        t = _token(rng)
        if len(" ".join(toks + [t])) > max_chars:
            break
        toks.append(t)
    return " ".join(toks) if toks else _token(rng)[:max_chars]


class LineDataset(Dataset):
    """n 行文字的图像与标签。图像 float32 (1, H, W)，墨迹 = 1；标签是字符串（由 collate 函数编码成编号）。
    文字太宽放不下时（render_line 抛出 ValueError）换一行重新生成。"""

    def __init__(self, seed, n, H=32, W=256, noise=0.0, distort=0.0, **text_kw):
        rng = np.random.default_rng(seed)
        xs, self.texts = [], []
        while len(self.texts) < n:
            t = sample_line_text(rng, **text_kw)
            try:
                img, _ = render_line(t, rng, H=H, W=W, noise=noise, distort=distort)
            except ValueError:
                continue
            xs.append(to_float(img, invert=True))
            self.texts.append(t)
        self.x = torch.from_numpy(np.stack(xs).astype(np.float32)).unsqueeze(1)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.x[i], self.texts[i]


def font_width(ch, size=19):
    """字符在给定字号下的前进宽度（像素），用来说明"每个字符占多宽是不固定的"。"""
    return font(size).getlength(ch)
