"""第 23 章：VLM 的指令数据。聊天模板、ShapesWorld 上的九种任务、多轮对话、多图、场景文字。

    apply_chat_template   把一段对话（多轮、可含多张图）变成 token 序列和标签：只在助手的回答上计算损失
    draw_text_scene       在场景图的空白处写一个大写单词（"场景文字"，TextVQA 的玩具版）
    box_text / parse_box  框 ↔ 文字 "[x0,y0,x1,y1]"（整数像素坐标，与 Qwen2.5-VL 的做法相同）
    make_task             一张（或两张）图上某个任务的一轮问答
    TASKS                 任务名列表

对话格式（字符级，第 16 章的字符表；没有专门的角色 token，用纯文字的角色前缀）：
    <bos>User: <img>…<img>What color is the leftmost object?\\nAssistant: red<eos>User: How many objects?\\nAssistant: 3<eos>
"""
import numpy as np
from PIL import Image, ImageDraw

from .render import font
from .shapes import COLORS, SHAPES, describe, sample_scene
from .text import WORDS

IMAGE = "<image>"                                     # 消息文字里的图像标记，套模板时换成 n_img 个 <img>
TASKS = ("caption", "count", "count_color", "left", "exists", "ground", "refer", "compare", "text")
SIGN_WORDS = sorted({w.upper() for w in WORDS if 3 <= len(w) <= 4})


# >>> 练习 23.1：聊天模板
def apply_chat_template(vocab, messages, n_img):
    """messages：[{"role": "user" | "assistant", "content": 字符串}, …]，用户和助手轮流发言，从用户开始。
    content 里的每个 IMAGE（"<image>"）换成 n_img 个 <img>。拼法：
        [<bos>]
        用户：  编码("User: ") + 内容 + 编码("\\n")
        助手：  编码("Assistant: ") + 编码(内容) + [<eos>]
    返回 (ids, targets)：targets[t] = ids[t + 1]，但只保留"预测助手内容和它后面的 <eos>"的位置，其余为 −100
    （最后一位补 −100）。即：用户说的话、角色前缀都不计算损失；每一轮助手的回答都计算。"""
    img = [vocab.stoi["<img>"]] * n_img
    ids, sup = [vocab.stoi["<bos>"]], [False]
    for m in messages:
        if m["role"] == "user":
            parts = m["content"].split(IMAGE)
            seq = vocab.encode("User: ")
            for i, p in enumerate(parts):
                seq += (img if i else []) + vocab.encode(p)
            seq += vocab.encode("\n")
            ids += seq
            sup += [False] * len(seq)
        else:
            head = vocab.encode("Assistant: ")
            body = vocab.encode(m["content"]) + [vocab.stoi["<eos>"]]
            ids += head + body
            sup += [False] * len(head) + [True] * len(body)
    targets = [ids[t + 1] if sup[t + 1] else -100 for t in range(len(ids) - 1)] + [-100]
    return ids, targets
# <<<


def box_text(box):
    return "[" + ",".join(str(int(v)) for v in box) + "]"


def parse_box(s):
    """"[12,40,28,56]" → (12, 40, 28, 56)；格式不对时返回 None。"""
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    try:
        v = [int(x) for x in s[1:-1].split(",")]
    except ValueError:
        return None
    return tuple(v) if len(v) == 4 else None


def name(o):
    return f"{o['size']} {o['color']} {o['shape']}"


def draw_text_scene(img, objs, rng, size=13):
    """在不与任何物体重叠的地方写一个 3–4 个字母的大写单词（黑色）。返回 (新图, 单词, 文字框)；放不下时返回 (img, None, None)。"""
    word = SIGN_WORDS[int(rng.integers(len(SIGN_WORDS)))]
    f = font(size)
    w, h = int(f.getlength(word)) + 1, size
    H, W = img.shape[:2]
    for _ in range(100):
        x, y = int(rng.integers(1, W - w)), int(rng.integers(1, H - h))
        box = (x, y, x + w, y + h)
        if all(box[2] + 1 <= o["box"][0] or o["box"][2] + 1 <= box[0] or box[3] + 1 <= o["box"][1] or o["box"][3] + 1 <= box[1]
               for o in objs):
            im = Image.fromarray(img)
            ImageDraw.Draw(im).text((x, y + h), word, font=f, fill=(0, 0, 0), anchor="ls")
            return np.array(im), word, box
    return img, None, None


def make_task(task, objs, rng, objs2=None, word=None, neg_ground=0.2):
    """返回 (问题, 答案)。问题里用 IMAGE 标记图的位置（compare 任务有两张图，objs2 是第二张图的物体；
    text 任务的图上写了单词 word，见 draw_text_scene）。
    ground / refer 只问"在图中唯一"的物体描述；ground 有 neg_ground 的概率问一个不存在的物体，答案是 "none"。"""
    q = IMAGE
    if task == "caption":
        return q + "Describe the image.", describe(objs)
    if task == "count":
        return q + "How many objects?", str(len(objs))
    if task == "count_color":
        c = list(COLORS)[int(rng.integers(len(COLORS)))]
        return q + f"How many {c} objects?", str(sum(o["color"] == c for o in objs))
    if task == "left":
        return q + "What color is the leftmost object?", min(objs, key=lambda o: o["box"][0])["color"]
    if task == "exists":
        present = {(o["color"], o["shape"]) for o in objs}
        if rng.random() < 0.5:
            c, s = sorted(present)[int(rng.integers(len(present)))]
        else:
            absent = [(c, s) for c in COLORS for s in SHAPES if (c, s) not in present]
            c, s = absent[int(rng.integers(len(absent)))]
        return q + f"Is there a {c} {s}?", "yes" if (c, s) in present else "no"
    names = [name(o) for o in objs]
    unique = [o for o in objs if names.count(name(o)) == 1]
    if task == "ground":
        if rng.random() < neg_ground or not unique:
            absent = [f"{z} {c} {s}" for z in ("small", "large") for c in COLORS for s in SHAPES if f"{z} {c} {s}" not in names]
            return q + f"Where is the {absent[int(rng.integers(len(absent)))]}?", "none"
        o = unique[int(rng.integers(len(unique)))]
        return q + f"Where is the {name(o)}?", box_text(o["box"])
    if task == "refer":
        o = objs[int(rng.integers(len(objs)))]
        return q + f"What is at {box_text(o['box'])}?", f"a {name(o)}"
    if task == "compare":
        a, b = len(objs), len(objs2)
        ans = "the first" if a > b else "the second" if b > a else "the same"
        return f"Image 1: {IMAGE}Image 2: {IMAGE}Which image has more objects?", ans
    if task == "text":
        return q + "What word is written in the image?", word
    raise ValueError(task)
