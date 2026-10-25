"""第 10 章：训练循环。之后每一章（CNN、CRNN、Transformer、VLM 微调）都用同一个骨架。

    evaluate          在一个 DataLoader 上算平均损失和准确率（不记录计算图）
    train_one_epoch   把训练集过一遍：清零 → 前向 → 损失 → 反向 → （裁剪）→ 更新
    fit               训练若干轮，每轮在验证集上评测，保留验证准确率最高的那一轮的参数
    predict           返回整个数据集的预测与真实标签
    confusion_matrix  混淆矩阵：C[i, j] = 真实类别 i 被预测成 j 的次数
    overfit_one_batch 体检：在同一批数据上反复训练，损失应当降到接近 0

所有函数都接受 loss_fn 参数，默认是 F.cross_entropy；第 10 章的"类别不均衡"实验会传入带权重的版本。
"""
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F


# >>> 练习 10.2：评测
def evaluate(model, loader, device, loss_fn=F.cross_entropy):
    """返回 (平均损失, 准确率)。

    要点：model.eval()（第 11 章的 Dropout/BatchNorm 在评测时行为不同）；torch.no_grad()；
    损失按样本数加权平均（最后一批可能较小）；评测结束后不必切回 train 模式，train_one_epoch 会自己切。
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += loss_fn(logits, yb).item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n
# <<<


# >>> 练习 10.3：训练一轮
def train_one_epoch(model, loader, opt, device, loss_fn=F.cross_entropy, clip=None):
    """把 loader 过一遍，返回这一轮的平均训练损失（按样本数加权）。

    每一批：搬到 device → opt.zero_grad() → logits = model(xb) → loss → loss.backward()
            → （clip 不为 None 时）torch.nn.utils.clip_grad_norm_(model.parameters(), clip) → opt.step()。
    记录损失用 loss.item()，不要把带计算图的张量存起来。
    """
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        total += loss.item() * len(yb)
        n += len(yb)
    return total / n
# <<<


# >>> 练习 10.4：完整的训练过程
def fit(model, train_loader, val_loader, opt, epochs, device, loss_fn=F.cross_entropy, clip=None, log=print):
    """训练 epochs 轮。每轮：train_one_epoch → 在训练集和验证集上 evaluate → 记录一行历史 → 打印。

    返回 history：列表，每个元素是 dict(epoch, train_loss, train_acc, val_loss, val_acc, seconds)。
    其中 train_loss 是训练过程中的平均损失，train_acc 是这一轮结束后在训练集上评测的准确率。
    训练结束时，把模型参数恢复成验证准确率最高的那一轮（copy.deepcopy(model.state_dict()) 保存，load_state_dict 恢复）。
    log=None 时不打印。
    """
    history, best_acc, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, opt, device, loss_fn, clip)
        _, train_acc = evaluate(model, train_loader, device, loss_fn)
        val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)
        row = dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                   val_loss=val_loss, val_acc=val_acc, seconds=time.perf_counter() - t0)
        history.append(row)
        if log:
            log(f"  第 {epoch:2d} 轮  训练损失 {train_loss:.4f}  训练准确率 {train_acc:6.1%}  "
                f"验证损失 {val_loss:.4f}  验证准确率 {val_acc:6.1%}")
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return history
# <<<


def predict(model, loader, device):
    """返回 (预测, 真实标签)，都是 NumPy 的 int64 数组。"""
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).argmax(1).cpu())
            ys.append(yb)
    return torch.cat(preds).numpy(), torch.cat(ys).numpy()


# >>> 练习 10.5：混淆矩阵
def confusion_matrix(pred, y, k):
    """C (k, k) int64：C[i, j] = 真实类别为 i、被预测为 j 的样本数。对角线是预测对的。
    提示：np.add.at(C, (y, pred), 1)，它能正确处理重复的下标（C[y, pred] += 1 不能）。"""
    C = np.zeros((k, k), dtype=np.int64)
    np.add.at(C, (y, pred), 1)
    return C
# <<<


def top_confusions(C, names, n=10):
    """混淆矩阵里最大的 n 个非对角元素：[(真实, 预测, 次数), ...]。"""
    D = C.copy()
    np.fill_diagonal(D, 0)
    idx = np.argsort(-D, axis=None, kind="stable")[:n]
    out = []
    for f in idx:
        i, j = divmod(int(f), C.shape[1])
        if D[i, j] > 0:
            out.append((names[i], names[j], int(D[i, j])))
    return out


def overfit_one_batch(model, xb, yb, opt, steps=200, loss_fn=F.cross_entropy):
    """体检：在同一批数据上训练 steps 步，返回每步的损失。正常的模型应当能把损失降到接近 0；
    降不下去，说明模型、损失或更新步骤有 bug（与数据量、泛化无关）。"""
    model.train()
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses
