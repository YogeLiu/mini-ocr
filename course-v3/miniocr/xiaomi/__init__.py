"""第 34 章：Xiaomi-OCR-0 的配方，用前面写过的零件做迷你复现。

    anchor   Q-Mask 式的文字锚定：按查询预测"答案在图里的哪些视觉 token 上"（NTP + Dice + CE）
    mtp      多 token 预测（MTP）头：从第 t 个位置同时预测第 t+2 个 token
    mixrl    Mix-RL 的按任务路由的奖励（区块、整页、KIE、VQA）和多因素挖掘的簇内抽样权重
"""
