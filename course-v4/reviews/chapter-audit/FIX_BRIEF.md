# 审核修正任务说明（每章一个 subagent，按顺序执行）

仓库根目录：/home/yoge/Dev/project/personal/xiaomi-ocr-study；课程在 course-v4/。
审核材料：course-v4/reviews/chapter-audit/（cross-chapter-review.md、findings-index.md、chapter-map.md、cross-checks/*.md、chapters/<章>.md）。

## 你负责一章

1. 完整阅读本章的逐章审核报告 `chapters/<章>.md`，以及 cross-chapter-review.md 与 cross-checks/*.md 中涉及本章的条目（按章号搜索，如“第22章”“22、”等）；再读 chapter-map.md 中本章的主审核记录（含降级裁定）。
2. 对每一条问题，回到教材原文（course-v4/book/**/<章>/chapter.md）及相关代码独立核实：审核是否成立？Codex 也可能错。可在 course-v4/.venv 中跑 CPU 小反例验证（不跑训练、不下载模型）。
3. 成立的问题：修改正文。原则：
   - 只收窄或补全断言、补限定条件、改正事实错误；不加新实验数字，不改历史数值，不编造结果。需要新实验才能确定的内容写成“尚未验证/待检验”。
   - 正文新引用的任何数字必须自己从仓库产物/原文核实过，不能照抄审核报告。
   - 不修改共享实现（miniocr/eval/*、数据管线）——改它会改变全书历史结果。对于 CER 边界碰撞（XC01）、单元素分组剥除（XC02）等实现缺陷，在正文中如实说明局限和影响范围，必要时在 exercises 中给出修复练习；不要改代码。
   - 文风：大学教材风格，简洁生动，中文；改动最小，不重写无关段落；保持原有节编号与格式。
   - 跨章引用（如“见第34章”）可以加，但只改本章文件。
4. 不成立的问题：不改，写清理由。
5. 改完后用 Codex 复核本章 diff（只采纳事实错误/不一致）：
   `cd /home/yoge/Dev/project/personal/xiaomi-ocr-study && git diff course-v4/book | codex exec -m gpt-6.1-sol -s read-only -o <scratch>/codex_<章>.txt "<请审查这份 diff 是否正确修复了审核问题、是否引入新错误>"`（若 stdin 不行，就把 diff 存文件让 codex 读）。Codex 不可用就跳过并注明。
6. 写结果到 course-v4/reviews/chapter-audit/fixes/<章>.md：每条问题一行：编号｜裁定（成立/部分成立/不成立）｜改动位置与摘要或不改理由。不要 git commit。
7. 最终回复只要 5 行内摘要。
