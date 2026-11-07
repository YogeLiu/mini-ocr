# 第32章逐章审查

## 阅读记录

源文件：`course-v4/book/part4_vlm/ch32_real_vlm/chapter.md`；SHA256：`d5e19dbe99f93b874eca38858ed41d93f105bca500265d7f889489cd88c544ce`。完整阅读1–240、241–480、481–720、721–950、951–1161行。核对本机transformers的smart_resize与Qwen3_5Model.get_rope_index实现。

## 章节主线与关键协议

本章不训练，解剖Qwen3.5-0.8B，辅以SmolVLM-256M和Qwen3-VL-2B。RGB静态图复制两帧，16×16时空patch展平成1536维；processor按2×2邻域优先排patch，merger每4patch拼接后MLP到1024。视觉塔12×768，语言侧24×1024混合18线性/6全注意力；加载模型共852985920参数，文件额外MTP不计入。smart_resize按取整后面积选分支、原图面积算比例，factor32；面积预算不是无条件硬夹子。32×298→96×800→300patch→75visual。Qwen3.5模板36文字token，输入111；M-RoPE图片从坐标4开始、3×25网格后文字坐标29–60。SmolVLM4×4重排，每512图块64token，块数不同。val180、可渲染配对80、test100，val规则选Qwen3-VL，仅胜者test。零样本低EM有抽取器删标点导致系统性误伤，保留冻结官方结果并单列诊断。

## 已确认问题

本章未确认P1/P2。

### CH32-01 · P3 · “连接器唯一能缩短长度”范围过宽

- 位置552行；短引文：“连接器是唯一能在进语言模型之前砍长度的地方”。
- 理由及反例：本章32.2和1096行自己展示processor调小面积预算可减少进入语言侧的视觉token；视觉塔内部也可合并/剪枝。只有“保持processor和视觉塔输出N固定”的三段比较中，连接器才是讨论的压缩位置。
- 建议：改成“在视觉塔已经输出固定N个token后，连接器可以在进入语言模型前再压缩长度”。

## 待验证线索、避免误报

- Qwen3.5是混合线性/全注意力模型，因此552行平方注意力代价承接第31章时宜说明只适用全注意力层；第466行已有18/6层声明，不能把全模型当24层标准GPT估算。
- 649行零输入输出不变不能逻辑上证明没接上输入（饱和、退化或某输入恰巧映射相同也可能）；该测试用于随机小连接器的冒烟有用，错误排查结论应改“优先检查连接”。
- 1032行单个漏符号例子“这才是没看清”是合理视觉候选，但仍不能排除语言侧解码选择；31章对这一点限定更谨慎。
- 209行穷举结论明确限定尺寸范围；没有把有限穷举当全尺寸定理，不报面积预算越界漏洞已遗漏。
- 400行逐位复刻只适用特定Pillow/RGB静态图路径，54行明确未复刻模型前向；均已披露。
- 1009行46条标点误伤是作者离线统计未另落盘，可要求补存诊断证据，但不是正文隐瞒指标问题；历史GPU推理和46条原始输出本轮未重跑。
- 1094行从本章一个候选转向33章两个模型、样本由180改100及预算更改已有说明，需作为跨章协议链核对，不能先认定偷换。

## 跨章关联表

|目标章|本章具体输出/依赖|跨章应检查命题|
|---|---|---|
|0/18/19/22|extract只删预测尾标点，目标不删|此前已知评测误伤，为何后续训练仍用旧协议；应分清指标读数与视觉能力|
|26/27|同一Qwen processor，不同面积预算|26/32 min65536；27 min1024；33 min16384，不能直接以token数差认实现矛盾|
|29/31|patch和位置插值、prefix接口|Qwen视觉表插值有二维结构；31全因果前缀与Qwen占位因果在依赖图方向一致|
|30|SmolVLM SigLIP预训练|视觉预训练来源不等同本地第30章小双塔权重|
|33|模型选择、图像预算、原始输出诊断|33新协议须声明双候选、相同val前100、不同图像预算和baseline重算；32 test不应反馈开发|
|36|参数/显存/延迟报告|852985920加载参数含共享去重不含MTP；allocated不等于整卡显存，并发计时不作架构速度排序|

## 事实核查来源与验证边界

实际打开官方Qwen配置 https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json 核对词表、宽度、层型、patch和merge；实际打开SmolVLM配置 https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/raw/main/config.json 核对scale_factor4、语言宽576。读取本地transformers smart_resize完整函数确认round/floor/ceil及原图缩放比例，读取get_rope_index核心实现确认图片后位置加max(grid_h,grid_w)/merge。实际打开模型卡 https://huggingface.co/zai-org/GLM-OCR 的Introduction核对CogViT、轻量下采样连接器、GLM解码器和外置layout；打开 https://huggingface.co/PaddlePaddle/PaddleOCR-VL 的Introduction核对NaViT风格与ERNIE4.5。以上是源码/文档核对，不代表复现性能；未加载GPU权重、未重跑历史实验。
