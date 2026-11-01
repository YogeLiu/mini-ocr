# 附录

这些附录**了解即可**，不是主线的前置。主线遇到相关概念时已有概括，想知道更多再来读。每个附录只讲要用到的部分，其余给参考资料。

每个附录都有 `chapter.md`（正文）和 `tour.py`（实验），全部能在 CPU 上运行：

```bash
python -m book.appendix.a_distributed.tour
python -m pytest -q book/appendix
```

| 附录 | 内容 | 学完能做到 | 和哪章衔接 |
|---|---|---|---|
| [A 分布式训练](a_distributed/chapter.md) | DDP、ZeRO 三阶段、FSDP、张量并行、流水线并行 | 说出每种方法切分了什么、省了多少显存 | 16.6、16.9、16.12 |
| [B FlashAttention](b_flash_attention/chapter.md) | 分块计算、online softmax、重算 | 写一个分块注意力，结果与标准注意力一致；说清它省的是显存访问，不是计算量 | 3.3、8.10、15.7、16.8 |
| [C 量化](c_quantization/chapter.md) | int8、int4、GPTQ、AWQ 的思路 | 说出量化误差从哪里来，QLoRA 用了哪种量化 | 16.1、20.6.4、20.8、20.9 |
| [D vLLM 推理](d_vllm/chapter.md) | PagedAttention、连续批处理、LoRA 挂载 | 按步骤部署微调后的模型（命令待验证），说清它为什么比逐条生成快 | 15.5、15.7、15.8、20.6、21.7 |
| [E 性能分析](e_profiling/chapter.md) | PyTorch profiler、计算受限与带宽受限 | 找出一段训练代码最慢的部分 | 15.7、16.10 |

**数字来源：** 和全书一样，分数学结论、本项目观察（注明 tour 部分号）、论文报告、待验证。

**没有做的事：** 本书没有在多卡上训练，没有在 GPU 上测 FlashAttention，没有实测 QLoRA，也没有运行 vLLM 服务。这些内容都标了待验证。
