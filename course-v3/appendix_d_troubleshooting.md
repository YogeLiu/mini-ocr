# 附录 D　环境与故障排查

本附录把全书各章分散的"环境问题"收在一起：先给出作者的环境和标准安装步骤，再按"出错的阶段"列出现象、原因和解决办法。各章自己的"常见错误"表格讲的是概念和代码上的错误，汇总在附录 B。

## D.1 作者的环境

本书所有"实测"数字都来自下面这套环境。你的数字和书上差一点（随机数、库版本、硬件不同）是正常的；差很多时，先对照这张表。

| 项目 | 作者的版本 | 说明 |
|---|---|---|
| 操作系统 | Linux（x86_64） | macOS、Windows（WSL2）也能跑前 32 章 |
| Python | 3.12 | 用 uv 创建 |
| PyTorch | 2.14.1+cu130 | 第 7 章起需要；CPU 版也可以 |
| numpy / pillow / matplotlib / pytest | 2.5.4 / 12.3.0 / 3.11.2 / 9.1.1 | `requirements.txt` |
| transformers / peft / trl | 5.19.0 / 0.21.2 / 1.15.0 | `requirements-hf.txt`；第 24、33–35 章需要，第 27 章第 3–4 部分、第 28 章第 1 部分、第 29 章第 5 部分、第 30 章第 4 部分、第 31 章第 4 部分、第 36 章第 5 部分也需要 |
| datasets / accelerate / huggingface_hub | 5.1.0 / 1.15.0 / 2.2.0 | 同上 |
| pyarrow | 26.0.0 | 同上；第 33 章第 4 部分、第 36 章第 5 部分直接读 parquet 文件 |
| 内存 | 32 GB | 第 35 章的检查峰值约 7.9 GB |
| GPU | 一块 NVIDIA 显卡，**但书里的实测都在 CPU 上做** | 见 D.4 |

没有列出的库（torchvision、scipy、flash-linear-attention、causal-conv1d）本书都**不需要**。遇到要求安装它们的报错，看 D.5。

## D.2 标准安装步骤

在 `course-v3/` 目录下：

```bash
uv venv --python 3.12 .venv          # 创建虚拟环境
source .venv/bin/activate            # 激活（Windows PowerShell：.venv\Scripts\activate）
uv pip install -r requirements.txt   # 第 0–23 章，以及后面各章不用真实模型的部分
python -m labs.ch00.check_env        # 检查环境
python -m pytest                     # 运行全部自动检查
```

需要 GPU 版 PyTorch 时，按 PyTorch 官网的选择器选出对应 CUDA 版本的安装命令，用 `uv pip install` 代替 `pip install`。用 `nvidia-smi` 右上角的 "CUDA Version" 确认驱动支持的最高 CUDA 版本，选不高于它的版本。

学到第 24 章时再装 Hugging Face 的库：

```bash
uv pip install -r requirements-hf.txt
```

## D.3 下载模型与离线运行

用到真实权重的章节：

| 章 | 模型 / 数据 | 本地缓存大小 |
|---|---|---|
| 24、35 | Qwen/Qwen3.5-0.8B | 1.7 GB |
| 24、35 | HuggingFaceTB/SmolVLM-256M-Instruct | 0.5 GB |
| 33、34 | zai-org/GLM-OCR | 2.5 GB |
| 33、34 | OleehyO/TexTeller | 4.6 GB |
| 33 | OleehyO/latex-formulas（数据集） | 全部约 2.8 GB，可以只下一个分片 |

下载命令（第 24 章开头"前置检查"一节）：

```bash
huggingface-cli download Qwen/Qwen3.5-0.8B
huggingface-cli download HuggingFaceTB/SmolVLM-256M-Instruct
```

文件存在 `~/.cache/huggingface/hub/` 下。下载完以后设置 `HF_HUB_OFFLINE=1`，transformers 只读本地缓存、不联网。本书的脚本默认这样设置。

- 下载前**先读模型卡上的许可证**，确认允许你的用途。
- 第 34 章不需要 GLM-OCR 和 TexTeller 的权重：三个模型在 60 个公式上的读数已经存在 `labs/ch34/data/` 里，第 33 章也引用这份数据。
- 下载中断时重新运行同一条命令，会接着下。
- 网络访问不了 huggingface.co 时，可以设置环境变量 `HF_ENDPOINT` 指向一个镜像站。

## D.4 CPU、GPU 与 Apple 芯片

各章演示脚本选设备的方式分三类：

| 章 | 默认设备 | 怎么改 |
|---|---|---|
| 9–13 | `pick_device()` 选出的设备：有 GPU 就用 GPU | 加 `--cpu` 强制只用 CPU |
| 16–23、25、26、28、35 | CPU | 加 `--gpu` 才用 GPU |
| 15、24、27、29–34、36，附录 F | 只用 CPU | 不需要 GPU |

第二类脚本的开头是：

```python
if "--gpu" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
```

这样即使你有 GPU，脚本也只用 CPU，加 `--gpu` 才用 GPU。这么做是因为作者的 GPU 同时在跑别的服务，显存只剩不到 1 GB。你的 GPU 空着的话，加 `--gpu` 会快很多；加之前先用 `nvidia-smi` 看一下显存还剩多少。第一类脚本在 GPU 被别的程序占满时可能报 `CUDA out of memory`，这时加 `--cpu`。

`CUDA_VISIBLE_DEVICES` 必须在 `import torch` **之前**设置才有效。在 Python 里中途改它不起作用。

**自己写的代码怎么选设备**：用 `miniocr.torch_basics.pick_device()`。有 NVIDIA GPU 返回 `cuda`，有 Apple 芯片返回 `mps`，否则返回 `cpu`。

**Apple 芯片（mps）的注意事项：**

- 有少数算子 mps 不支持。报 `NotImplementedError ... is not currently implemented for the MPS device` 时，在运行前设置环境变量 `PYTORCH_ENABLE_MPS_FALLBACK=1`，让这些算子退回 CPU。
- mps 不支持 float64。`torch.float64` 的张量要先转成 float32。
- mps 上的结果和 CPU 不是逐位一致的，测试里的容差可能需要放宽一点。本书的测试都在 CPU 上跑，不受影响。

## D.5 按阶段排查

### 安装与导入

| 现象 | 原因 | 怎么办 |
|---|---|---|
| `uv: command not found` | uv 没装好，或终端没重新打开 | 重新安装 uv，关掉终端再打开 |
| 提示符前面没有 `(.venv)` | 虚拟环境没激活 | `source .venv/bin/activate` |
| `ModuleNotFoundError: No module named 'numpy'`（或 PIL、torch） | 虚拟环境没激活，或库没装 | 激活后 `uv pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'miniocr'` | 不在 `course-v3/` 下运行，或用了 `python labs/xxx.py` | `cd` 到 `course-v3/`，用 `python -m labs.xxx`（点分隔，不带 `.py`） |
| `ModuleNotFoundError: No module named 'transformers'` | 没装 `requirements-hf.txt` | `uv pip install -r requirements-hf.txt` |
| 改了 `miniocr/` 里的代码，运行结果没变 | 运行的是另一份代码（比如学生版和完整版混了） | `python -c "import miniocr; print(miniocr.__file__)"` 看实际导入的是哪个文件 |

### PyTorch 与 GPU

| 现象 | 原因 | 怎么办 |
|---|---|---|
| `CUDA 可用: False`，但电脑有 NVIDIA 显卡 | 装成了 CPU 版的 PyTorch；驱动太旧；或者脚本把 `CUDA_VISIBLE_DEVICES` 设成了空 | 先看是不是没加 `--gpu`；再按官网选择器重装；`nvidia-smi` 看驱动 |
| `CUDA out of memory` | 显存被别的程序占着，或 batch / 图片太大 | `nvidia-smi` 看谁在占；减小 batch、图片分辨率、`max_pixels`；或者用 CPU |
| `Expected all tensors to be on the same device` | 模型在 GPU 上，数据在 CPU 上（或反过来） | 模型和每个输入张量都 `.to(device)` |
| CPU 上训练特别慢，CPU 占用只有 100% | 线程数被限制 | `torch.get_num_threads()` 看线程数；没有别的程序抢 CPU 时用 `torch.set_num_threads(物理核数)` |
| 同一个脚本两次运行结果不一样 | 没固定随机种子；或者 GPU 上有不确定的算子 | 用 `torch.manual_seed`；比较两种做法时用多个种子（第 25 章） |

### 加载 Hugging Face 模型

| 现象 | 原因 | 怎么办 |
|---|---|---|
| `AutoProcessor` 报 "requires the Torchvision library" | Qwen3.5、GLM-OCR 的 processor 同时会创建视频处理器，视频处理器依赖 torchvision | 不用 `AutoProcessor`：分词器用 `AutoTokenizer`，图像预处理用第 24 章自己写的函数（或只加载 image processor）；也可以装 torchvision |
| 打印 "falling back to its reference PyTorch implementation" | 没装 flash-linear-attention / causal-conv1d（Qwen3.5 的线性注意力层用它们加速） | 不是错误，结果正确，只是慢；需要速度时再装 |
| `OSError: ... couldn't connect to huggingface.co` / `LocalEntryNotFoundError` | 设了 `HF_HUB_OFFLINE=1`，但本地没有这个模型 | 先 `huggingface-cli download` 下载；或者检查模型名的大小写 |
| 加载时报 `Unrecognized model type` 或缺少某个类 | transformers 版本太旧，不认识这个架构 | 用 `requirements-hf.txt` 里的版本 |
| 进程被系统杀掉（`Killed`），没有 Python 报错 | 内存不够。float32 下每 10 亿参数约 4 GB：Qwen3.5-0.8B 约 3.4 GB，GLM-OCR 约 4.4 GB，LoRA 训练还要再加梯度和激活 | 关掉别的程序；一次只加载一个模型；用 `free -g` 看可用内存 |
| 模型输出全是乱码或重复同一个词 | 对话模板不对，或图像预处理和模型不匹配（第 24 章） | 用 `tokenizer.apply_chat_template`；对照模型的 preprocessor_config 检查尺寸、均值、方差 |
| 生成结果一开始就停了，或者输出了 `<think>` | 停止符或思考模式没设置好 | 看模型卡里推荐的生成参数和对话模板选项 |

### 训练

| 现象 | 原因 | 怎么办 |
|---|---|---|
| trl 的 `SFTTrainer` / `DPOTrainer` / `GRPOTrainer` 在 CPU 上第一步报 `RuntimeError: 0 active drivers ([]). There should only be one.` | trl 1.15 把 LM head 换成了只支持 CUDA / XPU 的融合 Triton kernel（第 35 章 35.4 节） | 在 CPU 上用本书自己写的训练循环；trl 的 trainer 在 GPU 上用 |
| LoRA 训练 loss 不动 | 优化器在解冻参数之前就建好了；或 target_modules 没有匹配到任何层 | 先解冻、再建优化器；打印可训练参数个数（附录 B） |
| Qwen3.5 的 LoRA 只加到了 6 层 | Qwen3.5 的 24 层里只有 6 层是全注意力（q/k/v/o_proj），另外 18 层线性注意力的投影叫 in_proj_qkv、in_proj_z、out_proj | target_modules 里把两种名字都列上（第 35 章） |
| loss 是 nan | 学习率太大；fp16 溢出；或者某个 batch 里所有标签都被 mask 掉了 | 看第一次出现 nan 的步；对全 mask 的 batch 跳过；fp16 换 bf16 |
| GRPO 的 reward 不涨、loss 一直是 0 | 同一组的回答 reward 全一样，优势全为 0 | 看每组 reward 的方差；调大温度或换更难/更易的题（第 31 章） |

### 测试与学生版

| 现象 | 原因 | 怎么办 |
|---|---|---|
| `python -m pytest` 有很多 `NotImplementedError` | 你在用学生版，练习还没做 | 正常。用 `python -m pytest labs/chXX` 只测当前章 |
| 测试通过，但书上的实验数字对不上 | 测试只查形状和关键性质，不查数值；随机种子或库版本不同 | 先对照 D.1 的版本；差别在几个百分点以内是正常的 |
| 某一章的测试很慢 | 有的测试会加载缓存里的真实模型 | 没有缓存时这些测试会自动跳过；只跑当前章 |

## D.6 求助前先收集这些信息

自己排查不出来时，把下面这些一起贴出来，别人才能帮你：

```bash
python -m labs.ch00.check_env                # 各个库的版本、设备
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi                                   # 有 NVIDIA 显卡时
free -g                                      # 内存
```

再加上：完整的命令、完整的报错（从 `Traceback` 开始到最后一行）、你改过哪些文件。只贴最后一行报错通常不够，真正的原因经常在中间。
