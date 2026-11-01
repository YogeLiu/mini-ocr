"""A1 接线契约。这里只导入并调用自己的 mechanism，不放算法。
张量接口保持输入 dtype/device，除特别说明外非法形状/参数抛 ValueError。
逐题说明见 assignments/a1/handout.md；学生从空 mechanism 开始。
"""


def make_value(data):
    """返回标量 Value：data/grad 可读写，支持 + - * /（含反向）、常数幂、tanh/relu/backward。
    backward 清中间梯度、累加叶子梯度；ReLU 在零处梯度为零。
    非数值输入抛 TypeError/ValueError，变量指数抛 TypeError/AssertionError，除零抛 ZeroDivisionError。
    """
    raise NotImplementedError


def run_numerical_grad(f, xs, h=1e-5):
    """f(float列表)->float 的中心差分，返回同长度列表，不修改 xs；h 必须有限且 >0。"""
    raise NotImplementedError


def run_softmax(z):
    """有限浮点 (N,C)，N,C>0，最后一维稳定 softmax；半精度用 fp32 中间计算。"""
    raise NotImplementedError


def run_cross_entropy(logits, targets):
    """有限浮点 (N,C)、long (N,) 且 0<=y<C，返回可反传的标量平均 CE；不支持忽略标签。"""
    raise NotImplementedError


def run_softmax_classifier(W, X, y, reg=0.0):
    """W(D,C), X(N,D), y(N,)；返回 (平均CE+reg*sum(W²), dW)。
    dW 必须手推，不调用 autograd；reg>=0。偏置由 X 补全1列实现，正则也作用于该行。
    """
    raise NotImplementedError


def run_mlp(x, w1, b1, w2, b2):
    """x(N,D), w1(D,H), b1(H,), w2(H,C), b2(C,) -> (N,C)。
    输出 relu(x@w1+b1)@w2+b2，不加最终softmax；保留对全部输入的计算图。
    """
    raise NotImplementedError


def get_classifier(input_dim, hidden_dim, n_classes):
    """返回 Module：输入 (N,D)，输出 logits(N,C)。hidden_dim=0 为线性，否则两层ReLU MLP。
    state_dict：线性 w(D,C),b(C)；MLP w1(D,H),b1(H),w2(H,C),b2(C)。
    偏置为零，矩阵随机非零；具体初始化分布自行声明。input_dim/classes>0, hidden_dim>=0。
    """
    raise NotImplementedError


def get_adamw(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    """返回有 step()/zero_grad() 与可写 lr 的优化器；AdamW先解耦衰减再更新，epsilon在根号外。
    grad=None 跳过（含衰减/计数），zero_grad置None；lr/decay>=0, eps>0, 0<=beta<1。
    """
    raise NotImplementedError


def run_schedule(step, total_steps, warmup_steps, base_lr, min_lr=0.0):
    """s从0起：s>=T返回min；s<W返回base*(s+1)/W；否则余弦进度(s-W)/max(1,T-W)。
    整数 s>=0,T>0,0<=W<=T；有限 base>=min>=0。
    """
    raise NotImplementedError


def run_clip_grad_norm(params, max_norm):
    """全局L2裁剪，返回裁剪前范数；超过上限乘 max_norm/(norm+1e-6)。
    忽略None，无梯度返回0；有限max_norm>=0。
    """
    raise NotImplementedError


def save_checkpoint(path, model, optimizer, step, generators=None, extra=None):
    """原子保存已完成步数、模型、优化器配置/矩状态、Python/NumPy/torch和具名生成器RNG、extra。
    文件至少含 model、step、extra；用于评测的extra包含model_cfg和data_cfg。path为str。
    """
    raise NotImplementedError


def load_checkpoint(path, model, optimizer=None, generators=None, map_location="cpu"):
    """恢复保存的全部状态，返回(step,extra)；下一步编号为step。文件不存在抛FileNotFoundError。
    同一CPU环境续训应逐位一致；MPS随机状态需自行补充并验证，不能直接声称跨设备一致。
    """
    raise NotImplementedError
