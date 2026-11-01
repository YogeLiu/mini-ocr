"""附录 D：vLLM 的两个核心想法。纯模拟，不导入 vLLM，不启动服务。

python -m book.appendix.d_vllm.tour [部分号 1–6 ...]

步数模型：所有请求同时到达；一步里每个在跑的请求生成一个 token；不计预填充，不计调度开销；
不同批量的一步耗时不相等，所以步数不是时间，也不是吞吐。
"""
import argparse
from collections import deque
import math

from book.appendix import emit
from miniocr.kvcache import kv_cache_bytes_for, local_model_configs, step_cost

# 固定的一组请求：(提示长度, 输出长度)。
REQUESTS = [(24, 3), (12, 8), (40, 2), (18, 5), (60, 16), (9, 4), (33, 7), (15, 1),
            (28, 12), (20, 6), (50, 9), (11, 2), (16, 64), (22, 3), (35, 5), (13, 4)]


def blocks_for(tokens, block_size):
    return math.ceil(tokens/block_size)


def fragmentation(lengths, block_size):
    """按需分块的内部碎片：每条请求最后一块没用满的位置。"""
    slots = [blocks_for(n, block_size)*block_size for n in lengths]
    wasted = sum(s-n for s, n in zip(slots, lengths))
    return dict(tokens=sum(lengths), blocks=sum(s//block_size for s in slots), slots=sum(slots),
                wasted=wasted, wasted_ratio=wasted/sum(slots))


def reservation(lengths, reserve):
    """每条请求按固定长度 reserve 预留连续空间。"""
    if any(n > reserve for n in lengths):
        raise ValueError("请求超过预留长度")
    slots = reserve*len(lengths)
    return dict(tokens=sum(lengths), slots=slots, wasted=slots-sum(lengths), wasted_ratio=(slots-sum(lengths))/slots)


class BlockPool:
    """固定大小的物理块池。块表记录每条请求用哪些块；引用计数支持共享整块前缀。"""

    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.free = deque(range(num_blocks))
        self.refs = {}
        self.tables = {}
        self.lengths = {}

    def _take(self):
        if not self.free:
            raise MemoryError("没有空闲块")
        block = self.free.popleft()
        self.refs[block] = 1
        return block

    def add(self, rid, shared_blocks=()):
        """新建请求。shared_blocks 是别的请求已填满的前缀块，只加引用计数。"""
        for block in shared_blocks:
            self.refs[block] += 1
        self.tables[rid] = list(shared_blocks)
        self.lengths[rid] = len(shared_blocks)*self.block_size

    def append(self, rid, n=1):
        """写入 n 个 token；当前块写满才拿新块。"""
        for _ in range(n):
            if self.lengths[rid] % self.block_size == 0:
                self.tables[rid].append(self._take())
            self.lengths[rid] += 1

    def release(self, rid):
        for block in self.tables.pop(rid):
            self.refs[block] -= 1
            if self.refs[block] == 0:
                del self.refs[block]
                self.free.append(block)
        del self.lengths[rid]

    def used(self):
        return len(self.refs)


def static_batching(lengths, batch_size):
    """按到达顺序分批；一批要等最长的那条结束。返回步数、槽位利用率、平均完成步。"""
    steps, finish = 0, []
    for i in range(0, len(lengths), batch_size):
        chunk = lengths[i:i+batch_size]
        finish += [steps+max(chunk)]*len(chunk)  # 一批一起返回
        steps += max(chunk)
    return _summary(lengths, batch_size, steps, finish)


def continuous_batching(lengths, batch_size):
    """有请求结束，下一步就把等待的请求放进空出来的槽位。"""
    waiting, running, steps, finish = deque(range(len(lengths))), {}, 0, [0]*len(lengths)
    while waiting or running:
        while waiting and len(running) < batch_size:
            i = waiting.popleft()
            running[i] = lengths[i]
        steps += 1
        for i in list(running):
            running[i] -= 1
            if running[i] == 0:
                del running[i]
                finish[i] = steps
    return _summary(lengths, batch_size, steps, finish)


def _summary(lengths, batch_size, steps, finish):
    return dict(steps=steps, utilization=sum(lengths)/(batch_size*steps), mean_finish=sum(finish)/len(finish),
                lower_bound=max(math.ceil(sum(lengths)/batch_size), max(lengths)))


def simulate_serving(requests, pool_blocks, block_size, max_batch, reserve_output=None):
    """连续批处理 + 有限块池。

    reserve_output=None：按需分块，块不够时抢占最晚进入的请求（释放全部块，从头重算，不计预填充步）。
    reserve_output=R：进入时一次预留 prompt+R 的块，永不抢占。
    """
    budget = [p+(o if reserve_output is None else reserve_output) for p, o in requests]
    if reserve_output is not None and any(o > reserve_output for _, o in requests):
        raise ValueError("输出超过预留长度")
    if any(blocks_for(n, block_size) > pool_blocks for n in budget):
        raise ValueError("单条请求就装不下")
    waiting = deque(range(len(requests)))
    running, produced, held = [], [0]*len(requests), [0]*len(requests)
    free, steps, peak_running, preemptions, redone = pool_blocks, 0, 0, 0, 0

    def initial_blocks(i):
        tokens = requests[i][0] if reserve_output is None else budget[i]
        return blocks_for(tokens, block_size)

    def preempt():
        nonlocal free, preemptions, redone
        victim = running.pop()
        free += held[victim]
        held[victim] = 0
        redone += produced[victim]
        produced[victim] = 0
        preemptions += 1
        waiting.appendleft(victim)
        return victim

    while waiting or running:
        while waiting and len(running) < max_batch and initial_blocks(waiting[0]) <= free:
            i = waiting.popleft()
            held[i] = initial_blocks(i)
            free -= held[i]
            running.append(i)
        steps += 1
        peak_running = max(peak_running, len(running))
        for i in list(running):
            if i not in running:     # 这一步已被抢占
                continue
            if reserve_output is None:
                need = blocks_for(requests[i][0]+produced[i]+1, block_size)-held[i]
                while need > free and preempt() != i:  # 抢占最晚进入的，轮到自己就停
                    pass
                if i not in running:
                    continue
                held[i] += need
                free -= need
            produced[i] += 1
            if produced[i] == requests[i][1]:
                running.remove(i)
                free += held[i]
                held[i] = 0
    return dict(steps=steps, peak_running=peak_running, preemptions=preemptions, redone_tokens=redone)


def adapter_bytes(N_base, N_adapter, adapters, bytes_per=2):
    """K 个 adapter：合并后各存一份，或一份基座加 K 份 adapter。"""
    return dict(merged=adapters*N_base*bytes_per, mounted=(N_base+adapters*N_adapter)*bytes_per)


def part1(_):
    emit(来源="数学结论", 请求长度=23, 块大小=16, 块数=blocks_for(23, 16), 槽位=32, 空位=9)
    lengths = [p+o for p, o in REQUESTS]
    emit(来源="数学结论", 请求数=len(lengths), 每条最终长度=lengths, token总数=sum(lengths), 最长=max(lengths))
    for block in (4, 16, 64):
        emit(来源="数学结论", 方式="分块", 块大小=block, **fragmentation(lengths, block), 每条平均空位=fragmentation(lengths, block)["wasted"]/len(lengths))
    for reserve in (128, 512, 4096):
        emit(来源="数学结论", 方式="按最大长度预留", 预留=reserve, **reservation(lengths, reserve))
    per_token = kv_cache_bytes_for(local_model_configs()[0], 1, 1, 2)
    emit(来源="数学结论", 模型="Qwen3-0.6B骨架", 每token缓存字节=per_token,
         预留512的浪费GiB=reservation(lengths, 512)["wasted"]*per_token/2**30,
         块16的浪费GiB=fragmentation(lengths, 16)["wasted"]*per_token/2**30)


def part2(_):
    pool = BlockPool(8, 16)
    pool.add("a")
    pool.append("a", 23)
    emit(来源="数学结论", 步骤="a写23个token", 块表a=pool.tables["a"], 已用块=pool.used())
    pool.add("b")
    pool.append("b", 17)
    pool.append("a", 10)
    emit(来源="数学结论", 步骤="b写17个，a再写10个", 块表a=pool.tables["a"], 块表b=pool.tables["b"], 已用块=pool.used(),
         说明="同一请求的块不必连续")
    pool.release("a")
    pool.add("c")
    pool.append("c", 40)
    emit(来源="数学结论", 步骤="a结束，c写40个", 块表b=pool.tables["b"], 块表c=pool.tables["c"], 已用块=pool.used())


def part3(_):
    alone, share = BlockPool(16, 16), BlockPool(16, 16)
    for pool, rid in ((alone, "x"), (alone, "y")):
        pool.add(rid)
        pool.append(rid, 32+8)
    share.add("x")
    share.append("x", 32+8)
    share.add("y", shared_blocks=share.tables["x"][:2])
    share.append("y", 8)
    emit(来源="数学结论", 场景="两条请求前32个token相同，再各写8个", 块大小=16, 不共享已用块=alone.used(), 共享已用块=share.used(),
         块表x=share.tables["x"], 块表y=share.tables["y"], 共享块引用数=[share.refs[b] for b in share.tables["x"][:2]])
    share.release("x")
    emit(来源="数学结论", 步骤="x结束", 剩余已用块=share.used(), 说明="共享块引用数减到1，不回收")


def part4(_):
    lengths = [3, 8, 2, 5]
    emit(来源="数学结论", 输出长度=lengths, B=2, 静态=static_batching(lengths, 2), 连续=continuous_batching(lengths, 2))
    outputs = [o for _, o in REQUESTS]
    emit(来源="数学结论", 请求数=len(outputs), 输出长度=outputs, token总数=sum(outputs), 最长=max(outputs))
    for B in (1, 2, 4, 8, 16):
        a, b = static_batching(outputs, B), continuous_batching(outputs, B)
        emit(来源="数学结论", B=B, 下界=a["lower_bound"], 静态步数=a["steps"], 连续步数=b["steps"], 步数比=b["steps"]/a["steps"],
             静态利用率=a["utilization"], 连续利用率=b["utilization"], 静态平均完成步=a["mean_finish"], 连续平均完成步=b["mean_finish"])


def part5(_):
    for pool in (64, 32, 16, 8, 6):
        paged = simulate_serving(REQUESTS, pool, 16, 16)
        try:
            reserved = simulate_serving(REQUESTS, pool, 16, 16, reserve_output=64)
        except ValueError as exc:
            reserved = str(exc)
        emit(来源="数学结论", 块池=pool, 块大小=16, 批上限=16, 按需分块=paged, 按提示加64预留=reserved)


def part6(_):
    cfg = local_model_configs()[0]
    for B in (1, 32):
        row = step_cost(cfg, 1, 512, B)
        emit(来源="数学结论（第15章粗算）", 模型=cfg.name, T_past=512, B=B, 每步FLOPs=row["flops"], 每步访存字节=row["bytes"],
             算术强度=row["intensity"], 每token访存字节=row["bytes"]//B)
    N, r4 = 596049920, 573440
    for K in (1, 3, 10):
        row = adapter_bytes(N, r4, K)
        emit(来源="数学结论", 基座参数=N, adapter参数=r4, K=K, 权重dtype="bf16", 合并各存一份GiB=row["merged"]/2**30, 共享基座GiB=row["mounted"]/2**30)
    emit(来源="待验证", 说明="vLLM启动和LoRA挂载命令本书未运行；这里不给吞吐数字")


PARTS = {1: (part1, "内部碎片与预留（D.4.1）"),
         2: (part2, "块表（D.4.2）"),
         3: (part3, "前缀共享（D.4.2）"),
         4: (part4, "静态与连续批处理（D.4.3）"),
         5: (part5, "有限块池下的调度（D.4.4）"),
         6: (part6, "为什么比逐条生成快；LoRA挂载的账（D.4.5）")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=int)
    args = parser.parse_args(argv)
    if any(part not in PARTS for part in args.parts):
        parser.error("部分号须为1–6")
    emit(来源="环境", 附录="D", 默认行为="纯模拟；确定性；不导入vLLM；不写文件")
    for part in args.parts or PARTS:
        fn, title = PARTS[part]
        emit(来源="环境", 部分=part, 内容=title)
        fn(None)


if __name__ == "__main__":
    main()
