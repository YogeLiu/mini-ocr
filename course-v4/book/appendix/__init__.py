"""附录 A–E 的小实验。emit 统一输出 key=value，并标注来源。"""


def emit(*, 来源="本项目观察", **values):
    """每行注明来源；float 保留七位有效数字。"""
    values = dict(来源=来源, **values)
    print("；".join(f"{key}={value:.7g}" if isinstance(value, float) else f"{key}={value}"
                   for key, value in values.items()), flush=True)
