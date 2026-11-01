"""VLM LoRA 冻结策略与参数审计；不修改模型原有接口。"""
import torch


# >>> 练习 33.2：由实际模块树选目标，不能假定所有层都有 q_proj/v_proj。
def adapter_spec(model, strategy='language'):
    """language / connector（完整 merger）/ vision（再加视觉 Linear LoRA）。"""
    if strategy not in ('language', 'connector', 'vision'): raise ValueError(strategy)
    modules = dict(model.named_modules())
    connectors = [n for n in modules if n.endswith('visual.merger') or '.visual.deepstack_merger_list.' in n and n.rsplit('.', 1)[1].isdigit()]
    targets = []
    for name, module in modules.items():
        if not isinstance(module, torch.nn.Linear): continue
        if '.language_model.' in name:
            targets.append(name)
        elif strategy == 'vision' and '.visual.' in name and not any(name.startswith(c+'.') for c in connectors):
            targets.append(name)
    if not targets: raise ValueError('找不到语言线性层')
    if strategy != 'language' and not connectors: raise ValueError('找不到连接器')
    return dict(target_modules=targets, modules_to_save=connectors if strategy != 'language' else None)
# <<<


def attach_lora(model, strategy='language', rank=8):
    """PEFT 保存完整连接器及 LoRA；冻结其余基座参数。"""
    from peft import LoraConfig, get_peft_model
    spec = adapter_spec(model, strategy)
    return get_peft_model(model, LoraConfig(r=rank, lora_alpha=2*rank, lora_dropout=0.,
                           bias='none', task_type='CAUSAL_LM', **spec)), spec


def parameter_audit(model):
    """按视觉/连接器/语言统计参数，记录可训练名称用于排查冻结错误。"""
    groups = {k: dict(total=0, trainable=0) for k in ('language', 'connector', 'vision')}
    names = []; dtypes = {}
    for name, p in model.named_parameters():
        group = 'connector' if '.visual.' in name and ('merger.' in name or 'deepstack_merger_list.' in name) else 'vision' if '.visual.' in name else 'language'
        groups[group]['total'] += p.numel()
        if p.requires_grad:
            groups[group]['trainable'] += p.numel(); names.append(name)
            dtypes[str(p.dtype)] = dtypes.get(str(p.dtype),0) + p.numel()
    return dict(groups=groups, trainable=sum(v['trainable'] for v in groups.values()),
                total=sum(v['total'] for v in groups.values()), trainable_names=names, trainable_dtypes=dtypes)


def move_batch(batch, device, dtype):
    """保持 token/网格/label 的整数 dtype，仅转换浮点图片。"""
    return {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype) for k, v in batch.items()}
