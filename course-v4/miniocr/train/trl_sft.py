"""TRL 训练层的显式监督桥接与数值审计，复用第 19 章的机制实现。

预编码 labels 后 skip_prepare_dataset，避免框架再次套模板/截断；不能仅凭
completion_only_loss=True 判断监督正确，必须检查真正送入模型的 batch。
"""
import torch

from miniocr.data.sft import SFTExample, collate_sft
from miniocr.train.sft import causal_loss


# >>> 练习 21.1：将第 19 章的监督合同交给框架，并检查真实 batch
def dataset_rows(examples):
    """转换为 TRL 预编码记录；保留 labels，不允许全 mask 或错位长度。"""
    rows = []
    for ex in examples:
        collate_sft([ex], 0)  # 使用既有合同校验，不重新实现 mask。
        rows.append(dict(input_ids=list(ex.input_ids), labels=list(ex.labels)))
    if not rows:
        raise ValueError('没有训练样本')
    return rows


def audit_batch(batch, tokenizer, examples):
    """输出每一行的每一个 token/label，并检查 padding 与同位置答案监督。"""
    ids, labels, attention = (batch[k].cpu() for k in ('input_ids', 'labels', 'attention_mask'))
    assert ids.shape == labels.shape == attention.shape
    assert (labels[attention == 0] == -100).all()
    assert torch.equal(ids[labels != -100], labels[labels != -100])
    expected = {tuple(e.input_ids): e for e in examples}
    rows = []
    for i in range(len(ids)):
        active = torch.where(labels[i] != -100)[0]
        assert len(active) and active[0] > 0
        end = int(attention[i].sum()) - 1
        reference = expected[tuple(ids[i, :end+1].tolist())]
        assert int(active[0]) == reference.prompt_length
        assert labels[i, :end+1].tolist() == list(reference.labels)
        assert labels[i, end] == tokenizer.eos_token_id
        assert (labels[i, active[0]:end+1] != -100).all()
        rows.append([dict(position=j, id=int(t), token=tokenizer.convert_ids_to_tokens(int(t)),
                          label=int(labels[i,j]), attention=int(attention[i,j])) for j,t in enumerate(ids[i])])
    return dict(supervised_tokens=int((labels[:,1:] != -100).sum()), rows=rows,
                padding_mask_ok=True, completion_only=True, eos_supervised=True)
# <<<


def make_trainer(model, tokenizer, examples, output_dir, *, device='cpu', dtype='fp32',
                 steps=2, batch_size=1, accumulation=1, lr=2e-4, seed=0, rank=4,
                 checkpointing=False, callbacks=()):
    """创建 SFTTrainer + PEFT；显式关闭 packing/隐式截断/外部日志。"""
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    config = SFTConfig(output_dir=str(output_dir), max_steps=steps,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=accumulation,
        learning_rate=lr, lr_scheduler_type='constant', warmup_steps=0, weight_decay=.01,
        optim='adamw_torch', bf16=dtype=='bf16', fp16=False, use_cpu=device=='cpu',
        gradient_checkpointing=checkpointing, gradient_checkpointing_kwargs={'use_reentrant':False},
        max_length=None, packing=False, padding_free=False, completion_only_loss=True,
        dataset_kwargs={'skip_prepare_dataset':True}, remove_unused_columns=False,
        logging_steps=10, logging_first_step=True, save_strategy='no', report_to='none',
        disable_tqdm=True, dataloader_pin_memory=False, seed=seed, data_seed=seed)
    # TRL 1.15 的 fused LM head 使用 Triton，没有 CPU/MPS kernel。
    # 本地仍用 SFTTrainer 的数据/优化器/训练循环，loss 退回 transformers
    # 原生 causal LM loss；CUDA 保留未经覆盖的 TRL compute_loss。
    class PortableSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(**inputs, use_cache=False, num_items_in_batch=num_items_in_batch)
            return (outputs.loss, outputs) if return_outputs else outputs.loss
    trainer_class = SFTTrainer if device == 'cuda' else PortableSFTTrainer
    return trainer_class(model=model, args=config, processing_class=tokenizer,
        train_dataset=Dataset.from_list(dataset_rows(examples)), callbacks=list(callbacks),
        peft_config=LoraConfig(r=rank,lora_alpha=2*rank,lora_dropout=0.,bias='none',
                              target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))


# >>> 练习 21.2：对拍 loss、累积梯度及一次 AdamW 更新
def loss_step_parity(trainer, batches):
    """同模型 FP32 下对拍 TRL fused NLL 与第 19 章 causal_loss。

    每个 microbatch 使用整个更新的监督 token 总数作分母。保存并恢复参数，
    不污染后续训练；验证非零更新、梯度与 AdamW 更新，而不仅比较 loss。
    """
    model = trainer.model
    device = next(model.parameters()).device
    batches = [{k:v.to(device) for k,v in b.items()} for b in batches]
    total = sum(int((b['labels'][:,1:] != -100).sum()) for b in batches)
    params = {k:p for k,p in model.named_parameters() if p.requires_grad}
    initial = {k:p.detach().clone() for k,p in params.items()}
    was_training = model.training
    model.eval()
    def run(framework):
        model.zero_grad(set_to_none=True)
        loss_value = 0.
        for b in batches:
            if framework:
                loss = trainer.compute_loss(model, dict(b), num_items_in_batch=torch.tensor(total,device=device))
            else:
                logits = model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False).logits
                loss = causal_loss(logits,b['labels'],reduction='sum') / total
            loss.backward()
            loss_value += loss.item()
        grads = {k:p.grad.detach().clone() for k,p in params.items()}
        torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
        # fused/完整词表矩阵乘的微小梯度舍入会被 Adam 的近零分母放大。
        # 对拍两边显式用 eps=1e-6；正式 Trainer 保持其已保存的默认 eps。
        opt = torch.optim.AdamW(list(params.values()),lr=1e-4,weight_decay=.01,eps=1e-6)
        opt.step()
        updated = {k:p.detach().clone() for k,p in params.items()}
        return loss_value, grads, updated
    try:
        a = run(True)
        with torch.no_grad():
            for k,p in params.items(): p.copy_(initial[k])
        b = run(False)
        torch.testing.assert_close(torch.tensor(a[0]),torch.tensor(b[0]),atol=2e-5,rtol=2e-5)
        for k in params:
            torch.testing.assert_close(a[1][k],b[1][k],atol=2e-5,rtol=2e-4)
            torch.testing.assert_close(a[2][k],b[2][k],atol=2e-6,rtol=2e-5)
        delta=max((a[2][k]-initial[k]).abs().max().item() for k in params)
        assert delta > 0
        return dict(trl_loss=a[0],manual_loss=b[0],loss_error=abs(a[0]-b[0]),
            max_gradient_error=max((a[1][k]-b[1][k]).abs().max().item() for k in params),
            max_update_error=max((a[2][k]-b[2][k]).abs().max().item() for k in params),
            max_parameter_delta=delta, supervised_tokens=total, microbatches=len(batches),
            precision='fp32', adam_epsilon=1e-6)
    finally:
        with torch.no_grad():
            for k,p in params.items(): p.copy_(initial[k])
        model.zero_grad(set_to_none=True)
        model.train(was_training)
# <<<


def tiny_fixture():
    """无权重下载的小 Qwen 与词表，仅用于框架/机制对拍。"""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {'<pad>':0,'<eos>':1,'<unk>':2, **{f't{i}':i for i in range(3,64)}},unk_token='<unk>')),
        pad_token='<pad>',eos_token='<eos>',unk_token='<unk>')
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        eos_token_id=1,pad_token_id=0))
    examples=[SFTExample((3,4,5,1),(-100,-100,5,1),2),
              SFTExample((3,6,7,8,9,1),(-100,-100,7,8,9,1),2)]
    return model,tokenizer,examples
