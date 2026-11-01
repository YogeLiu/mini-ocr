"""Small from-scratch OCR: bidirectional ViT → connector → causal GPT prefix.

Reuses chapter 29 patches/blocks and chapter 11 GPT without changing interfaces.
UTF-8 byte vocabulary: PAD=0, bytes=1..256, BOS=257, EOS=258. No weights/downloads.
"""
import torch
from torch import nn
from miniocr.nn.vit import patch_embed, VisionBlock
from miniocr.models.gpt import GPT, GPTConfig

PAD, BOS, EOS = 0, 257, 258


def targets_for(texts, max_length):
    """BOS+bytes as inputs; bytes+EOS as labels; padding is never supervised."""
    inputs = torch.zeros(len(texts), max_length+1, dtype=torch.long)
    labels = torch.full_like(inputs, -100)
    for i, text in enumerate(texts):
        raw = list(text.encode('utf-8'))
        if not raw or len(raw) > max_length:
            raise ValueError('nonempty formula must fit byte budget; never truncate')
        inputs[i, :len(raw)+1] = torch.tensor([BOS]+[b+1 for b in raw])
        labels[i, :len(raw)+1] = torch.tensor([b+1 for b in raw]+[EOS])
    return inputs, labels


# >>> 练习 31.1：连接器；投影视觉特征到 GPT 嵌入空间。
class Connector(nn.Module):
    """Token-wise two-layer MLP preserves the spatial sequence."""
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width,width),
                                 nn.GELU(), nn.Linear(width,width))

    def forward(self, visual):
        return self.net(visual)
# <<<


class FormulaOCR(nn.Module):
    """32×160 ink canvas; all parameters trained from random initialization."""
    def __init__(self, width=32, vision_depth=1, decoder_depth=2, patch=8, max_length=64):
        super().__init__()
        if 32 % patch or 160 % patch or width % 8:
            raise ValueError('patch must divide canvas; width must divide into even 4-head dimensions')
        self.config = dict(width=width,vision_depth=vision_depth,decoder_depth=decoder_depth,
                           patch=patch,max_length=max_length)
        self.prefix_length = (32//patch)*(160//patch)
        self.patch_weight = nn.Parameter(torch.empty(width,1,patch,patch))
        nn.init.xavier_uniform_(self.patch_weight.flatten(1))
        self.pos = nn.Parameter(torch.randn(1,self.prefix_length,width)*.02)
        self.vision = nn.Sequential(*(VisionBlock(width,4) for _ in range(vision_depth)))
        self.connector = Connector(width)
        self.gpt = GPT(GPTConfig(vocab_size=259,block_size=self.prefix_length+max_length+1,
                                d_model=width,n_layer=decoder_depth,n_head=4))

    def encode_image(self, images):
        return self.connector(self.vision(patch_embed(images,self.patch_weight)+self.pos))

    # >>> 练习 31.2：拼接、右移与监督 mask；视觉位置没有文本目标。
    def forward(self, images, inputs, labels=None):
        visual = self.encode_image(images)
        embeds = torch.cat((visual,self.gpt.tok(inputs)),dim=1)
        targets = None
        if labels is not None:
            targets = torch.cat((labels.new_full((len(labels),self.prefix_length),-100),labels),dim=1)
        logits, loss = self.gpt(inputs_embeds=embeds,targets=targets)
        return logits[:,self.prefix_length:], loss
    # <<<

    @torch.no_grad()
    def generate(self, images, max_new_tokens=None):
        """Greedy, cached prefix prefill; stop at EOS; PAD/BOS cannot be generated."""
        self.eval()
        steps = max_new_tokens or self.config['max_length']+1
        if steps > self.config['max_length']+1:
            raise ValueError('generation exceeds configured context')
        visual = self.encode_image(images)
        bos = torch.full((len(images),1),BOS,device=images.device,dtype=torch.long)
        caches = self.gpt.new_caches()
        logits,_ = self.gpt(inputs_embeds=torch.cat((visual,self.gpt.tok(bos)),1),caches=caches)
        done = torch.zeros(len(images),dtype=torch.bool,device=images.device)
        generated = []
        for t in range(steps):
            scores = logits[:,-1].clone(); scores[:,[PAD,BOS]] = -float('inf')
            token = scores.argmax(-1).masked_fill(done,EOS)
            generated.append(token); done |= token.eq(EOS)
            if done.all() or t+1 == steps: break
            logits,_ = self.gpt(token[:,None],caches=caches,pos0=self.prefix_length+1+t)
        result = []
        for row in torch.stack(generated,1).tolist():
            if EOS in row: row = row[:row.index(EOS)]
            result.append(bytes(t-1 for t in row).decode('utf-8',errors='replace'))
        return result


# >>> 练习 31.3：目标 token 的 log probability 对像素求导。
def token_saliency(model, image, inputs, position, token):
    """Absolute input gradient for a fixed teacher-forced token; not causal proof."""
    model.eval()
    x = image.detach().clone().requires_grad_(True)
    logits,_ = model(x,inputs)
    score = logits[0,position].float().log_softmax(-1)[token]
    gradient, = torch.autograd.grad(score,x)
    return gradient[0].abs().mean(0).detach()
# <<<
