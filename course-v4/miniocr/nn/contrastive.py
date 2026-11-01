"""Chapter 30: normalized dual encoders and explicit CLIP/SigLIP objectives.

Each batch must contain distinct formula keys, with the positive pair on the
diagonal. SigLIP sums B² pair losses then divides by B (not by B²).
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


def _check(scores):
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or not scores.numel():
        raise ValueError('expected nonempty square paired scores')


# >>> 练习 30.1：对称 InfoNCE，禁止调用 cross_entropy
def clip_loss(scores):
    """Stable symmetric row/column log-softmax CE; scores already scaled."""
    _check(scores)
    positive = scores.diagonal()
    return ((scores.logsumexp(1)-positive).mean() +
            (scores.logsumexp(0)-positive).mean()) / 2
# <<<


# >>> 练习 30.2：逐对 sigmoid 损失及偏置，禁止调用 BCE/logsigmoid
def siglip_loss(scores, bias=0.):
    """sum softplus(-label*(score+bias))/B, labels in {-1,+1}."""
    _check(scores)
    labels = 2 * torch.eye(len(scores), device=scores.device, dtype=scores.dtype) - 1
    z = -labels * (scores + bias)
    return torch.logaddexp(torch.zeros_like(z), z).sum() / len(scores)
# <<<


class DualEncoder(nn.Module):
    """Small spatial CNN + bidirectional byte Transformer, no pretrained weights.

    Padding id 0 is excluded from attention keys and mean pooling. Byte ids are
    1..256; no truncation is permitted by the dataset. Temperature stays fixed
    for controlled ablations; SigLIP bias starts at -log(B-1) and is learned.
    """
    def __init__(self, width=64, max_length=96, temperature=.07, batch_size=32):
        super().__init__()
        if width % 4 or width < 4 or max_length < 1 or temperature <= 0 or batch_size < 2:
            raise ValueError('invalid encoder settings')
        self.temperature = temperature
        self.image = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 20)), nn.Flatten(),
            nn.Linear(64*4*20, width), nn.LayerNorm(width))
        self.embedding = nn.Embedding(257, width, padding_idx=0)
        self.position = nn.Parameter(torch.randn(1, max_length, width)*.02)
        layer = nn.TransformerEncoderLayer(width, 4, 4*width, dropout=0.,
                                           batch_first=True, norm_first=True)
        self.text = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.project = nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width))
        self.bias = nn.Parameter(torch.tensor(-math.log(batch_size-1)))

    def encode_image(self, images):
        """Return unit-length image vectors."""
        return F.normalize(self.image(images), dim=-1)

    def encode_text(self, ids):
        """Bidirectional encoder with padding-safe pooling."""
        mask = ids.ne(0)
        if ids.ndim != 2 or ids.shape[1] > self.position.shape[1] or not mask.any(1).all():
            raise ValueError('empty text or invalid sequence length')
        h = self.embedding(ids) + self.position[:, :ids.shape[1]]
        h = self.text(h, src_key_padding_mask=~mask)
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return F.normalize(self.project(pooled), dim=-1)

    def forward(self, images, ids):
        return self.encode_image(images) @ self.encode_text(ids).T / self.temperature
