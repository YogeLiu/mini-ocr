"""维护者参考接线；现成层只存在于受检 mechanism 之外，与 B2 相同。"""
from miniocr.nn import cnn, vit
from miniocr.nn.contrastive import clip_loss
from miniocr.models.formula_ocr import Connector, FormulaOCR, targets_for

conv_forward = cnn.conv_forward
conv_backward = cnn.conv_backward


def _positive(*values):
    if any(type(v) is not int or v < 1 for v in values):
        raise ValueError('expected positive integers')


def patch_grid(height, width, patch, pad=False):
    _positive(height, width, patch)
    return vit.patch_grid(height, width, patch, pad)


def patchify(x, patch):
    _positive(patch)
    return vit.patchify(x, patch)


def patch_embed(x, weight, bias=None):
    if (x.ndim != 4 or weight.ndim != 4 or weight.shape[1] != x.shape[1]
            or weight.shape[-1] != weight.shape[-2]
            or (bias is not None and bias.shape != (weight.shape[0],))):
        raise ValueError('invalid patch shapes')
    _positive(*weight.shape)
    return vit.patch_embed(x, weight, bias)


def interpolate_positions(pos, old_grid, new_grid, has_cls=True):
    if len(old_grid) != 2 or len(new_grid) != 2:
        raise ValueError('expected two axes')
    _positive(*old_grid, *new_grid)
    return vit.interpolate_positions(pos, old_grid, new_grid, has_cls)


def get_vision_block(width, heads=4):
    _positive(width, heads)
    return vit.VisionBlock(width, heads)


def get_vit(cfg):
    _positive(*(cfg[k] for k in ('image_size','patch','width','depth','heads','classes')))
    class CheckedViT(vit.TinyViT):
        def forward(self, x):
            if x.ndim != 4 or x.shape[1] != 1:
                raise ValueError('expected gray NCHW')
            return super().forward(x)
    return CheckedViT(**cfg)


def get_connector(width):
    _positive(width)
    class CheckedConnector(Connector):
        def forward(self, x):
            if x.ndim != 3 or x.shape[-1] != width:
                raise ValueError('expected B,P,width')
            return super().forward(x)
    return CheckedConnector(width)


def get_ocr(cfg):
    return FormulaOCR(**cfg)
