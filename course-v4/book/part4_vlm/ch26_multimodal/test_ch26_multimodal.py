"""CPU 快验收：轴、因果隔离、视觉梯度、白底图片、token/评分协议。"""
from argparse import Namespace
import json

import numpy as np
from PIL import Image
import pytest
import torch

from miniocr.models.multimodal import VisualPrefix, CrossAttention, prefix_mask
from miniocr.models.vlm_anatomy import local_snapshot, load_image_only_processor
from book.part4_vlm.ch26_multimodal.tour import main, rendered_examples, infer_rows


def test_prefix_preserves_text_and_projects_width():
    model=VisualPrefix(3,4)
    vision=torch.randn(2,6,3,requires_grad=True)
    text=torch.randn(2,5,4,requires_grad=True)
    output=model(vision,text)
    assert output.shape==(2,11,4)
    torch.testing.assert_close(output[:,6:],text)
    output[:,:6].square().mean().backward()
    assert vision.grad.abs().sum()>0
    assert text.grad.abs().sum()==0


def test_prefix_cannot_leak_future_text():
    mask=prefix_mask(2,3)
    assert mask.tolist()==[[1,1,0,0,0],[1,1,0,0,0],[1,1,1,0,0],
                           [1,1,1,1,0],[1,1,1,1,1]]
    torch.manual_seed(26)
    x=torch.randn(1,5,4)
    def attend(x):
        weights=(x@x.transpose(-1,-2)).masked_fill(~mask,float("-inf")).softmax(-1)
        return weights@x
    changed=x.clone(); changed[:,4]+=100
    torch.testing.assert_close(attend(x)[:,:4],attend(changed)[:,:4])
    assert not torch.allclose(attend(x)[:,4],attend(changed)[:,4])


def test_cross_attention_matches_independent_sdpa_and_uses_vision():
    torch.manual_seed(26)
    model=CrossAttention(3,4)
    vision=torch.randn(2,6,3,requires_grad=True)
    text=torch.randn(2,5,4,requires_grad=True)
    output,weights=model(vision,text)
    expected=torch.nn.functional.scaled_dot_product_attention(model.q(text),model.k(vision),model.v(vision))
    torch.testing.assert_close(output,expected)
    assert output.shape==(2,5,4) and weights.shape==(2,5,6)
    torch.testing.assert_close(weights.sum(-1),torch.ones(2,5))
    output.square().mean().backward()
    assert vision.grad.abs().sum()>0 and text.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not torch.allclose(output,model(torch.zeros_like(vision),text)[0])


def test_zero_query_means_uniform_visual_average():
    model=CrossAttention(2,2)
    with torch.no_grad(): model.v.weight.copy_(torch.eye(2))
    vision=torch.tensor([[[2.,4.],[6.,8.]]])
    output,weights=model(vision,torch.zeros(1,3,2))
    torch.testing.assert_close(output,torch.tensor([[[4.,6.]]]).expand(1,3,2))
    torch.testing.assert_close(weights,torch.full((1,3,2),.5))


@pytest.mark.parametrize("shape",[(2,3),(3,4,8),(2,0,8)])
def test_bad_input_shapes(shape):
    with pytest.raises(ValueError): VisualPrefix()(torch.zeros(*shape),torch.zeros(2,5,12))


def test_render_polarity_and_cli_parts(tmp_path):
    main(["1","2","--out",str(tmp_path)])
    rows=json.loads((tmp_path/"rendered.json").read_text())["rows"]
    assert len(rows)==3
    for row in rows:
        image=np.asarray(Image.open(tmp_path/row["image"]))
        assert image.shape==(row["height"],row["width"],3)
        assert (image[:16]==255).all() and image.min()<50
        # 检查去掉外加白边后的内容；只有边框为白抓不住 render 极性漏反转。
        content=image[16:-16,16:-16]
        assert (content>240).mean()>.5
        assert (content<50).any()
    shapes=json.loads((tmp_path/"shapes.json").read_text())
    assert shapes["prefix"]==[2,11,12]
    assert shapes["cross_weights"]==[2,5,6]
    assert shapes["gradients_finite_nonzero"]
    assert json.loads((tmp_path/"run.json").read_text())["test_evaluated"] is False


def test_missing_weights_skip_offline(tmp_path):
    main(["3","--model-cache",str(tmp_path/"missing"),"--out",str(tmp_path)])
    assert json.loads((tmp_path/"controlled.json").read_text())["status"]=="skipped"


@pytest.mark.parametrize("args",[["5"],["--n","0"],["--visual-tokens","-1"],["--max-pixels","1"]])
def test_invalid_cli(args):
    with pytest.raises(SystemExit): main(args)


def test_raw_and_extracted_are_both_preserved(tmp_path):
    args=Namespace(out=tmp_path,dpi=160,n=20,device="cpu")
    rows=rendered_examples(args)[:1]
    class FakeInference:
        metadata={"model":"test double"}
        def load(self): return True
        def predict(self,image):
            assert image.mode=="RGB"
            return dict(raw="$x^2+y^2=z^2$",extracted="x^2+y^2=z^2",seconds=0.,
                        visual_tokens=4,generated_tokens=12,length_capped=False)
    infer_rows(args,FakeInference(),rows,"controlled")
    report=json.loads((tmp_path/"controlled.json").read_text())
    assert report["raw_metrics"]["em_raw"]==0
    assert report["extracted_metrics"]["em_raw"]==1
    row=json.loads((tmp_path/"controlled.jsonl").read_text())
    assert row["raw"]!=row["extracted"] and row["render_score"]["image_match"]


def test_cached_processor_token_count():
    path=local_snapshot("qwen35")
    if path is None: pytest.skip("本地缺少 Qwen3.5 大权重；不联网")
    processor=load_image_only_processor(path,"qwen35")
    prompt=processor.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":"Read."}]}],
                                         tokenize=False,add_generation_prompt=True,enable_thinking=False)
    batch=processor(text=prompt,images=[Image.new("RGB",(128,64),"white")],return_tensors="pt")
    image_id=processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    assert (batch["input_ids"]==image_id).sum()==batch["image_grid_thw"].prod()//processor.image_processor.merge_size**2
