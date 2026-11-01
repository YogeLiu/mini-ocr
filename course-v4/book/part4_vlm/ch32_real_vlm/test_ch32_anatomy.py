"""快 CPU 验收：独立布局锚点、HF 对拍、协议守卫及结构梯度。"""
from argparse import Namespace
import json
import numpy as np
from PIL import Image
import pytest
import torch
from miniocr.models.vlm_anatomy import smart_resize,replica_processor,MLPConnector,QueryConnector,local_snapshot
from book.part4_vlm.ch32_real_vlm.tour import main,selected_rows


@pytest.mark.parametrize("hw",[(1,1),(32,64),(37,239),(1024,2048),(16,3000)])
def test_processor_elementwise(hw):
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    im=Image.fromarray(np.random.default_rng(3).integers(0,256,(*hw,3),dtype=np.uint8))
    proc=Qwen2VLImageProcessorPil(patch_size=16,merge_size=2,temporal_patch_size=2,
                                 image_mean=[.5]*3,image_std=[.5]*3,
                                 size={"shortest_edge":65536,"longest_edge":262144})
    actual=replica_processor(im)
    expected=proc(images=im,return_tensors="np")
    for key in ("pixel_values","image_grid_thw"):
        np.testing.assert_array_equal(actual[key],expected[key])


def test_patch_order_independent_anchor():
    # 4x4 patches, p=1, merge=2, each pixel is an identifiable index.
    arr=np.repeat(np.arange(16,dtype=np.uint8).reshape(4,4,1),3,axis=2)
    result=replica_processor(Image.fromarray(arr),patch_size=1,merge_size=2,
                             min_pixels=16,max_pixels=16,mean=(0,0,0),std=(1,1,1))
    order=[0,1,4,5,2,3,6,7,8,9,12,13,10,11,14,15]
    np.testing.assert_allclose(result["pixel_values"][:,0]*255,order,atol=1e-6)
    # For each channel, the two static frames must be adjacent and identical.
    for c in range(3):
        np.testing.assert_array_equal(result["pixel_values"][:,2*c],result["pixel_values"][:,2*c+1])


@pytest.mark.parametrize("args",[(0,12),(1,201),(32,32,0),(32,32,32,65536,1024)])
def test_resize_rejects_invalid(args):
    with pytest.raises(ValueError): smart_resize(*args)


def test_resize_rounding_and_budget():
    assert smart_resize(80,80,min_pixels=1,max_pixels=100000)==(64,64) # half-to-even
    h,w=smart_resize(1000,2000,max_pixels=262144)
    assert h%32==w%32==0 and h*w<=262144
    # Lower-bound ceil may overshoot the nominal upper budget; do not promise a strict clamp.
    assert np.prod(smart_resize(37,239,min_pixels=65536,max_pixels=65536))>65536


@pytest.mark.parametrize("cls",[MLPConnector,QueryConnector])
def test_connectors_use_image_and_backprop(cls):
    torch.manual_seed(0)
    model=cls()
    x=torch.randn(2,16,32,requires_grad=True)
    result=model(x)
    assert result.shape==(2,16 if cls is MLPConnector else 8,48)
    result.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert not torch.allclose(result,model(torch.zeros_like(x)))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_test_guard_precedes_data_access():
    args=Namespace(split="test",final_test=False,models=["qwen35"],source="original",data=None)
    with pytest.raises(ValueError,match="test"): selected_rows(args)


def test_pairing_retains_ids(tmp_path):
    rows=[dict(id=str(i),rendered=None if i==1 else "x.png") for i in range(4)]
    (tmp_path/"val.json").write_text(json.dumps(rows))
    args=Namespace(split="val",source="rendered",data=tmp_path,n=2)
    rendered=selected_rows(args)
    args.source="paired-original"
    assert selected_rows(args)==rendered
    assert [r["id"] for r in rendered]==["0","2"]
    args.n=4
    with pytest.raises(ValueError): selected_rows(args)


def test_missing_model_is_offline(tmp_path):
    assert local_snapshot("qwen35",tmp_path) is None


def test_cached_real_processor():
    path=local_snapshot("qwen35")
    if path is None: pytest.skip("Qwen3.5 大权重未缓存")
    from miniocr.models.vlm_anatomy import load_image_only_processor
    processor=load_image_only_processor(path,"qwen35")
    assert processor.image_processor.patch_size==16
    assert processor.image_processor.merge_size==2


def test_cli_cpu_smoke(tmp_path):
    main(["1","3","6","--out",str(tmp_path)])
    assert json.loads((tmp_path/"processor.json").read_text())["rows"][0]["unequal_elements"]==0
    assert len(json.loads((tmp_path/"connectors.json").read_text()))==6


def test_open_source_image_extraction_preserves_pair(tmp_path,monkeypatch):
    from io import BytesIO
    import pyarrow as pa
    import pyarrow.parquet as pq
    import miniocr.data.latex_images_ch32 as module
    payload=BytesIO()
    Image.new("RGB",(40,20),"white").save(payload,format="PNG")
    parquet=tmp_path/"tiny.parquet"
    pq.write_table(pa.table({"latex_formula":["x", " y ","x"],
                            "image":[{"bytes":payload.getvalue(),"path":None}]*3}),parquet)
    monkeypatch.setattr(module,"_parquet_paths",lambda:[parquet])
    monkeypatch.setattr(module,"standard_split",lambda **kw:{"train":[],"val":["x","x"],"test":["y"]})
    out=tmp_path/"images"
    result=module.prepare_images(out,2)
    assert result["unique_counts"]=={"val":1,"test":1}
    val=json.loads((out/"val.json").read_text())[0]
    test=json.loads((out/"test.json").read_text())[0]
    assert (val["target"],val["row"],val["width"],val["height"])==("x",0,40,20)
    assert val["rendered"] and test["rendered"] is None
    assert val["normalization_key"]!=test["normalization_key"]


def test_placeholder_uses_merged_count():
    from miniocr.models.vlm_anatomy import expand_image_placeholder
    assert expand_image_placeholder("a<image>b","<image>",[[1,4,6]])=="a"+"<image>"*6+"b"
    with pytest.raises(ValueError): expand_image_placeholder("no image","<image>",[[1,4,6]])
    with pytest.raises(ValueError): expand_image_placeholder("<image>","<image>",[[1,3,3]])


def test_deep_generated_formula_is_unparseable_not_fatal():
    from miniocr.eval.render import can_parse
    from miniocr.eval.postcorrect_eval import score_postcorrect
    from miniocr.data.postcorrect import PostCorrectExample
    deep='{'*80+'x'+'}'*80
    assert can_parse(deep) is False
    result=score_postcorrect([PostCorrectExample('nested','val','x','x')],[deep],bootstrap_samples=0)
    assert result['summary']['parse_rate']['successes']==0
    assert can_parse('x^2') is True
