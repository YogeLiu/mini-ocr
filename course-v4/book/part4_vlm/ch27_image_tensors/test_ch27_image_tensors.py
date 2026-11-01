"""CPU 数值约束、评测协议和离线权重检查；不下载模型。"""
import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch
from miniocr.data.image_tensors import (image_to_chw, chw_to_image, normalize, area_weights,
                                        resize_chw, transform_image, pixel_metrics, fraction_bar_mask)
from miniocr.data.image_tensor_corpus import distinct_prefix
from book.part4_vlm.ch27_image_tensors import tour


def test_axis_order_alpha_and_roundtrip():
    a = np.array([[[1, 2, 3], [40, 50, 60]], [[80, 90, 100], [254, 255, 0]]], dtype=np.uint8)
    x = image_to_chw(Image.fromarray(a))
    assert x.shape == (3, 2, 2) and x.is_contiguous()
    torch.testing.assert_close(x[:, 0, 1], torch.tensor([40, 50, 60])/255)
    assert np.array_equal(np.asarray(chw_to_image(x)), a)
    assert image_to_chw(Image.new("RGBA", (2, 3), (0, 0, 0, 0))).eq(1).all()


def test_normalization_channelwise_inverse():
    x = torch.arange(18).float().reshape(3, 2, 3)/17
    mean, std = (.1, .3, .7), (.2, .4, .8)
    z = normalize(x, mean, std)
    assert z[0, 0, 0] == -.5
    torch.testing.assert_close(normalize(z, mean, std, inverse=True), x)
    with pytest.raises(ValueError):
        normalize(x, std=(1, 0, 1))
    with pytest.raises(ValueError):
        normalize(x, mean=(0,))


def test_fractional_area_is_coverage_not_adaptive_pool():
    expected = torch.tensor([[2/3, 1/3, 0], [0, 1/3, 2/3]])
    torch.testing.assert_close(area_weights(3, 2), expected)
    x = torch.tensor([[[0., 0., 1.]]])
    torch.testing.assert_close(resize_chw(x, (1, 2), "area"), torch.tensor([[[0., 2/3]]]))


@pytest.mark.parametrize("source,target", [(13, 5), (5, 13), (3, 3)])
def test_area_constant_and_integrated_ink(source, target):
    weights = area_weights(source, target)
    torch.testing.assert_close(weights.sum(1), torch.ones(target))
    values = torch.arange(source).float()
    assert (weights@values).mean().item() == pytest.approx(values.mean().item(), abs=1e-5)


@pytest.mark.parametrize("method", ["nearest", "bilinear", "area"])
def test_resize_identity_and_constant(method):
    x = torch.rand(3, 7, 11)
    torch.testing.assert_close(resize_chw(x, (7, 11), method), x)
    torch.testing.assert_close(resize_chw(torch.ones(3, 7, 11), (3, 5), method), torch.ones(3, 3, 5))


def test_nearest_and_bilinear_center_convention():
    x = torch.tensor([[[0., 1.]]])
    torch.testing.assert_close(resize_chw(x, (1, 4), "nearest"), torch.tensor([[[0., 0., 1., 1.]]]))
    torch.testing.assert_close(resize_chw(x, (1, 4), "bilinear"), torch.tensor([[[0., .25, .75, 1.]]]))


def test_letterbox_padding_and_identity_metric():
    original = Image.new("RGB", (7, 3), "black")
    image, meta = transform_image(original, 10)
    assert meta["resized_hw"] == [4, 10] and meta["top"] == 3
    array = np.asarray(image)
    assert np.all(array[:3] == 255) and np.all(array[3:7] == 0) and np.all(array[7:] == 255)
    assert pixel_metrics(original, image, meta) == dict(mae=0., ink_iou=1., ink_recall=1.)
    # Odd padding: 10 - round(10*2/7) = 7, top=3 and bottom=4.
    image, meta = transform_image(Image.new("RGB", (7, 2), "black"), 10)
    assert meta["resized_hw"] == [3, 10] and meta["top"] == 3
    assert np.all(np.asarray(image)[6:] == 255)


def test_line_loss_is_measured_in_original_coordinates():
    a = np.full((20, 40, 3), 255, dtype=np.uint8)
    a[10, 4:36] = 0
    original = Image.fromarray(a)
    mask = fraction_bar_mask(original)
    assert mask.sum() == 32
    processed, meta = transform_image(original, 40, geometry="letterbox")
    assert pixel_metrics(original, processed, meta, mask)["fraction_line_recall"] == 1
    white = Image.new("RGB", processed.size, "white")
    result = pixel_metrics(original, white, meta, mask)
    assert result["fraction_line_recall"] == 0 and result["fraction_line_soft"] == 0


def test_distinct_selection_uses_normalized_keys():
    assert distinct_prefix(["x^2", "x^{2}", "y"], 2) == ["x^2", "y"]
    with pytest.raises(ValueError):
        distinct_prefix(["x^2", "x^{2}"], 2)


def test_freeze_uses_val_and_cannot_overwrite(tmp_path):
    data, results = tmp_path/"data", tmp_path/"results"
    data.mkdir(); results.mkdir()
    base = dict(status="complete", metrics=dict(n=20, em_norm=.5, cer_char=.3),
                sample_ids_sha256="same", mean_visual_tokens=16, model="qwen35", revision="abc", config="letterbox128",
                dtype="bf16", batch_size=8, max_new_tokens=128, max_pixels=262144, seed=0, prompt=tour.PROMPT)
    (results/"qwen35_val_original_letterbox128.json").write_text(json.dumps(base))
    better = {**base, "model":"smol", "metrics":dict(n=20, em_norm=.6, cer_char=.4)}
    (results/"smol_val_original_letterbox128.json").write_text(json.dumps(better))
    (results/"qwen35_test_original_letterbox128.json").write_text(json.dumps({**base, "metrics":dict(n=20, em_norm=1, cer_char=0)}))
    args = SimpleNamespace(select_from=results, data=data, test_n=100)
    tour.freeze(args)
    assert json.loads((data/"frozen.json").read_text())["model"] == "smol"
    with pytest.raises(FileExistsError):
        tour.freeze(args)


def test_cli_test_guard_and_small_default(tmp_path):
    with pytest.raises(SystemExit):
        tour.main(["5", "--out", str(tmp_path)])
    tour.main(["1", "--out", str(tmp_path)])
    record = json.loads((tmp_path/"run.json").read_text())
    assert record["config"]["n"] == 20 and not record["test_evaluated"]
    assert json.loads((tmp_path/"tensor.json").read_text())["inverse_max_abs"] < 1e-6


def test_offline_processor_if_weights_exist():
    from miniocr.models.vlm_anatomy import local_snapshot, load_image_only_processor
    path = local_snapshot("qwen35")
    if path is None:
        pytest.skip("large local Qwen weights absent; no download")
    pytest.importorskip("transformers")
    p = load_image_only_processor(path, "qwen35")
    assert p.image_processor.merge_size > 0


def test_bootstrap_is_deterministic_and_uses_corpus_cer():
    rows = [dict(em_norm=True, ed_char=0, len_char=2), dict(em_norm=False, ed_char=4, len_char=8)]
    a = tour.bootstrap_intervals(rows, 0, repeats=100)
    assert a == tour.bootstrap_intervals(rows, 0, repeats=100)
    assert a["cer_char"] == [0, .5]


def test_batch_padding_trim():
    assert tour.trim_after_eos([5, 6, 2, 0, 0], [2]) == [5, 6, 2]
    assert tour.trim_after_eos([5, 6, 2, 0], [2, 3]) == [5, 6, 2]
    assert tour.trim_after_eos([5, 6], [2]) == [5, 6]


def test_artifact_audit_recomputes_scores_and_rejects_different_images(tmp_path):
    import hashlib
    from book.part4_vlm.ch27_image_tensors.summarize_runs import audit
    from miniocr.eval.metrics import score_pair, summarize
    (tmp_path/"data").mkdir(); (tmp_path/"original").mkdir()
    manifest = dict(val_original=[dict(key_sha256="a")], val_rendered=[dict(key_sha256="a")],
                    test_original=[dict(key_sha256="b")])
    (tmp_path/"data/manifest.json").write_text(json.dumps(manifest))
    line = dict(id="a", ref="x", pred="y", capped=False, input_image_sha256="pixels")
    result = dict(status="complete", split="val", source="original", config="native", capped=0,
                  sample_ids_sha256=hashlib.sha256(json.dumps(["a"]).encode()).hexdigest(),
                  metrics=summarize([score_pair("x", "y")]))
    for model in ("qwen35", "smol"):
        path = tmp_path/"original"/f"{model}.json"
        path.write_text(json.dumps(result))
        path.with_suffix(".jsonl").write_text(json.dumps(line)+"\n")
    audit(tmp_path)
    assert json.loads((tmp_path/"audit.json").read_text())["status"] == "passed"
    (tmp_path/"original/smol.jsonl").write_text(json.dumps({**line, "input_image_sha256":"other"})+"\n")
    with pytest.raises(AssertionError, match="identical images"):
        audit(tmp_path)
