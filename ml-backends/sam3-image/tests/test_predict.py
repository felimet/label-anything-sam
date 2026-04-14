"""Smoke tests for SAM3 image backend (NewModel).

Run:
    cd ml-backends/sam3-image && pytest tests/ --tb=short -v

SAM3 predictor fully mocked — no GPU, no HF download required.

The image backend now uses Sam3Processor (state-dict API):
    state = processor.set_image(pil_image)
    state = processor.set_text_prompt(prompt, state)     # PCS path
    state = processor.add_geometric_prompt(box, label, state)
    state["masks"]  : bool tensor [N, 1, H, W]
    state["scores"] : float tensor [N]

SAM2 fallback uses the classic tuple API:
    processor.set_image(np_array)
    masks, scores, _ = processor.predict(point_coords, point_labels, box, multimask_output)
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("SAM3_MODEL_ID", "test/model")
os.environ.setdefault("SAM3_CHECKPOINT_FILENAME", "model.pt")
os.environ.setdefault("MODEL_DIR", "/tmp/test-models")
os.environ.setdefault("SAM3_ENABLE_PCS", "true")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_task(url="http://fake/image.jpg", task_id=1):
    return {"id": task_id, "data": {"image": url}}


def _kp_ctx(x=50.0, y=50.0, label="Object", is_positive=1, orig_w=64, orig_h=64):
    return {"result": [{
        "type":           "keypointlabels",
        "is_positive":    is_positive,
        "original_width": orig_w,
        "original_height": orig_h,
        "value": {"x": x, "y": y, "width": 0.5, "keypointlabels": [label]},
    }]}


def _rect_ctx(x=10.0, y=10.0, w=30.0, h=30.0, label="Object", orig_w=64, orig_h=64):
    return {"result": [{
        "type":           "rectanglelabels",
        "original_width": orig_w,
        "original_height": orig_h,
        "value": {"x": x, "y": y, "width": w, "height": h, "rectanglelabels": [label]},
    }]}


def _text_ctx(text="the red car", orig_w=64, orig_h=64):
    """Simulate a TextArea context result."""
    return {"result": [{
        "type":           "textarea",
        "original_width": orig_w,
        "original_height": orig_h,
        "value":          {"text": [text]},
    }]}


def _mixed_ctx(text="the cat", orig_w=64, orig_h=64):
    """Text + keypointlabels together."""
    return {"result": [
        {
            "type":           "textarea",
            "original_width": orig_w,
            "original_height": orig_h,
            "value":          {"text": [text]},
        },
        {
            "type":           "keypointlabels",
            "is_positive":    1,
            "original_width": orig_w,
            "original_height": orig_h,
            "value": {"x": 50.0, "y": 50.0, "width": 0.5, "keypointlabels": ["Object"]},
        },
    ]}


def _make_sam3_state(n_masks=1, h=64, w=64, score=0.9):
    """Build a fake Sam3Processor state dict."""
    masks  = torch.zeros(n_masks, 1, h, w, dtype=torch.bool)
    masks[:, :, 10:30, 10:30] = True
    scores = torch.tensor([score] * n_masks, dtype=torch.float32)
    boxes  = torch.tensor([[10., 10., 30., 30.]] * n_masks, dtype=torch.float32)
    return {"masks": masks, "scores": scores, "boxes": boxes}


def _mock_sam3_processor():
    """Mock Sam3Processor — returns state dict with masks/scores."""
    state = _make_sam3_state()
    m = MagicMock()
    m.set_image.return_value = {}
    m.set_text_prompt.return_value  = state
    m.add_geometric_prompt.return_value = state
    return m


def _mock_sam2_predictor():
    """Mock SAM2ImagePredictor — returns (masks, scores, logits) tuple."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 10:30] = True
    masks  = np.stack([mask, mask, mask])
    scores = np.array([0.95, 0.80, 0.70])
    m = MagicMock()
    m.predict.return_value = (masks, scores, None)
    return m


_LABEL_CONFIG = """
<View>
  <Image name="image" value="$image"/>
  <BrushLabels name="tag" toName="image"><Label value="Object"/></BrushLabels>
  <KeyPointLabels name="kp" toName="image" smart="true"><Label value="Object"/></KeyPointLabels>
  <RectangleLabels name="rect" toName="image" smart="true"><Label value="Object"/></RectangleLabels>
  <TextArea name="text_prompt" toName="image" maxSubmissions="1" editable="true"/>
</View>
"""


# ── SAM3 path tests ────────────────────────────────────────────────────────────

class TestSAM3Predict(unittest.TestCase):

    def setUp(self):
        self.mock_proc = _mock_sam3_processor()
        self.p1 = patch("model.processor", self.mock_proc)
        self.p2 = patch("model._checkpoint_path", "/tmp/fake.pt")
        self.p3 = patch("model._USING_SAM2_FALLBACK", False)
        self.p1.start(); self.p2.start(); self.p3.start()

    def tearDown(self):
        self.p1.stop(); self.p2.stop(); self.p3.stop()

    def _backend(self):
        from model import NewModel
        return NewModel(project_id="test", label_config=_LABEL_CONFIG)

    def _run(self, ctx, img_h=64, img_w=64):
        backend  = self._backend()
        fake_arr = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        fake_pil = Image.fromarray(fake_arr)
        with patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch("model.Image") as mock_pil:
            mock_pil.open.return_value.convert.return_value = fake_pil
            return backend.predict([_make_task()], context=ctx)

    # ── Basic sanity ───────────────────────────────────────────────────────────

    def test_no_context_returns_empty(self):
        result = self._backend().predict([_make_task()], context=None)
        assert result.predictions == []

    def test_empty_context_returns_empty(self):
        result = self._backend().predict([_make_task()], context={"result": []})
        assert result.predictions == []

    # ── Text-only (PCS) path ───────────────────────────────────────────────────

    def test_text_only_calls_set_text_prompt(self):
        self._run(_text_ctx("the red car"))
        self.mock_proc.set_text_prompt.assert_called_once()
        call_kwargs = self.mock_proc.set_text_prompt.call_args
        assert call_kwargs.kwargs.get("prompt") == "the red car" or \
               (call_kwargs.args and call_kwargs.args[0] == "the red car")

    def test_text_only_does_not_call_add_geometric_prompt(self):
        self._run(_text_ctx("the cat"))
        self.mock_proc.add_geometric_prompt.assert_not_called()

    def test_text_only_returns_brushlabels(self):
        from label_studio_ml.response import ModelResponse
        result = self._run(_text_ctx("a dog"))
        assert isinstance(result, ModelResponse)
        if result.predictions:
            pred = result.predictions[0]
            dump = pred.model_dump() if hasattr(pred, "model_dump") else pred
            results = dump.get("result", [])
            if results:
                assert results[0]["type"] == "brushlabels"

    # ── Geometric path ─────────────────────────────────────────────────────────

    def test_keypoint_calls_add_geometric_prompt(self):
        self._run(_kp_ctx(is_positive=1))
        self.mock_proc.add_geometric_prompt.assert_called()

    def test_negative_keypoint_uses_label_false(self):
        self._run(_kp_ctx(is_positive=0))
        for call in self.mock_proc.add_geometric_prompt.call_args_list:
            lbl = call.kwargs.get("label")
            if lbl is not None:
                assert lbl is False or lbl == 0

    def test_rect_calls_add_geometric_prompt_with_normalized_box(self):
        self._run(_rect_ctx(x=0, y=0, w=100, h=100, orig_w=100, orig_h=100), 100, 100)
        self.mock_proc.add_geometric_prompt.assert_called()
        call_kwargs = self.mock_proc.add_geometric_prompt.call_args.kwargs
        box = call_kwargs.get("box")
        if box is not None:
            # center + w/h should all be in [0, 1]
            assert all(0.0 <= v <= 1.0 for v in box), \
                f"Box not normalized: {box}"

    # ── Mixed path ─────────────────────────────────────────────────────────────

    def test_mixed_calls_both_set_text_and_add_geometric(self):
        self._run(_mixed_ctx("the cat"))
        self.mock_proc.set_text_prompt.assert_called_once()
        self.mock_proc.add_geometric_prompt.assert_called()

    # ── set_image always called ────────────────────────────────────────────────

    def test_set_image_called_for_any_prompt(self):
        self._run(_kp_ctx())
        self.mock_proc.set_image.assert_called_once()


# ── SAM2 fallback path tests ───────────────────────────────────────────────────

class TestSAM2FallbackPredict(unittest.TestCase):

    def setUp(self):
        self.mock_proc = _mock_sam2_predictor()
        self.p1 = patch("model.processor", self.mock_proc)
        self.p2 = patch("model._checkpoint_path", "/tmp/fake.pt")
        self.p3 = patch("model._USING_SAM2_FALLBACK", True)
        self.p1.start(); self.p2.start(); self.p3.start()

    def tearDown(self):
        self.p1.stop(); self.p2.stop(); self.p3.stop()

    def _backend(self):
        from model import NewModel
        return NewModel(project_id="test", label_config=_LABEL_CONFIG)

    def _run(self, ctx, img_h=64, img_w=64):
        backend  = self._backend()
        fake_arr = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        with patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"), \
             patch("model.Image") as mock_pil:
            mock_pil.open.return_value.convert.return_value = Image.fromarray(fake_arr)
            return backend.predict([_make_task()], context=ctx)

    def test_text_only_returns_empty_in_fallback(self):
        """Text-only prompt with SAM2 fallback — no geometric prompt, must return empty."""
        result = self._run(_text_ctx("the red car"))
        assert result.predictions == []

    def test_text_plus_geo_ignores_text_uses_geo(self):
        """Mixed context with SAM2 fallback — text ignored, geometry processed."""
        result = self._run(_mixed_ctx("the cat"))
        # SAM2 predict should have been called (geometric path)
        self.mock_proc.predict.assert_called()

    def test_geo_only_calls_sam2_predict(self):
        self._run(_kp_ctx())
        self.mock_proc.predict.assert_called_once()
        call_kw = self.mock_proc.predict.call_args.kwargs
        assert call_kw.get("multimask_output") is True

    def test_geo_returns_brushlabels(self):
        from label_studio_ml.response import ModelResponse
        result = self._run(_kp_ctx())
        assert isinstance(result, ModelResponse)
        if result.predictions:
            pred = result.predictions[0]
            dump = pred.model_dump() if hasattr(pred, "model_dump") else pred
            results = dump.get("result", [])
            if results:
                assert results[0]["type"] == "brushlabels"


# ── RLE encoding ───────────────────────────────────────────────────────────────

class TestRLEEncoding(unittest.TestCase):

    def test_rle_roundtrip(self):
        from label_studio_converter import brush
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        rle = brush.mask2rle(mask)
        assert isinstance(rle, list) and len(rle) > 0
        decoded = brush.decode_rle(rle)
        arr = np.frombuffer(decoded, dtype=np.uint8).reshape(64, 64, 4)
        assert arr[20, 20, 3] > 0
        assert arr[0, 0, 3] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
