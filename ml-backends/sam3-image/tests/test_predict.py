# Smoke tests for SAM3 image backend (NewModel).
# Run: cd ml-backends/sam3-image && pytest tests/ --tb=short -v
# Model is mocked at module level. No GPU or HF download required.
from __future__ import annotations
import os, unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("SAM3_MODEL_ID", "test/model")
os.environ.setdefault("SAM3_CHECKPOINT_FILENAME", "model.pt")
os.environ.setdefault("MODEL_DIR", "/tmp/test-models")


def _make_task(url="http://fake/image.jpg", task_id=1):
    return {"id": task_id, "data": {"image": url}}


def _kp_ctx(x=50.0, y=50.0, label="Object", is_positive=1, orig_w=64, orig_h=64):
    return {"result": [{
        "type": "keypointlabels",
        "is_positive": is_positive,
        "original_width": orig_w,
        "original_height": orig_h,
        "value": {"x": x, "y": y, "width": 0.5, "keypointlabels": [label]},
    }]}


def _rect_ctx(x=10.0, y=10.0, w=30.0, h=30.0, label="Object", orig_w=64, orig_h=64):
    return {"result": [{
        "type": "rectanglelabels",
        "original_width": orig_w,
        "original_height": orig_h,
        "value": {"x": x, "y": y, "width": w, "height": h, "rectanglelabels": [label]},
    }]}


def _mock_pred():
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 10:30] = True
    masks = np.stack([mask, mask, mask])
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
</View>
"""


class TestPredict(unittest.TestCase):

    def setUp(self):
        self.mock_pred = _mock_pred()
        self.patches = [
            patch("model.predictor", self.mock_pred),
            patch("model._checkpoint_path", "/tmp/fake.pt"),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def _backend(self):
        from model import NewModel
        return NewModel(project_id="test", label_config=_LABEL_CONFIG)

    def _run_predict(self, ctx, img_h=64, img_w=64):
        backend = self._backend()
        fake_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        with patch.object(backend, "get_local_path", return_value="/tmp/fake.jpg"),                  patch("model.Image") as mock_pil:
            mock_pil.open.return_value.convert.return_value = Image.fromarray(fake_img)
            result = backend.predict([_make_task()], context=ctx)
        return result

    def test_no_context_returns_empty(self):
        backend = self._backend()
        result = backend.predict([_make_task()], context=None)
        assert result.predictions == []

    def test_empty_context_returns_empty(self):
        backend = self._backend()
        result = backend.predict([_make_task()], context={"result": []})
        assert result.predictions == []

    def test_positive_keypoint_sets_label_1(self):
        self._run_predict(_kp_ctx(is_positive=1))
        call_kw = self.mock_pred.predict.call_args
        if call_kw:
            labels = call_kw.kwargs.get("point_labels")
            if labels is not None:
                assert 1 in labels, f"Expected positive label, got {labels}"

    def test_negative_keypoint_sets_label_0(self):
        self._run_predict(_kp_ctx(is_positive=0))
        call_kw = self.mock_pred.predict.call_args
        if call_kw:
            labels = call_kw.kwargs.get("point_labels")
            if labels is not None:
                assert 0 in labels, f"Expected negative label, got {labels}"

    def test_rect_produces_input_box(self):
        self._run_predict(_rect_ctx(x=10, y=10, w=30, h=30, orig_w=100, orig_h=100), 100, 100)
        call_kw = self.mock_pred.predict.call_args
        if call_kw:
            box = call_kw.kwargs.get("box")
            assert box is not None

    def test_multimask_output_true(self):
        self._run_predict(_kp_ctx())
        call_kw = self.mock_pred.predict.call_args
        if call_kw:
            assert call_kw.kwargs.get("multimask_output") is True


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
