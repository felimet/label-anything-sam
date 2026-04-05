"""Smoke tests for SAM3 image backend (NewModel).

Run:
    cd ml-backends/sam3-image && pytest tests/ --tb=short -v

Tests use a synthetic 64x64 image — no real download required.
GPU is not required (tests run on CPU via DEVICE=cpu env var).
SAM3 model is fully mocked — no weights download occurs.
"""
from __future__ import annotations

import os
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Force CPU and dummy model ID for tests
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("SAM3_MODEL_ID", "facebook/sam3.1")


def _make_synthetic_image(width: int = 64, height: int = 64) -> Image.Image:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _make_task(image_url: str = "http://fake/image.jpg") -> dict:
    return {"id": 1, "data": {"image": image_url}}


def _make_keypoint_context(
    x_pct: float = 50.0,
    y_pct: float = 50.0,
    label: str = "Object",
    orig_w: int = 64,
    orig_h: int = 64,
) -> dict:
    return {
        "result": [
            {
                "type": "keypointlabels",
                "original_width": orig_w,
                "original_height": orig_h,
                "value": {
                    "x": x_pct,
                    "y": y_pct,
                    "width": 0.5,
                    "keypointlabels": [label],
                },
            }
        ]
    }


def _make_rect_context(
    x: float = 10.0, y: float = 10.0,
    width: float = 30.0, height: float = 30.0,
    label: str = "Object",
    orig_w: int = 64,
    orig_h: int = 64,
) -> dict:
    return {
        "result": [
            {
                "type": "rectanglelabels",
                "original_width": orig_w,
                "original_height": orig_h,
                "value": {
                    "x": x, "y": y,
                    "width": width, "height": height,
                    "rectanglelabels": [label],
                },
            }
        ]
    }


# ─── Context parsing tests ────────────────────────────────────────────────────


class TestParseContext(unittest.TestCase):
    """Test _parse_context without model inference."""

    def _backend(self):
        from model import NewModel
        return object.__new__(NewModel)

    def test_positive_keypoint(self):
        backend = self._backend()
        ctx = _make_keypoint_context(x_pct=25.0, y_pct=75.0, orig_w=200, orig_h=100)
        prompts = backend._parse_context(ctx, 200, 100)

        assert len(prompts["points"]) == 1
        assert prompts["points"][0] == pytest.approx([50.0, 75.0])  # 25%*200, 75%*100
        assert prompts["labels"][0] == 1  # positive
        assert prompts["text"] == "Object"  # label name as text prompt

    def test_negative_keypoint(self):
        backend = self._backend()
        ctx = _make_keypoint_context(label="background", orig_w=100, orig_h=100)
        prompts = backend._parse_context(ctx, 100, 100)

        assert prompts["labels"][0] == 0  # negative
        assert prompts["text"] is None    # negative label not used as text prompt

    def test_rectangle_parsed(self):
        backend = self._backend()
        ctx = _make_rect_context(x=10.0, y=20.0, width=30.0, height=40.0,
                                  orig_w=200, orig_h=200)
        prompts = backend._parse_context(ctx, 200, 200)

        assert len(prompts["boxes"]) == 1
        box = prompts["boxes"][0]
        assert box[0] == pytest.approx(20.0)   # x1 = 10%*200
        assert box[1] == pytest.approx(40.0)   # y1 = 20%*200
        assert box[2] == pytest.approx(80.0)   # x2 = x1 + 30%*200
        assert box[3] == pytest.approx(120.0)  # y2 = y1 + 40%*200
        assert prompts["text"] == "Object"

    def test_empty_context(self):
        backend = self._backend()
        prompts = backend._parse_context({}, 100, 100)
        assert prompts["points"] == []
        assert prompts["labels"] == []
        assert prompts["boxes"] == []
        assert prompts["text"] is None


# ─── Label config parsing tests ──────────────────────────────────────────────


class TestLabelConfigParsing(unittest.TestCase):
    def _backend_with_config(self, config: str):
        from model import NewModel
        b = object.__new__(NewModel)
        b.label_config = config
        return b

    def test_parses_brush_and_image_names(self):
        backend = self._backend_with_config("""
        <View>
          <Image name="img" value="$image"/>
          <BrushLabels name="brush" toName="img">
            <Label value="Car"/>
          </BrushLabels>
        </View>
        """)
        brush_name, img_name, label = backend._parse_label_config()
        assert brush_name == "brush"
        assert img_name == "img"
        assert label == "Car"

    def test_fallback_on_missing_config(self):
        from model import NewModel
        b = object.__new__(NewModel)
        b.label_config = None
        brush_name, img_name, label = b._parse_label_config()
        assert brush_name == "tag"
        assert img_name == "image"
        assert label == "Object"


# ─── Full predict path (mocked SAM3) ─────────────────────────────────────────


class TestPredictWithMockedModel(unittest.TestCase):
    """Full predict() path with mocked facebookresearch/sam3 processor."""

    def _make_backend(self):
        from model import NewModel
        import torch

        backend = object.__new__(NewModel)
        backend.label_config = """
        <View>
          <Image name="image" value="$image"/>
          <BrushLabels name="tag" toName="image">
            <Label value="Object"/>
          </BrushLabels>
        </View>
        """
        backend._image_cache = {}

        # Mock Sam3Processor
        mock_processor = MagicMock()
        mock_state = MagicMock()
        mock_processor.set_image.return_value = mock_state

        # SAM3 PCS output: 2 instances, both with a non-empty 64x64 mask
        n_instances = 2
        h, w = 64, 64
        masks = torch.zeros(n_instances, h, w, dtype=torch.bool)
        masks[0, 10:30, 10:30] = True   # first instance
        masks[1, 35:55, 35:55] = True   # second instance
        scores = torch.tensor([0.95, 0.87])

        mock_out = {"masks": masks, "scores": scores}
        mock_processor.set_text_prompt.return_value = mock_out
        mock_processor.set_point_prompt.return_value = mock_out
        mock_processor.set_box_prompt.return_value = mock_out

        backend._processor = mock_processor

        def mock_get(key):
            return "sam3-image:sam3.1" if key == "model_version" else None

        backend.get = mock_get
        return backend

    def test_predict_returns_model_response(self):
        from label_studio_ml.response import ModelResponse

        backend = self._make_backend()
        with patch.object(backend, "_fetch_image", return_value=_make_synthetic_image()):
            result = backend.predict([_make_task()], context=_make_keypoint_context())

        assert isinstance(result, ModelResponse)
        assert len(result.predictions) == 1

    def test_predict_no_context_returns_empty(self):
        backend = self._make_backend()
        result = backend.predict([_make_task()], context=None)
        assert result.predictions[0]["result"] == []

    def test_predict_returns_multiple_instances(self):
        """SAM3 PCS mode should return one BrushLabels result per instance."""
        backend = self._make_backend()
        ctx = _make_keypoint_context()

        with patch.object(backend, "_fetch_image", return_value=_make_synthetic_image()):
            result = backend.predict([_make_task()], context=ctx)

        pred = result.predictions[0]
        # Both instances have non-empty masks → should get 2 results
        assert len(pred["result"]) == 2
        for r in pred["result"]:
            assert r["type"] == "brushlabels"
            assert "rle" in r["value"]
            assert isinstance(r["value"]["rle"], list)

    def test_text_prompt_has_priority_over_point(self):
        """When label name is set, set_text_prompt should be called."""
        backend = self._make_backend()
        ctx = _make_keypoint_context(label="car")

        with patch.object(backend, "_fetch_image", return_value=_make_synthetic_image()):
            backend.predict([_make_task()], context=ctx)

        backend._processor.set_text_prompt.assert_called_once()
        backend._processor.set_point_prompt.assert_not_called()

    def test_box_prompt_fallback_when_no_text(self):
        """rectanglelabels with non-text-worthy label triggers box prompt."""
        backend = self._make_backend()
        # Use background label so text=None, then rect for box
        ctx = {
            "result": [
                {
                    "type": "keypointlabels",
                    "original_width": 64, "original_height": 64,
                    "value": {"x": 50, "y": 50, "width": 0.5, "keypointlabels": ["background"]},
                },
                {
                    "type": "rectanglelabels",
                    "original_width": 64, "original_height": 64,
                    "value": {"x": 10, "y": 10, "width": 30, "height": 30,
                               "rectanglelabels": ["background"]},
                },
            ]
        }
        with patch.object(backend, "_fetch_image", return_value=_make_synthetic_image()):
            backend.predict([_make_task()], context=ctx)

        # text=None (background is negative), so box prompt should be used
        backend._processor.set_box_prompt.assert_called_once()
        backend._processor.set_text_prompt.assert_not_called()


# ─── RLE encoding ────────────────────────────────────────────────────────────


class TestRLEEncoding(unittest.TestCase):
    def test_rle_roundtrip(self):
        from label_studio_converter import brush

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 255

        rle = brush.mask2rle(mask)
        assert isinstance(rle, list)
        assert len(rle) > 0

        decoded = brush.decode_rle(rle)
        decoded_2d = np.frombuffer(decoded, dtype=np.uint8).reshape(64, 64, 4)
        assert decoded_2d[20, 20, 3] > 0  # inside square
        assert decoded_2d[0, 0, 3] == 0   # outside square


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
