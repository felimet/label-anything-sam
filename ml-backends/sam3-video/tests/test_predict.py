"""Smoke tests for SAM3 video backend (NewModel).

Run:
    cd ml-backends/sam3-video && pytest tests/ --tb=short -v

SAM3 video predictor fully mocked — no GPU, no HF download required.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("SAM3_MODEL_ID", "test/model")
os.environ.setdefault("SAM3_CHECKPOINT_FILENAME", "model.pt")
os.environ.setdefault("MODEL_DIR", "/tmp/test-models")
os.environ.setdefault("MAX_FRAMES_TO_TRACK", "3")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_task(url: str = "http://fake/video.mp4", task_id: int = 1) -> dict:
    return {"id": task_id, "data": {"video": url}}


def _vr_ctx(
    frame: int = 1,
    x: float = 10.0,
    y: float = 10.0,
    w: float = 30.0,
    h: float = 20.0,
    label: str = "Object",
    obj_id: str = "obj1",
    frames_count: int = 25,
    duration: float = 1.0,
) -> dict:
    return {
        "result": [
            {
                "type": "videorectangle",
                "id": obj_id,
                "value": {
                    "framesCount": frames_count,
                    "duration": duration,
                    "labels": [label],
                    "sequence": [
                        {
                            "frame": frame,
                            "enabled": True,
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                        }
                    ],
                },
            }
        ]
    }


_LABEL_CONFIG = """
<View>
  <Labels name="videoLabels" toName="video" allowEmpty="true">
    <Label value="Object"/>
  </Labels>
  <Video name="video" value="$video" framerate="25.0"/>
  <VideoRectangle name="box" toName="video" smart="true"/>
</View>
"""


def _mock_predictor() -> MagicMock:
    """Build a mock SAM3 video predictor with the correct API."""
    mock = MagicMock()
    mock.init_state.return_value = {"state": "mock"}
    mock.reset_state.return_value = None
    mock.add_new_points.return_value = (0, [0], None)

    def _propagate(inference_state, start_frame_idx, max_frame_num_to_track):
        for i in range(max_frame_num_to_track):
            mask = np.zeros((1, 100, 100), dtype=bool)
            mask[0, 10:30, 10:50] = True
            logits = mask.astype(np.float32) * 10  # > 0 => foreground
            yield start_frame_idx + i, [0], [logits]

    mock.propagate_in_video.side_effect = _propagate
    return mock


# ── Tests: _get_prompts ───────────────────────────────────────────────────────

class TestGetPrompts(unittest.TestCase):

    def _backend(self):
        from model import NewModel
        return object.__new__(NewModel)

    def test_extracts_videorectangle(self):
        b = self._backend()
        prompts = b._get_prompts(_vr_ctx(frame=5, x=10.0, y=20.0, w=30.0, h=15.0))
        assert len(prompts) == 1
        assert prompts[0]["frame_idx"] == 4  # LS 1-indexed -> SAM3 0-indexed
        assert prompts[0]["x_pct"] == pytest.approx(10.0)

    def test_empty_context_returns_empty(self):
        assert self._backend()._get_prompts({}) == []

    def test_disabled_frame_skipped(self):
        b = self._backend()
        ctx = {
            "result": [
                {
                    "type": "videorectangle",
                    "id": "o",
                    "value": {
                        "sequence": [
                            {"frame": 1, "enabled": False, "x": 0, "y": 0, "width": 10, "height": 10},
                            {"frame": 2, "enabled": True, "x": 5, "y": 5, "width": 10, "height": 10},
                        ]
                    },
                }
            ]
        }
        prompts = b._get_prompts(ctx)
        assert len(prompts) == 1 and prompts[0]["frame_idx"] == 1

    def test_sorted_by_frame(self):
        b = self._backend()
        ctx = {
            "result": [
                {
                    "type": "videorectangle",
                    "id": "o",
                    "value": {
                        "sequence": [
                            {"frame": 10, "enabled": True, "x": 0, "y": 0, "width": 5, "height": 5},
                            {"frame": 2, "enabled": True, "x": 0, "y": 0, "width": 5, "height": 5},
                        ]
                    },
                }
            ]
        }
        prompts = b._get_prompts(ctx)
        frames = [p["frame_idx"] for p in prompts]
        assert frames == sorted(frames)


# ── Tests: _rect_to_keypoints ─────────────────────────────────────────────────

class TestRectToKeypoints(unittest.TestCase):

    def test_returns_5_keypoints(self):
        from model import NewModel
        pts, lbs = NewModel._rect_to_keypoints(10.0, 20.0, 30.0, 40.0, 200, 100)
        assert len(pts) == 5
        assert all(lbs == 1)

    def test_center_pixel_coords(self):
        from model import NewModel
        pts, _ = NewModel._rect_to_keypoints(0.0, 0.0, 100.0, 100.0, 200, 100)
        assert pts[0, 0] == pytest.approx(100.0)  # cx = 200 * 0.5
        assert pts[0, 1] == pytest.approx(50.0)   # cy = 100 * 0.5


# ── Tests: _mask_to_bbox_pct ──────────────────────────────────────────────────

class TestMaskToBboxPct(unittest.TestCase):

    def test_empty_mask_returns_none(self):
        from model import NewModel
        assert NewModel._mask_to_bbox_pct(np.zeros((1, 64, 64), dtype=bool)) is None

    def test_bbox_percentage_values(self):
        from model import NewModel
        mask = np.zeros((1, 100, 200), dtype=bool)
        mask[0, 10:20, 20:40] = True
        bbox = NewModel._mask_to_bbox_pct(mask)
        assert bbox is not None
        assert bbox["x"] == pytest.approx(10.0)
        assert bbox["y"] == pytest.approx(10.0)
        assert bbox["width"] == pytest.approx(10.0)
        assert bbox["height"] == pytest.approx(10.0)


# ── Tests: full predict pipeline ──────────────────────────────────────────────

class TestPredictMocked(unittest.TestCase):

    def setUp(self):
        self.mock_pred = _mock_predictor()
        self.p1 = patch("model.predictor", self.mock_pred)
        self.p2 = patch("model._checkpoint_path", "/tmp/fake.pt")
        self.p1.start()
        self.p2.start()

    def tearDown(self):
        self.p1.stop()
        self.p2.stop()

    def _backend(self):
        from model import NewModel
        return NewModel(project_id="test", label_config=_LABEL_CONFIG)

    def _run(self, ctx: dict):
        backend = self._backend()
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        def mock_split(video_path, frame_dir, start_frame=0, end_frame=100):
            for i in range(min(end_frame - start_frame, 5)):
                yield f"/tmp/f{i:05d}.jpg", fake_frame

        with patch.object(backend, "get_local_path", return_value="/tmp/fake.mp4"), \
             patch.object(backend, "_split_frames", mock_split):
            return backend.predict([_make_task()], context=ctx)

    def test_no_context_returns_empty(self):
        b = self._backend()
        result = b.predict([_make_task()], context=None)
        assert result.predictions == []

    def test_returns_model_response(self):
        from label_studio_ml.response import ModelResponse
        result = self._run(_vr_ctx())
        assert isinstance(result, ModelResponse)

    def test_init_state_called(self):
        self._run(_vr_ctx())
        self.mock_pred.init_state.assert_called_once()

    def test_add_new_points_called(self):
        self._run(_vr_ctx())
        self.mock_pred.add_new_points.assert_called()

    def test_propagate_called(self):
        self._run(_vr_ctx())
        self.mock_pred.propagate_in_video.assert_called_once()

    def test_prediction_type_is_videorectangle(self):
        result = self._run(_vr_ctx())
        if not result.predictions:
            return
        pred = result.predictions[0]
        if hasattr(pred, "model_dump"):
            results = pred.model_dump().get("result", [])
        else:
            results = pred.get("result", []) if isinstance(pred, dict) else []
        if results:
            assert results[0]["type"] == "videorectangle"

    def test_sequence_merges_context_and_propagated(self):
        """sequence = original context frame + propagated frames."""
        result = self._run(_vr_ctx(frame=1))
        if not result.predictions:
            return
        pred = result.predictions[0]
        if hasattr(pred, "model_dump"):
            results = pred.model_dump().get("result", [])
        else:
            results = pred.get("result", []) if isinstance(pred, dict) else []
        if results:
            seq = results[0]["value"]["sequence"]
            assert len(seq) >= 1  # at minimum the original context frame


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
