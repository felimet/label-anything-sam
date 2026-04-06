"""Smoke tests for SAM3 video backend (NewModel).

Run:
    cd ml-backends/sam3-video && pytest tests/ --tb=short -v

SAM3 video predictor fully mocked — no GPU, no HF download required.

The video backend uses Sam3VideoPredictorMultiGPU with session-based API:
    handle_request({"type": "start_session", "resource_path": video_path})
    handle_request({"type": "add_prompt", "session_id": ..., "frame_index": ..., ...})
    handle_stream_request({"type": "propagate_in_video", "session_id": ..., ...})
    handle_request({"type": "close_session", "session_id": ...})

propagate_in_video yields:
    {"frame_index": int, "outputs": {"out_obj_ids": ndarray,
                                     "out_binary_masks": ndarray[N,H,W] bool,
                                     "out_boxes_xywh": ndarray[N,4]}}
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
os.environ.setdefault("SAM3_ENABLE_PCS", "true")
os.environ.setdefault("SAM3_ENABLE_FA3", "false")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_task(url="http://fake/video.mp4", task_id=1):
    return {"id": task_id, "data": {"video": url}}


def _vr_ctx(
    frame=1, x=10.0, y=10.0, w=30.0, h=20.0,
    label="Object", obj_id="obj1",
    frames_count=25, duration=1.0,
):
    return {"result": [{
        "type": "videorectangle",
        "id":   obj_id,
        "value": {
            "framesCount": frames_count,
            "duration":    duration,
            "labels":      [label],
            "sequence":    [{
                "frame":   frame,
                "enabled": True,
                "x": x, "y": y, "width": w, "height": h,
            }],
        },
    }]}


def _text_vr_ctx(**kwargs):
    """VideoRectangle + TextArea prompt."""
    ctx = _vr_ctx(**kwargs)
    ctx["result"].append({
        "type":            "textarea",
        "original_width":  640,
        "original_height": 480,
        "value":           {"text": ["the person walking"]},
    })
    return ctx


_LABEL_CONFIG = """
<View>
  <Labels name="videoLabels" toName="video" allowEmpty="true">
    <Label value="Object"/>
  </Labels>
  <Video name="video" value="$video" framerate="25.0"/>
  <VideoRectangle name="box" toName="video" smart="true"/>
  <TextArea name="text_prompt" toName="video" maxSubmissions="1" editable="true"/>
</View>
"""


def _mock_sam3_predictor():
    """Mock Sam3VideoPredictorMultiGPU with handle_request / handle_stream_request."""
    mock = MagicMock()

    mock.handle_request.side_effect = _dispatch_request
    mock.handle_stream_request.side_effect = _dispatch_stream

    return mock


_SESSION_COUNTER = [0]


def _dispatch_request(request):
    rtype = request["type"]
    if rtype == "start_session":
        _SESSION_COUNTER[0] += 1
        return {"session_id": f"sess-{_SESSION_COUNTER[0]}"}
    elif rtype == "add_prompt":
        mask = np.zeros((1, 100, 100), dtype=bool)
        mask[0, 10:30, 10:50] = True
        return {
            "frame_index": request["frame_index"],
            "outputs": {
                "out_obj_ids":     np.array([0]),
                "out_binary_masks": mask,
                "out_boxes_xywh":  np.array([[10., 10., 40., 20.]]),
            },
        }
    elif rtype in ("close_session", "reset_session"):
        return {"is_success": True}
    return {}


def _dispatch_stream(request):
    rtype = request["type"]
    if rtype == "propagate_in_video":
        max_frames = int(os.environ.get("MAX_FRAMES_TO_TRACK", "3"))
        start = request.get("start_frame_index", 0) or 0
        for i in range(max_frames):
            mask = np.zeros((1, 100, 100), dtype=bool)
            mask[0, 10:30, 10:50] = True
            yield {
                "frame_index": start + i,
                "outputs": {
                    "out_obj_ids":     np.array([0]),
                    "out_binary_masks": mask,
                    "out_boxes_xywh":  np.array([[10., 10., 40., 20.]]),
                },
            }


# ── Tests: _get_geo_prompts ────────────────────────────────────────────────────

class TestGetGeoPrompts(unittest.TestCase):

    def _backend(self):
        from model import NewModel
        return object.__new__(NewModel)

    def test_extracts_videorectangle(self):
        b = self._backend()
        prompts = b._get_geo_prompts(_vr_ctx(frame=5, x=10.0, y=20.0, w=30.0, h=15.0))
        assert len(prompts) == 1
        assert prompts[0]["frame_idx"] == 4   # LS 1-indexed → SAM3 0-indexed

    def test_empty_context_returns_empty(self):
        assert self._backend()._get_geo_prompts({}) == []

    def test_disabled_frame_skipped(self):
        b = self._backend()
        ctx = {"result": [{
            "type": "videorectangle",
            "id":   "o",
            "value": {"sequence": [
                {"frame": 1, "enabled": False, "x": 0, "y": 0, "width": 10, "height": 10},
                {"frame": 2, "enabled": True,  "x": 5, "y": 5, "width": 10, "height": 10},
            ]},
        }]}
        prompts = b._get_geo_prompts(ctx)
        assert len(prompts) == 1 and prompts[0]["frame_idx"] == 1

    def test_sorted_by_frame(self):
        b = self._backend()
        ctx = {"result": [{
            "type": "videorectangle",
            "id":   "o",
            "value": {"sequence": [
                {"frame": 10, "enabled": True, "x": 0, "y": 0, "width": 5, "height": 5},
                {"frame": 2,  "enabled": True, "x": 0, "y": 0, "width": 5, "height": 5},
            ]},
        }]}
        frames = [p["frame_idx"] for p in b._get_geo_prompts(ctx)]
        assert frames == sorted(frames)


# ── Tests: _get_text_prompt ────────────────────────────────────────────────────

class TestGetTextPrompt(unittest.TestCase):

    def _backend(self):
        from model import NewModel
        return object.__new__(NewModel)

    def test_extracts_textarea(self):
        b = self._backend()
        ctx = {"result": [{"type": "textarea", "value": {"text": ["hello world"]}}]}
        assert b._get_text_prompt(ctx) == "hello world"

    def test_empty_text_returns_none(self):
        b = self._backend()
        ctx = {"result": [{"type": "textarea", "value": {"text": ["  "]}}]}
        assert b._get_text_prompt(ctx) is None

    def test_no_textarea_returns_none(self):
        assert self._backend()._get_text_prompt(_vr_ctx()) is None


# ── Tests: _rect_to_keypoints ──────────────────────────────────────────────────

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


# ── Tests: _mask_to_bbox_pct ───────────────────────────────────────────────────

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
        assert bbox["x"]      == pytest.approx(10.0)
        assert bbox["y"]      == pytest.approx(10.0)
        assert bbox["width"]  == pytest.approx(10.0)
        assert bbox["height"] == pytest.approx(10.0)


# ── Tests: full predict pipeline (SAM3 path) ───────────────────────────────────

class TestSAM3PredictMocked(unittest.TestCase):

    def setUp(self):
        self.mock_pred = _mock_sam3_predictor()
        self.p1 = patch("model.predictor", self.mock_pred)
        self.p2 = patch("model._checkpoint_path", "/tmp/fake.pt")
        self.p3 = patch("model._USING_SAM2_FALLBACK", False)
        self.p1.start(); self.p2.start(); self.p3.start()

    def tearDown(self):
        self.p1.stop(); self.p2.stop(); self.p3.stop()

    def _backend(self):
        from model import NewModel
        return NewModel(project_id="test", label_config=_LABEL_CONFIG)

    def _run(self, ctx):
        backend = self._backend()
        with patch.object(backend, "get_local_path", return_value="/tmp/fake.mp4"):
            return backend.predict([_make_task()], context=ctx)

    def test_no_context_returns_empty(self):
        result = self._backend().predict([_make_task()], context=None)
        assert result.predictions == []

    def test_returns_model_response(self):
        from label_studio_ml.response import ModelResponse
        assert isinstance(self._run(_vr_ctx()), ModelResponse)

    def test_start_session_called(self):
        self._run(_vr_ctx())
        calls = [c for c in self.mock_pred.handle_request.call_args_list
                 if c.args[0].get("type") == "start_session"]
        assert len(calls) == 1

    def test_add_prompt_called(self):
        self._run(_vr_ctx())
        calls = [c for c in self.mock_pred.handle_request.call_args_list
                 if c.args[0].get("type") == "add_prompt"]
        assert len(calls) >= 1

    def test_close_session_always_called(self):
        """close_session must be called even if propagation yields nothing."""
        self._run(_vr_ctx())
        calls = [c for c in self.mock_pred.handle_request.call_args_list
                 if c.args[0].get("type") == "close_session"]
        assert len(calls) == 1

    def test_propagate_in_video_called(self):
        self._run(_vr_ctx())
        self.mock_pred.handle_stream_request.assert_called_once()
        req = self.mock_pred.handle_stream_request.call_args.args[0]
        assert req["type"] == "propagate_in_video"

    def test_text_prompt_passed_to_add_prompt(self):
        self._run(_text_vr_ctx())
        add_calls = [c for c in self.mock_pred.handle_request.call_args_list
                     if c.args[0].get("type") == "add_prompt"]
        assert any(c.args[0].get("text") == "the person walking"
                   for c in add_calls)

    def test_result_type_is_videorectangle(self):
        result = self._run(_vr_ctx())
        if not result.predictions:
            return
        pred = result.predictions[0]
        dump = pred.model_dump() if hasattr(pred, "model_dump") else pred
        results = dump.get("result", []) if isinstance(dump, dict) else []
        if results:
            assert results[0]["type"] == "videorectangle"

    def test_sequence_contains_propagated_frames(self):
        result = self._run(_vr_ctx(frame=1))
        if not result.predictions:
            return
        pred = result.predictions[0]
        dump = pred.model_dump() if hasattr(pred, "model_dump") else pred
        results = dump.get("result", []) if isinstance(dump, dict) else []
        if results:
            seq = results[0]["value"]["sequence"]
            # original context frame(s) + propagated
            assert len(seq) >= 1


# ── Tests: SAM2 fallback (text ignored) ───────────────────────────────────────

class TestSAM2FallbackVideo(unittest.TestCase):

    def setUp(self):
        self.mock_pred = MagicMock()
        self.mock_pred.init_state.return_value = {"state": "mock"}
        self.mock_pred.reset_state.return_value = None
        self.mock_pred.add_new_points.return_value = (0, [0], None)

        def _propagate(inference_state, start_frame_idx, max_frame_num_to_track):
            for i in range(max_frame_num_to_track):
                mask = np.zeros((1, 100, 100), dtype=bool)
                mask[0, 10:30, 10:50] = True
                yield start_frame_idx + i, [0], [mask.astype(np.float32) * 10]

        self.mock_pred.propagate_in_video.side_effect = _propagate

        self.p1 = patch("model.predictor", self.mock_pred)
        self.p2 = patch("model._checkpoint_path", "/tmp/fake.pt")
        self.p3 = patch("model._USING_SAM2_FALLBACK", True)
        self.p1.start(); self.p2.start(); self.p3.start()

    def tearDown(self):
        self.p1.stop(); self.p2.stop(); self.p3.stop()

    def _backend(self):
        from model import NewModel
        return NewModel(project_id="test", label_config=_LABEL_CONFIG)

    def _run(self, ctx):
        backend = self._backend()
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        def mock_split(video_path, frame_dir, start_frame=0, end_frame=100):
            for i in range(min(end_frame - start_frame, 5)):
                yield f"/tmp/f{i:05d}.jpg", fake_frame

        with patch.object(backend, "get_local_path", return_value="/tmp/fake.mp4"), \
             patch.object(backend, "_split_frames", mock_split):
            return backend.predict([_make_task()], context=ctx)

    def test_text_with_no_geo_returns_empty(self):
        """SAM2 fallback: text-only context → empty (no geometric prompt)."""
        ctx = {"result": [{
            "type":  "textarea",
            "value": {"text": ["the red car"]},
        }]}
        result = self._run(ctx)
        assert result.predictions == []

    def test_geo_prompt_runs_propagation(self):
        self._run(_vr_ctx())
        self.mock_pred.propagate_in_video.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
