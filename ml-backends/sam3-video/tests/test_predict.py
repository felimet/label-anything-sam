"""Smoke tests for SAM3 video backend (NewModel).

Run:
    cd ml-backends/sam3-video && pytest tests/ --tb=short -v

SAM3 video predictor is fully mocked — no weights download, no GPU required.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("SAM3_MODEL_ID", "facebook/sam3.1")
os.environ.setdefault("MAX_FRAMES_TO_TRACK", "3")


def _make_task(video_url: str = "http://fake/video.mp4", task_id: int = 1) -> dict:
    return {"id": task_id, "data": {"video": video_url}}


def _make_video_context(
    frame: int = 0,
    x: float = 10.0, y: float = 10.0,
    width: float = 30.0, height: float = 20.0,
    label: str = "Object",
) -> dict:
    return {
        "result": [
            {
                "type": "videorectangle",
                "value": {
                    "labels": [label],
                    "sequence": [
                        {
                            "frame": frame,
                            "enabled": True,
                            "x": x, "y": y,
                            "width": width, "height": height,
                        }
                    ],
                },
            }
        ]
    }


# ─── Prompt extraction tests ──────────────────────────────────────────────────


class TestGetPrompts(unittest.TestCase):
    def _backend(self):
        from model import NewModel
        return object.__new__(NewModel)

    def test_extracts_videorectangle(self):
        backend = self._backend()
        ctx = _make_video_context(frame=5, x=10.0, y=20.0, width=30.0, height=15.0)
        prompts = backend._get_prompts(ctx)

        assert len(prompts) == 1
        assert prompts[0]["frame"] == 5
        assert prompts[0]["label"] == "Object"
        assert prompts[0]["box_pct"] == pytest.approx([10.0, 20.0, 30.0, 15.0])

    def test_empty_context(self):
        backend = self._backend()
        assert backend._get_prompts({}) == []

    def test_disabled_frame_skipped(self):
        backend = self._backend()
        ctx = {
            "result": [
                {
                    "type": "videorectangle",
                    "value": {
                        "labels": ["car"],
                        "sequence": [
                            {"frame": 0, "enabled": False, "x": 5, "y": 5, "width": 10, "height": 10},
                            {"frame": 1, "enabled": True,  "x": 6, "y": 6, "width": 11, "height": 11},
                        ],
                    },
                }
            ]
        }
        prompts = backend._get_prompts(ctx)
        assert len(prompts) == 1
        assert prompts[0]["frame"] == 1

    def test_sorted_by_frame(self):
        backend = self._backend()
        ctx = {
            "result": [
                {
                    "type": "videorectangle",
                    "value": {
                        "labels": ["person"],
                        "sequence": [
                            {"frame": 10, "enabled": True, "x": 0, "y": 0, "width": 10, "height": 10},
                            {"frame": 2,  "enabled": True, "x": 0, "y": 0, "width": 10, "height": 10},
                        ],
                    },
                }
            ]
        }
        prompts = backend._get_prompts(ctx)
        frames = [p["frame"] for p in prompts]
        assert frames == sorted(frames)


# ─── Full predict path (mocked SAM3 video predictor) ─────────────────────────


class TestPredictWithMockedPredictor(unittest.TestCase):
    """Full predict() path with mocked build_sam3_video_predictor."""

    def _make_backend(self):
        from model import NewModel

        backend = object.__new__(NewModel)

        mock_predictor = MagicMock()

        # start_session returns session_id
        mock_predictor.handle_request.side_effect = self._mock_handle_request

        backend._predictor = mock_predictor
        backend._sessions = {}

        def mock_get(key):
            return "sam3-video:sam3.1" if key == "model_version" else None

        backend.get = mock_get
        backend.label_interface = MagicMock()
        backend.label_interface.get_first_tag_occurence.return_value = (
            "box", "video", None
        )
        return backend

    _call_count = 0

    def _mock_handle_request(self, req):
        req_type = req.get("type")
        if req_type == "start_session":
            return {"session_id": "mock-session-001"}
        elif req_type == "add_prompt":
            return {"status": "ok"}
        elif req_type == "get_output":
            fi = req.get("frame_index", 0)
            return {"bbox_pct": [10.0 + fi, 10.0, 30.0, 20.0]}
        return {}

    def test_predict_returns_model_response(self):
        from label_studio_ml.response import ModelResponse

        backend = self._make_backend()
        ctx = _make_video_context()

        with patch.object(backend, "get_local_path", return_value="/tmp/fake.mp4"):
            result = backend.predict([_make_task()], context=ctx)

        assert isinstance(result, ModelResponse)
        assert len(result.predictions) == 1

    def test_predict_no_context_returns_empty(self):
        backend = self._make_backend()
        result = backend.predict([_make_task()], context=None)
        assert result.predictions[0]["result"] == []

    def test_predict_returns_videorectangle_sequence(self):
        backend = self._make_backend()
        ctx = _make_video_context()

        with patch.object(backend, "get_local_path", return_value="/tmp/fake.mp4"):
            result = backend.predict([_make_task()], context=ctx)

        pred = result.predictions[0]
        assert len(pred["result"]) == 1
        r = pred["result"][0]
        assert r["type"] == "videorectangle"
        assert "sequence" in r["value"]
        # MAX_FRAMES_TO_TRACK=3 in env → at most 3 frames
        assert len(r["value"]["sequence"]) <= 3

    def test_session_created(self):
        backend = self._make_backend()
        ctx = _make_video_context()

        with patch.object(backend, "get_local_path", return_value="/tmp/fake.mp4"):
            backend.predict([_make_task(task_id=42)], context=ctx)

        assert 42 in backend._sessions
        assert backend._sessions[42] == "mock-session-001"

    def test_no_video_url_returns_empty(self):
        backend = self._make_backend()
        task_no_video = {"id": 1, "data": {}}
        ctx = _make_video_context()

        result = backend.predict([task_no_video], context=ctx)
        assert result.predictions[0]["result"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
