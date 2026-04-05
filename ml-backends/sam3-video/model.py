"""SAM3 video segmentation backend for Label Studio (skeleton — minimally viable).

Implements LabelStudioMLBase using facebookresearch/sam3 video predictor:
  - Input:  VideoRectangle prompt from Label Studio (first annotated frame)
  - Output: VideoRectangle sequence (tracked bounding boxes across N frames)

SAM3 video API uses a session-based interface:
    predictor.handle_request({"type": "start_session", "resource_path": video_path})
    predictor.handle_request({"type": "add_prompt", "session_id": ..., ...})
    predictor.handle_request({"type": "get_output", "session_id": ..., "frame_index": i})

Known limitations (skeleton scope):
  - Session eviction: sessions live in memory dict; no TTL / LRU eviction.
  - Max frames: MAX_FRAMES_TO_TRACK (env, default 10).
  - No multi-object tracking in a single session.
  - No mask → polygon conversion; outputs VideoRectangle (bbox) only.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

# SAM3 package — installed via `pip install -e .` from facebookresearch/sam3
from sam3.model_builder import build_sam3_video_predictor

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID: str = os.getenv("SAM3_MODEL_ID", "facebook/sam3.1")
CHECKPOINT_FILENAME: str = os.getenv("SAM3_CHECKPOINT_FILENAME", "sam3.1.pt")
MODEL_DIR: str = os.getenv("MODEL_DIR", "/data/models")
MAX_FRAMES_TO_TRACK: int = int(os.getenv("MAX_FRAMES_TO_TRACK", "10"))


class NewModel(LabelStudioMLBase):
    """SAM3 video tracking backend.

    Workflow:
    1. User draws a VideoRectangle on a frame in Label Studio.
    2. predict() extracts the annotated frame + bbox, initialises a SAM3 session,
       adds the prompt, then queries outputs for MAX_FRAMES_TO_TRACK frames.
    3. Returns a VideoRectangle sequence (normalized bboxes per frame).
    """

    def setup(self) -> None:
        hf_token: Optional[str] = os.getenv("HF_TOKEN") or None

        logger.info("Downloading SAM3 checkpoint '%s' …", MODEL_ID)
        os.makedirs(MODEL_DIR, exist_ok=True)
        try:
            checkpoint_path = hf_hub_download(
                repo_id=MODEL_ID,
                filename=CHECKPOINT_FILENAME,
                local_dir=MODEL_DIR,
                token=hf_token,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download SAM3 checkpoint from '{MODEL_ID}'. "
                "Ensure HF_TOKEN is set and you have accepted the model license at "
                f"https://huggingface.co/{MODEL_ID}"
            ) from exc

        logger.info("Loading SAM3 video predictor on %s from %s", DEVICE, checkpoint_path)
        self._predictor = build_sam3_video_predictor(
            checkpoint_path=checkpoint_path, device=DEVICE
        )

        # task_id → session_id (in-memory; no eviction in skeleton scope)
        self._sessions: dict[int, str] = {}

        self.set("model_version", f"sam3-video:{MODEL_ID.split('/')[-1]}")
        logger.info("SAM3 video predictor loaded. Device: %s", DEVICE)

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        tasks: list[dict],
        context: Optional[dict] = None,
        **kwargs,
    ) -> ModelResponse:
        if not context or not context.get("result"):
            return ModelResponse(
                predictions=[{"result": [], "score": 0.0} for _ in tasks]
            )

        try:
            from_name, to_name, _ = self.label_interface.get_first_tag_occurence(
                "VideoRectangle", "Video"
            )
        except Exception:
            from_name, to_name = "box", "video"

        predictions = []
        for task in tasks:
            pred = self._predict_single(task, context, from_name, to_name)
            predictions.append(pred)
        return ModelResponse(predictions=predictions)

    def fit(self, event: str, data: dict, **kwargs) -> None:
        logger.info("Received event '%s' (fine-tuning not implemented)", event)

    # ── Private: single-task inference ───────────────────────────────────────

    def _predict_single(
        self,
        task: dict,
        context: dict,
        from_name: str,
        to_name: str,
    ) -> dict:
        video_url = task.get("data", {}).get("video")
        if not video_url:
            logger.warning("No video URL in task: %s", list(task.get("data", {}).keys()))
            return {"result": [], "score": 0.0}

        # Download video via Label Studio SDK helper
        try:
            video_path = self.get_local_path(video_url, task_id=task.get("id"))
        except Exception as exc:
            logger.error("Failed to download video: %s", exc)
            return {"result": [], "score": 0.0}

        # Parse VideoRectangle prompts from context
        prompt_frames = self._get_prompts(context)
        if not prompt_frames:
            return {"result": [], "score": 0.0}

        # Use first annotated frame as the prompt
        first = prompt_frames[0]
        frame_idx = first["frame"]
        label_name = first["label"]
        box_px = first.get("box_pct")  # [x_pct, y_pct, w_pct, h_pct]

        # SAM3 video session
        task_id = task.get("id", 0)
        try:
            session_resp = self._predictor.handle_request({
                "type": "start_session",
                "resource_path": video_path,
            })
            session_id = session_resp["session_id"]
            self._sessions[task_id] = session_id

            prompt_req: dict = {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_idx,
            }
            if label_name:
                prompt_req["text"] = label_name
            if box_px is not None:
                prompt_req["box_pct"] = box_px  # pass normalised % coords to SAM3

            self._predictor.handle_request(prompt_req)

            # Collect outputs for subsequent frames
            sequence: list[dict] = []
            for i in range(frame_idx, frame_idx + MAX_FRAMES_TO_TRACK):
                try:
                    out = self._predictor.handle_request({
                        "type": "get_output",
                        "session_id": session_id,
                        "frame_index": i,
                    })
                    bbox_pct = out.get("bbox_pct")  # [x, y, w, h] in %
                    if bbox_pct is None:
                        break
                    sequence.append({
                        "frame": i,
                        "enabled": True,
                        "x": bbox_pct[0],
                        "y": bbox_pct[1],
                        "width": bbox_pct[2],
                        "height": bbox_pct[3],
                        "rotation": 0,
                    })
                except Exception as exc:
                    logger.debug("Frame %d query error: %s", i, exc)
                    break

        except Exception as exc:
            logger.error("SAM3 video session error: %s", exc)
            return {"result": [], "score": 0.0}

        if not sequence:
            return {"result": [], "score": 0.0}

        result = {
            "from_name": from_name,
            "to_name": to_name,
            "type": "videorectangle",
            "value": {
                "sequence": sequence,
                "labels": [label_name] if label_name else [],
                "framesCount": sequence[-1]["frame"] + 1,
            },
        }
        return {
            "result": [result],
            "score": 0.9,
            "model_version": self.get("model_version"),
        }

    # ── Private: prompt extraction ───────────────────────────────────────────

    def _get_prompts(self, context: dict) -> list[dict]:
        """Extract VideoRectangle prompts from Label Studio context.

        Returns list of dicts: [{frame, label, box_pct}, ...]
        sorted by frame index.
        """
        prompts = []
        for item in context.get("result", []):
            if item.get("type") != "videorectangle":
                continue
            value = item.get("value", {})
            labels = value.get("labels", [])
            label_name = labels[0] if labels else None

            for seq in value.get("sequence", []):
                if not seq.get("enabled", True):
                    continue
                prompts.append({
                    "frame": seq.get("frame", 0),
                    "label": label_name,
                    "box_pct": [
                        seq.get("x", 0.0),
                        seq.get("y", 0.0),
                        seq.get("width", 0.0),
                        seq.get("height", 0.0),
                    ],
                })

        return sorted(prompts, key=lambda p: p["frame"])
