"""SAM3 video tracking backend for Label Studio.

Follows the official SAM2 video example from HumanSignal/label-studio-ml-backend.

Model is loaded at module scope (singleton) — same reason as image backend:
label_studio_ml.api instantiates MODEL_CLASS on every /predict request.

SAM3 video API (assumed identical to SAM2 video API):
  predictor.init_state(video_path=frame_dir)
  predictor.add_new_points(inference_state, frame_idx, obj_id, points, labels)
  predictor.propagate_in_video(inference_state, start_frame_idx, max_frame_num_to_track)

If SAM3 package is unavailable, falls back to SAM2 video predictor.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID: str = os.getenv("SAM3_MODEL_ID", "facebook/sam3.1")
CHECKPOINT_FILENAME: str = os.getenv("SAM3_CHECKPOINT_FILENAME", "sam3.1.pt")
MODEL_DIR: str = os.getenv("MODEL_DIR", "/data/models")
MAX_FRAMES_TO_TRACK: int = int(os.getenv("MAX_FRAMES_TO_TRACK", "10"))

# ── CUDA optimisations ────────────────────────────────────────────────────────
if DEVICE == "cuda":
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# ── Model loading (module-level singleton) ────────────────────────────────────
try:
    from huggingface_hub import hf_hub_download

    hf_token: Optional[str] = os.getenv("HF_TOKEN") or None
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info("Downloading SAM3 video checkpoint '%s/%s' …", MODEL_ID, CHECKPOINT_FILENAME)
    _checkpoint_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=CHECKPOINT_FILENAME,
        local_dir=MODEL_DIR,
        token=hf_token,
    )
    logger.info("Checkpoint at: %s", _checkpoint_path)
except Exception as _hf_err:
    raise RuntimeError(
        f"Failed to download SAM3 checkpoint from '{MODEL_ID}'. "
        "Ensure HF_TOKEN is set and you have accepted the model license at "
        f"https://huggingface.co/{MODEL_ID}"
    ) from _hf_err

try:
    # Real SAM3 package
    from sam3.model_builder import build_sam3_video_predictor  # type: ignore[import]

    logger.info("Loading SAM3 video predictor on %s …", DEVICE)
    predictor = build_sam3_video_predictor(_checkpoint_path, device=DEVICE)
    logger.info("SAM3 video predictor loaded.")

except ImportError:
    # Fallback to SAM2 video predictor (same API)
    logger.warning(
        "SAM3 package not found — falling back to SAM2 video predictor. "
        "Swap imports when facebookresearch/sam3 is released."
    )
    ROOT_DIR = os.getcwd()
    sys.path.insert(0, ROOT_DIR)
    from sam2.build_sam import build_sam2_video_predictor  # type: ignore[import]

    predictor = build_sam2_video_predictor(
        os.getenv("MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml"),
        _checkpoint_path,
        device=DEVICE,
    )
    logger.info("SAM2 video predictor loaded (SAM3 fallback).")

# ── Inference state cache (module-level, process-shared via gunicorn preload) ─
# Key: frame directory path → inference state.
# TODO: add LRU eviction for production use.
_state_cache_key: str = ""
_inference_state = None


def _get_inference_state(frame_dir: str):
    global _state_cache_key, _inference_state
    if _state_cache_key != frame_dir:
        _state_cache_key = frame_dir
        _inference_state = predictor.init_state(video_path=frame_dir)
    return _inference_state


# ── Backend class ─────────────────────────────────────────────────────────────

class NewModel(LabelStudioMLBase):
    """SAM3 video tracking backend.

    Workflow:
    1. User draws VideoRectangle on a frame in Label Studio.
    2. predict() downloads the video, splits it to frames, initialises SAM3 state.
    3. Rectangle prompt → 5 keypoints (center + cardinal) passed to add_new_points().
    4. propagate_in_video() runs for MAX_FRAMES_TO_TRACK frames.
    5. Per-frame masks → bboxes → VideoRectangle sequence returned to Label Studio.

    The original context sequence (user-drawn boxes) is prepended to the output
    so annotators keep their reference frames.
    """

    def setup(self) -> None:
        self.set("model_version", f"sam3-video:{MODEL_ID.split('/')[-1]}")

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        tasks: List[Dict],
        context: Optional[Dict] = None,
        **kwargs,
    ) -> ModelResponse:

        from_name, to_name, value = self.get_first_tag_occurence("VideoRectangle", "Video")

        if not context or not context.get("result"):
            return ModelResponse(predictions=[])

        task = tasks[0]
        task_id = task.get("id")
        video_url = task["data"][value]

        # ── Download / resolve video ──────────────────────────────────────────
        try:
            video_path = self.get_local_path(video_url, task_id=task_id)
        except Exception as exc:
            logger.error("Failed to download video: %s", exc)
            return ModelResponse(predictions=[])

        # ── Extract fps metadata from context ─────────────────────────────────
        frames_count = context["result"][0]["value"].get("framesCount", 0)
        duration = context["result"][0]["value"].get("duration", 1.0)
        fps = frames_count / duration if duration else 25.0

        # ── Parse VideoRectangle prompts ──────────────────────────────────────
        prompts = self._get_prompts(context)
        if not prompts:
            return ModelResponse(predictions=[])

        all_obj_ids = set(p["obj_id"] for p in prompts)
        # Map string obj_id → integer for SAM3
        obj_id_map = {oid: i for i, oid in enumerate(all_obj_ids)}

        first_frame_idx = min(p["frame_idx"] for p in prompts)
        last_frame_idx = max(p["frame_idx"] for p in prompts)

        # ── Split video to frames ─────────────────────────────────────────────
        with tempfile.TemporaryDirectory() as frame_dir:
            frames = list(self._split_frames(
                video_path, frame_dir,
                start_frame=first_frame_idx,
                end_frame=last_frame_idx + MAX_FRAMES_TO_TRACK + 1,
            ))
            if not frames:
                logger.error("No frames extracted from video: %s", video_path)
                return ModelResponse(predictions=[])

            _frame_path, first_frame_img = frames[0]
            height, width, _ = first_frame_img.shape
            logger.debug("Video dimensions: %dx%d, frames extracted: %d", width, height, len(frames))

            # ── Initialise SAM3 inference state ───────────────────────────────
            inference_state = _get_inference_state(frame_dir)
            predictor.reset_state(inference_state)

            # ── Add prompts ───────────────────────────────────────────────────
            for prompt in prompts:
                # Convert rect → 5 keypoints (center + 4 cardinal midpoints)
                # All in absolute pixel coordinates
                pts, lbs = self._rect_to_keypoints(
                    prompt["x_pct"], prompt["y_pct"],
                    prompt["w_pct"], prompt["h_pct"],
                    width, height,
                )
                logger.debug(
                    "Adding %d points for obj_id=%s at frame %d",
                    len(pts), prompt["obj_id"], prompt["frame_idx"],
                )
                predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=prompt["frame_idx"],
                    obj_id=obj_id_map[prompt["obj_id"]],
                    points=pts,
                    labels=lbs,
                )

            # ── Propagate and collect outputs ─────────────────────────────────
            sequence: list[dict] = []
            logger.info(
                "Propagating from frame %d for %d frames …",
                last_frame_idx, MAX_FRAMES_TO_TRACK,
            )
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=last_frame_idx,
                max_frame_num_to_track=MAX_FRAMES_TO_TRACK,
            ):
                real_frame_idx = out_frame_idx + first_frame_idx
                for i, _obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                    bbox = self._mask_to_bbox_pct(mask)
                    if bbox:
                        sequence.append({
                            "frame": real_frame_idx + 1,  # LS is 1-indexed
                            "x": bbox["x"],
                            "y": bbox["y"],
                            "width": bbox["width"],
                            "height": bbox["height"],
                            "enabled": True,
                            "rotation": 0,
                            "time": out_frame_idx / fps,
                        })

        # ── Merge original context with new sequence ──────────────────────────
        context_sequence = context["result"][0]["value"].get("sequence", [])
        full_sequence = context_sequence + sequence

        prediction = PredictionValue(
            result=[{
                "value": {
                    "framesCount": frames_count,
                    "duration": duration,
                    "sequence": full_sequence,
                },
                "from_name": from_name,
                "to_name": to_name,
                "type": "videorectangle",
                "origin": "manual",
                "id": list(all_obj_ids)[0],
            }]
        )

        return ModelResponse(predictions=[prediction])

    def fit(self, event: str, data: dict, **kwargs) -> None:
        logger.info("Received event '%s' (fit not implemented)", event)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_prompts(self, context: dict) -> list[dict]:
        """Parse VideoRectangle context into prompt dicts."""
        prompts: list[dict] = []
        for item in context.get("result", []):
            if item.get("type") != "videorectangle":
                continue
            obj_id = item["id"]
            for seq in item["value"].get("sequence", []):
                if not seq.get("enabled", True):
                    continue
                # frame in LS context is 1-indexed; SAM2/3 is 0-indexed
                frame_idx = seq.get("frame", 1) - 1
                prompts.append({
                    "obj_id": obj_id,
                    "frame_idx": max(frame_idx, 0),
                    "x_pct": seq.get("x", 0.0),
                    "y_pct": seq.get("y", 0.0),
                    "w_pct": seq.get("width", 0.0),
                    "h_pct": seq.get("height", 0.0),
                })
        return sorted(prompts, key=lambda p: p["frame_idx"])

    @staticmethod
    def _rect_to_keypoints(
        x_pct: float, y_pct: float, w_pct: float, h_pct: float,
        width: int, height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert percentage rectangle to 5 keypoints in absolute pixel coords.

        Mirrors the official SAM2 video example: center + 4 cardinal midpoints.
        All keypoints are foreground (label=1).
        """
        x = x_pct / 100
        y = y_pct / 100
        w = w_pct / 100
        h = h_pct / 100
        kps = [
            [x + w / 2, y + h / 2],           # center
            [x + w / 4, y + h / 2],            # left-center
            [x + 3 * w / 4, y + h / 2],        # right-center
            [x + w / 2, y + h / 4],            # top-center
            [x + w / 2, y + 3 * h / 4],        # bottom-center
        ]
        pts = np.array(kps, dtype=np.float32)
        pts[:, 0] *= width
        pts[:, 1] *= height
        labels = np.ones(len(kps), dtype=np.int32)
        return pts, labels

    @staticmethod
    def _mask_to_bbox_pct(mask: np.ndarray) -> Optional[dict]:
        """Convert binary mask to percentage bounding box.

        Returns None if mask is empty.
        """
        mask = mask.squeeze()
        y_idx, x_idx = np.where(mask == 1)
        if len(x_idx) == 0:
            return None
        h, w = mask.shape
        return {
            "x": round(float(np.min(x_idx)) / w * 100, 2),
            "y": round(float(np.min(y_idx)) / h * 100, 2),
            "width": round(float(np.max(x_idx) - np.min(x_idx) + 1) / w * 100, 2),
            "height": round(float(np.max(y_idx) - np.min(y_idx) + 1) / h * 100, 2),
        }

    @staticmethod
    def _split_frames(
        video_path: str,
        frame_dir: str,
        start_frame: int = 0,
        end_frame: int = 100,
    ):
        """Split video into JPEG frames, yield (path, frame_array) tuples.

        SAM2/3 video predictor requires a directory of sequentially named frames
        (e.g. 00000.jpg, 00001.jpg …). Frame filenames are 0-indexed relative
        to `start_frame`, matching the offset passed to init_state().
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.debug("Video total frames: %d, extracting [%d, %d)", total, start_frame, end_frame)

        frame_count = 0
        rel_count = 0  # relative index within our extraction window
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_count < start_frame:
                frame_count += 1
                continue
            if frame_count >= end_frame:
                break

            fname = os.path.join(frame_dir, f"{rel_count:05d}.jpg")
            if not os.path.exists(fname):
                cv2.imwrite(fname, frame)
            yield fname, frame
            frame_count += 1
            rel_count += 1

        cap.release()
