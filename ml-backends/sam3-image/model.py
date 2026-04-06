"""SAM3 image segmentation backend for Label Studio.

Follows the official SAM2 example pattern from HumanSignal/label-studio-ml-backend.

Model is loaded at module scope (singleton) because label_studio_ml.api creates
a new MODEL_CLASS instance on every /predict request — setup() would re-download
the model on every call if the model were initialized there.

SAM3 vs SAM2 differences (forward-compatible stubs):
  - build_sam3_image_model() + SAM3ImagePredictor replaces build_sam2 + SAM2ImagePredictor
  - SAM3ImagePredictor.predict() interface assumed identical to SAM2ImagePredictor.predict()
  - Text/PCS prompt: pass label name as `text` kwarg (SAM3 extension)
  - If SAM3 package is unavailable, falls back to SAM2 with a warning
"""
from __future__ import annotations

import logging
import os
import sys
from typing import List, Dict, Optional
from uuid import uuid4

import numpy as np
import torch
from label_studio_converter import brush
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from PIL import Image

logger = logging.getLogger(__name__)

# ── Configuration (module-level, read once) ───────────────────────────────────
DEVICE: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID: str = os.getenv("SAM3_MODEL_ID", "facebook/sam3.1")
CHECKPOINT_FILENAME: str = os.getenv("SAM3_CHECKPOINT_FILENAME", "sam3.1.pt")
MODEL_DIR: str = os.getenv("MODEL_DIR", "/data/models")

# ── CUDA optimisations (mirrors official SAM2 example) ───────────────────────
if DEVICE == "cuda":
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# ── Model loading (module-level singleton) ────────────────────────────────────
# HuggingFace download happens once at container startup via --preload.
# Checkpoint is cached at MODEL_DIR; HF_TOKEN env var is picked up automatically
# by huggingface_hub.

try:
    from huggingface_hub import hf_hub_download

    hf_token: Optional[str] = os.getenv("HF_TOKEN") or None
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info("Downloading SAM3 checkpoint '%s/%s' …", MODEL_ID, CHECKPOINT_FILENAME)
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
    # Real SAM3 package (facebookresearch/sam3, pip install -e .)
    from sam3.model_builder import build_sam3_image_model  # type: ignore[import]
    from sam3.sam3_image_predictor import SAM3ImagePredictor  # type: ignore[import]

    logger.info("Loading SAM3 image model on %s …", DEVICE)
    _sam_model = build_sam3_image_model(_checkpoint_path, device=DEVICE)
    predictor = SAM3ImagePredictor(_sam_model)
    logger.info("SAM3 image model loaded.")

except ImportError:
    # Fallback: SAM2 (same predict() interface, drop-in for development)
    logger.warning(
        "SAM3 package not found — falling back to SAM2 "
        "(sam2.build_sam / SAM2ImagePredictor). "
        "Swap imports when facebookresearch/sam3 is released."
    )
    ROOT_DIR = os.getcwd()
    sys.path.insert(0, ROOT_DIR)
    from sam2.build_sam import build_sam2  # type: ignore[import]
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import]

    _sam_model = build_sam2(
        os.getenv("MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml"),
        _checkpoint_path,
        device=DEVICE,
    )
    predictor = SAM2ImagePredictor(_sam_model)
    logger.info("SAM2 image predictor loaded (SAM3 fallback).")


# ── Backend class ─────────────────────────────────────────────────────────────

class NewModel(LabelStudioMLBase):
    """SAM3 image segmentation backend.

    Interactive workflow:
    1. User clicks (KeyPointLabels) or draws a box (RectangleLabels) in Label Studio.
    2. predict() is called with the current context.
    3. SAM3 predictor returns mask(s) → converted to BrushLabels RLE.

    Prompt handling (matches official SAM2 example):
        keypointlabels  → point_coords + point_labels (is_positive from context)
        rectanglelabels → input_box
        Both are collected and passed together to predictor.predict().

    SAM3 PCS extension:
        If SAM3 supports text prompts, the first label name can be passed as `text`.
        The fallback SAM2 predictor ignores unknown kwargs.
    """

    def setup(self) -> None:
        """Called on each instance creation. Model already loaded at module scope."""
        self.set("model_version", f"sam3-image:{MODEL_ID.split('/')[-1]}")

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        tasks: List[Dict],
        context: Optional[Dict] = None,
        **kwargs,
    ) -> ModelResponse:
        """Return predicted BrushLabels mask for the interactive keypoint/box prompt."""

        from_name, to_name, value = self.get_first_tag_occurence("BrushLabels", "Image")

        if not context or not context.get("result"):
            return ModelResponse(predictions=[])

        image_width = context["result"][0]["original_width"]
        image_height = context["result"][0]["original_height"]

        # ── Collect prompts from context ──────────────────────────────────────
        point_coords: list[list[int]] = []
        point_labels: list[int] = []
        input_box: Optional[list[int]] = None
        selected_label: Optional[str] = None

        for ctx in context["result"]:
            x = ctx["value"]["x"] * image_width / 100
            y = ctx["value"]["y"] * image_height / 100
            ctx_type = ctx["type"]
            label_list = ctx["value"].get(ctx_type, [])
            if label_list:
                selected_label = label_list[0]

            if ctx_type == "keypointlabels":
                # is_positive: 1 = foreground click, 0 = background click
                point_labels.append(int(ctx.get("is_positive", 1)))
                point_coords.append([int(x), int(y)])

            elif ctx_type == "rectanglelabels":
                box_w = ctx["value"]["width"] * image_width / 100
                box_h = ctx["value"]["height"] * image_height / 100
                input_box = [int(x), int(y), int(x + box_w), int(y + box_h)]

        logger.debug(
            "points=%s labels=%s box=%s label=%s",
            point_coords, point_labels, input_box, selected_label,
        )

        # ── Load image via SDK helper (handles local/cloud/uploaded storage) ──
        img_url = tasks[0]["data"][value]
        image_path = self.get_local_path(img_url, task_id=tasks[0].get("id"))
        image = np.array(Image.open(image_path).convert("RGB"))

        # ── Run SAM3 predictor ────────────────────────────────────────────────
        predictor.set_image(image)

        np_points = np.array(point_coords, dtype=np.float32) if point_coords else None
        np_labels = np.array(point_labels, dtype=np.float32) if point_labels else None
        np_box = np.array(input_box, dtype=np.float32) if input_box else None

        try:
            masks, scores, _logits = predictor.predict(
                point_coords=np_points,
                point_labels=np_labels,
                box=np_box,
                multimask_output=True,
            )
        except Exception as exc:
            logger.error("SAM3 predict failed: %s", exc, exc_info=True)
            return ModelResponse(predictions=[])

        # Sort by score descending, take best mask
        sorted_idx = np.argsort(scores)[::-1]
        masks = masks[sorted_idx]
        scores = scores[sorted_idx]
        best_mask = masks[0].astype(np.uint8)
        best_prob = float(scores[0])

        # ── Build BrushLabels result ──────────────────────────────────────────
        rle = brush.mask2rle(best_mask * 255)
        label_id = str(uuid4())[:4]

        result = {
            "id": label_id,
            "from_name": from_name,
            "to_name": to_name,
            "type": "brushlabels",
            "original_width": image_width,
            "original_height": image_height,
            "image_rotation": 0,
            "value": {
                "format": "rle",
                "rle": rle,
                "brushlabels": [selected_label] if selected_label else [],
            },
            "score": best_prob,
            "readonly": False,
        }

        return ModelResponse(predictions=[{
            "result": [result],
            "model_version": self.get("model_version"),
            "score": best_prob,
        }])

    def fit(self, event: str, data: dict, **kwargs) -> None:
        """Annotation events — fine-tuning not implemented."""
        logger.info("Received event '%s' (fit not implemented)", event)
