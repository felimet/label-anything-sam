"""SAM3 image segmentation backend for Label Studio.

Implements LabelStudioMLBase using facebookresearch/sam3:
  - Prompt types: Text (open-vocab PCS) > Rectangle box > KeyPoint
  - Output: BrushLabels with Label Studio RLE encoding (one result per detected instance)
  - Image cache: PIL Image LRU to avoid repeated downloads

SAM3 vs SAM2 key differences:
  - text prompt: label name is used as open-vocabulary concept prompt
  - PCS mode returns N instances per text prompt (masks: [N, H, W])
  - Uses build_sam3_image_model() + Sam3Processor, NOT SAM2ImagePredictor
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from io import BytesIO
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from label_studio_converter import brush
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from PIL import Image

# SAM3 package — installed via `pip install -e .` from facebookresearch/sam3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID: str = os.getenv("SAM3_MODEL_ID", "facebook/sam3.1")
# SAM3 checkpoint filename on HuggingFace (sam3.pt for sam3, sam3.1.pt for sam3.1)
CHECKPOINT_FILENAME: str = os.getenv("SAM3_CHECKPOINT_FILENAME", "sam3.1.pt")
MODEL_DIR: str = os.getenv("MODEL_DIR", "/data/models")

IMAGE_CACHE_SIZE: int = int(os.getenv("IMAGE_CACHE_SIZE", "50"))
IMAGE_CACHE_TTL: float = float(os.getenv("IMAGE_CACHE_TTL", "300"))

# Label names interpreted as negative (background) prompts
_NEGATIVE_LABELS = frozenset({"background", "negative", "neg", "背景"})


class NewModel(LabelStudioMLBase):
    """SAM3 image segmentation backend.

    Workflow:
    1. User clicks/draws on image in Label Studio →
       context.result = [keypointlabels|rectanglelabels]
    2. predict() parses prompts, runs Sam3Processor, returns BrushLabels RLE masks.
    3. Label Studio displays mask overlays; user can refine with more clicks.

    Prompt priority (higher wins):
        text (label name as concept) > box (rectanglelabels) > point (keypointlabels)

    SAM3 PCS note:
        Text prompts return N instance masks. All instances are returned as
        separate BrushLabels results, each with its own IoU score.
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

        logger.info("Loading SAM3 model on %s from %s", DEVICE, checkpoint_path)
        model = build_sam3_image_model(checkpoint_path=checkpoint_path, device=DEVICE)
        self._processor = Sam3Processor(model)

        self.set("model_version", f"sam3-image:{MODEL_ID.split('/')[-1]}")
        # PIL Image LRU cache: {url_md5: (timestamp, PIL.Image)}
        self._image_cache: dict[str, tuple[float, Image.Image]] = {}
        logger.info("SAM3 image model loaded. Device: %s", DEVICE)

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        tasks: list[dict],
        context: Optional[dict] = None,
        **kwargs,
    ) -> ModelResponse:
        # Interactive model: requires user prompt — no auto-segmentation.
        if not context or not context.get("result"):
            return ModelResponse(
                predictions=[{"result": [], "score": 0.0} for _ in tasks]
            )

        predictions = [self._predict_single(task, context) for task in tasks]
        return ModelResponse(predictions=predictions)

    def fit(self, event: str, data: dict, **kwargs) -> None:
        """Annotation events received — fine-tuning not implemented."""
        logger.info("Received event '%s' (fine-tuning not implemented)", event)

    # ── Private: single-task inference ───────────────────────────────────────

    def _predict_single(self, task: dict, context: dict) -> dict:
        image_url = self._get_image_url(task)
        if not image_url:
            logger.warning(
                "No image URL in task data: %s", list(task.get("data", {}).keys())
            )
            return {"result": [], "score": 0.0}

        image = self._load_image(image_url)
        orig_w, orig_h = image.size

        prompts = self._parse_context(context, orig_w, orig_h)
        if not prompts["text"] and not prompts["points"] and not prompts["boxes"]:
            return {"result": [], "score": 0.0}

        state = self._processor.set_image(image)

        # Prompt priority: text > box > point
        if prompts["text"]:
            out = self._processor.set_text_prompt(state=state, prompt=prompts["text"])
        elif prompts["boxes"]:
            out = self._processor.set_box_prompt(state=state, boxes=prompts["boxes"])
        else:
            out = self._processor.set_point_prompt(
                state=state,
                points=prompts["points"],
                labels=prompts["labels"],
            )

        brush_from_name, image_to_name, label_name = self._parse_label_config()

        # SAM3 PCS: out["masks"] is [N, H, W] bool Tensor, out["scores"] is [N]
        masks: torch.Tensor = out.get("masks", torch.zeros(0, orig_h, orig_w, dtype=torch.bool))
        scores_raw = out.get("scores", torch.zeros(masks.shape[0]))

        results = []
        for i in range(masks.shape[0]):
            mask_np = masks[i].cpu().numpy().astype(bool)
            if not mask_np.any():
                continue  # skip empty masks

            score = float(scores_raw[i].item()) if i < len(scores_raw) else 0.9
            mask_uint8 = (mask_np * 255).astype(np.uint8)
            rle = brush.mask2rle(mask_uint8)

            results.append({
                "from_name": brush_from_name,
                "to_name": image_to_name,
                "type": "brushlabels",
                "original_width": orig_w,
                "original_height": orig_h,
                "image_rotation": 0,
                "value": {
                    "format": "rle",
                    "rle": rle,
                    "brushlabels": [label_name],
                },
            })

        overall_score = float(scores_raw.max().item()) if len(scores_raw) > 0 else 0.0
        return {
            "result": results,
            "score": overall_score,
            "model_version": self.get("model_version"),
        }

    # ── Private: prompt parsing ───────────────────────────────────────────────

    def _parse_context(
        self,
        context: dict,
        orig_w: int,
        orig_h: int,
    ) -> dict:
        """Extract prompts from Label Studio interactive context.

        Returns dict with keys:
            text   – str | None   (label name used as SAM3 text/concept prompt)
            points – list[[x,y]]  (pixel coords)
            labels – list[int]    (1=positive, 0=negative)
            boxes  – list[[x1,y1,x2,y2]] (pixel coords)

        Label Studio sends percentage-based coordinates (0–100).
        SAM3 needs absolute pixel coordinates.
        """
        text: Optional[str] = None
        points: list[list[float]] = []
        labels: list[int] = []
        boxes: list[list[float]] = []

        for item in context.get("result", []):
            itype = item.get("type", "")
            value = item.get("value", {})
            iw = item.get("original_width", orig_w) or orig_w
            ih = item.get("original_height", orig_h) or orig_h

            if itype == "keypointlabels":
                x_px = value["x"] / 100.0 * iw
                y_px = value["y"] / 100.0 * ih
                kp_labels: list[str] = value.get("keypointlabels", ["Object"])
                is_neg = any(k.lower() in _NEGATIVE_LABELS for k in kp_labels)

                if not is_neg and text is None:
                    # Use first positive label name as SAM3 text concept
                    text = kp_labels[0]

                points.append([x_px, y_px])
                labels.append(0 if is_neg else 1)

            elif itype == "rectanglelabels":
                x1 = value["x"] / 100.0 * iw
                y1 = value["y"] / 100.0 * ih
                x2 = x1 + value["width"] / 100.0 * iw
                y2 = y1 + value["height"] / 100.0 * ih
                boxes.append([x1, y1, x2, y2])

                rect_labels: list[str] = value.get("rectanglelabels", ["Object"])
                if text is None and rect_labels:
                    text = rect_labels[0]

        return {"text": text, "points": points, "labels": labels, "boxes": boxes}

    # ── Private: image loading ────────────────────────────────────────────────

    def _get_image_url(self, task: dict) -> Optional[str]:
        data = task.get("data", {})
        return data.get("image") or data.get("img") or data.get("url")

    def _load_image(self, url: str) -> Image.Image:
        """Download image with LRU + TTL cache. Handles LS-internal /data/ URLs."""
        cache_key = hashlib.md5(url.encode()).hexdigest()
        now = time.monotonic()

        # Evict expired entries
        self._image_cache = {
            k: v for k, v in self._image_cache.items()
            if now - v[0] < IMAGE_CACHE_TTL
        }
        # Evict oldest when at capacity
        if len(self._image_cache) >= IMAGE_CACHE_SIZE and cache_key not in self._image_cache:
            oldest = min(self._image_cache, key=lambda k: self._image_cache[k][0])
            del self._image_cache[oldest]

        if cache_key in self._image_cache:
            return self._image_cache[cache_key][1]

        image = self._fetch_image(url)
        self._image_cache[cache_key] = (now, image)
        return image

    def _fetch_image(self, url: str) -> Image.Image:
        ls_url = os.getenv("LABEL_STUDIO_URL", "http://label-studio:8080").rstrip("/")
        api_key = os.getenv("LABEL_STUDIO_API_KEY", "")
        headers = {"Authorization": f"Token {api_key}"} if api_key else {}

        # Resolve LS-internal relative paths
        if url.startswith("/data/") or url.startswith("/tasks/"):
            url = f"{ls_url}{url}"

        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")

    # ── Private: label config parsing ────────────────────────────────────────

    def _parse_label_config(self) -> tuple[str, str, str]:
        """Parse XML label config to extract BrushLabels and Image tag names.

        Returns (brush_from_name, image_to_name, first_label_value).
        Falls back to ("tag", "image", "Object") if parsing fails.
        """
        brush_from_name = "tag"
        image_to_name = "image"
        default_label = "Object"

        try:
            config = getattr(self, "label_config", None) or "<View/>"
            root = ET.fromstring(config)

            for elem in root.iter("BrushLabels"):
                brush_from_name = elem.get("name", brush_from_name)
                lbl_values = [lbl.get("value", "Object") for lbl in elem.findall("Label")]
                if lbl_values:
                    default_label = lbl_values[0]
                break  # use first BrushLabels tag

            for elem in root.iter("Image"):
                image_to_name = elem.get("name", image_to_name)
                break

        except ET.ParseError as exc:
            logger.warning("Failed to parse label config: %s", exc)

        return brush_from_name, image_to_name, default_label
