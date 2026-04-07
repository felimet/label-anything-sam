# SAM3 Video Backend for Label Studio

VideoRectangle tracking + optional text (PCS) prompts using [SAM 3.1](https://github.com/facebookresearch/sam3).

## Features

- **Text-only path**: TextArea prompt without a box → `add_prompt` at frame 0 with text only; SAM3 detects and tracks all matching instances forward.
- **Text + Box (mixed)**: text passed alongside VideoRectangle geometry for precise grounding.
- **Exclude box**: `Labels value="Exclude"` on a VideoRectangle → `bounding_box_labels=[0]` (negative prompt).
- **Scores display**: `add_prompt` responses written to `TextArea name="scores"` after each prediction.
- **SAM2 fallback**: if `sam3` package is unavailable, falls back to SAM2 video predictor; text prompts are ignored with a WARNING in that mode.

## Usage

1. Set `.env.ml`: `LABEL_STUDIO_API_KEY`, `HF_TOKEN`
2. `docker compose -f ../../docker-compose.yml -f ../../docker-compose.ml.yml up -d sam3-video-backend`
3. Label Studio: Add Model URL `http://sam3-video-backend:9090`
4. Use `labeling_config.xml` as project interface (includes `<TextArea>` for text prompts, `Exclude` label, scores area)

## Key Variables

| Variable | Default | Description |
|---|---|---|
| `SAM3_VIDEO_MODEL_ID` | `facebook/sam3.1` | HF model ID (video backend). Fallback: `SAM3_MODEL_ID` |
| `SAM3_VIDEO_CHECKPOINT_FILENAME` | `sam3.1_multiplex.pt` | Checkpoint filename (~3.5 GB). Requires `sam3.1` branch build arg. Fallback: `SAM3_CHECKPOINT_FILENAME` |
| `MAX_FRAMES_TO_TRACK` | `10` | **Dual role:** (1) max frames to propagate per request (`max_frame_num_to_track` passed to `propagate_in_video`); (2) memory budget — only frames `[start_frame, last_frame + MAX_FRAMES_TO_TRACK + 1)` are extracted to a temporary image folder before opening the SAM3 session. Lower this value to reduce RAM usage on long/high-resolution videos. |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `HF_TOKEN` | — | HF gated token (required) |
| `SAM3_ENABLE_PCS` | `true` | Enable text prompts. Supports text-only (no box required) and text+box mixed mode. Set `false` for geometry-only tracking. |
| `SAM3_ENABLE_FA3` | `false` | Flash Attention 3 (requires `--build-arg ENABLE_FA3=true` at build time) |

## Predict Paths

| Input | Path | Notes |
|-------|------|-------|
| TextArea only | Text-only PCS | `add_prompt` at frame 0 with `text` only; no box required. Video metadata probed from file via `cv2`. |
| VideoRectangle (Object) | Geometric | `bounding_box_labels=[1]` per prompted frame |
| VideoRectangle (Exclude) | Negative geometric | `bounding_box_labels=[0]`; tells SAM3 to exclude the region |
| TextArea + VideoRectangle | Mixed | `text` + `bounding_boxes` in same `add_prompt` call |

## Session Lifecycle

The SAM3 video predictor uses a session API:

```
extract frames → start_session (image folder) → add_prompt (per frame) → propagate_in_video → close_session
```

**Frame extraction (OOM fix):** Instead of passing the full video file to `start_session`, the backend first extracts only the needed frames — from `start_frame` up to `last_frame + MAX_FRAMES_TO_TRACK + 1` — into a `tempfile.TemporaryDirectory` using `_extract_frames()`. `start_session` then receives the image folder path. This prevents loading hundreds of high-resolution frames (e.g. 400 × 1080p ≈ 2+ GB) into RAM.

`close_session` is guaranteed via `finally` even on errors. The temporary frame directory is cleaned up automatically when the `with tempfile.TemporaryDirectory()` block exits after `close_session`. Video dimensions are probed via `cv2.VideoCapture` to convert percentage coordinates to normalised `[x0, y0, w, h]` required by `add_prompt`. For text-only requests (no VideoRectangle), metadata is probed from the video file before frame extraction.

**`propagate_in_video` error handling:** Any exception raised inside the SAM3 generator (e.g. internal `NoneType` errors when no object is detected) is caught and logged as a WARNING instead of crashing the request. An empty sequence is returned in that case.

**Symlink for extension-less cache files:** Label Studio's download-and-cache may strip file extensions. If the resolved `video_path` has no extension, a symlink with the correct extension is created (parsed from the `?d=` query parameter in the task URL) so that SAM3's session validator accepts the file.

## Known Behaviors / Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| CUDA OOM on long videos | Too many frames loaded for a high-resolution clip | Lower `MAX_FRAMES_TO_TRACK` (e.g. `5`); each unit saves ~1 frame × resolution of RAM |
| Empty prediction, WARNING in logs: `propagate_in_video raised an error` | SAM3 internal error (no detections or boundary condition) | Expected for zero-detection cases; check that prompts cover at least one object |
| `start_session` rejects the file | Extension-less cache file symlink failed | Check container filesystem permissions; ensure `video_path` directory is writable |
| Text prompt ignored, WARNING in logs | `sam3` package not installed; SAM2 fallback active | Rebuild image with `sam3.1` branch; or use geometric prompts only |

## Tests

```bash
pip install -r requirements-test.txt
pytest tests/ -v --tb=short
```
