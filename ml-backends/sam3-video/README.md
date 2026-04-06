# SAM3 Video Backend for Label Studio

VideoRectangle tracking + optional text (PCS) prompts using [SAM 3.1](https://github.com/facebookresearch/sam3).

## Usage

1. Set `.env`: `LABEL_STUDIO_API_KEY`, `HF_TOKEN`
2. `docker compose -f ../../docker-compose.yml -f ../../docker-compose.ml.yml up -d sam3-video-backend`
3. Label Studio: Add Model URL `http://sam3-video-backend:9090`
4. Use `labeling_config.xml` as project interface (includes `<TextArea>` for text prompts)

## Key Variables

| Variable | Default | Description |
|---|---|---|
| `SAM3_MODEL_ID` | `facebook/sam3.1` | HF model ID |
| `MAX_FRAMES_TO_TRACK` | `10` | Max frames to propagate per request |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `HF_TOKEN` | — | HF gated token (required) |
| `SAM3_ENABLE_PCS` | `true` | Enable text prompt alongside VideoRectangle |
| `SAM3_ENABLE_FA3` | `false` | Flash Attention 3 (requires `--build-arg ENABLE_FA3=true` at build time) |

## Session Lifecycle

The SAM3 video predictor uses a session API:

```
start_session → add_prompt (per frame, with text + pixel bbox) → propagate_in_video → close_session
```

`close_session` is guaranteed via `finally` even on errors. Video dimensions are probed via `cv2.VideoCapture` to convert percentage coordinates to pixel `[x0, y0, w, h]` required by `add_prompt`.

## Tests

```bash
pip install -r requirements-test.txt
pytest tests/ -v --tb=short
```
