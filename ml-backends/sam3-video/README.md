# SAM3 Video Backend for Label Studio (Skeleton)

VideoRectangle → SAM3 multi-frame tracking using [SAM 3.1](https://github.com/facebookresearch/sam3).

## Usage

1. Set `.env`: `LABEL_STUDIO_API_KEY`, `HF_TOKEN`
2. `docker compose -f ../../docker-compose.yml -f ../../docker-compose.ml.yml up -d sam3-video-backend`
3. Label Studio: Add Model URL `http://sam3-video-backend:9090`
4. Use `labeling_config.xml` as project interface

## Key Variables

| Variable | Default | Description |
|---|---|---|
| `SAM3_MODEL_ID` | `facebook/sam3.1` | HF model ID |
| `MAX_FRAMES_TO_TRACK` | `10` | Max frames to propagate |
| `DEVICE` | `cuda` | Device |
| `HF_TOKEN` | — | HF gated token |

## Tests

```bash
pip install -r requirements-test.txt
pytest tests/ -v --tb=short
```
