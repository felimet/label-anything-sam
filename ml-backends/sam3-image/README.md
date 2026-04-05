# SAM3 Image Backend for Label Studio

Interactive image segmentation using [SAM 3.1](https://github.com/facebookresearch/sam3) (Segment Anything Model 3) with Label Studio.

## Features

- **Text prompt** (SAM3 open-vocabulary PCS): label name → find all matching instances
- **Point prompt**: KeyPointLabels (positive / negative clicks)
- **Box prompt**: RectangleLabels (bounding box)
- **Output**: BrushLabels with Label Studio RLE encoding

## Prerequisites

1. NVIDIA GPU with driver ≥ 535.x (CUDA 12.6)
2. HuggingFace account with access to [facebook/sam3.1](https://huggingface.co/facebook/sam3.1)
   - Accept the model license at https://huggingface.co/facebook/sam3.1
   - Generate a token at https://huggingface.co/settings/tokens

## Quick Start

```bash
# 1. Set environment variables
cp ../../.env.example ../../.env
# Edit .env: set LABEL_STUDIO_API_KEY, HF_TOKEN

# 2. Start with main Label Studio stack
docker compose -f ../../docker-compose.yml -f ../../docker-compose.ml.yml up -d --build sam3-image-backend

# 3. Register in Label Studio UI
#    Project → Settings → Machine Learning → Add Model
#    URL: http://sam3-image-backend:9090
```

## Labeling Config

Use `labeling_config.xml` as your project's labeling interface.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SAM3_MODEL_ID` | `facebook/sam3.1` | HuggingFace model ID |
| `SAM3_CHECKPOINT_FILENAME` | `sam3.1.pt` | Checkpoint filename on HF |
| `MODEL_DIR` | `/data/models` | Local checkpoint cache directory |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `HF_TOKEN` | — | HuggingFace access token (required for gated model) |
| `LABEL_STUDIO_URL` | `http://label-studio:8080` | Label Studio internal URL |
| `LABEL_STUDIO_API_KEY` | — | Label Studio API token |
| `IMAGE_CACHE_SIZE` | `50` | Max cached PIL images |
| `IMAGE_CACHE_TTL` | `300` | Cache TTL in seconds |

## Notes on Text Prompt

SAM3 uses **Promptable Concept Segmentation (PCS)**. When you click a KeyPoint or draw a Rectangle with a label like `"car"`, the label name is passed as the text concept to SAM3, which finds all matching instances in the image. Use descriptive English noun phrases for best results.

## Running Tests

```bash
pip install -r requirements-test.txt
pytest tests/ -v --tb=short
```
