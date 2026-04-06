# ============================================================
# SAM3 Image Backend
# Base: pytorch/pytorch (includes torch + CUDA + Python 3.12)
# Build: single-stage with BuildKit pip cache mounts
#
# Requirements: NVIDIA driver >= 535.x (CUDA 12.6)
# Build: DOCKER_BUILDKIT=1 docker compose build sam3-image-backend
# ============================================================

# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/root/.cache/pip \
    PORT=9090 \
    WORKERS=1 \
    THREADS=4 \
    DEVICE=cuda \
    LOG_LEVEL=INFO \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    SAM3_MODEL_ID=facebook/sam3.1 \
    SAM3_CHECKPOINT_FILENAME=sam3.1.pt \
    MODEL_DIR=/data/models

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        git gcc g++ \
        ffmpeg \
        libglib2.0-0 libgl1 libsm6 libxrender1 libxext6 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Base deps: gunicorn + label-studio-ml (git) ──────────────
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# ── App deps ─────────────────────────────────────────────────
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

# ── SAM3 from source (no PyPI package) ───────────────────────
# Pin SAM3_COMMIT to a known-good commit for reproducibility.
ARG SAM3_COMMIT=main
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    git clone --depth 1 --branch ${SAM3_COMMIT} \
        https://github.com/facebookresearch/sam3.git /sam3 \
    && rm -rf /sam3/.git /sam3/notebooks /sam3/demo \
    && pip install -e /sam3

# ── App files ────────────────────────────────────────────────
COPY . .
RUN chmod +x start.sh

# Non-root user
RUN useradd -m -u 1001 appuser \
    && mkdir -p /data/models /root/.cache/huggingface \
    && chown -R appuser:appuser /app /data /root/.cache/huggingface

USER appuser

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=5 \
    CMD curl -sf http://localhost:9090/health || exit 1

CMD ["/app/start.sh"]
