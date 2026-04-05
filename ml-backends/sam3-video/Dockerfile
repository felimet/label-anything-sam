# ============================================================
# SAM3 Video Backend — Multi-stage CUDA build
# Requirements: torch 2.7+, CUDA 12.6+, Python 3.12+
# NVIDIA driver >= 535.x required for CUDA 12.6
# ============================================================

# ── Stage 1: dependency builder ─────────────────────────────
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install SAM3 from source (no PyPI package)
ARG SAM3_COMMIT=main
RUN git clone https://github.com/facebookresearch/sam3.git \
    && cd sam3 \
    && git checkout ${SAM3_COMMIT} \
    && pip install --no-cache-dir -e .

# Install label-studio-ml from official GitHub
RUN pip install --no-cache-dir \
    "label-studio-ml @ git+https://github.com/HumanSignal/label-studio-ml-backend.git"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=9090 \
    WORKERS=1 \
    THREADS=8 \
    DEVICE=cuda \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    SAM3_MODEL_ID=facebook/sam3.1 \
    SAM3_CHECKPOINT_FILENAME=sam3.1.pt \
    MODEL_DIR=/data/models \
    MAX_FRAMES_TO_TRACK=10

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/conda/lib/python3.12/site-packages \
                    /opt/conda/lib/python3.12/site-packages
COPY --from=builder /opt/conda/bin /opt/conda/bin
COPY --from=builder /build/sam3 /opt/sam3

WORKDIR /app
COPY . .

RUN useradd -m -u 1001 appuser \
    && mkdir -p /data/models /root/.cache/huggingface \
    && chown -R appuser:appuser /app /data /root/.cache/huggingface

USER appuser

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=5 \
    CMD curl -sf http://localhost:9090/health || exit 1

CMD ["gunicorn", \
     "--preload", \
     "--bind", "0.0.0.0:9090", \
     "--workers", "1", \
     "--threads", "8", \
     "--worker-class", "gthread", \
     "--timeout", "0", \
     "--graceful-timeout", "30", \
     "--log-level", "info", \
     "_wsgi:app"]
