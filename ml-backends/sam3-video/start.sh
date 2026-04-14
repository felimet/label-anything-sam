#!/bin/bash
# SAM3 video backend entrypoint
#
# GPU assignment:
#   SAM3_VIDEO_GPU_INDEX (set in .env.ml) — comma-separated nvidia-smi GPU index(es)
#   to expose to this container (e.g. "1", "1,2,3").  Empty = all GPUs visible.
#   NOTE: Docker Compose ${VAR} substitution reads from the HOST's .env, NOT from
#   env_file:.  So CUDA_VISIBLE_DEVICES must be derived here, where env_file vars
#   are already in scope.
#
# Worker count:
#   SAM3_VIDEO_WORKERS (set in .env.ml) — gunicorn workers for this service.
#   Set equal to the number of GPUs in SAM3_VIDEO_GPU_INDEX.
#   Falls back to WORKERS (Dockerfile default 1) if SAM3_VIDEO_WORKERS is unset.
#   Note: video propagation is sequential per request; THREADS allows concurrent
#   requests to be queued without dropping connections.
#
# NOTE: Do NOT use --preload — same CUDA fork issue as image backend.

# ── GPU pinning ───────────────────────────────────────────────────────────────
# Only override CUDA_VISIBLE_DEVICES when SAM3_VIDEO_GPU_INDEX is explicitly set.
# Leaving CUDA_VISIBLE_DEVICES unset exposes all GPUs (correct for single-GPU nodes).
# Setting it to "" (empty string) would hide all GPUs — we avoid that here.
if [ -n "${SAM3_VIDEO_GPU_INDEX:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${SAM3_VIDEO_GPU_INDEX}"
fi

exec gunicorn \
  --config /app/gunicorn.conf.py \
  --bind ":${PORT:-9090}" \
  --workers "${SAM3_VIDEO_WORKERS:-${WORKERS:-1}}" \
  --threads "${THREADS:-8}" \
  --worker-class gthread \
  --timeout "${TIMEOUT:-300}" \
  --graceful-timeout 30 \
  --log-level "${LOG_LEVEL:-info}" \
  _wsgi:app
