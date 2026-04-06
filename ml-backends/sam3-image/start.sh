#!/bin/bash
# SAM3 image backend entrypoint
#
# Tuning guide:
#   WORKERS  — 1 per GPU (model is preloaded once per worker via --preload;
#               multiple workers on a single GPU cause duplicate VRAM usage).
#               Raise to N for N-GPU nodes, e.g. WORKERS=2 for dual-GPU.
#   THREADS  — threads share the same worker process and model weights;
#               safe to raise on modern CPUs. 8 is a good baseline for 8-core
#               machines; raise to 16 for 16+ core servers.
#
# Override via environment: ML_WORKERS / ML_THREADS in .env or docker-compose.
exec gunicorn \
  --bind ":${PORT:-9090}" \
  --workers "${WORKERS:-1}" \
  --threads "${THREADS:-8}" \
  --worker-class gthread \
  --timeout "${TIMEOUT:-120}" \
  --graceful-timeout 30 \
  --log-level "${LOG_LEVEL:-info}" \
  --preload \
  _wsgi:app
