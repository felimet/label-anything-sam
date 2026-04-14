#!/bin/bash
# SAM3 image backend entrypoint
#
# GPU assignment:
#   SAM3_IMAGE_GPU_INDEX (set in .env.ml) — comma-separated nvidia-smi GPU index(es)
#   to expose to this container (e.g. "0", "0,1").  Empty = all GPUs visible.
#   NOTE: Docker Compose ${VAR} substitution reads from the HOST's .env, NOT from
#   env_file:.  So CUDA_VISIBLE_DEVICES must be derived here, where env_file vars
#   are already in scope.
#
# Worker count:
#   SAM3_IMAGE_WORKERS (set in .env.ml) — gunicorn workers for this service.
#   Set equal to the number of GPUs in SAM3_IMAGE_GPU_INDEX.
#   Falls back to WORKERS (Dockerfile default 1) if SAM3_IMAGE_WORKERS is unset.
#
# NOTE: Do NOT use --preload. CUDA must be initialised inside each worker
# process (after fork), not in the gunicorn master. --preload causes the
# master to import the app first, which can trigger CUDA init via transitive
# imports, making all forked workers fail with:
#   "Cannot re-initialize CUDA in forked subprocess"

# ── GPU pinning ───────────────────────────────────────────────────────────────
# Only override CUDA_VISIBLE_DEVICES when SAM3_IMAGE_GPU_INDEX is explicitly set.
# Leaving CUDA_VISIBLE_DEVICES unset exposes all GPUs (correct for single-GPU nodes).
# Setting it to "" (empty string) would hide all GPUs — we avoid that here.
if [ -n "${SAM3_IMAGE_GPU_INDEX:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${SAM3_IMAGE_GPU_INDEX}"
fi

exec gunicorn \
  --config /app/gunicorn.conf.py \
  --bind ":${PORT:-9090}" \
  --workers "${SAM3_IMAGE_WORKERS:-${WORKERS:-1}}" \
  --threads "${THREADS:-8}" \
  --worker-class gthread \
  --timeout "${TIMEOUT:-120}" \
  --graceful-timeout 30 \
  --log-level "${LOG_LEVEL:-info}" \
  _wsgi:app
