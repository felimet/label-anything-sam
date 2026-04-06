#!/bin/bash
# SAM3 video backend entrypoint — reads WORKERS/THREADS/PORT from environment
exec gunicorn \
  --bind ":${PORT:-9090}" \
  --workers "${WORKERS:-1}" \
  --threads "${THREADS:-4}" \
  --worker-class gthread \
  --timeout 0 \
  --graceful-timeout 30 \
  --log-level "${LOG_LEVEL:-info}" \
  --preload \
  _wsgi:app
