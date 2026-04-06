#!/bin/sh
# MinIO initialisation script
# Creates bucket, sets policy, and configures CORS.
# Runs once via the minio-init service on `docker compose up`.
set -eu

MINIO_ALIAS="local"
MINIO_ENDPOINT="http://minio:9000"
BUCKET="${MINIO_BUCKET:-label-studio-bucket}"

echo "[init-minio] Waiting for MinIO to be ready..."
until mc alias set "${MINIO_ALIAS}" "${MINIO_ENDPOINT}" \
        "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}" 2>/dev/null; do
    sleep 2
done
echo "[init-minio] Connected to MinIO."

# ── Create bucket ──────────────────────────────────────────
echo "[init-minio] Creating bucket: ${BUCKET}"
mc mb --ignore-existing "${MINIO_ALIAS}/${BUCKET}"

# ── Policy: download (anonymous GET, no LIST) ──────────────
# This means only presigned URLs work for reading — no public directory listing.
echo "[init-minio] Setting download policy on ${BUCKET}"
mc anonymous set download "${MINIO_ALIAS}/${BUCKET}"

# ── CORS ──────────────────────────────────────────────────
# MinIO open-source editions >= 2024 removed the S3 PutBucketCors API.
# CORS is now controlled via MINIO_API_CORS_ALLOW_ORIGIN env var (set in
# docker-compose.yml on the minio service) — no mc command needed here.
echo "[init-minio] CORS handled via MINIO_API_CORS_ALLOW_ORIGIN server env var."

echo "[init-minio] Verifying bucket..."
mc ls "${MINIO_ALIAS}/${BUCKET}" && echo "[init-minio] Done."
