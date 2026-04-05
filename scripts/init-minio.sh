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
# Label Studio UI (browser) needs CORS to PUT files and GET presigned URLs.
echo "[init-minio] Setting CORS on ${BUCKET}"
mc cors set "${MINIO_ALIAS}/${BUCKET}" - <<'EOF'
{
  "CORSRules": [
    {
      "AllowedHeaders": ["*"],
      "AllowedMethods": ["GET", "PUT", "POST", "DELETE", "HEAD"],
      "AllowedOrigins": ["*"],
      "ExposeHeaders": ["ETag", "x-amz-request-id"],
      "MaxAgeSeconds": 3000
    }
  ]
}
EOF

echo "[init-minio] Verifying bucket..."
mc ls "${MINIO_ALIAS}/${BUCKET}" && echo "[init-minio] Done."
