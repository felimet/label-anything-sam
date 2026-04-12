#!/bin/sh
# MinIO initialisation script
# Creates bucket, sets anonymous download policy, provisions a
# least-privilege service account for Label Studio, and optionally
# sets a bucket quota.
# Idempotent — safe to re-run.
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
# Presigned URLs remain readable by anyone who holds the URL,
# but public directory listing is disabled.
echo "[init-minio] Setting anonymous download policy on ${BUCKET}"
mc anonymous set download "${MINIO_ALIAS}/${BUCKET}"

# ── CORS ──────────────────────────────────────────────────
# MinIO open-source editions >= 2024 removed the S3 PutBucketCors API.
# CORS is now controlled via MINIO_API_CORS_ALLOW_ORIGIN env var (set in
# docker-compose.yml on the minio service) — no mc command needed here.
echo "[init-minio] CORS handled via MINIO_API_CORS_ALLOW_ORIGIN server env var."

# ── Service account for Label Studio ──────────────────────
# Provision a dedicated access key scoped to this bucket only.
# Label Studio should use MINIO_LS_ACCESS_ID / MINIO_LS_SECRET_KEY
# (not the root credentials) when configuring Cloud Storage in the UI.
if [ -n "${MINIO_LS_ACCESS_ID:-}" ] && [ -n "${MINIO_LS_SECRET_KEY:-}" ]; then
    echo "[init-minio] Provisioning Label Studio service account..."

    # Write bucket-scoped IAM policy
    cat > /tmp/ls-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::${BUCKET}/*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
      "Resource": "arn:aws:s3:::${BUCKET}"
    }
  ]
}
EOF

    # Create policy (skip if already exists)
    mc admin policy info "${MINIO_ALIAS}" ls-bucket-policy > /dev/null 2>&1 || \
        mc admin policy create "${MINIO_ALIAS}" ls-bucket-policy /tmp/ls-policy.json

    # Create user (skip if already exists)
    mc admin user info "${MINIO_ALIAS}" "${MINIO_LS_ACCESS_ID}" > /dev/null 2>&1 || \
        mc admin user add "${MINIO_ALIAS}" "${MINIO_LS_ACCESS_ID}" "${MINIO_LS_SECRET_KEY}"

    # Attach policy to user (idempotent)
    mc admin policy attach "${MINIO_ALIAS}" ls-bucket-policy \
        --user "${MINIO_LS_ACCESS_ID}" 2>/dev/null || true

    echo "[init-minio] Service account ready: ${MINIO_LS_ACCESS_ID}"
    echo "[init-minio] Use this key when configuring Label Studio Cloud Storage (S3)."
else
    echo "[init-minio] MINIO_LS_ACCESS_ID/SECRET_KEY not set — skipping service account."
    echo "[init-minio] WARNING: Label Studio will use root credentials for storage access."
fi

# ── Bucket quota (optional) ───────────────────────────────
if [ -n "${MINIO_BUCKET_QUOTA_GB:-}" ]; then
    echo "[init-minio] Setting bucket quota: ${MINIO_BUCKET_QUOTA_GB} GiB"
    mc quota set "${MINIO_ALIAS}/${BUCKET}" --size "${MINIO_BUCKET_QUOTA_GB}GiB"
fi

echo "[init-minio] Verifying bucket..."
mc ls "${MINIO_ALIAS}/${BUCKET}" && echo "[init-minio] Done."
