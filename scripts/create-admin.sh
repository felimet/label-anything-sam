#!/bin/bash
# Create Label Studio admin user via Django management command.
# Reads credentials from .env if present.
set -euo pipefail

if [ -f .env ]; then
    source .env
fi

USERNAME="${LABEL_STUDIO_USERNAME:-admin@example.com}"
PASSWORD="${LABEL_STUDIO_PASSWORD:-}"

if [ -z "$PASSWORD" ]; then
    echo "ERROR: LABEL_STUDIO_PASSWORD not set in .env"
    exit 1
fi

echo "Creating admin user: ${USERNAME}"
docker compose exec label-studio label-studio user create \
    --username "${USERNAME}" \
    --password "${PASSWORD}" \
    --is-staff

echo "Done. Log in at: ${LABEL_STUDIO_HOST:-http://localhost:8080}"
