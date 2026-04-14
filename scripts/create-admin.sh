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
docker compose exec label-studio python /label-studio/label_studio/manage.py shell -c "
from users.models import User
email = '${USERNAME}'
password = '${PASSWORD}'
if User.objects.filter(email=email).exists():
    print(f'User {email} already exists — skipping creation')
else:
    u = User.objects.create_superuser(email=email, password=password)
    print(f'Admin user {email} created (id={u.id})')
"

echo "Done. Log in at: ${LABEL_STUDIO_HOST:-http://localhost:8080}"
