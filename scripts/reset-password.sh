#!/bin/bash
# Reset Label Studio admin password.
# Verifies current password before allowing change.
# Reads username from .env if present; all password input is interactive.
set -euo pipefail

if [ -f .env ]; then
    source .env
fi

USERNAME="${LABEL_STUDIO_USERNAME:-admin@example.com}"

echo "Resetting password for: ${USERNAME}"
docker compose exec -it label-studio python /label-studio/label_studio/manage.py shell -c "
import getpass, sys
from users.models import User
email = '${USERNAME}'
try:
    u = User.objects.get(email=email)
except User.DoesNotExist:
    print(f'ERROR: user {email} not found')
    sys.exit(1)
current = getpass.getpass('Current password: ')
if not u.check_password(current):
    print('ERROR: current password is incorrect')
    sys.exit(1)
pw = getpass.getpass('New password: ')
pw2 = getpass.getpass('Confirm new password: ')
if pw != pw2:
    print('ERROR: passwords do not match')
    sys.exit(1)
u.set_password(pw)
u.save()
print(f'Password updated for {email}')
"
