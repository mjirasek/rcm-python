#!/usr/bin/env bash
set -euo pipefail

DOMAIN="${DOMAIN:-rcm-api.michaeljirasek.com}"
EMAIL="${EMAIL:-}"

if [ -z "$EMAIL" ]; then
  echo "Set EMAIL first, for example:"
  echo "EMAIL=you@example.com bash deploy/oracle/enable_https.sh"
  exit 1
fi

sudo nginx -t
sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL" --redirect
sudo systemctl reload nginx
echo "HTTPS enabled: https://$DOMAIN/api/config"
