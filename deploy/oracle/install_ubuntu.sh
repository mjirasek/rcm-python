#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/rcm-python}"
APP_USER="${APP_USER:-${SUDO_USER:-$USER}}"
APP_GROUP="$(id -gn "$APP_USER")"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git rsync nginx certbot python3-certbot-nginx

sudo mkdir -p "$APP_DIR"
if [ "$REPO_DIR" != "$APP_DIR" ]; then
  sudo rsync -a --delete \
    --exclude .git --exclude .venv --exclude build --exclude dist \
    --exclude .github --exclude frontend --exclude tests \
    --exclude __pycache__ --exclude .pytest_cache --exclude .pytest-tmp \
    "$REPO_DIR"/ "$APP_DIR"/
fi
sudo chown -R "$APP_USER:$APP_GROUP" "$APP_DIR"

cd "$APP_DIR"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install ".[api]"

sed \
  -e "s|__APP_DIR__|$APP_DIR|g" \
  -e "s|__APP_USER__|$APP_USER|g" \
  -e "s|__APP_GROUP__|$APP_GROUP|g" \
  deploy/oracle/rcm-api.service.template | sudo tee /etc/systemd/system/rcm-api.service >/dev/null

sudo cp deploy/oracle/nginx-rcm.conf /etc/nginx/sites-available/rcm-api
sudo ln -sf /etc/nginx/sites-available/rcm-api /etc/nginx/sites-enabled/rcm-api
sudo rm -f /etc/nginx/sites-enabled/default

sudo systemctl daemon-reload
sudo systemctl enable --now rcm-api
sudo nginx -t
sudo systemctl reload nginx

echo "RCM API backend is installed."
echo "API:  http://SERVER_IP/api/config"
echo "After DNS points rcm-api.michaeljirasek.com here, run: bash deploy/oracle/enable_https.sh"
