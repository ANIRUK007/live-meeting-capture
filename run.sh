#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p certs
if [[ ! -f certs/cert.pem ]]; then
  openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -keyout certs/key.pem -out certs/cert.pem \
    -subj "/CN=live-meeting-capture" >/dev/null 2>&1
  echo "Generated self-signed cert in certs/ (browsers will show a one-time warning per phone)."
fi

exec python3 -m server.main
