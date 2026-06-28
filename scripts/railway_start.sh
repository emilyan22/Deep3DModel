#!/bin/sh
set -e
cd /app

PYTHON=/app/.venv/bin/python
UVICORN=/app/.venv/bin/uvicorn

# Best-effort model fetch; API still boots for /api/health if this fails.
"$PYTHON" scripts/download_deep3d_model.py || \
  echo "WARN: model download failed — /api/convert will error until weights exist."

exec "$UVICORN" backend.app:app --host 0.0.0.0 --port "${PORT:-8000}"
