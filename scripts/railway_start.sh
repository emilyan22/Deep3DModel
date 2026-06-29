#!/bin/sh
cd /app

PYTHON=/app/.venv/bin/python
UVICORN=/app/.venv/bin/uvicorn

# Do not block startup: gdown can hang for minutes and Railway health checks will fail.
(
  "$PYTHON" scripts/download_deep3d_model.py || \
    echo "WARN: model download failed — /api/convert will error until weights exist."
) &

exec "$UVICORN" backend.app:app --host 0.0.0.0 --port "${PORT:-8000}"
