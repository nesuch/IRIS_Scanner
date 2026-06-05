#!/bin/sh
# IRIS — container entrypoint.
# 1) Restore the SQLite DB from the GCS replica (if one exists), else seed it from
#    the image baseline baked in at build time.
# 2) Launch the app under `litestream replicate -exec`, which streams the WAL to
#    GCS continuously and performs a final sync on shutdown (Cloud Run SIGTERM).
set -e

DB="${IRIS_DB_PATH:-/data/iris.db}"
mkdir -p "$(dirname "$DB")"

if [ ! -f "$DB" ]; then
  echo "[entrypoint] No local DB at $DB — attempting restore from replica..."
  litestream restore -if-replica-exists -o "$DB" "$DB" || true
  if [ -f "$DB" ]; then
    echo "[entrypoint] Restored DB from GCS replica."
  else
    echo "[entrypoint] No replica found — seeding from image baseline."
    cp /app/seed/iris.db "$DB"
  fi
fi

echo "[entrypoint] Starting gunicorn under Litestream replication..."
exec litestream replicate -exec \
  "gunicorn --bind :${PORT:-8080} --workers 1 --threads 8 --timeout 120 app:app"
