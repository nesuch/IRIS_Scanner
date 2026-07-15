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
  # Restore from the GCS replica when one exists, else seed from the baked image.
  #
  # We rely on litestream's built-in `-if-replica-exists`, NOT on parsing the text
  # output of `litestream generations` (an earlier hand-rolled `... | tail -n +2 |
  # grep` check silently mis-reported an EXISTING single-generation replica as
  # absent, so every cold start reseeded the baked image and discarded all
  # replicated writes — passwords, audit logs, etc.).
  #
  # Semantics of `-if-replica-exists`:
  #   * no replica          -> exit 0, no file written  -> we seed the baseline
  #   * replica, restore ok  -> exit 0, file written     -> restored
  #   * replica, restore FAILS -> non-zero exit          -> we fail closed (exit 1),
  #                                                         never seed over good data
  echo "[entrypoint] Checking for GCS replica to restore..."
  if litestream restore -if-replica-exists -o "$DB" "$DB"; then
    if [ -f "$DB" ]; then
      echo "[entrypoint] Restored DB from GCS replica."
    else
      echo "[entrypoint] No replica found — seeding from image baseline."
      cp /app/iris.db "$DB"
    fi
  else
    echo "[entrypoint] ERROR: replica exists but restore failed — refusing to seed" >&2
    echo "[entrypoint] over it (fail closed). Crashing so the platform retries." >&2
    exit 1
  fi
fi

# Version-gated swap of the handbook-sourced financial tables from the baked seed
# into the live DB. financial_metrics (+ derived dim/flag tables) are purely
# handbook data, so replacing just those tables brings the reconciliation to
# production WITHOUT touching operational tables (users/flags/departments/clauses).
# Runs BEFORE replicate so the change lands in a fresh GCS generation; idempotent.
echo "[entrypoint] Applying financial-data migration (if needed)..."
python /app/migrate_financial.py --live "$DB" --seed /app/iris.db || \
  echo "[entrypoint] financial migration skipped/failed (continuing)"

echo "[entrypoint] Starting gunicorn under Litestream replication..."
exec litestream replicate -exec \
  "gunicorn --bind :${PORT:-8080} --workers 1 --threads 8 --timeout 120 app:app"
