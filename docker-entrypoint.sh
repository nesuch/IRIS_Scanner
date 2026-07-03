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
  # Only seed from the baked image when there is GENUINELY no replica (first deploy
  # / fresh path). If a replica EXISTS but the restore fails, fail closed: crash the
  # container (set -e) instead of silently seeding the baked baseline and then
  # replicating that over good data. (A transient restore failure once wiped prod
  # back to the 8-doc seed — never again.)
  if litestream generations "$DB" 2>/dev/null | tail -n +2 | grep -q .; then
    echo "[entrypoint] Replica found — restoring (mandatory, no fallback)."
    litestream restore -o "$DB" "$DB"
    echo "[entrypoint] Restored DB from GCS replica."
  else
    echo "[entrypoint] No replica found — seeding from image baseline."
    cp /app/iris.db "$DB"
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
