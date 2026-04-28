# IRIS Scanner

IRDAI's Regulatory Intelligence System.

## Why the app can work once and fail later

This project currently has **unpinned dependencies** in `requirements.txt`. That means running `pip install -r requirements.txt` on different days can install different transitive versions (Flask/Werkzeug/SQLAlchemy/pandas stack), even when `app.py` is unchanged. This is the most common root cause when production is stable but local re-installs break.

Other common local-only drift sources:

- Different Python runtime than deployment (for example macOS Python 3.12 locally vs Cloud Run/VM Python 3.10).
- Missing environment variables in a new shell (`DATABASE_URL`, `IRIS_AUTH_DATABASE_URL`, secrets).
- Local DB file drift (`iris.db` schema/data changed, permissions, stale WAL/SHM files).
- Different startup path / working directory affecting relative paths.

## Fast diagnostics (run these every time)

### 1) Capture runtime + package + env snapshot

```bash
python scripts/diagnose_env.py > local_env_snapshot.json
```

This reports:
- Python executable/version/platform
- Whether you are in the expected virtualenv
- Required package versions (and missing packages)
- Key environment-variable presence (values masked)
- Local SQLite health/table visibility

### 2) Capture installed dependency graph

```bash
pip freeze > requirements.lock.current.txt
```

Keep this file from the **known-good** environment and compare later.

### 3) Verify effective runtime config

```bash
python -c "import os;print('DATABASE_URL set:',bool(os.getenv('DATABASE_URL')));print('IRIS_AUTH_DATABASE_URL set:',bool(os.getenv('IRIS_AUTH_DATABASE_URL')));print('IRIS_DB_PATH:',os.getenv('IRIS_DB_PATH'))"
```

## Reproducible local workflow (recommended)

1. Use one Python version everywhere (match production exactly).
2. Recreate virtualenv from scratch:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Store local env vars in a file that is loaded explicitly per shell session (for example with direnv or a sourced script).
4. Generate and commit a lock snapshot whenever dependencies change:

```bash
pip freeze > requirements.lock.txt
```

5. Before deploying, run `python scripts/diagnose_env.py` and archive output with the release.

## Recover exact working state from GCP

If production is currently healthy, treat it as source of truth.

### A) Recover dependency versions from deployed image/container

If using Cloud Run:

```bash
gcloud run services describe <SERVICE_NAME> --region <REGION> --format='value(spec.template.spec.containers[0].image)'
```

Then inspect that image locally (or in CI) and export package versions:

```bash
docker run --rm <IMAGE_URL> python -m pip freeze > requirements.from_gcp.txt
```

### B) Recover deployed env vars (non-secret + references)

```bash
gcloud run services describe <SERVICE_NAME> --region <REGION> --format export > cloudrun_service_export.yaml
```

Review `env` entries and secret references; mirror them into your local shell (without committing secrets).

### C) Recover DB connectivity assumptions

- Confirm local points to the same intended DB backend (`IRIS_AUTH_DATABASE_URL`/`DATABASE_URL`).
- If local should use SQLite, back up and reset safely:

```bash
cp iris.db iris.db.bak.$(date +%Y%m%d_%H%M%S)
```

## Branch recovery strategy when local is messy

When branch history is confusing, recover from a known deploy commit:

```bash
git fetch origin
git checkout -b recovery/<date> <KNOWN_GOOD_COMMIT>
python scripts/diagnose_env.py > recovery_env_snapshot.json
```

If you only know “what is in production”, identify deployed commit via Cloud Build/Cloud Run labels and check out that exact SHA.

## Minimal team policy to avoid future drift

- Always pin runtime (Python version) in deployment + local tooling.
- Always save `pip freeze` artifact for every release.
- Keep `.env.example` current with required variables.
- Add a pre-release checklist item: attach `scripts/diagnose_env.py` output.


## Database migrations (authoritative schema management)

This repo now uses **Flask-Migrate/Alembic** for schema changes. Do not rely on `db.create_all()` as your migration strategy.

### Apply migrations (local/prod)

```bash
export FLASK_APP=app.py
flask db upgrade
```

### Create a new migration after model changes

```bash
export FLASK_APP=app.py
flask db migrate -m "describe change"
flask db upgrade
```

### Current migration

- `20260428_0001_add_user_email_to_system_logs` adds `system_logs.user_email` for older SQLite files.

## Keep schema consistent across environments

- Run `flask db upgrade` as part of startup/release pipeline before serving traffic.
- Never edit production schema manually.
- Store migration files in version control and deploy them with code.
- Keep one source of truth for DB URL configuration (`IRIS_AUTH_DATABASE_URL` preferred).
