# IRIS_Scanner — IRDAI's Regulatory Intelligence System

A Flask backend (knowledge-base search, financial data intelligence, compliance,
analytics, auth/admin) with a modern **React (Vite) single-page frontend**.

## Architecture

- **Backend (unchanged business logic):** `app.py` + `iris_brain.py` hold all the
  logic — KB search, data filtering, Excel export, compliance, analytics, auth
  (Flask-Login, session cookies), admin, background sync. The DB schema is
  untouched.
- **JSON API layer:** `api.py` is a thin Flask blueprint mounted at `/api/*`. Every
  handler calls the *same* `iris_brain`/auth functions and returns JSON. It adds no
  business logic.
- **React SPA:** `frontend/` is a Vite + React + React Router app that consumes
  `/api/*`. It is the frontend for every page (search, health/life, data explorer,
  compliance, analytics, admin, profile, feedback, auth).
- **Serving:** Flask serves the built SPA shell (`frontend/dist`) for all non-API
  browser navigations and the hashed assets at `/assets/*`. The React app is the
  only UI; the legacy Jinja templates have been removed.

## Develop

Two processes — Flask API on `:8080`, Vite dev server on `:5173` (proxies `/api`
and `/static` to Flask):

    # terminal 1 — backend
    python app.py                      # http://127.0.0.1:8080

    # terminal 2 — frontend (hot reload)
    cd frontend
    npm install
    npm run dev                        # http://localhost:5173  (use this while developing)

## Build & run (production / single server)

    cd frontend && npm install && npm run build   # emits frontend/dist
    cd .. && python app.py                          # Flask serves API + the built SPA at :8080

Under gunicorn (as in `Procfile`) nothing changes — Flask serves `frontend/dist`
directly, so **no Node is required on the server**. The built `frontend/dist` is
committed (see `.gitignore`) so deploys need only the existing Python stack.

## Deploy to Cloud Run (SQLite + Litestream, persistent)

The app stays on SQLite (fast, in-process), and **Litestream** replicates the DB
file to a GCS bucket so data survives redeploys/restarts. Keep everything in the
**same region** (e.g. `asia-south1`) to avoid cross-region latency.

Files: `Dockerfile`, `docker-entrypoint.sh` (restore-on-boot → run under
`litestream replicate -exec`), `litestream.yml`. `app.py` allows SQLite in the
cloud only when `IRIS_PERSIST_SQLITE=1`.

    # 1) one-time: bucket in the SAME region as Cloud Run
    gcloud storage buckets create gs://YOUR_IRIS_DB_BUCKET \
        --location=asia-south1 --uniform-bucket-level-access

    # 2) let the Cloud Run service account read/write the bucket
    gcloud storage buckets add-iam-policy-binding gs://YOUR_IRIS_DB_BUCKET \
        --member=serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com \
        --role=roles/storage.objectAdmin

    # 3) deploy (single instance = single writer; CPU always on for Litestream)
    gcloud run deploy iris-scanner --source . --region asia-south1 \
        --min-instances 1 --max-instances 1 --no-cpu-throttling \
        --memory 1Gi --cpu 1 \
        --set-env-vars IRIS_PERSIST_SQLITE=1,IRIS_DB_PATH=/data/iris.db,\
    LITESTREAM_BUCKET=YOUR_IRIS_DB_BUCKET,IRIS_SESSION_SECRET=CHANGE_ME_STABLE

Notes:
- `--max-instances 1` is **required** — SQLite/Litestream is single-writer.
- `--min-instances 1 --no-cpu-throttling` keeps the instance warm and lets
  Litestream replicate the WAL between requests (don't skip `--no-cpu-throttling`).
- **Remove any old Neon/Postgres env vars** (`IRIS_AUTH_DATABASE_URL`,
  `DATABASE_URL`) so the app uses SQLite. If a Postgres URL is set, it wins.
- Use a **stable** `IRIS_SESSION_SECRET` (ideally via Secret Manager) — changing
  it invalidates all sessions.
- First deploy seeds the DB from the image baseline; afterward the GCS replica is
  authoritative. Update regulatory content via **Admin → Sync Data** (which
  rebuilds the tables and Litestream replicates them).

## Default admin

Seeded on first run: `admin@irdai.gov.in` / `admin12345` (override with
`ADMIN_EMAIL` / `ADMIN_PASSWORD`). Accounts are restricted to the `@irdai.gov.in`
domain.
