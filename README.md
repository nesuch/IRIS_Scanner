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

## Default admin

Seeded on first run: `admin@irdai.gov.in` / `admin12345` (override with
`ADMIN_EMAIL` / `ADMIN_PASSWORD`). Accounts are restricted to the `@irdai.gov.in`
domain.
