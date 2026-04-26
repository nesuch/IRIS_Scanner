# IRIS Authentication + Authorization (Node/Express + MongoDB + React)

This folder contains a complete auth/RBAC implementation with:
- Email/password registration and login
- bcrypt password hashing
- JWT access token (15m) and refresh token rotation
- Tokens in HTTP-only cookies (no localStorage)
- Backend-enforced RBAC middleware
- Login rate limiting, CORS credentials, basic input validation

## Quick start

### Backend
```bash
cd auth-system/backend
cp .env.example .env
npm install
npm run dev
```

### Seed admin (optional but recommended)
```bash
npm run seed:admin
```

### Frontend
```bash
cd ../frontend
npm install
npm run dev
```

## Route mapping to requested IRIS modules

- Admin-only: `/api/admin`, `/api/system-analytics`
- Admin + Analyst: `/api/data-explorer`, `/api/compliance-cockpit`
- Viewer (and above): `/api/universal-module`
- All authenticated users: `/api/dashboard`

## Security notes
- Do **not** let clients send role during registration.
- Admin role is controlled by `ADMIN_EMAIL` (or seed script).
- Keep `JWT_SECRET` and `REFRESH_TOKEN_SECRET` strong and private.
