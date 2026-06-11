from flask import Flask, request, send_file, jsonify, session, send_from_directory
from typing import List
import re
import pandas as pd
import io
import os
import json
import time
import traceback
import hashlib
import threading # Required for Background Sync
from datetime import datetime, timedelta
import secrets
from zoneinfo import ZoneInfo
from werkzeug.exceptions import HTTPException
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from sqlalchemy import inspect, text
import iris_brain as brain

app = Flask(__name__)
app.secret_key = os.getenv("IRIS_SESSION_SECRET", "iris-dev-session-secret")
# Session cookie lifetime — server-side backstop for idle auto-logout. The SPA
# enforces a shorter inactivity timeout client-side; this caps the absolute age.
app.permanent_session_lifetime = timedelta(hours=int(os.getenv("IRIS_SESSION_HOURS", "8")))
DB_NAME = os.path.abspath(os.getenv("IRIS_DB_PATH", "iris.db"))
os.makedirs(os.path.dirname(DB_NAME), exist_ok=True)
IRIS_AUTH_DATABASE_URL = (os.getenv("IRIS_AUTH_DATABASE_URL") or "").strip()
DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
AUTH_DATABASE_URL = IRIS_AUTH_DATABASE_URL or DATABASE_URL
running_in_cloud = any(os.getenv(flag) for flag in ("K_SERVICE", "GAE_ENV", "GOOGLE_CLOUD_PROJECT"))
# Opt-in: SQLite is durable in the cloud when Litestream replicates the DB file
# to object storage (restore-on-boot + continuous WAL replication; see Dockerfile
# / docker-entrypoint.sh). This keeps SQLite's in-process speed without losing
# data on redeploy. Without it, a cloud deploy still requires a Postgres URL.
persistent_sqlite = os.getenv("IRIS_PERSIST_SQLITE", "").lower() in {"1", "true", "yes"}

if AUTH_DATABASE_URL:
    if AUTH_DATABASE_URL.startswith("postgres://"):
        AUTH_DATABASE_URL = AUTH_DATABASE_URL.replace("postgres://", "postgresql://", 1)
    app.config["SQLALCHEMY_DATABASE_URI"] = AUTH_DATABASE_URL
elif running_in_cloud and not persistent_sqlite:
    raise RuntimeError(
        "Persistent storage is required in cloud deployments. Either set "
        "IRIS_AUTH_DATABASE_URL (Postgres), or set IRIS_PERSIST_SQLITE=1 to run "
        "SQLite with Litestream replication (see Dockerfile)."
    )
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_NAME}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
# SPA-only: there is no server-rendered login page. Unauthorized API calls are
# handled in before_request (401 JSON); page requests get the SPA shell.
login_manager.login_view = None
login_manager.login_message = None

ALLOWED_EMAIL_DOMAIN = "@irdai.gov.in"

# --- React SPA serving -------------------------------------------------------
# The Vite build is emitted to frontend/dist. Flask serves the SPA shell for all
# non-API browser navigations; the React app is the only UI (the legacy Jinja
# templates were removed). The JSON API lives under /api/* (see api.py).
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
SPA_INDEX = os.path.join(FRONTEND_DIST, "index.html")


class User(UserMixin, db.Model):
    __tablename__ = "auth_users"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    is_admin = db.Column(db.Boolean, nullable=False, default=False)
    reset_token = db.Column(db.String(64), nullable=True)
    reset_token_expiry = db.Column(db.DateTime, nullable=True)
    session_version = db.Column(db.Integer, nullable=False, default=0)
    display_name = db.Column(db.String(120), nullable=True)
    avatar = db.Column(db.Text, nullable=True)  # small base64 data-URL (resized client-side)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class SystemLog(db.Model):
    __tablename__ = "system_logs"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    endpoint = db.Column(db.String(255), nullable=True)
    method = db.Column(db.String(16), nullable=True)
    ip = db.Column(db.String(64), nullable=True)
    status = db.Column(db.Integer, nullable=True)
    error_msg = db.Column(db.Text, nullable=True)
    user_email = db.Column(db.String(255), nullable=True)


class UserSession(db.Model):
    __tablename__ = "user_sessions"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("auth_users.id"), nullable=False, index=True)
    session_token = db.Column(db.String(255), nullable=False, index=True)
    ip = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.String(300), nullable=True)
    active = db.Column(db.Boolean, nullable=False, default=True, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    last_seen_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class PasswordResetAudit(db.Model):
    __tablename__ = "password_reset_audit"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), nullable=False)
    reset_link = db.Column(db.Text, nullable=False)
    requested_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)


class AdminAuditLog(db.Model):
    __tablename__ = "admin_audit_logs"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), nullable=True)
    action_type = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(64), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class FeedbackEntry(db.Model):
    __tablename__ = "feedback_entries"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("auth_users.id"), nullable=False, index=True)
    category = db.Column(db.String(32), nullable=False, default="Suggestion")
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(16), nullable=False, default="Open")  # Open | Done | Ignored
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class SearchLog(db.Model):
    """Records each knowledge-base search query for admin usage visibility."""
    __tablename__ = "search_logs"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(255), nullable=True, index=True)
    module = db.Column(db.String(32), nullable=True)
    # NB: attribute is `query_text` (not `query`) — a `query` attribute would
    # shadow Flask-SQLAlchemy's Model.query and break SearchLog.query reads.
    query_text = db.Column("query", db.Text, nullable=True)
    result_count = db.Column(db.Integer, nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class Announcement(db.Model):
    """Team → user communications (updates, notices) shown in the SPA bell menu."""
    __tablename__ = "announcements"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    body = db.Column(db.Text, nullable=False)
    level = db.Column(db.String(16), nullable=False, default="info")  # info | success | warning
    created_by = db.Column(db.String(255), nullable=True)
    active = db.Column(db.Boolean, nullable=False, default=True, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class FeedbackComment(db.Model):
    """Follow-up thread on a feedback entry (user follow-ups + admin replies)."""
    __tablename__ = "feedback_comments"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    feedback_id = db.Column(db.Integer, db.ForeignKey("feedback_entries.id"), nullable=False, index=True)
    author_email = db.Column(db.String(255), nullable=True)
    is_admin = db.Column(db.Boolean, nullable=False, default=False)
    body = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class FlagComment(db.Model):
    """Follow-up thread on a flag (user follow-ups + admin replies)."""
    __tablename__ = "flag_comments"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    flag_id = db.Column(db.Integer, db.ForeignKey("flags.id"), nullable=False, index=True)
    author_email = db.Column(db.String(255), nullable=True)
    is_admin = db.Column(db.Boolean, nullable=False, default=False)
    body = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class Flag(db.Model):
    """User-reported issue on a specific clause or financial data row."""
    __tablename__ = "flags"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(255), nullable=True, index=True)
    kind = db.Column(db.String(16), nullable=False, default="clause")  # clause | financial
    reason = db.Column(db.String(64), nullable=False)
    description = db.Column(db.Text, nullable=True)
    target = db.Column(db.String(300), nullable=True)   # source/clause id or report context
    detail = db.Column(db.Text, nullable=True)          # snapshot (clause text / row json)
    screenshot = db.Column(db.Text, nullable=True)      # base64 PNG data URL of the user's screen
    status = db.Column(db.String(16), nullable=False, default="Open")  # Open | Resolved | Dismissed
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class PqDocument(db.Model):
    """A Parliamentary Question reply, rendered to HTML for faithful in-app reading
    (bold/italic/underline/tables preserved) with the original .docx kept for download."""
    __tablename__ = "pq_documents"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    pq_no = db.Column(db.String(40), nullable=True, index=True)
    house = db.Column(db.String(24), nullable=True)         # Lok Sabha / Rajya Sabha
    title = db.Column(db.String(400), nullable=False)
    subject = db.Column(db.Text, nullable=True)
    doc_date = db.Column(db.String(40), nullable=True)
    tags = db.Column(db.Text, nullable=True)                # comma-separated
    html = db.Column(db.Text, nullable=False)               # mammoth-rendered body
    body_text = db.Column(db.Text, nullable=True)           # plain text (search/preview)
    docx_filename = db.Column(db.String(300), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class EditingSession(db.Model):
    """Advisory presence: who currently has a document open in Studio (heartbeat)."""
    __tablename__ = "editing_sessions"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    source_doc = db.Column(db.String(255), nullable=False, index=True)
    email = db.Column(db.String(255), nullable=True)
    last_seen = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class DocumentAsset(db.Model):
    """An original source PDF an admin attached to a document — shown in the
    Studio pane and offered to users as a download for that document's clauses."""
    __tablename__ = "document_assets"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    source_doc = db.Column(db.String(255), nullable=False, unique=True, index=True)
    pdf_filename = db.Column(db.String(300), nullable=True)
    uploaded_by = db.Column(db.String(255), nullable=True)
    uploaded_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class ClauseVersion(db.Model):
    """Audit/restore snapshot of a clause's content, written before each in-app
    edit. Lets admins see history and revert (important for regulatory text)."""
    __tablename__ = "clause_versions"
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    clause_id = db.Column(db.String(120), nullable=False, index=True)
    source_doc = db.Column(db.String(255), nullable=False, index=True)
    html = db.Column(db.Text, nullable=True)
    body_text = db.Column(db.Text, nullable=True)
    tags = db.Column(db.Text, nullable=True)
    edited_by = db.Column(db.String(255), nullable=True)
    edited_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# ==========================================
# CRITICAL FIX: FORCE DATA LOAD ON STARTUP
# ==========================================
# This ensures that as soon as you run python app.py, 
# the system loads the data from SQL/Files into memory.
print("--- IRIS: Initializing Data Engine ---")
try:
    # 1. Load Knowledge Base (Text Search)
    brain.load_knowledge_base()
    # 2. Load Master Data Engine (Financial Data)
    brain.load_master_data_engine()
except Exception as e:
    print(f"[!] Warning: Data Engine load failed on startup: {e}")

CHAT_HISTORY = []
JUST_REDIRECTED = False
ADMIN_ONLY_PATHS = {"/admin", "/admin/sync_start", "/admin/sync_status", "/clear_logs"}
PUBLIC_AUTH_PATHS = {"/login", "/logout", "/forgot-password"}
# The SPA talks to /api/*, so usage/analytics is tracked on the API endpoints the
# React pages actually hit (the old Jinja page routes are gone). /api/me and other
# polling/auth calls are intentionally excluded so they don't inflate the numbers.
TRACKED_MODULE_ENDPOINTS = {
    "/api/search": "Knowledge Search",
    "/api/data/filter": "Data Explorer",
    "/api/data/download": "Data Export",
    "/api/compliance": "Compliance Cockpit",
    "/api/analytics": "System Analytics",
    "/api/admin/overview": "Admin Panel",
}
DISPLAY_TZ = ZoneInfo("Asia/Kolkata")


def _format_dt_local(dt_obj: datetime) -> str:
    if dt_obj is None:
        return None
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=ZoneInfo("UTC"))
    return dt_obj.astimezone(DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S IST")


def _route_exists(path: str) -> bool:
    try:
        app.url_map.bind("").match(path, method=request.method)
        return True
    except Exception:
        return False


def _safe_next_url(next_url: str) -> str:
    if not next_url or not next_url.startswith("/"):
        return "/"
    if next_url in PUBLIC_AUTH_PATHS:
        return "/"
    if not _route_exists(next_url):
        return "/"
    return next_url


def _is_allowed_email(email: str) -> bool:
    return email.endswith(ALLOWED_EMAIL_DOMAIN)


def _hash_reset_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _seed_admin_user():
    admin_email = (os.getenv("ADMIN_EMAIL") or f"admin{ALLOWED_EMAIL_DOMAIN}").strip().lower()
    admin_password = os.getenv("ADMIN_PASSWORD") or "admin12345"
    if not _is_allowed_email(admin_email):
        admin_email = f"admin{ALLOWED_EMAIL_DOMAIN}"

    existing = User.query.filter_by(email=admin_email).first()
    if existing:
        existing.is_admin = True
        if not existing.password_hash:
            existing.password_hash = generate_password_hash(admin_password)
    else:
        db.session.add(
            User(
                email=admin_email,
                password_hash=generate_password_hash(admin_password),
                is_admin=True,
                is_active=True,
                created_at=datetime.utcnow(),
            )
        )
        print(f"[+] Default admin user created: {admin_email}")
    db.session.commit()


def _start_user_session(user: User):
    session_token = secrets.token_urlsafe(32)
    now = datetime.utcnow()
    db.session.add(
        UserSession(
            user_id=user.id,
            session_token=session_token,
            ip=get_client_ip(),
            user_agent=(request.headers.get("User-Agent") or "")[:300],
            active=True,
            created_at=now,
            last_seen_at=now,
        )
    )
    db.session.commit()
    session.permanent = True
    session["auth_token"] = session_token
    session["session_version"] = user.session_version


# A session idle longer than this is treated as logged-out (server-side expiry),
# so the active-device count reflects reality instead of growing forever.
SESSION_IDLE_MINUTES = int(os.getenv("IRIS_SESSION_IDLE_MIN", "720"))  # 12h default


def _expire_stale_sessions(user_id: int = None):
    """Deactivate sessions whose last activity is older than the idle window."""
    cutoff = datetime.utcnow() - timedelta(minutes=SESSION_IDLE_MINUTES)
    q = UserSession.query.filter(UserSession.active.is_(True), UserSession.last_seen_at < cutoff)
    if user_id is not None:
        q = q.filter(UserSession.user_id == user_id)
    if q.update({"active": False}, synchronize_session=False):
        db.session.commit()


def _is_session_active(user_id: int, session_token: str) -> bool:
    if not session_token:
        return False
    row = UserSession.query.filter_by(
        user_id=user_id, session_token=session_token, active=True
    ).first()
    if row is None:
        return False
    now = datetime.utcnow()
    # Idle too long → expire this session (server-side auto-logout).
    if row.last_seen_at and row.last_seen_at < now - timedelta(minutes=SESSION_IDLE_MINUTES):
        row.active = False
        db.session.commit()
        return False
    # Keep-alive: refresh last_seen at most once a minute to limit writes.
    if not row.last_seen_at or (now - row.last_seen_at).total_seconds() > 60:
        row.last_seen_at = now
        db.session.commit()
    return True


def _deactivate_session_token(user_id: int, session_token: str):
    if not session_token:
        return
    UserSession.query.filter_by(user_id=user_id, session_token=session_token).update(
        {"active": False, "last_seen_at": datetime.utcnow()}
    )
    db.session.commit()


def _deactivate_session_by_id(user_id: int, session_id: int):
    UserSession.query.filter_by(id=session_id, user_id=user_id).update(
        {"active": False, "last_seen_at": datetime.utcnow()}
    )
    db.session.commit()


def _list_user_sessions(user_id: int):
    rows = (
        UserSession.query.with_entities(
            UserSession.id,
            UserSession.ip,
            UserSession.user_agent,
            UserSession.active,
            UserSession.created_at,
            UserSession.last_seen_at,
        )
        .filter_by(user_id=user_id, active=True)
        .order_by(UserSession.id.desc())
        .limit(30)
        .all()
    )
    return [
        {
            "id": row.id,
            "ip": row.ip,
            "user_agent": row.user_agent,
            "active": row.active,
            "created_at": _format_dt_local(row.created_at),
            "last_seen_at": _format_dt_local(row.last_seen_at),
        }
        for row in rows
    ]


def _deactivate_all_user_sessions(user_id: int):
    UserSession.query.filter_by(user_id=user_id).update(
        {"active": False, "last_seen_at": datetime.utcnow()}
    )
    db.session.commit()


def _get_active_device_count(user_id: int) -> int:
    _expire_stale_sessions(user_id)  # drop idle sessions so the count is accurate
    count = UserSession.query.filter_by(user_id=user_id, active=True).count()
    return int(count or 0)


def _purge_user_dependents(user_id: int):
    """Remove rows that FK-reference a user so the user can be hard-deleted.

    SQLite ignores foreign keys by default (delete "just works"), but Postgres
    (cloud) enforces them — deleting a user with sessions/feedback raises an
    IntegrityError. Clear the child rows first so delete works on both.
    """
    UserSession.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    # Delete comments on this user's feedback before the feedback rows (FK order).
    fb_ids = [f.id for f in FeedbackEntry.query.with_entities(FeedbackEntry.id)
              .filter_by(user_id=user_id).all()]
    if fb_ids:
        FeedbackComment.query.filter(FeedbackComment.feedback_id.in_(fb_ids)).delete(synchronize_session=False)
    FeedbackEntry.query.filter_by(user_id=user_id).delete(synchronize_session=False)


def _record_admin_audit(email: str, action_type: str, status: str):
    try:
        db.session.add(
            AdminAuditLog(
                email=(email or "").strip().lower() or None,
                action_type=action_type,
                status=status,
                timestamp=datetime.utcnow(),
            )
        )
        db.session.commit()
    except Exception as e:
        app.logger.warning("Unable to record admin audit log: %s", e)


def _record_search(email: str, module: str, query: str, result_count: int = None):
    """Best-effort logging of a search query for admin usage visibility."""
    q = (query or "").strip()
    if not q or q.startswith("__DEEP_SCAN__"):
        return
    try:
        db.session.add(
            SearchLog(
                user_email=(email or "").strip().lower() or None,
                module=(module or "")[:32],
                query_text=q[:1000],
                result_count=result_count,
                timestamp=datetime.utcnow(),
            )
        )
        db.session.commit()
    except Exception as e:
        app.logger.warning("Unable to record search log: %s", e)


@login_manager.user_loader
def load_user(user_id: str):
    if not user_id.isdigit():
        return None
    return db.session.get(User, int(user_id))


def _ensure_database_schema():
    """Backfills missing columns/tables when database files are out of sync with models."""
    db.create_all()
    inspector = inspect(db.engine)

    required_columns = {
        "auth_users": {
            "is_active": "BOOLEAN NOT NULL DEFAULT 1",
            "is_admin": "BOOLEAN NOT NULL DEFAULT 0",
            "reset_token": "VARCHAR(64)",
            "reset_token_expiry": "DATETIME",
            "session_version": "INTEGER NOT NULL DEFAULT 0",
            "display_name": "VARCHAR(120)",
            "avatar": "TEXT",
            "created_at": "DATETIME",
        },
        "system_logs": {
            "user_email": "VARCHAR(255)",
        },
        "user_sessions": {
            "active": "BOOLEAN NOT NULL DEFAULT 1",
            "created_at": "DATETIME",
            "last_seen_at": "DATETIME",
        },
        "feedback_entries": {
            "status": "VARCHAR(16) NOT NULL DEFAULT 'Open'",
        },
        "flags": {
            "screenshot": "TEXT",
        },
        "regulatory_clauses": {
            "clause_html": "TEXT",          # rich edited body (HTML); null => render clause_text
            "updated_at": "DATETIME",
            "updated_by": "VARCHAR(255)",
            "sort_order": "INTEGER",        # clause ordering within a document (Studio)
        },
    }

    for table_name, cols in required_columns.items():
        if not inspector.has_table(table_name):
            continue
        existing_cols = {c["name"] for c in inspector.get_columns(table_name)}
        for col_name, col_type in cols.items():
            if col_name in existing_cols:
                continue
            db.session.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"))
            app.logger.warning("Backfilled missing column %s.%s", table_name, col_name)

    # Seed clause ordering from row insertion order where not yet set.
    if inspector.has_table("regulatory_clauses"):
        try:
            db.session.execute(text("UPDATE regulatory_clauses SET sort_order = rowid WHERE sort_order IS NULL"))
        except Exception:
            pass

    db.session.commit()


with app.app_context():
    _ensure_database_schema()
    _seed_admin_user()


@app.context_processor
def inject_device_count():
    if current_user.is_authenticated:
        return {"current_device_count": _get_active_device_count(current_user.id)}
    return {"current_device_count": 0}


@app.before_request
def iris_auth_gatekeeper():
    # Static files and the Vite build assets pass straight through.
    if request.path.startswith(("/static", "/assets")) or request.path == "/favicon.ico":
        return None

    # --- JSON API gate -------------------------------------------------------
    # The React SPA talks to /api/*. Public auth endpoints are open; everything
    # else requires an authenticated, still-valid session (else 401 JSON, so the
    # SPA can route to its login screen).
    if request.path.startswith("/api/"):
        public_api = {"/api/login", "/api/logout", "/api/me", "/api/forgot-password"}
        if request.path in public_api or request.path.startswith("/api/reset-password/"):
            return None
        if not current_user.is_authenticated:
            return jsonify({"error": "auth_required"}), 401
        expected_version = getattr(current_user, "session_version", 0)
        current_version = session.get("session_version", 0)
        session_token = session.get("auth_token")
        if current_version != expected_version or not _is_session_active(current_user.id, session_token):
            logout_user()
            session.clear()
            return jsonify({"error": "session_invalid"}), 401
        return None

    # --- Everything else is the React SPA ------------------------------------
    # Serve the built shell for all non-API page requests; React Router renders
    # the route and auth is enforced client-side (/api/me) + on every /api call.
    if os.path.exists(SPA_INDEX):
        return send_file(SPA_INDEX)
    return ("Frontend build not found. Run `npm run build` in frontend/.", 503)

# ==========================================
# 0. SYSTEM ANALYTICS (MIDDLEWARE)
# ==========================================

def get_client_ip():
    """
    Extracts the real client IP when app is behind reverse proxies/load balancers.
    Falls back to Flask's remote_addr for local/dev usage.
    """
    # X-Forwarded-For may contain a chain of IPs: client, proxy1, proxy2
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        first_ip = forwarded_for.split(",")[0].strip()
        if first_ip:
            return first_ip

    # Common alternative header used by some proxies/CDNs
    real_ip = request.headers.get("X-Real-IP", "").strip()
    if real_ip:
        return real_ip

    return request.remote_addr


def log_interaction(status_code, error_msg=None):
    """
    Records every request to the database (system_logs table).
    Filters out static assets and favicons to keep analytics clean.
    """
    # --- FILTER: Ignore static files AND favicon ---
    if request.path.startswith('/static') or request.path == '/favicon.ico': 
        return
    
    try:
        db.session.add(
            SystemLog(
                timestamp=datetime.utcnow(),
                endpoint=request.path,
                method=request.method,
                ip=get_client_ip(),
                status=status_code,
                error_msg=error_msg,
                user_email=current_user.email if current_user.is_authenticated else None,
            )
        )
        db.session.commit()
    except Exception as e:
        print(f"Logging Failed: {e}") 

@app.after_request
def record_success(response):
    """Logs successful requests (200, 302, 404, etc.)"""
    # We only log here if it's NOT a 500 (500s are handled by handle_crash)
    if response.status_code < 500:
        log_interaction(response.status_code)
    return response

@app.errorhandler(Exception)
def handle_crash(e):
    """
    Catches CRASHES (500 errors), logs them with the traceback, 
    and keeps IRIS alive instead of crashing the server.
    """
    # Let Flask handle standard HTTP errors (404/405/etc.) normally.
    if isinstance(e, HTTPException):
        return e

    # 1. Capture the full traceback to know EXACTLY where it failed
    error_trace = str(traceback.format_exc())
    print(f"🔥 IRIS CRASHED: {error_trace}") # Print to terminal for debugging
    
    # 2. Extract the specific error line for the UI log (last non-empty line)
    detailed_error = error_trace.strip().split('\n')[-1]
    
    log_interaction(500, error_msg=detailed_error) # Log detailed error
    return "<h3>IRIS System Error</h3><p>The system encountered an error. It has been logged for the admin.</p>", 500

# ==========================================
# CONFIGURATION
# ==========================================
PDF_MAP = {
    "HEALTH MASTER CIRCULAR 2024": ["documents/health/health_master_circular_2024.pdf", "documents/health/HEALTH_MC_2024.pdf"],
    "HEALTH MC 2024": ["documents/health/HEALTH_MC_2024.pdf", "documents/health/health_master_circular_2024.pdf"],
    "PRODUCT REGULATIONS 2024": ["documents/health/product_regulations_2024.pdf", "documents/health/PRODUCT_REGS_2024.pdf"],
    "PRODUCT REGS 2024": ["documents/health/PRODUCT_REGS_2024.pdf", "documents/health/product_regulations_2024.pdf"],
    "PPHI REGULATIONS 2024": ["documents/health/PPHI_REGS_2024.pdf"],
    "PPHI REGS 2024": ["documents/health/PPHI_REGS_2024.pdf"],
    "PPHI MASTER CIRCULAR 2024": ["documents/health/PPHI_MC_2024.pdf"],
    "PPHI MC 2024": ["documents/health/PPHI_MC_2024.pdf"],
    "INSURANCE ACT 1938": ["documents/health/INSURANCE_ACT_1938.pdf"],
    "IRDAI ACT 1999": ["documents/health/IRDAI_ACT_1999.pdf"]
}


def resolve_pdf_path(doc_name_key):
    candidates = PDF_MAP.get(doc_name_key, [])
    for rel_path in candidates:
        abs_path = os.path.join(app.static_folder, rel_path)
        if os.path.exists(abs_path):
            return rel_path
    return None

TYPE_STYLES = {
    "ACT": {"color": "#856404", "bg": "#fff3cd", "border": "#ffeeba", "label": "ACT (The Law)"},
    "REGULATION": {"color": "#004085", "bg": "#cce5ff", "border": "#b8daff", "label": "REGULATION"},
    "MASTER": {"color": "#155724", "bg": "#d4edda", "border": "#c3e6cb", "label": "MASTER CIRCULAR"},
    "CIRCULAR": {"color": "#0c5460", "bg": "#d1ecf1", "border": "#bee5eb", "label": "CIRCULAR"},
    "GUIDELINE": {"color": "#383d41", "bg": "#e2e3e5", "border": "#d6d8db", "label": "GUIDELINE"},
    "UNKNOWN": {"color": "#666", "bg": "#f2f2f2", "border": "#ddd", "label": "DOCUMENT"}
}

# ==========================================
# ASYNC SYNC ENGINE (BACKGROUND THREADS)
# ==========================================
# Global state to track the background job.
# The Admin UI polls this variable.
SYNC_STATE = {
    "status": "idle",       # idle, running, complete, error
    "message": "System ready.",
    "timestamp": None
}

def run_background_sync():
    """Executes the heavy data aggregation logic in a separate thread."""
    global SYNC_STATE
    try:
        print("--- BACKGROUND SYNC STARTED ---")
        SYNC_STATE["status"] = "running"
        SYNC_STATE["message"] = "Syncing financial + regulatory data from knowledge_base..."

        financial_msg = brain.aggregate_submissions()
        regulatory_msg = brain.aggregate_regulatory_documents()
        result_msg = f"{financial_msg} | {regulatory_msg}"

        SYNC_STATE["status"] = "complete"
        SYNC_STATE["message"] = result_msg
        SYNC_STATE["timestamp"] = time.strftime("%H:%M:%S")
        print("--- BACKGROUND SYNC FINISHED ---")
        
    except Exception as e:
        print(f"--- SYNC ERROR: {e} ---")
        SYNC_STATE["status"] = "error"
        SYNC_STATE["message"] = f"Error: {str(e)}"


def _collect_admin_usage_insights():
    user_rows = []
    module_totals = {name: 0 for name in TRACKED_MODULE_ENDPOINTS.values()}
    try:
        rows = (
            SystemLog.query.with_entities(SystemLog.timestamp, SystemLog.endpoint, SystemLog.user_email)
            .filter(SystemLog.endpoint.in_(tuple(TRACKED_MODULE_ENDPOINTS.keys())))
            .filter(SystemLog.user_email.isnot(None), SystemLog.user_email != "")
            .order_by(SystemLog.timestamp.asc())
            .all()
        )

        per_user = {}
        for row in rows:
            email = (row.user_email or "").strip().lower()
            endpoint = row.endpoint
            module_label = TRACKED_MODULE_ENDPOINTS.get(endpoint, endpoint)
            module_totals[module_label] = module_totals.get(module_label, 0) + 1
            event_time = row.timestamp
            if event_time is None:
                continue

            if email not in per_user:
                per_user[email] = {
                    "email": email,
                    "total_requests": 0,
                    "estimated_minutes": 0.0,
                    "last_seen": _format_dt_local(event_time),
                    "module_counts": {},
                    "_last_event_dt": event_time,
                }

            user_entry = per_user[email]
            user_entry["total_requests"] += 1
            user_entry["last_seen"] = _format_dt_local(event_time)
            user_entry["module_counts"][module_label] = user_entry["module_counts"].get(module_label, 0) + 1

            delta_seconds = (event_time - user_entry["_last_event_dt"]).total_seconds()
            if 0 < delta_seconds <= 600:
                user_entry["estimated_minutes"] += delta_seconds / 60.0
            user_entry["_last_event_dt"] = event_time

        for data in per_user.values():
            top_module = max(data["module_counts"], key=data["module_counts"].get) if data["module_counts"] else "-"
            user_rows.append(
                {
                    "email": data["email"],
                    "total_requests": data["total_requests"],
                    "estimated_minutes": round(data["estimated_minutes"], 1),
                    "top_module": top_module,
                    "last_seen": data["last_seen"],
                }
            )

        user_rows.sort(key=lambda row: row["total_requests"], reverse=True)
        user_rows = user_rows[:15]
    except Exception as e:
        print(f"Admin insights error: {e}")

    module_rows = [{"module": module, "count": count} for module, count in module_totals.items()]
    return {"users": user_rows, "modules": module_rows}



# ==========================================
# REACT SPA — additive JSON API blueprint
# ==========================================
# Registers /api/* endpoints that wrap the same brain/auth logic used above.
# Injects this module so api.py can reach the models/helpers without a circular
# import or re-executing app.py as a second module.
import sys
import api as iris_api
iris_api.init_api(sys.modules[__name__])
app.register_blueprint(iris_api.api_bp)


# Serve the Vite build's hashed assets (JS/CSS) when the SPA is enabled.
@app.route("/assets/<path:filename>")
def spa_assets(filename):
    return send_from_directory(os.path.join(FRONTEND_DIST, "assets"), filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8080)
