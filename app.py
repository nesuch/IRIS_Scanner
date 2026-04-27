from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, session
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
from werkzeug.exceptions import HTTPException
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from sqlalchemy import inspect, text
import iris_brain as brain

app = Flask(__name__)
app.secret_key = os.getenv("IRIS_SESSION_SECRET", "iris-dev-session-secret")
DB_NAME = os.path.abspath(os.getenv("IRIS_DB_PATH", "iris.db"))
os.makedirs(os.path.dirname(DB_NAME), exist_ok=True)
IRIS_AUTH_DATABASE_URL = (os.getenv("IRIS_AUTH_DATABASE_URL") or "").strip()
DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
AUTH_DATABASE_URL = IRIS_AUTH_DATABASE_URL or DATABASE_URL
running_in_cloud = any(os.getenv(flag) for flag in ("K_SERVICE", "GAE_ENV", "GOOGLE_CLOUD_PROJECT"))

if AUTH_DATABASE_URL:
    if AUTH_DATABASE_URL.startswith("postgres://"):
        AUTH_DATABASE_URL = AUTH_DATABASE_URL.replace("postgres://", "postgresql://", 1)
    app.config["SQLALCHEMY_DATABASE_URI"] = AUTH_DATABASE_URL
elif running_in_cloud:
    raise RuntimeError(
        "Persistent auth database is required in cloud deployments. "
        "Set IRIS_AUTH_DATABASE_URL (preferred) or DATABASE_URL."
    )
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_NAME}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = "Please sign in to continue."

ALLOWED_EMAIL_DOMAIN = "@irdai.gov.in"


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
TRACKED_MODULE_ENDPOINTS = {
    "/": "Universal Search",
    "/health": "Health Dept",
    "/life": "Life Dept",
    "/data": "Data Explorer",
    "/compliance": "Compliance Cockpit",
    "/admin": "Admin Panel",
}


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

def _safe_generate_password_hash(password: str) -> str:
    """
    Use PBKDF2 explicitly so password hashing works on environments where
    hashlib.scrypt is unavailable.
    """
    return generate_password_hash(password, method="pbkdf2:sha256")

def _safe_check_password_hash(stored_hash: str, raw_password: str) -> bool:
    try:
        return bool(stored_hash) and check_password_hash(stored_hash, raw_password)
    except Exception:
        return False

def _ensure_schema_compatibility():
    """
    Backfill columns that may be missing in existing deployments with older DB schema.
    """
    try:
        inspector = inspect(db.engine)
        if "system_logs" not in inspector.get_table_names():
            return

        existing_cols = {c["name"] for c in inspector.get_columns("system_logs")}
        if "user_email" not in existing_cols:
            db.session.execute(text("ALTER TABLE system_logs ADD COLUMN user_email VARCHAR(255)"))
            db.session.commit()
            app.logger.info("Schema patch applied: added system_logs.user_email")
    except Exception as e:
        db.session.rollback()
        app.logger.warning("Schema compatibility patch skipped/failed: %s", e)


def _seed_admin_user():
    admin_email = (os.getenv("ADMIN_EMAIL") or f"admin{ALLOWED_EMAIL_DOMAIN}").strip().lower()
    admin_password = (os.getenv("ADMIN_PASSWORD") or "admin12345").strip()
    if not _is_allowed_email(admin_email):
        admin_email = f"admin{ALLOWED_EMAIL_DOMAIN}"

    existing = User.query.filter_by(email=admin_email).first()
    if existing:
        existing.is_admin = True
        existing.is_active = True
        # Keep default/dev admin login usable even if legacy hashes exist
        # or hashing algorithms changed across releases.
        if (not existing.password_hash) or (not _safe_check_password_hash(existing.password_hash, admin_password)):
            existing.password_hash = _safe_generate_password_hash(admin_password)
    else:
        db.session.add(
            User(
                email=admin_email,
                password_hash=_safe_generate_password_hash(admin_password),
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
    session["auth_token"] = session_token
    session["session_version"] = user.session_version


def _is_session_active(user_id: int, session_token: str) -> bool:
    if not session_token:
        return False
    return (
        UserSession.query.filter_by(
            user_id=user_id,
            session_token=session_token,
            active=True,
        ).first()
        is not None
    )


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
            "created_at": row.created_at.strftime("%Y-%m-%d %H:%M:%S") if row.created_at else None,
            "last_seen_at": row.last_seen_at.strftime("%Y-%m-%d %H:%M:%S") if row.last_seen_at else None,
        }
        for row in rows
    ]


def _deactivate_all_user_sessions(user_id: int):
    UserSession.query.filter_by(user_id=user_id).update(
        {"active": False, "last_seen_at": datetime.utcnow()}
    )
    db.session.commit()


def _get_active_device_count(user_id: int) -> int:
    count = UserSession.query.filter_by(user_id=user_id, active=True).count()
    return int(count or 0)


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


@login_manager.user_loader
def load_user(user_id: str):
    if not user_id.isdigit():
        return None
    return db.session.get(User, int(user_id))


with app.app_context():
    db.create_all()
    _ensure_schema_compatibility()
    _seed_admin_user()


@app.context_processor
def inject_device_count():
    if current_user.is_authenticated:
        return {"current_device_count": _get_active_device_count(current_user.id)}
    return {"current_device_count": 0}


@app.before_request
def iris_auth_gatekeeper():
    if request.path.startswith("/static") or request.path == "/favicon.ico":
        return None

    if request.path in PUBLIC_AUTH_PATHS or request.path.startswith("/reset-password/"):
        return None

    if not _route_exists(request.path):
        return None

    if not current_user.is_authenticated:
        return redirect(url_for("login", next=request.path))

    if request.path in ADMIN_ONLY_PATHS and not current_user.is_admin:
        if request.path == "/admin":
            return redirect(url_for("index"))
        if request.path.startswith("/admin"):
            return jsonify({"message": "Admin access required"}), 403
        return "<h3>Forbidden</h3><p>Admin access required.</p>", 403

    expected_version = getattr(current_user, "session_version", 0)
    current_version = session.get("session_version", 0)
    session_token = session.get("auth_token")
    if current_version != expected_version or not _is_session_active(current_user.id, session_token):
        logout_user()
        session.clear()
        return redirect(url_for("login", next=request.path))

    return None

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

# --- CLEAR LOGS ROUTE ---
@app.route("/clear_logs", methods=["POST"])
def clear_logs():
    """
    Wipes ONLY crashes and favicon noise from the database.
    Keeps legitimate user traffic stats intact.
    """
    try:
        SystemLog.query.filter(
            (SystemLog.status >= 500) | (SystemLog.endpoint == "/favicon.ico")
        ).delete(synchronize_session=False)
        db.session.commit()
            
    except Exception as e:
        print(f"Error clearing logs: {e}")
        
    return redirect(url_for('analytics_dashboard'))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    next_url = _safe_next_url(request.args.get("next") or request.form.get("next") or "/")

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        app.logger.info("Login attempt for %s", email or "<blank>")

        user = User.query.filter(db.func.lower(User.email) == email).first()
        invalid = "Invalid credentials."

        if not _is_allowed_email(email):
            error = invalid
            _record_admin_audit(email, "login_attempt", "failure")
        elif user and user.is_active and _safe_check_password_hash(user.password_hash, password):
            login_user(user)
            _start_user_session(user)
            app.logger.info("Login success for %s", email)
            _record_admin_audit(email, "login_attempt", "success")
            return redirect(_safe_next_url(next_url))
        else:
            # Recovery path: allow configured admin credentials even when legacy
            # hash formats can't be verified in the current Python runtime.
            admin_email = (os.getenv("ADMIN_EMAIL") or f"admin{ALLOWED_EMAIL_DOMAIN}").strip().lower()
            admin_password = (os.getenv("ADMIN_PASSWORD") or "admin12345").strip()
            if email == admin_email and password == admin_password:
                admin_user = user or User.query.filter(db.func.lower(User.email) == admin_email).first()
                if not admin_user:
                    admin_user = User(
                        email=admin_email,
                        password_hash=_safe_generate_password_hash(admin_password),
                        is_admin=True,
                        is_active=True,
                        created_at=datetime.utcnow(),
                    )
                    db.session.add(admin_user)
                else:
                    admin_user.is_admin = True
                    admin_user.is_active = True
                    admin_user.password_hash = _safe_generate_password_hash(admin_password)
                db.session.commit()
                login_user(admin_user)
                _start_user_session(admin_user)
                app.logger.info("Admin login recovery success for %s", email)
                _record_admin_audit(email, "login_attempt", "success")
                return redirect(_safe_next_url(next_url))
            app.logger.warning("Login failed for %s", email or "<blank>")
            _record_admin_audit(email, "login_attempt", "failure")
            error = invalid

    return render_template("login.html", error=error, next_url=next_url)


@app.route("/register", methods=["GET", "POST"])
def register():
    return "<h3>Signup disabled</h3><p>Please contact an administrator.</p>", 403


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    error = None
    success = None

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        app.logger.info("Password reset request for %s", email or "<blank>")
        generic_msg = "If the account is eligible, a password reset link has been generated."

        if not email or not _is_allowed_email(email):
            success = generic_msg
            app.logger.warning("Password reset rejected for invalid domain: %s", email or "<blank>")
        else:
            user = User.query.filter(db.func.lower(User.email) == email).first()
            if user and user.is_active:
                raw_token = secrets.token_urlsafe(32)
                user.reset_token = _hash_reset_token(raw_token)
                user.reset_token_expiry = datetime.utcnow() + timedelta(minutes=15)
                db.session.commit()
                reset_link = url_for("reset_password", token=raw_token, _external=True)
                app.logger.info("Password reset link generated for %s: %s", email, reset_link)
                try:
                    db.session.add(
                        PasswordResetAudit(
                            email=email,
                            reset_link=reset_link,
                            requested_at=datetime.utcnow(),
                            expires_at=user.reset_token_expiry,
                        )
                    )
                    db.session.commit()
                except Exception as e:
                    app.logger.warning("Unable to audit password reset request: %s", e)
                _record_admin_audit(email, "password_reset_request", "success")
            else:
                app.logger.warning("Password reset request not fulfilled for %s", email)
                _record_admin_audit(email, "password_reset_request", "failure")
            success = generic_msg

    return render_template("forgot_password.html", error=error, success=success)


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    token_hash = _hash_reset_token(token)
    now_utc = datetime.utcnow()
    user = User.query.filter_by(reset_token=token_hash).first()
    valid_token = (
        user
        and user.is_active
        and user.reset_token_expiry
        and user.reset_token_expiry >= now_utc
    )

    if not valid_token:
        app.logger.warning("Invalid or expired reset token used.")
        return render_template("reset_password.html", error="Invalid or expired reset link.", success=None), 400

    if request.method == "POST":
        new_password = request.form.get("new_password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if len(new_password) < 8:
            return render_template("reset_password.html", error="Password must be at least 8 characters.", success=None), 400
        if new_password != confirm_password:
            return render_template("reset_password.html", error="Passwords do not match.", success=None), 400

        user.password_hash = _safe_generate_password_hash(new_password)
        user.reset_token = None
        user.reset_token_expiry = None
        db.session.commit()
        app.logger.info("Password reset completed for user id %s", user.id)
        _record_admin_audit(user.email, "password_reset_complete", "success")
        return redirect(url_for("login"))

    return render_template("reset_password.html", error=None, success=None)


@app.route("/create-user", methods=["POST"])
@login_required
def create_user():
    if not current_user.is_admin:
        return jsonify({"message": "Admin access required"}), 403

    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    is_admin = (request.form.get("is_admin") or "").lower() in {"1", "true", "yes", "on"}

    if not _is_allowed_email(email):
        return jsonify({"message": "Email must use @irdai.gov.in domain."}), 400
    if len(password) < 8:
        return jsonify({"message": "Password must be at least 8 characters."}), 400
    if User.query.filter(db.func.lower(User.email) == email).first():
        return jsonify({"message": "User already exists."}), 409

    new_user = User(
        email=email,
        password_hash=_safe_generate_password_hash(password),
        is_active=True,
        is_admin=is_admin,
    )
    db.session.add(new_user)
    db.session.commit()
    app.logger.info("Admin %s created user %s", current_user.email, email)
    _record_admin_audit(email, "user_create", "success")
    return jsonify({"message": "User created successfully."}), 201


@app.route("/logout", methods=["POST", "GET"])
def logout():
    if request.method == "GET":
        if not current_user.is_authenticated:
            return redirect(url_for("login"))
        return render_template("logout_confirm.html")
    if current_user.is_authenticated:
        _deactivate_session_token(current_user.id, session.get("auth_token"))
    logout_user()
    session.clear()
    return redirect(url_for("login"))


@app.route("/logout-all", methods=["POST"])
@login_required
def logout_all():
    user = db.session.get(User, current_user.id)
    user.session_version = (user.session_version or 0) + 1
    db.session.commit()
    _deactivate_all_user_sessions(user.id)
    _record_admin_audit(user.email, "logout_all_devices", "success")
    logout_user()
    session.clear()
    return redirect(url_for("login"))


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    error = None
    success = None
    if request.method == "POST":
        current_password = request.form.get("current_password") or ""
        new_password = request.form.get("new_password") or ""
        confirm_password = request.form.get("confirm_password") or ""
        if not check_password_hash(current_user.password_hash, current_password):
            error = "Current password is incorrect."
        elif len(new_password) < 8:
            error = "New password must be at least 8 characters."
        elif new_password != confirm_password:
            error = "New password and confirm password do not match."
        else:
            user = db.session.get(User, current_user.id)
            user.password_hash = _safe_generate_password_hash(new_password)
            user.session_version = (user.session_version or 0) + 1
            db.session.commit()
            _deactivate_all_user_sessions(user.id)
            _record_admin_audit(user.email, "password_change", "success")
            success = "Password changed. Please login again."
            logout_user()
            session.clear()
            return redirect(url_for("login"))
    sessions = _list_user_sessions(current_user.id)
    return render_template("profile.html", sessions=sessions, error=error, success=success)


@app.route("/profile/session/<int:session_id>/logout", methods=["POST"])
@login_required
def profile_logout_session(session_id):
    _deactivate_session_by_id(current_user.id, session_id)
    if session_id and session.get("auth_token"):
        # if current session row is disabled by user, enforce immediate logout
        if not _is_session_active(current_user.id, session.get("auth_token")):
            logout_user()
            session.clear()
            return redirect(url_for("login"))
    return redirect(url_for("profile"))

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
                    "last_seen": event_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "module_counts": {},
                    "_last_event_dt": event_time,
                }

            user_entry = per_user[email]
            user_entry["total_requests"] += 1
            user_entry["last_seen"] = event_time.strftime("%Y-%m-%d %H:%M:%S")
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
# ADMIN ROUTES (SYNC CONTROL)
# ==========================================

@app.route("/admin", methods=["GET"])
@login_required
def admin_panel():
    if not current_user.is_admin:
        return "<h3>Forbidden</h3><p>Admin access required.</p>", 403
    usage_insights = _collect_admin_usage_insights()
    reset_audit = []
    try:
        reset_audit_rows = (
            PasswordResetAudit.query.with_entities(
                PasswordResetAudit.email,
                PasswordResetAudit.reset_link,
                PasswordResetAudit.requested_at,
                PasswordResetAudit.expires_at,
            )
            .order_by(PasswordResetAudit.id.desc())
            .limit(20)
            .all()
        )
        reset_audit = [
            {
                "email": row.email,
                "reset_link": row.reset_link,
                "requested_at": row.requested_at.strftime("%Y-%m-%d %H:%M:%S") if row.requested_at else None,
                "expires_at": row.expires_at.strftime("%Y-%m-%d %H:%M:%S") if row.expires_at else None,
            }
            for row in reset_audit_rows
        ]
    except Exception as e:
        app.logger.warning("Unable to load password reset audit entries: %s", e)
    users = User.query.order_by(User.created_at.desc()).all()
    user_device_counts = {u.id: _get_active_device_count(u.id) for u in users}
    audit_logs = []
    try:
        audit_log_rows = (
            AdminAuditLog.query.with_entities(
                AdminAuditLog.email,
                AdminAuditLog.action_type,
                AdminAuditLog.status,
                AdminAuditLog.timestamp,
            )
            .order_by(AdminAuditLog.id.desc())
            .limit(100)
            .all()
        )
        audit_logs = [
            {
                "email": row.email,
                "action_type": row.action_type,
                "status": row.status,
                "timestamp": row.timestamp.strftime("%Y-%m-%d %H:%M:%S") if row.timestamp else None,
            }
            for row in audit_log_rows
        ]
    except Exception as e:
        app.logger.warning("Unable to load admin audit logs: %s", e)
    return render_template("admin.html", sync_state=SYNC_STATE, usage_insights=usage_insights, reset_audit=reset_audit, users=users, audit_logs=audit_logs, user_device_counts=user_device_counts)


@app.route("/admin/user/<int:user_id>/toggle-active", methods=["POST"])
@login_required
def admin_toggle_user_active(user_id):
    if not current_user.is_admin:
        return jsonify({"message": "Admin access required"}), 403
    user = User.query.get_or_404(user_id)
    user.is_active = not user.is_active
    db.session.commit()
    _record_admin_audit(user.email, "user_activate" if user.is_active else "user_deactivate", "success")
    return redirect(url_for("admin_panel"))


@app.route("/admin/user/<int:user_id>/delete", methods=["POST"])
@login_required
def admin_delete_user(user_id):
    if not current_user.is_admin:
        return jsonify({"message": "Admin access required"}), 403
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        return redirect(url_for("admin_panel"))
    _deactivate_all_user_sessions(user.id)
    _record_admin_audit(user.email, "user_delete", "success")
    db.session.delete(user)
    db.session.commit()
    return redirect(url_for("admin_panel"))


@app.route("/admin/user/<int:user_id>/logout-all", methods=["POST"])
@login_required
def admin_logout_user_all_devices(user_id):
    if not current_user.is_admin:
        return jsonify({"message": "Admin access required"}), 403
    user = User.query.get_or_404(user_id)
    user.session_version = (user.session_version or 0) + 1
    db.session.commit()
    _deactivate_all_user_sessions(user.id)
    _record_admin_audit(user.email, "logout_all_devices", "success")
    return redirect(url_for("admin_panel"))


@app.route("/admin/user/<int:user_id>/trigger-reset", methods=["POST"])
@login_required
def admin_trigger_user_reset(user_id):
    if not current_user.is_admin:
        return jsonify({"message": "Admin access required"}), 403
    user = User.query.get_or_404(user_id)
    raw_token = secrets.token_urlsafe(32)
    user.reset_token = _hash_reset_token(raw_token)
    user.reset_token_expiry = datetime.utcnow() + timedelta(minutes=15)
    db.session.commit()
    reset_link = url_for("reset_password", token=raw_token, _external=True)
    try:
        db.session.add(
            PasswordResetAudit(
                email=user.email.lower(),
                reset_link=reset_link,
                requested_at=datetime.utcnow(),
                expires_at=user.reset_token_expiry,
            )
        )
        db.session.commit()
    except Exception as e:
        app.logger.warning("Unable to write admin-triggered reset audit: %s", e)
    _record_admin_audit(user.email, "password_reset_request", "success")
    return redirect(url_for("admin_panel"))


@app.route("/admin/clear-reset-audit", methods=["POST"])
@login_required
def admin_clear_reset_audit():
    if not current_user.is_admin:
        return jsonify({"message": "Admin access required"}), 403
    PasswordResetAudit.query.delete()
    db.session.commit()
    _record_admin_audit(current_user.email, "clear_reset_audit", "success")
    return redirect(url_for("admin_panel"))


@app.route("/admin/clear-audit-logs", methods=["POST"])
@login_required
def admin_clear_audit_logs():
    if not current_user.is_admin:
        return jsonify({"message": "Admin access required"}), 403
    AdminAuditLog.query.delete()
    db.session.commit()
    _record_admin_audit(current_user.email, "clear_admin_audit_logs", "success")
    return redirect(url_for("admin_panel"))

@app.route("/admin/sync_start", methods=["POST"])
@login_required
def sync_start():
    """Kicks off the background sync thread."""
    global SYNC_STATE
    
    if SYNC_STATE["status"] == "running":
        return jsonify({"status": "error", "message": "Sync already in progress."})

    # Reset State & Start
    SYNC_STATE["status"] = "starting"
    SYNC_STATE["message"] = "Initializing background process..."
    
    thread = threading.Thread(target=run_background_sync)
    thread.daemon = True # Ensures thread dies if app restarts
    thread.start()
    
    return jsonify({"status": "started"})

@app.route("/admin/sync_status", methods=["GET"])
@login_required
def sync_status():
    """Frontend polls this to update progress bars."""
    return jsonify(SYNC_STATE)


# ==========================================
# ROUTES (MODULES)
# ==========================================

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    return handle_search("universal")

@app.route("/health", methods=["GET", "POST"])
@login_required
def health_module():
    return handle_search("health")

@app.route("/life", methods=["GET", "POST"])
@login_required
def life_module():
    return handle_search("life")

# --- DATA MODULE (UPDATED) ---
@app.route("/data", methods=["GET", "POST"])
@login_required
def data_module():
    filter_options = brain.get_filter_options()
    
    if request.method == "POST":
        # Capture all filters from the form
        filters = {
            "dimension": request.form.get("dimension", "Insurer"),
            "entities": request.form.getlist("entities"),
            "metrics": request.form.getlist("metrics"),
            "years": request.form.getlist("years"),
            "quarters": request.form.getlist("quarters"),
            "lobs": request.form.getlist("lobs"),
            "classes": request.form.getlist("classes")
        }
        
        # Process filters through the brain
        report_data = brain.filter_data(filters)
        
        # Return only the table fragment (HTMX-style update)
        return render_template("components/data_table.html", report=report_data)
        
    # GET Request: Render the full dashboard
    return render_template("data_dashboard.html", options=filter_options, active_module="data")

@app.route("/download_data", methods=["POST"])
@login_required
def download_data():
    """Generates and downloads the Excel report based on active filters."""
    filters = {
        "dimension": request.form.get("dimension", "Insurer"),
        "entities": request.form.getlist("entities"),
        "metrics": request.form.getlist("metrics"),
        "years": request.form.getlist("years"),
        "quarters": request.form.getlist("quarters"),
        "lobs": request.form.getlist("lobs"),
        "classes": request.form.getlist("classes")
    }
    
    excel_file = brain.generate_excel(filters)
    
    if excel_file:
        return send_file(
            excel_file,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='IRIS_Financial_Report.xlsx'
        )
    return "No data found for these filters.", 400

# --- COMPLIANCE ROUTE (UPDATED FOR YEAR FILTER) ---
@app.route("/compliance", methods=["GET"])
@login_required
def compliance_dashboard():
    # 1. Get available years for the dropdown
    available_years = brain.get_compliance_years()
    
    # 2. Get selected year from URL (e.g. ?year=2023-24)
    selected_year = request.args.get("year")
    
    # 3. Get Dashboard Data based on selection
    compliance_data = brain.get_compliance_dashboard(target_year=selected_year)
    
    return render_template("compliance.html", 
                           companies=compliance_data, 
                           years=available_years,
                           active_year=selected_year if selected_year else "Latest",
                           active_module="compliance")

# --- ANALYTICS ROUTE (SQL INTEGRATED) ---
@app.route("/analytics")
@login_required
def analytics_dashboard():
    logs = []
    try:
        rows = SystemLog.query.order_by(SystemLog.id.desc()).all()
        logs = [
            {
                "id": row.id,
                "timestamp": row.timestamp.strftime("%Y-%m-%d %H:%M:%S") if row.timestamp else None,
                "endpoint": row.endpoint,
                "method": row.method,
                "ip": row.ip,
                "status": row.status,
                "error_msg": row.error_msg,
                "user_email": row.user_email,
            }
            for row in rows
        ]
    except Exception as e:
        print(f"DB Error in Analytics: {e}")
        logs = []

    # --- 1. PRE-PROCESS LOGS ---
    module_logs = [l for l in logs if l.get("endpoint") in TRACKED_MODULE_ENDPOINTS]
    valid_logs = [l for l in logs if l.get("endpoint") != "/favicon.ico"]

    # --- 2. CALCULATE BASIC STATS ---
    total_requests = len(module_logs)
    unique_users = db.session.query(db.func.count(db.func.distinct(db.func.lower(User.email)))).scalar() or 0
    errors = [l for l in valid_logs if l['status'] >= 500]
    error_count = len(errors)
    can_view_crash_details = bool(getattr(current_user, "is_admin", False))
    visible_errors = errors if can_view_crash_details else []
    
    # --- 3. CALCULATE ENDPOINT USAGE (PIE CHART) ---
    endpoints = {}

    for l in module_logs:
        ep = l['endpoint']
        label = TRACKED_MODULE_ENDPOINTS.get(ep, ep.lstrip("/").replace("_", " ").title())
        endpoints[label] = endpoints.get(label, 0) + 1
        
    chart_labels = list(endpoints.keys())
    chart_data = list(endpoints.values())

    # --- 4. CALCULATE MONTHLY & YEARLY TRENDS ---
    current_year = datetime.now().year
    
    # Initialize the buckets for Jan-Dec
    monthly_labels = [datetime(current_year, m, 1).strftime('%b %Y') for m in range(1, 13)]
    monthly_data = [0] * 12
    yearly_stats = []

    try:
        if module_logs:
            df = pd.DataFrame(module_logs)
            df['dt'] = pd.to_datetime(df['timestamp'])
            
            # -- MONTHLY LOGIC --
            # Filter for CURRENT YEAR ONLY
            df_this_year = df[df['dt'].dt.year == current_year]
            
            # Group by Month Number (1-12)
            counts = df_this_year['dt'].dt.month.value_counts()
            
            # Populate array (index 0 is Jan)
            for month_num, count in counts.items():
                if 1 <= month_num <= 12:
                    monthly_data[month_num - 1] = int(count)

            # -- YEARLY LOGIC --
            yearly_counts = df['dt'].dt.year.value_counts().sort_index()
            yearly_stats = [{"year": y, "count": c} for y, c in yearly_counts.items()]

    except Exception as e:
        print(f"Analytics Data Processing Error: {e}")

    # Render template with all processed data
    return render_template("analytics.html", 
                           stats={"total": total_requests, "users": unique_users, "errors": error_count},
                           logs=visible_errors,
                           can_view_crash_details=can_view_crash_details,
                           chart={"labels": chart_labels, "data": chart_data},
                           monthly={"labels": monthly_labels, "data": monthly_data},
                           yearly=yearly_stats,
                           active_module="analytics")

# ==========================================
# CORE SEARCH LOGIC (TEXT ONLY)
# ==========================================

def handle_search(active_module):
    """
    Central handler for Universal, Health, and Life search logic.
    Manages Keywords, Deep Scans, and Rendering Results.
    """
    global CHAT_HISTORY, JUST_REDIRECTED

    # Brain now handles SQL retrieval internally
    KB_DF = brain.load_knowledge_base()
    vocab = brain.get_autocomplete_data()

    # GET Request: Just show the search page (with history if applicable)
    if request.method == "GET":
        if not JUST_REDIRECTED: CHAT_HISTORY = []
        JUST_REDIRECTED = False
        return render_template("index.html", history=CHAT_HISTORY, vocab=vocab, active_module=active_module)

    # POST Request: Process the query
    query = request.form.get("query", "").strip()
    
    if not query: 
        if active_module == "universal": return redirect(url_for("index"))
        return redirect(url_for(f"{active_module}_module"))

    # --- 1. DEEP SCAN HANDLER (Triggered by Chips) ---
    if query.startswith("__DEEP_SCAN__:"):
        raw_payload = query.replace("__DEEP_SCAN__:", "")
        pairs = raw_payload.split("||")
        # Reconstruct the tuple list: [('hospital', 'hospit'), ('cashless', 'cashless')]
        keyword_tuples = [(p.split("|")[0], p.split("|")[1]) for p in pairs if len(p.split("|"))==2]
        
        display_kws = [t[0] for t in keyword_tuples]
        
        # First, check tags again to get context IDs
        tag_matches = brain.search_tags_only(keyword_tuples, KB_DF, module=active_module)
        exclude_ids = [m['id'] for m in tag_matches]
        
        # Then, perform the expensive full-text scan
        matches = brain.deep_scan_brain(keyword_tuples, KB_DF, exclude_ids=exclude_ids, module=active_module)
        
        response = ""
        if matches:
            response = f"<div class='analysis-text'>Deep Scan results in <strong>{active_module.upper()}</strong>: <strong>{', '.join(display_kws)}</strong></div>"
            response += build_results_html(matches, display_kws)
        else:
            response = f"No additional matches found in <strong>{active_module.capitalize()}</strong> module."
            
        CHAT_HISTORY.append({"query": "Deep Scan", "response": response})
        JUST_REDIRECTED = True
        
        if active_module == "universal": return redirect(url_for("index"))
        return redirect(url_for(f"{active_module}_module"))

    # --- 2. GREETING CHECK ---
    if brain.check_greeting(query):
        response = "<strong>Hello!</strong> I am <strong>IRIS</strong>. Ask me anything related to IRDAI Acts, Regulations, Circulars, or Guidelines."
    
    else:
        # --- 3. STANDARD SEARCH ---
        # Step A: NLP Processing (Cleaning & Stemming)
        kw_tuples = brain.get_clean_keywords(query)
        display_kws = [t[0] for t in kw_tuples]
        
        if not kw_tuples:
            response = "Query rejected. Please use regulatory terms."
        else:
            # Step B: Tag-Based Search (Fast & Precise)
            tag_matches = brain.search_tags_only(kw_tuples, KB_DF, module=active_module)
            highlight_kws = [raw for (raw, clean) in kw_tuples if clean in brain.ALL_UNIQUE_TAGS]

            if tag_matches:
                response = f"<div style='font-size:12px; color:#888; margin-bottom:10px;'>Found via <strong>Tags</strong>: {', '.join(display_kws)}</div>"
                response += build_results_html(tag_matches, highlight_kws)
            else:
                if active_module == "life":
                    response = "<div><strong>Life Department:</strong> No documents currently loaded.</div>"
                elif active_module == "data":
                    response = "<div><strong>Data Module:</strong> Text search is disabled here.</div>"
                else:
                    response = f"<div>No matches found in {active_module.capitalize()} for: <strong>{', '.join(display_kws)}</strong></div>"

            # Step C: Offer Deep Scan Chips
            response += build_chips_html(kw_tuples, query)

    CHAT_HISTORY.append({"query": query, "response": response})
    JUST_REDIRECTED = True
    
    if active_module == "universal": return redirect(url_for("index"))
    return redirect(url_for(f"{active_module}_module"))

# =========================================================
# UTILS (Rendering Helpers)
# =========================================================

def build_chips_html(kw_tuples, original_query):
    """
    Generates the 'Deep Scan' buttons.
    FIX: Removes redundant 'Phrase Search' if it matches a single keyword.
    """
    if not kw_tuples: return ""

    html = """<hr><div style="font-size:12px; color:#666; margin-bottom:8px;">
            Not finding what you need? <strong>Deep Scan specific terms:</strong></div>
            <div style="display: flex; flex-wrap: wrap; gap: 6px;">"""
    
    # Track what we have shown to avoid duplicates
    shown_labels = set()

    # 1. Chips for individual keywords
    for raw, clean in kw_tuples:
        label = f'Search "{raw}"'
        if label in shown_labels: continue
        
        payload = f"{raw}|{clean}"
        html += f"""<form method="POST" style="margin:0;" onsubmit="return showDeepScanLoading(event, this)"><input type="hidden" name="query" value="__DEEP_SCAN__:{payload}">
                <button type="submit" style="background:#e8eaf6; border:1px solid #3f51b5; color:#1a237e; padding:6px 12px; border-radius:16px; font-size:11px; cursor:pointer;">
                {label}</button></form>"""
        shown_labels.add(label)
    
    # 2. Chip for the Exact Phrase (ONLY if multi-word AND not already shown)
    clean_original = " ".join(original_query.split()).strip()
    phrase_label = f'Search Phrase "{clean_original}"'
    
    has_special_compound = bool(re.search(r"[-/]", clean_original))
    if (len(clean_original.split()) > 1 or has_special_compound) and phrase_label not in shown_labels:
        html += f"""<form method="POST" style="margin:0;" onsubmit="return showDeepScanLoading(event, this)"><input type="hidden" name="query" value="__DEEP_SCAN__:{clean_original}|{clean_original}">
                <button type="submit" style="background:#e3f2fd; border:1px solid #2196f3; color:#0d47a1; padding:6px 12px; border-radius:16px; font-weight:700; font-size:11px; cursor:pointer;">
                {phrase_label}</button></form>"""
    
    # 3. Chip for 'Search All' (combined) - Only if we have multiple distinct keywords
    if len(kw_tuples) > 1:
        all_payload = "||".join([f"{t[0]}|{t[1]}" for t in kw_tuples])
        html += f"""<form method="POST" style="margin:0;" onsubmit="return showDeepScanLoading(event, this)"><input type="hidden" name="query" value="__DEEP_SCAN__:{all_payload}">
                <button type="submit" style="background:#fff; border:1px solid #999; color:#666; padding:6px 12px; border-radius:16px; font-size:11px; cursor:pointer;">
                Search All</button></form>"""
                
    html += "</div>"
    return html

def highlight_keywords(text, keywords):
    """Wraps found keywords in a highlighting span."""
    if not keywords: return text
    expanded = set(keywords)
    # Simple stemming for highlight matching (e.g., 'insurer' -> 'insurers')
    for k in keywords:
        k = k.lower()
        expanded.add(k + "s"); expanded.add(k + "ed"); expanded.add(k + "ing")
    
    for kw in sorted(list(expanded), key=len, reverse=True):
        if len(kw) < 3: continue
        pattern = re.compile(rf"\b({re.escape(kw)})\b", re.IGNORECASE)
        text = pattern.sub(r"<span class='iris-highlight'>\1</span>", text)
    return text

def convert_markdown_to_html(text):
    """Simple parser to handle basic Markdown tables and line breaks."""
    lines = text.splitlines()
    new_lines: List[str] = []
    in_table = False
    table_rows: List[str] = []
    
    def flush_table(rows):
        if not rows: return ""
        html = '<table border="1" style="border-collapse: collapse; width: 100%; border-color: #ddd; font-size: 11px; margin-top:5px; margin-bottom:5px;">'
        for i, row in enumerate(rows):
            # Skip separator lines like |---|---|
            if re.match(r"^\|[\s\-:\|]+\|$", row): continue
            tag = "th" if i == 0 else "td"
            bg = 'style="background-color: #f2f2f2; padding: 5px;"' if i == 0 else 'style="padding: 5px;"'
            cells = [c.strip() for c in row.strip("|").split("|")]
            html += "<tr>"
            for c in cells: html += f"<{tag} {bg}>{c}</{tag}>"
            html += "</tr>"
        html += "</table>"
        return html

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True; table_rows.append(stripped)
        else:
            if in_table: 
                new_lines.append(flush_table(table_rows))
                table_rows = []; in_table = False
            new_lines.append(line)
            
    if in_table: new_lines.append(flush_table(table_rows))
    return "\n".join(new_lines)

def format_verbatim(raw_text, keywords):
    """Formats raw text for display while preserving heading style without highlights."""
    if not raw_text: return ""
    text_with_tables = convert_markdown_to_html(raw_text)

    lines = text_with_tables.splitlines()
    formatted_lines: List[str] = []

    for line in lines:
        stripped = line.strip()

        # Keep table/html lines untouched
        if stripped.startswith("<table") or stripped.startswith("<tr") or stripped.startswith("<th") or stripped.startswith("<td") or stripped.startswith("</"):
            formatted_lines.append(line)
            continue

        # Headings keep original formatting (bold only), no keyword highlighting
        if stripped.endswith(":"):
            formatted_lines.append(f"<strong>{line}</strong>")
            continue

        # Regular lines get keyword highlighting
        formatted_lines.append(highlight_keywords(line, keywords))

    final_text = "\n".join(formatted_lines)
    return f'<div style="white-space: pre-wrap; font-family: inherit;">{final_text}</div>'

def build_results_html(matches, keywords):
    """
    Constructs the HTML card for each search result.
    FIX: Makes PDF lookup case-insensitive.
    """
    html = ""
    current_doc_type = None
    
    for i, m in enumerate(matches):
        doc_type = m['type']
        # Header for new document types
        if doc_type != current_doc_type:
            style = TYPE_STYLES.get(doc_type, TYPE_STYLES["UNKNOWN"])
            html += f"""<div style="margin-top:15px;margin-bottom:10px;background-color:{style['bg']};border-left:5px solid {style['color']};padding:8px 12px;font-family:'Segoe UI';color:{style['color']};font-weight:bold;font-size:13px;text-transform:uppercase;border-radius:4px;">{style['label']}</div>"""
            current_doc_type = doc_type
        else: 
            html += "<hr style='border: 0; border-top: 1px dashed #999; margin: 15px 0;'>"
            
        formatted_body = format_verbatim(m['raw_text'], keywords)
        
        # --- FIX: Case-Insensitive PDF Lookup ---
        doc_name_key = m['source'].strip().upper()
        pdf_path = resolve_pdf_path(doc_name_key)

        pdf_btn = ""
        if pdf_path:
            pdf_btn = f"""<a href='/static/{pdf_path}' target='_blank' 
                        style='float:right; margin-left:8px; font-size:10px; font-weight:bold; text-decoration:none; color:#d32f2f; background:#fff; padding:2px 6px; border:1px solid #d32f2f; border-radius:3px;'>
                        <i class="fas fa-file-pdf"></i> PDF</a>"""
        
        # Copy Button Logic
        safe_source = re.sub(r"[^a-zA-Z0-9_-]", "_", str(m.get('source', 'doc')))[:24]
        safe_clause = re.sub(r"[^a-zA-Z0-9_-]", "_", str(m.get('id', i)))[:24]
        content_id = f"clause_text_{safe_source}_{safe_clause}_{i}_{time.time_ns()}"
        copy_btn = f"""<button onclick="copyToClipboard('{content_id}')" title="Copy Clause" style="float:right; margin-right: 8px; background:none; border:none; color:#666; cursor:pointer; font-size:14px;"><i class="far fa-copy"></i></button>"""
        
        html += f"""<div style="margin-bottom:6px; font-size:11px; color:#555;"><span style="font-weight:800; color:#333;">{m['source']}</span> | <span style="color:#0056b3;">{m['header']}</span> | Clause: {m['id']} {pdf_btn} {copy_btn}</div><div id="{content_id}" style="line-height:1.5; color:#222; font-size:14px;">{formatted_body}</div>"""
    return html

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8080)
