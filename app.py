"""
Minimal secure authentication foundation for IRIS.

This single-file Flask app includes:
- Flask-Login setup
- SQLAlchemy User model
- secure login/logout
- protected home route
"""

import os
import secrets
import hashlib
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urljoin

from flask import Flask, request, redirect, url_for, render_template_string
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash

# ---------------------------------
# App and database configuration
# ---------------------------------
app = Flask(__name__)

# SECRET_KEY is required to cryptographically sign session cookies.
# Without this, session integrity cannot be trusted.
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///iris_auth.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ---------------------------------
# Flask-Login setup
# ---------------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page"
login_manager.session_protection = "strong"


# ---------------------------------
# User model compatible with Flask-Login
# ---------------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    reset_token = db.Column(db.String(255), nullable=True)
    reset_token_expiry = db.Column(db.DateTime, nullable=True)


@login_manager.user_loader
def load_user(user_id):
    """Load a user from SQLAlchemy by ID for Flask-Login session handling."""
    return db.session.get(User, int(user_id))


def is_safe_url(target: str) -> bool:
    """Prevent open redirects by only allowing same-host URLs."""
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ("http", "https") and ref_url.netloc == test_url.netloc


with app.app_context():
    # Ensure DB tables exist automatically at startup.
    db.create_all()
    # Backfill new reset columns for existing SQLite DBs created before this change.
    cols = [c[1] for c in db.session.execute(db.text("PRAGMA table_info(user)")).fetchall()]
    if "reset_token" not in cols:
        db.session.execute(db.text("ALTER TABLE user ADD COLUMN reset_token VARCHAR(255)"))
    if "reset_token_expiry" not in cols:
        db.session.execute(db.text("ALTER TABLE user ADD COLUMN reset_token_expiry DATETIME"))
    db.session.commit()

    # Seed one local admin-style test account for first run.
    # Email domain is restricted to @irdai.gov.in.
    if not User.query.filter_by(email="admin@irdai.gov.in").first():
        db.session.add(
            User(
                email="admin@irdai.gov.in",
                password_hash=generate_password_hash("Admin@12345"),
                is_active=True,
                is_admin=True,
            )
        )
        db.session.commit()


@app.route("/create-user", methods=["POST"])
@login_required
def create_user():
    """
    Admin-only user creation route.
    Security controls:
    - Requires authenticated admin user
    - Accepts email/password from POST form data only
    - Validates domain and non-empty password
    - Stores hashed password only
    """
    if not current_user.is_admin:
        return "Unauthorized", 403

    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    client_ip = request.remote_addr

    # Use WARNING for user-input/authorization issues (not system faults).
    if not email:
        app.logger.warning(f"{current_user.email} create-user validation failed (missing email) from {client_ip}")
        return "Request could not be completed.", 400
    if "@" not in email or not email.endswith("@irdai.gov.in"):
        app.logger.warning(f"{current_user.email} create-user validation failed (invalid email) from {client_ip}")
        return "Request could not be completed.", 400
    if email == current_user.email:
        app.logger.warning(f"{current_user.email} create-user blocked (self-creation) from {client_ip}")
        return "Request could not be completed.", 400
    if not password.strip():
        app.logger.warning(f"{current_user.email} create-user validation failed (missing password) from {client_ip}")
        return "Request could not be completed.", 400
    if len(password) < 6:
        app.logger.warning(f"{current_user.email} create-user validation failed (short password) from {client_ip}")
        return "Request could not be completed.", 400

    # Prevent duplicate users.
    existing = User.query.filter_by(email=email).first()
    if existing:
        app.logger.warning(f"{current_user.email} create-user blocked (duplicate) from {client_ip}")
        return "Request could not be completed.", 409

    try:
        db.session.add(
            User(
                email=email,
                password_hash=generate_password_hash(password),
                is_active=True,
                # Security decision: this endpoint can only create standard users.
                # Admin creation should use a separate, more tightly controlled flow.
                is_admin=False,
            )
        )
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        # Use ERROR only for server-side/system failures.
        app.logger.error(f"{current_user.email} create-user system error for {email} from {client_ip}: {exc}")
        return "Internal error.", 500

    app.logger.info(f"{current_user.email} created user {email} from {client_ip}")
    return "User created successfully.", 201


# ---------------------------------
# Protected home route
# ---------------------------------
@app.route("/")
@login_required
def home():
    return render_template_string(
        """
        <h2>IRIS Home</h2>
        <p>Welcome, {{ user.email }}</p>
        <p><a href="{{ url_for('logout') }}">Logout</a></p>
        """,
        user=current_user,
    )


# ---------------------------------
# Login route (GET + POST)
# ---------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    # Better UX: authenticated users should not see the login form again.
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "GET":
        return render_template_string(
            """
            <h2>Login</h2>
            <form method="post">
              <label>Email</label><br>
              <input type="email" name="email" required><br><br>

              <label>Password</label><br>
              <input type="password" name="password" required><br><br>

              <button type="submit">Login</button>
            </form>
            """
        )

    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    if not email.endswith("@irdai.gov.in"):
        return "Only @irdai.gov.in email addresses are allowed.", 403

    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        return "Invalid credentials.", 401

    if not user.is_active:
        return "Account is inactive.", 403

    # Keep users signed in across browser restarts for dev convenience.
    login_user(user, remember=True)
    next_page = request.args.get("next")
    if next_page and is_safe_url(next_page):
        return redirect(next_page)
    return redirect(url_for("home"))


# ---------------------------------
# Logout route
# ---------------------------------
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    """
    Security behavior:
    - Always return generic response, never reveal if user exists.
    - If user exists, create short-lived single-use reset token.
    - Print reset link to console (no email integration yet).
    """
    if request.method == "GET":
        return render_template_string(
            """
            <h2>Forgot Password</h2>
            <form method="post">
              <label>Email</label><br>
              <input type="email" name="email" required><br><br>
              <button type="submit">Request reset</button>
            </form>
            """
        )

    email = (request.form.get("email") or "").strip().lower()
    client_ip = request.remote_addr
    user = User.query.filter_by(email=email).first()

    app.logger.info(f"[RESET_REQUEST] email={email} ip={client_ip}")
    if user:
        token = secrets.token_urlsafe(32)
        # Store only a hashed reset token (never raw token) to reduce impact
        # if DB contents are exposed.
        hashed_token = hashlib.sha256(token.encode()).hexdigest()
        # Invalidate any previous token first so only one token is valid at a time.
        user.reset_token = None
        user.reset_token_expiry = None
        user.reset_token = hashed_token
        user.reset_token_expiry = datetime.now(timezone.utc) + timedelta(minutes=15)
        try:
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            app.logger.error(f"[RESET_ERROR] email={email} ip={client_ip} stage=token_save error={exc}")
            return "Internal error.", 500
        # Development-only behavior: print link locally.
        # In production, replace this with a proper email/SMS delivery service.
        print(f"http://127.0.0.1:5000/reset-password/{token}")

    return "If the account exists, a reset link has been generated.", 200


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    hashed_token = hashlib.sha256(token.encode()).hexdigest()
    user = User.query.filter_by(reset_token=hashed_token).first()
    current_time = datetime.now(timezone.utc)

    # Validate token existence and expiry before showing/accepting reset form.
    if not user or not user.reset_token_expiry or user.reset_token_expiry < current_time:
        # Log only a short token-hash prefix for traceability without exposing full token data.
        hashed_token_prefix = hashed_token[:8]
        app.logger.warning(
            f"[RESET_INVALID] token_prefix={hashed_token_prefix} ip={request.remote_addr} reason=invalid_or_expired_token"
        )
        # Small delay helps slow brute-force token probing attempts.
        time.sleep(0.5)
        return "Invalid or expired reset token.", 400
    if not user.is_active:
        app.logger.warning(f"[RESET_INVALID] email={user.email} ip={request.remote_addr} reason=inactive_user")
        return "Invalid or expired reset token.", 400

    if request.method == "GET":
        return render_template_string(
            """
            <h2>Reset Password</h2>
            <form method="post">
              <label>New Password</label><br>
              <input type="password" name="password" required><br><br>
              <button type="submit">Set new password</button>
            </form>
            """
        )

    new_password = (request.form.get("password") or "").strip()
    # Password is already stripped above; warn and delay on weak attempts.
    if not new_password or len(new_password) < 8:
        app.logger.warning(
            f"[RESET_INVALID] email={user.email} ip={request.remote_addr} reason=weak_password"
        )
        time.sleep(0.3)
        return "Password must be at least 8 characters.", 400

    user.password_hash = generate_password_hash(new_password)
    # Single-use token: clear immediately after successful reset.
    user.reset_token = None
    user.reset_token_expiry = None
    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        app.logger.error(f"[RESET_ERROR] email={user.email} ip={request.remote_addr} stage=password_commit error={exc}")
        return "Internal error.", 500

    app.logger.info(f"[RESET_SUCCESS] email={user.email} ip={request.remote_addr}")
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True, port=8080)
