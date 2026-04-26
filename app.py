"""
Minimal IRIS Gatekeeper application.

Requirements implemented:
- Flask + SQLAlchemy
- werkzeug.security password hashing
- No public signup route
- Admin-only user creation at /admin/create-user
- @irdai.gov.in email domain restriction
- Token-based password reset with 15-minute expiry
"""

from datetime import datetime, timedelta, timezone
import secrets

from flask import Flask, request, redirect, session, url_for, render_template_string, abort
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# -----------------------------
# App + DB configuration
# -----------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-in-production"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///iris_auth.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

ALLOWED_DOMAIN = "@irdai.gov.in"
RESET_TOKEN_MINUTES = 15


# -----------------------------
# Database model
# -----------------------------
class User(db.Model):
    """User model required by spec."""

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    reset_token = db.Column(db.String(128), nullable=True)
    reset_token_expiry = db.Column(db.DateTime(timezone=True), nullable=True)


# -----------------------------
# Helper functions
# -----------------------------
def is_valid_irdai_email(email: str) -> bool:
    """Return True only for @irdai.gov.in addresses."""
    return bool(email) and email.lower().endswith(ALLOWED_DOMAIN)


def current_user() -> User | None:
    """Load current logged-in user from session."""
    uid = session.get("user_id")
    if not uid:
        return None
    return db.session.get(User, uid)


def require_login() -> User:
    """Require login for protected routes."""
    user = current_user()
    if not user:
        abort(401, "Please login first.")
    return user


def require_admin() -> User:
    """Require admin rights for protected admin routes."""
    user = require_login()
    if not user.is_admin:
        abort(403, "Admin access required.")
    return user


def create_default_admin() -> None:
    """Create one default admin if DB has none (for first run)."""
    if User.query.filter_by(is_admin=True).first():
        return

    admin_email = "admin@irdai.gov.in"
    admin_password = "Admin@12345"

    user = User(
        email=admin_email,
        password_hash=generate_password_hash(admin_password),
        is_active=True,
        is_admin=True,
    )
    db.session.add(user)
    db.session.commit()
    print("[IRIS] Default admin created")
    print(f"[IRIS] Email: {admin_email}")
    print(f"[IRIS] Password: {admin_password}")


# -----------------------------
# App initialization
# -----------------------------
with app.app_context():
    db.create_all()
    create_default_admin()


# -----------------------------
# Basic home page (protected)
# -----------------------------
@app.route("/")
def home():
    user = require_login()
    return render_template_string(
        """
        <h2>IRIS Home</h2>
        <p>Welcome, {{ user.email }}</p>
        <p>Admin: {{ 'Yes' if user.is_admin else 'No' }}</p>
        <a href="{{ url_for('logout') }}">Logout</a>
        {% if user.is_admin %}
          <p><a href="{{ url_for('admin_create_user') }}">Create User (Admin)</a></p>
        {% endif %}
        """,
        user=user,
    )


# -----------------------------
# Login route
# -----------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    """
    Validate:
    a) email exists
    b) password matches
    c) is_active is True
    d) email ends with @irdai.gov.in
    """
    message = ""

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        if not is_valid_irdai_email(email):
            message = "Only @irdai.gov.in email addresses are allowed."
        else:
            user = User.query.filter_by(email=email).first()
            if not user:
                message = "Invalid credentials."
            elif not check_password_hash(user.password_hash, password):
                message = "Invalid credentials."
            elif not user.is_active:
                message = "Account is inactive."
            else:
                session["user_id"] = user.id
                return redirect(url_for("home"))

    return render_template_string(
        """
        <h2>Login</h2>
        {% if message %}<p style="color:red;">{{ message }}</p>{% endif %}
        <form method="post">
          <input name="email" type="email" placeholder="name@irdai.gov.in" required><br><br>
          <input name="password" type="password" placeholder="Password" required><br><br>
          <button type="submit">Login</button>
        </form>
        <p><a href="{{ url_for('forgot_password') }}">Forgot Password?</a></p>
        """,
        message=message,
    )


# -----------------------------
# Logout route
# -----------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# -----------------------------
# Admin-only user creation route
# -----------------------------
@app.route("/admin/create-user", methods=["GET", "POST"])
def admin_create_user():
    """
    Admin-only route.
    Allows creation of users with email + password.
    Public signup is intentionally disabled.
    """
    require_admin()
    message = ""

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        make_admin = request.form.get("is_admin") == "on"

        if not is_valid_irdai_email(email):
            message = "Email must end with @irdai.gov.in"
        elif len(password) < 8:
            message = "Password must be at least 8 characters."
        elif User.query.filter_by(email=email).first():
            message = "User already exists."
        else:
            user = User(
                email=email,
                password_hash=generate_password_hash(password),
                is_active=True,
                is_admin=make_admin,
            )
            db.session.add(user)
            db.session.commit()
            message = f"User created: {email}"

    return render_template_string(
        """
        <h2>Admin - Create User</h2>
        {% if message %}<p>{{ message }}</p>{% endif %}
        <form method="post">
          <input name="email" type="email" placeholder="name@irdai.gov.in" required><br><br>
          <input name="password" type="password" placeholder="Temporary password" required><br><br>
          <label><input name="is_admin" type="checkbox"> Is admin</label><br><br>
          <button type="submit">Create User</button>
        </form>
        <p><a href="{{ url_for('home') }}">Back Home</a></p>
        """,
        message=message,
    )


# -----------------------------
# Forgot password route
# -----------------------------
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    """
    On request:
    - generate secure token (secrets module)
    - store token + 15-min expiry
    - print reset link in console
    """
    message = ""

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()

        if not is_valid_irdai_email(email):
            message = "Only @irdai.gov.in email addresses are allowed."
        else:
            user = User.query.filter_by(email=email).first()
            # Do not reveal too much; still keep behavior simple for this minimal app
            if user:
                token = secrets.token_urlsafe(32)
                user.reset_token = token
                user.reset_token_expiry = datetime.now(timezone.utc) + timedelta(minutes=RESET_TOKEN_MINUTES)
                db.session.commit()

                reset_link = url_for("reset_password", token=token, _external=True)
                print(f"[IRIS] Password reset link for {email}: {reset_link}")

            message = "If the account exists, a reset link has been generated and printed in server console."

    return render_template_string(
        """
        <h2>Forgot Password</h2>
        {% if message %}<p>{{ message }}</p>{% endif %}
        <form method="post">
          <input name="email" type="email" placeholder="name@irdai.gov.in" required><br><br>
          <button type="submit">Generate Reset Link</button>
        </form>
        <p><a href="{{ url_for('login') }}">Back to Login</a></p>
        """,
        message=message,
    )


# -----------------------------
# Token-based reset route
# -----------------------------
@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    """
    Allow reset only if:
    - token exists
    - token not expired
    Invalidate token after successful reset.
    """
    user = User.query.filter_by(reset_token=token).first()

    if not user:
        return "Invalid reset token.", 400

    if not user.reset_token_expiry or user.reset_token_expiry < datetime.now(timezone.utc):
        return "Reset token expired.", 400

    message = ""
    if request.method == "POST":
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm") or ""

        if len(password) < 8:
            message = "Password must be at least 8 characters."
        elif password != confirm:
            message = "Passwords do not match."
        else:
            user.password_hash = generate_password_hash(password)
            user.reset_token = None
            user.reset_token_expiry = None
            db.session.commit()
            return redirect(url_for("login"))

    return render_template_string(
        """
        <h2>Reset Password</h2>
        {% if message %}<p style="color:red;">{{ message }}</p>{% endif %}
        <form method="post">
          <input name="password" type="password" placeholder="New password" required><br><br>
          <input name="confirm" type="password" placeholder="Confirm password" required><br><br>
          <button type="submit">Set New Password</button>
        </form>
        """,
        message=message,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8080)
