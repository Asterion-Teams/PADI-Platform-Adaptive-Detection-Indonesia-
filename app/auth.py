"""
Simple authentication module for SmartTraffic AI.
Two roles: admin (full dashboard) and operator (map view).

Security: Set DEMO_MODE=1 to enable demo bypass.
         Set ADMIN_PASSWORD or ADMIN_PASSWORD_HASH for admin credentials.
         Set OPERATOR_PASSWORD or OPERATOR_PASSWORD_HASH for operator credentials.
         Set SECRET_KEY for session security.
"""
import hashlib
import os
import secrets
from functools import wraps
from flask import session, redirect, request, jsonify


def _sha256(value):
    return hashlib.sha256(str(value or "").encode()).hexdigest()

def _resolve_username(prefix):
    return str(os.environ.get(f"{prefix}_USERNAME") or "").strip()


def _resolve_password_hash(prefix):
    direct_hash = str(os.environ.get(f"{prefix}_PASSWORD_HASH") or "").strip()
    if direct_hash:
        return direct_hash

    plain = os.environ.get(f"{prefix}_PASSWORD")
    if plain:
        return _sha256(plain)

    return ""


def _build_users():
    users = {}

    # Admin user
    admin_username = _resolve_username("ADMIN") or "admin"
    admin_hash = _resolve_password_hash("ADMIN")
    if admin_username and admin_hash:
        users[admin_username] = {
            "password_hash": admin_hash,
            "role": "admin",
            "name": str(os.environ.get("ADMIN_DISPLAY_NAME") or "Administrator").strip() or "Administrator",
        }

    # Operator user
    operator_username = _resolve_username("OPERATOR") or "operator"
    operator_hash = _resolve_password_hash("OPERATOR")
    if operator_username and operator_hash:
        users[operator_username] = {
            "password_hash": operator_hash,
            "role": "operator",
            "name": str(os.environ.get("OPERATOR_DISPLAY_NAME") or "Operator").strip() or "Operator",
        }

    return users


def _resolve_secret_key():
    secret = str(os.environ.get("SECRET_KEY") or "").strip()
    if secret:
        return secret
    # Warn in production: ephemeral key means sessions are invalidated on restart
    if str(os.environ.get("FLASK_ENV") or "").lower() == "production":
        raise RuntimeError(
            "[AUTH] FATAL: SECRET_KEY env must be set in production! "
            "Sessions will be invalidated on every restart otherwise."
        )
    print("[AUTH] WARN: SECRET_KEY env not set. Using an ephemeral generated secret.")
    print("[AUTH] WARN: Set SECRET_KEY=<random-string> for persistent sessions.")
    return secrets.token_urlsafe(48)


# ── DEMO_MODE ────────────────────────────────────────────────────────────────
# Set DEMO_MODE=1 to bypass authentication (development/demo only).
# In production, ALWAYS set DEMO_MODE=0 or leave it unset.
_DEMO_MODE = str(os.environ.get("DEMO_MODE") or "0").strip().lower() in {
    "1", "true", "yes", "on"
}

if _DEMO_MODE:
    print("[AUTH] WARNING: DEMO_MODE is ENABLED — authentication is bypassed.")
    print("[AUTH] WARNING: Do NOT enable DEMO_MODE in production!")


USERS = _build_users()
SECRET_KEY = _resolve_secret_key()
DEMO_USER = {
    "username": "demo",
    "role": "admin",
    "name": "Demo Admin",
}


def is_demo_auth_bypass():
    """Check if demo mode bypass is enabled.

    Returns True ONLY if DEMO_MODE env is explicitly set to 1/true/yes/on.
    When disabled (default), full authentication is required.
    """
    return _DEMO_MODE


def ensure_demo_user_session():
    if is_demo_auth_bypass() and not session.get("user"):
        session["user"] = dict(DEMO_USER)
    return session.get("user")


def check_login(username, password):
    """Verify credentials. Returns user dict or None."""
    # In demo mode, return demo user without password check
    if is_demo_auth_bypass():
        return dict(DEMO_USER)

    user = USERS.get(username)
    if not user:
        return None

    pw_hash = _sha256(password)
    if pw_hash == user["password_hash"]:
        return {"username": username, "role": user["role"], "name": user["name"]}
    return None


def login_required(f):
    """Decorator: redirect to login if not authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not ensure_demo_user_session():
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({"status": "error", "message": "Authentication required"}), 401
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """Decorator: only allow admin role."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = ensure_demo_user_session()
        if not user:
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({"status": "error", "message": "Authentication required"}), 401
            return redirect("/login")
        if user.get("role") != "admin":
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({"status": "error", "message": "Admin access required"}), 403
            return redirect("/operator")
        return f(*args, **kwargs)
    return decorated


def get_current_user():
    """Get current logged-in user from session."""
    return ensure_demo_user_session()
