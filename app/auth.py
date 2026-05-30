"""
Simple authentication module for SmartTraffic AI.
Two roles: admin (full dashboard) and operator (map view).
"""
import hashlib
import os
import secrets
from functools import wraps
from flask import session, redirect, request, jsonify


def _sha256(value):
    return hashlib.sha256(str(value or "").encode()).hexdigest()

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

    admin_username = str(os.environ.get("ADMIN_USERNAME") or "admin").strip() or "admin"
    admin_hash = _resolve_password_hash("ADMIN")
    if admin_hash:
        users[admin_username] = {
            "password_hash": admin_hash,
            "role": "admin",
            "name": str(os.environ.get("ADMIN_DISPLAY_NAME") or "Administrator").strip() or "Administrator",
        }

    operator_username = str(os.environ.get("OPERATOR_USERNAME") or "operator").strip() or "operator"
    operator_hash = _resolve_password_hash("OPERATOR")
    if operator_hash:
        users[operator_username] = {
            "password_hash": operator_hash,
            "role": "operator",
            "name": str(os.environ.get("OPERATOR_DISPLAY_NAME") or "Operator").strip() or "Operator",
        }

    if not users:
        print("[AUTH] WARN: No login users configured. Enabling local bootstrap credentials temporarily.")
        users = {
            "admin": {
                "password_hash": _sha256("admin123"),
                "role": "admin",
                "name": "Administrator",
            },
            "operator": {
                "password_hash": _sha256("operator123"),
                "role": "operator",
                "name": "Operator",
            },
        }

    return users


def _resolve_secret_key():
    secret = str(os.environ.get("SECRET_KEY") or "").strip()
    if secret:
        return secret
    print("[AUTH] WARN: SECRET_KEY env not set. Using an ephemeral generated secret for this process.")
    return secrets.token_urlsafe(48)


USERS = _build_users()
SECRET_KEY = _resolve_secret_key()


def check_login(username, password):
    """Verify credentials. Returns user dict or None."""
    user = USERS.get(username)
    if not user:
        return None
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    if pw_hash == user["password_hash"]:
        return {"username": username, "role": user["role"], "name": user["name"]}
    return None


def login_required(f):
    """Decorator: redirect to login if not authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user"):
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({"status": "error", "message": "Authentication required"}), 401
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """Decorator: only allow admin role."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = session.get("user")
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
    return session.get("user")
