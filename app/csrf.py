"""
CSRF Protection module for SmartTraffic AI.

Implements secret-key based CSRF token validation:
- Generates HMAC-based token per session
- Requires token on all state-changing requests (POST, PUT, PATCH, DELETE)
- Tokens are session-bound and short-lived (1 hour)

Usage:
    from app.csrf import csrf_required, generate_csrf_token

    @app.route("/api/add_camera", methods=["POST"])
    @csrf_required
    def add_camera():
        ...

In templates:
    {{ csrf_token() }}   -- outputs the session's CSRF token
    <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
"""
import hashlib
import hmac
import os
import secrets
import time
from functools import wraps
from flask import session, request, jsonify, current_app


# ── Token generation ────────────────────────────────────────────────────────────

def _get_secret():
    """Get CSRF secret from app config or session."""
    secret = current_app.config.get("SECRET_KEY") or session.get("csrf_secret")
    if not secret:
        secret = secrets.token_hex(24)
        session["csrf_secret"] = secret
    return secret


def generate_csrf_token():
    """
    Generate and return the CSRF token for the current session.
    Cached per session — regenerated only if expired or missing.
    """
    secret = _get_secret()
    # Token: HMAC-SHA256 of session user identifier + timestamp bucket (1-hour windows)
    user_id = str(session.get("user", {}).get("username", "anonymous") or "anon")
    # Use 1-hour timestamp bucket so tokens expire after 1 hour
    ts_bucket = int(time.time() // 3600)
    message = f"{user_id}:{ts_bucket}:{secret}"
    token = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()
    return f"{token[:32]}_{ts_bucket}"


def validate_csrf_token(token_str):
    """
    Validate a CSRF token.
    Returns True if valid, False otherwise.
    Accepts current bucket OR previous bucket (grace period of 1 hour).
    """
    if not token_str or not isinstance(token_str, str):
        return False

    parts = token_str.rsplit("_", 1)
    if len(parts) != 2:
        return False

    provided_token = parts[0]
    provided_bucket = parts[1]

    try:
        provided_bucket = int(provided_bucket)
    except ValueError:
        return False

    # Check against current bucket AND previous bucket (1-hour grace)
    current_bucket = int(time.time() // 3600)
    if abs(provided_bucket - current_bucket) > 1:
        return False  # Token too old (more than 2 hours old)

    secret = _get_secret()
    user_id = str(session.get("user", {}).get("username", "anonymous") or "anon")

    for bucket in (current_bucket, current_bucket - 1):
        message = f"{user_id}:{bucket}:{secret}"
        expected = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()
        if hmac.compare_digest(provided_token, expected[:32]):
            return True

    return False


# ── Decorators ──────────────────────────────────────────────────────────────────

def csrf_required(f):
    """
    Decorator: validate CSRF token on state-changing requests.
    Checks X-CSRF-Token header OR csrf_token form field.

    Usage:
        @app.route("/api/add_camera", methods=["POST"])
        @csrf_required
        def add_camera():
            ...
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # Skip CSRF in DEMO_MODE (for competition convenience)
        # In production, remove this check
        from app.auth import is_demo_auth_bypass
        if is_demo_auth_bypass():
            return f(*args, **kwargs)

        # Only check state-changing methods
        if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
            return f(*args, **kwargs)

        # Get token from header or form
        token = None

        # Try X-CSRF-Token header (for JSON/SPA clients)
        if request.is_json:
            data = request.get_json(silent=True) or {}
            token = data.get("csrf_token") or data.get("csrf")
        else:
            # Try form field
            token = request.form.get("csrf_token") or request.form.get("csrf")

        # Try header (for fetch/axios clients)
        if not token:
            token = request.headers.get("X-CSRF-Token") or request.headers.get("X-CSRFHeader-Token")

        if not token:
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({
                    "status": "error",
                    "message": "CSRF token required. Include 'csrf_token' in request body or 'X-CSRF-Token' header."
                }), 403
            return jsonify({"status": "error", "message": "CSRF token missing"}), 403

        if not validate_csrf_token(token):
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({
                    "status": "error",
                    "message": "Invalid or expired CSRF token. Refresh the page and try again."
                }), 403
            return jsonify({"status": "error", "message": "Invalid CSRF token"}), 403

        return f(*args, **kwargs)
    return decorated


def csrf_exempt(f):
    """
    Decorator: mark a route as exempt from CSRF validation.
    Use sparingly — only for truly public endpoints that need POST access.
    """
    f._csrf_exempt = True
    return f


# ── Template helper ─────────────────────────────────────────────────────────────

def csrf_token():
    """Jinja2 template filter: output the CSRF token for current session."""
    return generate_csrf_token()


# ── Jinja2 integration ────────────────────────────────────────────────────────

def init_csrf(app):
    """Register CSRF helpers into Jinja2 environment and before-request handler."""
    @app.before_request
    def csrf_protect():
        """Ensure session has a CSRF secret on every request."""
        if "csrf_secret" not in session:
            session["csrf_secret"] = secrets.token_hex(24)

    @app.context_processor
    def csrf_context():
        """Make csrf_token() available in all templates."""
        return {"csrf_token": generate_csrf_token}
