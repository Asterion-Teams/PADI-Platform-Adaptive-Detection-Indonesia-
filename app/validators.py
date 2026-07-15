"""
Input validation decorators and helpers for SmartTraffic AI.

Provides type-safe parameter validation for route handlers:
- Integer, float, string, boolean, list parameters
- Range constraints (min/max)
- Length constraints (for strings)
- Enum constraints
- UUID/safe string sanitization

Usage:
    from app.validators import validate_int, validate_float, validate_str

    @app.route("/api/violations/<int:vid>")
    def get_violation(vid: int):
        ...

    # For query/form parameters:
    @app.route("/api/stats")
    @validate_int("limit", min=1, max=1000)
    @validate_int("offset", min=0)
    def get_stats(limit: int, offset: int):
        ...
"""
import re
import uuid
from functools import wraps
from flask import request, jsonify


# ── Safe string sanitization ────────────────────────────────────────────────────

def safe_string(value, max_length=None, allow_dots=True):
    """
    Sanitize a string input to prevent injection attacks.
    - Strips control characters
    - Limits length
    - Removes path traversal patterns
    """
    if not isinstance(value, str):
        return ""
    # Strip control characters except newlines/tabs
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)
    # Remove path traversal
    s = re.sub(r"\.\.[/\\]", "", s)
    # Remove null bytes
    s = s.replace("\x00", "")
    # Strip and limit length
    s = s.strip()
    if max_length:
        s = s[:int(max_length)]
    return s


def safe_camera_id(value):
    """Validate and sanitize a camera ID."""
    s = safe_string(value, max_length=64)
    # Must be alphanumeric + underscore/hyphen
    if not re.match(r"^[\w\-]+$", s):
        return None
    return s


def safe_integer(value, default=0):
    """Parse integer safely, returning default on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


# ── Validation decorators ───────────────────────────────────────────────────────

class ValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(self, param: str, message: str):
        self.param = param
        self.message = message
        super().__init__(f"{param}: {message}")


def _get_param_value(param: str):
    """Get parameter from JSON body, form data, or query args."""
    # JSON body
    if request.is_json:
        data = request.get_json(silent=True) or {}
        if param in data:
            return data[param]
    # Form data
    if param in request.form:
        return request.form[param]
    # Query args
    return request.args.get(param)


def validate_int(param: str, min_val=None, max_val=None, default=None, required=True):
    """
    Decorator: validate an integer parameter from request.

    Args:
        param: parameter name
        min_val: minimum value (inclusive)
        max_val: maximum value (inclusive)
        default: default value if missing or invalid
        required: if True, missing params return 400 error
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            raw = _get_param_value(param)
            if raw is None:
                if required and default is None:
                    return jsonify({
                        "status": "error",
                        "message": f"Missing required parameter: {param}",
                        "param": param,
                    }), 400
                kwargs[param] = default if default is not None else 0
                return f(*args, **kwargs)

            try:
                value = int(raw)
            except (ValueError, TypeError):
                return jsonify({
                    "status": "error",
                    "message": f"Parameter '{param}' must be an integer",
                    "param": param,
                    "received": str(raw)[:100],
                }), 400

            if min_val is not None and value < min_val:
                return jsonify({
                    "status": "error",
                    "message": f"Parameter '{param}' must be >= {min_val}",
                    "param": param,
                    "value": value,
                    "min": min_val,
                }), 400

            if max_val is not None and value > max_val:
                return jsonify({
                    "status": "error",
                    "message": f"Parameter '{param}' must be <= {max_val}",
                    "param": param,
                    "value": value,
                    "max": max_val,
                }), 400

            kwargs[param] = value
            return f(*args, **kwargs)
        return decorated
    return decorator


def validate_float(param: str, min_val=None, max_val=None, default=None, required=True):
    """Decorator: validate a float parameter from request."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            raw = _get_param_value(param)
            if raw is None:
                if required and default is None:
                    return jsonify({
                        "status": "error",
                        "message": f"Missing required parameter: {param}",
                        "param": param,
                    }), 400
                kwargs[param] = default if default is not None else 0.0
                return f(*args, **kwargs)

            try:
                value = float(raw)
            except (ValueError, TypeError):
                return jsonify({
                    "status": "error",
                    "message": f"Parameter '{param}' must be a number",
                    "param": param,
                    "received": str(raw)[:100],
                }), 400

            if min_val is not None and value < min_val:
                return jsonify({
                    "status": "error",
                    "message": f"Parameter '{param}' must be >= {min_val}",
                    "param": param,
                }), 400

            if max_val is not None and value > max_val:
                return jsonify({
                    "status": "error",
                    "message": f"Parameter '{param}' must be <= {max_val}",
                    "param": param,
                }), 400

            kwargs[param] = value
            return f(*args, **kwargs)
        return decorated
    return decorator


def validate_str(param: str, max_length=None, pattern=None, allowed_chars=None,
                 default=None, required=True):
    """
    Decorator: validate a string parameter from request.

    Args:
        param: parameter name
        max_length: maximum character length
        pattern: regex pattern (string must match)
        allowed_chars: set of allowed characters
        default: default value if missing
        required: if True, missing params return 400 error
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            raw = _get_param_value(param)
            if raw is None:
                if required and default is None:
                    return jsonify({
                        "status": "error",
                        "message": f"Missing required parameter: {param}",
                        "param": param,
                    }), 400
                kwargs[param] = default or ""
                return f(*args, **kwargs)

            value = safe_string(raw, max_length=max_length)

            if allowed_chars:
                invalid = set(value) - set(allowed_chars)
                if invalid:
                    return jsonify({
                        "status": "error",
                        "message": f"Parameter '{param}' contains invalid characters",
                        "param": param,
                        "invalid_chars": list(invalid)[:10],
                    }), 400

            if pattern:
                if not re.match(pattern, value):
                    return jsonify({
                        "status": "error",
                        "message": f"Parameter '{param}' has invalid format",
                        "param": param,
                        "pattern": pattern,
                    }), 400

            kwargs[param] = value
            return f(*args, **kwargs)
        return decorated
    return decorator


def validate_enum(param: str, allowed_values: list, default=None, required=True,
                  case_insensitive=False):
    """Decorator: validate a parameter against allowed enum values."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            raw = _get_param_value(param)
            if raw is None:
                if required and default is None:
                    return jsonify({
                        "status": "error",
                        "message": f"Missing required parameter: {param}",
                        "param": param,
                    }), 400
                kwargs[param] = default or allowed_values[0]
                return f(*args, **kwargs)

            raw_str = str(raw).strip()
            if case_insensitive:
                allowed_lower = {str(v).lower() for v in allowed_values}
                if raw_str.lower() not in allowed_lower:
                    return jsonify({
                        "status": "error",
                        "message": f"Parameter '{param}' must be one of: {allowed_values}",
                        "param": param,
                        "received": raw_str,
                    }), 400
                # Find the canonical value
                value = next((str(v) for v in allowed_values if str(v).lower() == raw_str.lower()), raw_str)
            else:
                if raw_str not in [str(v) for v in allowed_values]:
                    return jsonify({
                        "status": "error",
                        "message": f"Parameter '{param}' must be one of: {allowed_values}",
                        "param": param,
                        "received": raw_str,
                    }), 400
                value = raw_str

            kwargs[param] = value
            return f(*args, **kwargs)
        return decorated
    return decorator


def validate_uuid(param: str, default=None, required=True):
    """Decorator: validate a UUID parameter."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            raw = _get_param_value(param)
            if raw is None:
                if required and default is None:
                    return jsonify({
                        "status": "error",
                        "message": f"Missing required parameter: {param}",
                        "param": param,
                    }), 400
                kwargs[param] = default
                return f(*args, **kwargs)

            try:
                uuid_obj = uuid.UUID(str(raw).strip())
                kwargs[param] = str(uuid_obj)
                return f(*args, **kwargs)
            except ValueError:
                return jsonify({
                    "status": "error",
                    "message": f"Parameter '{param}' must be a valid UUID",
                    "param": param,
                    "received": str(raw)[:100],
                }), 400
        return decorated
    return decorator
