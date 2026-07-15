"""
Rate Limiting module for SmartTraffic AI.

In-memory rate limiting with sliding window algorithm:
- Per-IP tracking for public endpoints
- Per-session tracking for authenticated endpoints
- Configurable limits per endpoint type
- Returns 429 Too Many Requests when exceeded

Usage:
    from app.ratelimit import rate_limit, RateLimitExceeded

    @app.route("/api/chat", methods=["POST"])
    @rate_limit(calls=10, period=60)
    def chat():
        ...

    # Custom limit for sensitive endpoints:
    @app.route("/api/cameras", methods=["POST"])
    @rate_limit(calls=3, period=60)
    @admin_required
    def add_camera():
        ...
"""
import threading
import time
from collections import defaultdict, deque
from functools import wraps
from flask import request, jsonify


# ── Rate limit store ─────────────────────────────────────────────────────────────

class _RateLimitStore:
    """Thread-safe in-memory rate limit store with sliding window."""

    def __init__(self):
        self._lock = threading.RLock()
        # key: (identifier, endpoint) -> deque of timestamps
        self._requests = defaultdict(deque)
        self._last_cleanup = time.time()

    def _cleanup(self):
        """Remove entries older than 1 hour to prevent memory leak."""
        now = time.time()
        if now - self._last_cleanup < 300:  # Cleanup max every 5 minutes
            return
        self._last_cleanup = now
        cutoff = now - 7200  # 2 hours
        keys_to_delete = []
        for key, times in self._requests.items():
            # Trim old entries
            while times and times[0] < cutoff:
                times.popleft()
            if not times:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            try:
                del self._requests[key]
            except Exception:
                pass

    def _get_key(self, identifier, endpoint):
        """Generate a unique key for this request."""
        return (str(identifier), str(endpoint))

    def is_allowed(self, identifier, endpoint, calls, period):
        """
        Check if a request is allowed under the rate limit.
        Uses sliding window: only counts requests within the last `period` seconds.

        Returns: (allowed: bool, remaining: int, reset_in: float)
        """
        with self._lock:
            self._cleanup()
            key = self._get_key(identifier, endpoint)
            now = time.time()
            cutoff = now - float(period)
            times = self._requests[key]

            # Remove timestamps outside the window
            while times and times[0] < cutoff:
                times.popleft()

            current_count = len(times)
            remaining = max(0, int(calls) - current_count)

            if current_count >= int(calls):
                # Rate limit exceeded
                oldest = times[0] if times else now
                reset_in = max(0.0, oldest + float(period) - now)
                return False, 0, reset_in

            # Allow this request
            times.append(now)
            remaining = max(0, int(calls) - len(times) - 1)
            reset_in = float(period)
            return True, remaining, reset_in

    def get_status(self, identifier, endpoint, calls, period):
        """Get current rate limit status without consuming a call."""
        with self._lock:
            key = self._get_key(identifier, endpoint)
            now = time.time()
            cutoff = now - float(period)
            times = self._requests[key]
            while times and times[0] < cutoff:
                times.popleft()
            current = len(times)
            remaining = max(0, int(calls) - current)
            reset_in = float(period)
            return current, remaining, reset_in


# Global store instance
_store = _RateLimitStore()


# ── Default rate limits ──────────────────────────────────────────────────────────

DEFAULT_LIMITS = {
    # Global defaults
    "default":       (30, 60),   # 30 calls per 60 seconds

    # Chat API (expensive — AI calls)
    "chat":          (5, 60),    # 5 messages per minute

    # Violations API (DB queries)
    "violations":    (30, 60),   # 30 per minute
    "violations_list": (60, 60),  # 60 per minute (list is lighter)

    # Camera management (sensitive)
    "camera_add":    (3, 60),    # 3 per minute
    "camera_delete": (2, 60),    # 2 per minute
    "camera_edit":   (5, 60),    # 5 per minute

    # Stats (lightweight)
    "stats":         (60, 60),   # 60 per minute
    "history":       (30, 60),   # 30 per minute
    "density":       (30, 60),   # 30 per minute

    # Prediction (ML — expensive)
    "predict":       (5, 60),    # 5 per minute

    # Social media
    "twitter":       (10, 60),   # 10 per minute
    "crm":           (10, 60),   # 10 per minute

    # AI test (very expensive)
    "ai_test":       (2, 60),    # 2 per minute
    "ocr_test":      (3, 60),    # 3 per minute
}


def rate_limit(calls=None, period=None, key=None, limit_name=None):
    """
    Decorator: apply rate limiting to a route.

    Args:
        calls:   max calls allowed (overrides limit_name)
        period:  time window in seconds (overrides limit_name)
        key:     custom key function (default: ip for public, username for auth)
        limit_name: name in DEFAULT_LIMITS dict (overrides calls/period)

    Usage:
        @rate_limit(limit_name="chat")       # uses DEFAULT_LIMITS["chat"]
        @rate_limit(calls=10, period=60)     # custom: 10 per minute
        @rate_limit(calls=5, period=60, key=lambda: request.headers.get("X-API-Key"))
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            from app.auth import get_current_user

            # Determine limit parameters
            if limit_name and limit_name in DEFAULT_LIMITS:
                lim_calls, lim_period = DEFAULT_LIMITS[limit_name]
            elif calls is not None and period is not None:
                lim_calls = int(calls)
                lim_period = int(period)
            else:
                lim_calls, lim_period = DEFAULT_LIMITS["default"]

            # Determine identifier
            if callable(key):
                identifier = key()
            else:
                user = None
                try:
                    user = get_current_user()
                except Exception:
                    pass
                if user and isinstance(user, dict):
                    identifier = f"user:{user.get('username', 'unknown')}"
                else:
                    # Fallback: use IP address
                    identifier = f"ip:{_get_client_ip()}"

            endpoint = f"{request.method}:{request.path}"

            allowed, remaining, reset_in = _store.is_allowed(
                identifier, endpoint, lim_calls, lim_period
            )

            if not allowed:
                response = jsonify({
                    "status": "error",
                    "message": f"Rate limit exceeded. Try again in {int(reset_in)} seconds.",
                    "retry_after": int(reset_in),
                    "limit": lim_calls,
                    "period": lim_period,
                })
                response.status_code = 429
                response.headers["Retry-After"] = str(int(reset_in))
                response.headers["X-RateLimit-Limit"] = str(lim_calls)
                response.headers["X-RateLimit-Remaining"] = "0"
                response.headers["X-RateLimit-Reset"] = str(int(time.time() + reset_in))
                return response

            # Proceed with the request
            result = f(*args, **kwargs)

            # Add rate limit headers to response (if it's a Response object)
            if hasattr(result, "headers"):
                result.headers["X-RateLimit-Limit"] = str(lim_calls)
                result.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
                result.headers["X-RateLimit-Reset"] = str(int(time.time() + reset_in))

            return result
        return decorated
    return decorator


def _get_client_ip():
    """Get the real client IP, checking X-Forwarded-For header."""
    # Check for proxy headers
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # X-Forwarded-For: client, proxy1, proxy2
        return forwarded.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    return request.remote_addr or "0.0.0.0"


def rate_limit_status(limit_name=None, calls=None, period=None):
    """
    Get current rate limit status without consuming a call.
    Useful for displaying rate limit info in the UI.
    """
    from app.auth import get_current_user

    if limit_name and limit_name in DEFAULT_LIMITS:
        lim_calls, lim_period = DEFAULT_LIMITS[limit_name]
    elif calls is not None and period is not None:
        lim_calls = int(calls)
        lim_period = int(period)
    else:
        lim_calls, lim_period = DEFAULT_LIMITS["default"]

    user = get_current_user()
    if user and isinstance(user, dict):
        identifier = f"user:{user.get('username', 'unknown')}"
    else:
        identifier = f"ip:{_get_client_ip()}"

    endpoint = f"status:{limit_name or 'unknown'}"
    current, remaining, reset_in = _store.get_status(identifier, endpoint, lim_calls, lim_period)

    return {
        "limit": lim_calls,
        "period": lim_period,
        "current": current,
        "remaining": remaining,
        "reset_in": int(reset_in),
        "exhausted": current >= lim_calls,
    }
