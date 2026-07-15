"""
Shared state and helpers for all route modules.

All route modules import from here to share state without circular imports.
Blueprint registration happens here, imported by routes.py.
"""
import os
import time
import threading
from flask import Blueprint

# ── Module-level cached state (shared across all route modules) ──────────────

_ollama_cached_model = None
_ollama_cached_at = 0.0
_ollama_cached_models = None
_ollama_cached_models_at = 0.0
_ollama_last_success_at = 0.0
_geocode_cache = {}

# Chat context store (thread-safe)
_chat_ctx_store = {}
_chat_ctx_lock = threading.Lock()

# Planner allowed action types
_PLANNER_ACTION_TYPES = {
    "navigate", "select_camera", "set_period",
    "show_prediction_popup", "show_forecast_popup"
}

# ── Shared helper functions (used by multiple route modules) ────────────────

def _chat_grounded_mode_enabled():
    v = str(os.environ.get("CHAT_GROUNDED_MODE") or "").strip().lower()
    return v not in ("0", "false", "no", "off")


def _chat_ctx_key(session_id, req):
    sid = str(session_id or "").strip()
    if sid:
        return sid
    ip = str(getattr(req, "remote_addr", "") or "")
    ua = str((req.headers.get("User-Agent") if hasattr(req, "headers") else "") or "")
    return f"{ip}|{ua}" if (ip or ua) else "anon"


def _chat_ctx_get(key, ttl_s=3600):
    now = time.time()
    with _chat_ctx_lock:
        item = _chat_ctx_store.get(key) or {}
        ts = float(item.get("ts") or 0.0)
        if ts and (now - ts) > float(ttl_s or 0):
            try:
                del _chat_ctx_store[key]
            except Exception:
                pass
            return {}
        ctx = item.get("ctx") if isinstance(item.get("ctx"), dict) else {}
        return dict(ctx or {})


def _chat_ctx_set(key, ctx):
    now = time.time()
    with _chat_ctx_lock:
        _chat_ctx_store[key] = {"ts": now, "ctx": dict(ctx or {})}


def _planner_enabled():
    v = str(os.environ.get("CHAT_PLANNER") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _planner_validate_actions(actions):
    out = []
    for a in (actions or []):
        if not isinstance(a, dict):
            continue
        t = str(a.get("type") or "").strip()
        if t not in _PLANNER_ACTION_TYPES:
            continue
        if t == "navigate":
            path = str(a.get("path") or "").strip()
            if not path.startswith("/"):
                continue
            out.append({"type": "navigate", "path": path})
            continue
        if t == "select_camera":
            cam_id = str(a.get("camera_id") or "").strip()
            cam_name = str(a.get("camera_name") or cam_id or "").strip()
            if not cam_id:
                continue
            out.append({"type": "select_camera", "camera_id": cam_id, "camera_name": cam_name})
            continue
        if t == "set_period":
            period = str(a.get("period") or "").strip()
            if not period:
                continue
            out.append({"type": "set_period", "period": period})
            continue
        if t == "show_prediction_popup":
            cam_id = str(a.get("camera_id") or "").strip()
            if not cam_id:
                continue
            keep = {k: a.get(k) for k in (
                "camera_id", "camera_name", "lat", "lng", "target_time_local",
                "timezone_name", "minutes_ahead", "vehicles_per_hour", "status",
                "thresholds", "nearby_congested"
            )}
            keep["type"] = "show_prediction_popup"
            out.append(keep)
            continue
        if t == "show_forecast_popup":
            cam_id = str(a.get("camera_id") or "").strip()
            if not cam_id:
                continue
            keep = {k: a.get(k) for k in (
                "camera_id", "camera_name", "lat", "lng", "timezone_name",
                "days", "start_date_local", "end_date_local",
                "daily_forecast", "nearby_congested"
            )}
            keep["type"] = "show_forecast_popup"
            out.append(keep)
            continue
    return out


def _ollama_base_url():
    return (os.environ.get("OLLAMA_URL") or "http://localhost:11434").strip().rstrip("/")


def _ollama_env_model():
    return str(os.environ.get("OLLAMA_MODEL") or "").strip()


def _ollama_resolve_model():
    global _ollama_cached_model, _ollama_cached_at
    now = time.time()
    if _ollama_cached_model and (now - _ollama_cached_at) < 300:
        return _ollama_cached_model
    env = _ollama_env_model()
    if env:
        _ollama_cached_model = env
        _ollama_cached_at = now
        return env
    # Auto-detect available model
    models = _ollama_fetch_models(cache_seconds=60)
    if not models:
        return None
    prefer = _ollama_sanitize_model_name("qwen2.5:3b")
    if prefer in models:
        _ollama_cached_model = prefer
    else:
        _ollama_cached_model = models[0]
    _ollama_cached_at = now
    return _ollama_cached_model


def _ollama_sanitize_model_name(name):
    return str(name or "").strip().lower().replace(":", "-")


def _ollama_reachable(timeout_s=3):
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(
            f"{_ollama_base_url()}/api/tags",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=timeout_s)
        return True
    except Exception:
        return False


def _ollama_fetch_models(timeout_s=4, force=False):
    global _ollama_cached_models, _ollama_cached_models_at
    now = time.time()
    cache_seconds = 120
    if force:
        cache_seconds = 0
    if not force and _ollama_cached_models and (now - _ollama_cached_models_at) < cache_seconds:
        return _ollama_cached_models

    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(
            f"{_ollama_base_url()}/api/tags",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        parsed = _safe_json_loads(raw) if raw else {}
        if not isinstance(parsed, dict):
            return _ollama_cached_models or []
        items = parsed.get("models") or []
        names = [_ollama_sanitize_model_name(m.get("name") or "") for m in items]
        _ollama_cached_models = [n for n in names if n]
        _ollama_cached_models_at = now
        return _ollama_cached_models
    except Exception:
        return _ollama_cached_models or []


def _safe_json_loads(text):
    import json
    try:
        return json.loads(text)
    except Exception:
        return {}


def _haversine_km(lat1, lon1, lat2, lon2):
    import math
    try:
        R = 6371.0
        lat1_f = float(lat1); lon1_f = float(lon1)
        lat2_f = float(lat2); lon2_f = float(lon2)
        dlat = math.radians(lat2_f - lat1_f)
        dlon = math.radians(lon2_f - lon1_f)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1_f)) * math.cos(math.radians(lat2_f)) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    except Exception:
        return None


def _geocode_place(query):
    import urllib.request
    import urllib.error
    global _geocode_cache
    q = str(query or "").strip()
    if not q:
        return None
    if q in _geocode_cache:
        return _geocode_cache[q]
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(q + ', Indonesia')}&format=json&limit=1&accept-language=id"
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "SmartTrafficAI/1.0")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = _safe_json_loads(resp.read().decode("utf-8", errors="ignore"))
        if isinstance(data, list) and data:
            first = data[0]
            result = {
                "name": q,
                "lat": float(first.get("lat") or 0),
                "lng": float(first.get("lon") or 0),
                "display_name": first.get("display_name", ""),
            }
            _geocode_cache[q] = result
            return result
    except Exception:
        pass
    return None


def _tokenize_query(q):
    import re
    stop = {"yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk", "ada", "adalah", "oleh", "pada", "akan", "sudah", "sedang", "lagi", "nya"}
    tokens = re.findall(r"[\w]+", q.lower())
    return [t for t in tokens if t not in stop and len(t) > 1]


def _normalize_text(text):
    import re
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _extract_origin_destination(text):
    import re
    q = str(text or "").strip()
    origin = None
    dest = None
    m = re.search(r"\bfrom\s+(.+?)\s+(?:to|tujuan|ke)\s+(.+)", q, re.IGNORECASE)
    if m:
        origin = _clean_place_phrase(m.group(1))
        dest = _clean_place_phrase(m.group(2))
    else:
        m2 = re.search(r"\b(dari|from)\s+(.+?)\s+(?:to|tujuan|ke)\s+(.+)", q, re.IGNORECASE)
        if m2:
            origin = _clean_place_phrase(m2.group(2))
            dest = _clean_place_phrase(m2.group(3))
        else:
            m3 = re.search(r"\b(?:to|tujuan|ke)\s+(.+?)(?:\s+(?:from|dari)|\s*$)", q, re.IGNORECASE)
            if m3:
                dest = _clean_place_phrase(m3.group(1))
    return origin, dest


def _clean_place_phrase(text):
    import re
    if not text:
        return ""
    t = re.sub(r"\b(ke|di|tujuan|ke\s+|di\s+)\s*", "", str(text).strip().lower())
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _live_camera_snapshot():
    from app.utils import load_config
    import app.globals as globals_state
    cfg = load_config() or []
    result = []
    for cam in cfg:
        cid = cam.get("id")
        if not cid:
            continue
        stats = {}
        try:
            stats = globals_state.global_stats.get(cid) or {}
        except Exception:
            pass
        count = int(stats.get("current_count") or 0)
        cam_status = str(stats.get("status") or "offline").strip().lower()
        if cam_status == "online":
            if count <= 3:
                cong = "LANCAR"
            elif count <= 8:
                cong = "PADAT LANCAR"
            elif count <= 15:
                cong = "MACET"
            else:
                cong = "MACET TOTAL"
        else:
            cong = "-"
        result.append({
            "camera_id": cid,
            "camera_name": cam.get("name") or cid,
            "lat": cam.get("lat"),
            "lng": cam.get("lng"),
            "current_count": count,
            "status": cam_status,
            "congestion": cong,
        })
    return result


def _top_cameras_30d(limit=5):
    from app.database import get_totals_by_camera, get_aggregated_stats, get_history_range
    import time
    cutoff = time.time() - (30 * 24 * 3600)
    all_totals = get_totals_by_camera(start_ts=cutoff) or {}
    rows = []
    for cid, t in all_totals.items():
        if isinstance(t, dict):
            rows.append({
                "camera_id": cid,
                "camera_name": t.get("camera_name") or cid,
                "total_30d": int(t.get("accumulated_count") or 0),
                "cars_30d": int(t.get("cars") or 0),
                "motors_30d": int(t.get("motorcycles") or 0),
            })
    rows.sort(key=lambda x: x.get("total_30d") or 0, reverse=True)
    return rows[:limit]


def _top_cameras_live_density(limit=5):
    snap = _live_camera_snapshot()
    online = [r for r in snap if str(r.get("status") or "") == "online"]
    online.sort(key=lambda x: x.get("current_count") or 0, reverse=True)
    return online[:limit]


# ── Route modules (scaffold for future migration) ──────────────────────────────
# These modules contain the same routes as routes.py but organized into separate files.
# They are NOT registered by default — routes.py is still the single source of truth.
# To enable modular routing in the future:
#   1. Remove conflicting routes from routes.py
#   2. Uncomment register_blueprints() in app/__init__.py
#
# Modular route files:
#   app/routes/auth.py         → login, logout, operator dashboard
#   app/routes/cameras.py     → camera management, stats, history, prediction
#   app/routes/enforcement.py → violations, zones, CRM, evidence, executive
#   app/routes/twitter.py     → Twitter/social search
#   app/routes/settings.py    → Settings page & runtime config API
#
# The blueprint registry below is prepared but NOT activated to avoid
# duplicate URL conflicts with routes.py (single source of truth).
#
# def register_blueprints(app):
#     from app.routes.auth import bp as auth_bp
#     from app.routes.cameras import bp as cameras_bp
#     from app.routes.enforcement import bp as enforce_bp
#     from app.routes.twitter import bp as twitter_bp
#     from app.routes.settings import bp as settings_bp
#     app.register_blueprint(auth_bp)
#     app.register_blueprint(cameras_bp)
#     app.register_blueprint(enforce_bp)
#     app.register_blueprint(twitter_bp)
#     app.register_blueprint(settings_bp)
