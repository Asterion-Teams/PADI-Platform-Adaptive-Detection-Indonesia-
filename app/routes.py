import os
import json
import time
import datetime
import csv
import io
import uuid
import re
import threading
import numpy as np
from collections import deque
from flask import Blueprint, render_template, Response, jsonify, request, g, stream_with_context, current_app, redirect, session
import urllib.request
import urllib.error
import urllib.parse
from app.config import DATA_DIR, CONFIG_FILE, HISTORY_MAX_LEN
from app.globals import CCTV_SOURCES
from app.services.camera import generate_frames, CameraAgent
from app.auth import admin_required, login_required
from app.csrf import csrf_required
from app.ratelimit import rate_limit
from app.validators import safe_camera_id, safe_string
from app.database import (
    predict_future_traffic,
    get_history_range,
    get_aggregated_stats,
    clear_all_history,
    get_total_lifetime,
    get_totals_by_camera,
    get_chat_profile,
    upsert_chat_profile,
    add_chat_message,
    get_recent_chat_messages,
    get_db_connection,
    # Case 1 - E-TLE support
    insert_zone,
    update_zone,
    delete_zone,
    get_zones_for_camera,
    get_all_zones,
    insert_violation,
    list_violations,
    get_violation,
    update_violation,
    violation_summary,
    violation_heatmap_by_camera,
    insert_crm_report,
    list_crm_reports,
    update_crm_report,
    crm_summary,
    recommend_enforcement_points,
)
from app.utils import backfill_camera_history, get_datalake_stats, load_config, save_config, save_stats, sync_stats_with_config
import app.globals as globals_state
from app.config import EVIDENCE_DIR, VIOLATION_TYPES, ZONE_TYPES
from app.services.enforcement import auto_classify_crm

bp = Blueprint('main', __name__)

_ollama_cached_model = None
_ollama_cached_at = 0.0
_ollama_cached_models = None
_ollama_cached_models_at = 0.0
_geocode_cache = {}
_ollama_last_success_at = 0.0
_chat_ctx_store = {}
_chat_ctx_lock = threading.Lock()
_PLANNER_ACTION_TYPES = {"navigate", "select_camera", "set_period", "show_prediction_popup", "show_forecast_popup"}

def _chat_grounded_mode_enabled():
    v = str(os.environ.get("CHAT_GROUNDED_MODE") or "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True

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
            keep = {k: a.get(k) for k in ("camera_id", "camera_name", "lat", "lng", "target_time_local", "timezone_name", "minutes_ahead", "vehicles_per_hour", "status", "thresholds", "nearby_congested")}
            keep["type"] = "show_prediction_popup"
            out.append(keep)
            continue
        if t == "show_forecast_popup":
            cam_id = str(a.get("camera_id") or "").strip()
            if not cam_id:
                continue
            keep = {k: a.get(k) for k in ("camera_id", "camera_name", "lat", "lng", "timezone_name", "days", "start_date_local", "end_date_local", "daily_forecast", "nearby_congested")}
            keep["type"] = "show_forecast_popup"
            out.append(keep)
            continue
    return out

def _ollama_base_url():
    return (os.environ.get("OLLAMA_URL") or "http://localhost:11434").strip().rstrip("/")

def _ollama_sanitize_model_name(name):
    s = str(name or "").strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()
    return s

def _ollama_env_model():
    return _ollama_sanitize_model_name(os.environ.get("OLLAMA_MODEL") or "")

def _ollama_fetch_models(timeout_s=4):
    base_url = _ollama_base_url()
    req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    parsed = json.loads(raw) if raw else {}
    models = []
    for m in (parsed.get("models") or []):
        name = (m or {}).get("name")
        if name:
            models.append(str(name))
    return models

def _ollama_get_models_cached(cache_seconds=30, timeout_s=4, force=False):
    global _ollama_cached_models, _ollama_cached_models_at
    now = time.time()
    if not force and _ollama_cached_models is not None and (now - float(_ollama_cached_models_at or 0)) < float(cache_seconds or 0):
        return list(_ollama_cached_models)
    models = _ollama_fetch_models(timeout_s=timeout_s)
    _ollama_cached_models = list(models or [])
    _ollama_cached_models_at = now
    return list(_ollama_cached_models)

def _ollama_resolve_model(prefer="mistral:7b", cache_seconds=60):
    global _ollama_cached_model, _ollama_cached_at

    env_model = _ollama_env_model()
    if env_model:
        try:
            models = _ollama_get_models_cached(cache_seconds=cache_seconds, timeout_s=2)
        except Exception:
            models = None
        if not models:
            return env_model
        if env_model in models:
            _ollama_cached_model = env_model
            _ollama_cached_at = time.time()
            return env_model

    now = time.time()
    if _ollama_cached_model and (now - float(_ollama_cached_at or 0)) < float(cache_seconds or 0):
        return _ollama_cached_model

    try:
        models = _ollama_get_models_cached(cache_seconds=cache_seconds, timeout_s=4, force=True)
    except Exception:
        models = []

    chosen = None
    if prefer and prefer in models:
        chosen = prefer
    elif "mistral:7b" in models:
        chosen = "mistral:7b"
    elif "qwen2.5:3b" in models:
        chosen = "qwen2.5:3b"
    elif models:
        chosen = models[0]

    _ollama_cached_model = chosen
    _ollama_cached_at = now
    return chosen

def _ollama_reachable(timeout_s=4):
    try:
        _ = _ollama_get_models_cached(cache_seconds=5, timeout_s=timeout_s, force=True)
        return True
    except Exception:
        return False

def _top_cameras_30d(limit=5):
    config = load_config() or []
    cam_ids = [c.get("id") for c in config if c.get("id")]
    if not cam_ids:
        return []
    cutoff = time.time() - (30 * 24 * 3600)
    totals = get_totals_by_camera(camera_ids=cam_ids, start_ts=cutoff) or {}
    rows = []
    for cam in config:
        cid = cam.get("id")
        if not cid:
            continue
        t = totals.get(cid) or {}
        rows.append(
            {
                "camera_id": cid,
                "camera_name": cam.get("name") or cid,
                "total_30d": int(t.get("accumulated_count") or 0),
                "cars_30d": int(t.get("cars") or 0),
                "motors_30d": int(t.get("motorcycles") or 0),
            }
        )
    rows.sort(key=lambda r: r.get("total_30d", 0), reverse=True)
    return rows[: int(limit or 5)]

def _top_cameras_live_density(limit=5):
    config = load_config() or []
    names = {c.get("id"): (c.get("name") or c.get("id")) for c in config if c.get("id")}
    sources = {}
    try:
        stats_path = os.path.join(DATA_DIR, "traffic_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f) or {}
            sources = (stats.get("sources") if isinstance(stats, dict) else {}) or {}
    except Exception:
        sources = {}
    rows = []
    if isinstance(sources, dict):
        for cam_id, s in sources.items():
            if not cam_id:
                continue
            rows.append(
                {
                    "camera_id": cam_id,
                    "camera_name": names.get(cam_id, cam_id),
                    "current_count": int((s or {}).get("current_count") or 0),
                    "status": (s or {}).get("status") or "-",
                }
            )
    rows.sort(key=lambda r: r.get("current_count", 0), reverse=True)
    return rows[: int(limit or 5)]

def _normalize_text(s):
    t = str(s or "").lower()
    t = re.sub(r"[^a-z0-9\s_-]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _tokenize_query(s):
    stop = {
        "coba",
        "tolong",
        "dong",
        "ya",
        "untuk",
        "di",
        "ke",
        "kedepan",
        "depan",
        "selama",
        "sekarang",
        "saat",
        "ini",
        "hari",
        "minggu",
        "bulan",
        "tahun",
        "menit",
        "mnt",
        "min",
        "minute",
        "minutes",
        "jam",
        "hour",
        "hrs",
        "hr",
        "traffic",
        "traffict",
        "kendaraan",
        "prediksi",
        "predict",
        "forecast",
        "daerah",
        "lokasi",
        "titik",
        "tempat",
        "area",
    }
    t = _normalize_text(s)
    toks = []
    for w in t.split():
        if w in stop:
            continue
        if w.isdigit():
            continue
        if len(w) < 3:
            continue
        toks.append(w)
    return toks

def _find_best_cameras(query, limit=5):
    q_norm = _normalize_text(query)
    q_tokens = _tokenize_query(q_norm)
    if not q_norm and not q_tokens:
        return []

    config = load_config() or []
    scored = []
    for cam in config:
        name = cam.get("name") or cam.get("id") or ""
        name_norm = _normalize_text(name)
        if not name_norm:
            continue

        score = 0
        if q_norm and q_norm in name_norm:
            score += 10
        for tok in q_tokens:
            if tok in name_norm:
                score += 3

        if score <= 0:
            continue

        scored.append((score, -len(name_norm), cam))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [c for _, __, c in scored[: int(limit or 5)]]

def _clean_place_phrase(s):
    t = str(s or "").strip()
    t = re.sub(r"[\n\r\t]+", " ", t)
    t = re.sub(r"[\"'`]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""
    cut_words = [
        " kalau ",
        " kalo ",
        " jika ",
        " bila ",
        " ketika ",
        " saat ",
        " sekarang ",
        " hari ini ",
        " barusan ",
        " tadi ",
        " besok ",
        " lusa ",
        " seperti ",
        " kayak ",
        " kaya ",
        " gimana ",
        " bagaimana ",
        " gimana sih ",
        " seperti apa ",
        " apakah ",
        " gimana ",
        " bagaimana ",
        " macet ",
        " padat ",
        " lancar ",
        " ramai ",
        " senggang ",
        " sepi ",
        " rute ",
        " rekomendasi ",
        " jalur ",
        " arah ",
        " menuju ",
        " terus ",
        " lalu ",
        " kemudian ",
        " lewat ",
        " via ",
        " dan ",
        " yang ",
        " karena ",
        " sebab ",
        " agar ",
    ]
    lowered = " " + t.lower() + " "
    cut_idx = None
    for w in cut_words:
        i = lowered.find(w)
        if i >= 0:
            cut_idx = i if cut_idx is None else min(cut_idx, i)
    if cut_idx is not None:
        t = t[:cut_idx].strip(" ,.;:-")
    return t.strip(" ,.;:-")

def _extract_origin_destination(user_text):
    q = str(user_text or "").strip()
    if not q:
        return None, None
    q_norm = _normalize_text(q)

    m = re.search(r"\bdari\s+(.+?)\s+ke\s+(.+)$", q_norm)
    if m:
        origin = _clean_place_phrase(m.group(1))
        dest = _clean_place_phrase(m.group(2))
        return origin or None, dest or None

    m = re.search(r"\bke\s+(.+)$", q_norm)
    if m:
        dest = _clean_place_phrase(m.group(1))
        return None, dest or None

    m = re.search(r"\bdi\s+(.+)$", q_norm)
    if m:
        dest = _clean_place_phrase(m.group(1))
        return None, dest or None

    m = re.search(r"\bmenuju\s+(.+)$", q_norm)
    if m:
        dest = _clean_place_phrase(m.group(1))
        return None, dest or None

    return None, None

def _geocode_place(place, timeout_s=6, ttl_s=600):
    q = str(place or "").strip()
    if not q:
        return None
    key = _normalize_text(q)
    now = time.time()
    cached = _geocode_cache.get(key)
    if cached and (now - float(cached.get("ts") or 0)) < float(ttl_s or 0):
        return cached.get("value")

    try:
        url = (
            "https://nominatim.openstreetmap.org/search?format=json&limit=1"
            f"&countrycodes=id&addressdetails=0&q={urllib.parse.quote(q)}"
        )
        req = urllib.request.Request(
            url,
            method="GET",
            headers={
                "Accept": "application/json",
                "User-Agent": "SmartTrafficAI/1.0 (route-advice; contact=local)",
            },
        )
        with urllib.request.urlopen(req, timeout=float(timeout_s or 6)) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        parsed = json.loads(raw) if raw else []
        first = parsed[0] if isinstance(parsed, list) and parsed else None
        if not first:
            _geocode_cache[key] = {"ts": now, "value": None}
            return None
        lat = first.get("lat")
        lon = first.get("lon")
        if lat is None or lon is None:
            _geocode_cache[key] = {"ts": now, "value": None}
            return None
        value = {
            "lat": float(lat),
            "lng": float(lon),
            "name": str(first.get("display_name") or q),
        }
        _geocode_cache[key] = {"ts": now, "value": value}
        return value
    except Exception:
        _geocode_cache[key] = {"ts": now, "value": None}
        return None

def _haversine_km(lat1, lon1, lat2, lon2):
    try:
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)
    except Exception:
        return None
    r = 6371.0
    p = 3.141592653589793 / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (pow(__import__("math").sin(dlat / 2), 2) +
         __import__("math").cos(lat1 * p) * __import__("math").cos(lat2 * p) * pow(__import__("math").sin(dlon / 2), 2))
    c = 2 * __import__("math").asin(min(1.0, __import__("math").sqrt(a)))
    return r * c

def _load_thresholds():
    thresholds_path = os.path.join(DATA_DIR, "camera_thresholds.json")
    if not os.path.exists(thresholds_path):
        return {}
    try:
        with open(thresholds_path, "r") as f:
            data = json.load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _congestion_label(count, thr):
    try:
        c = int(count or 0)
    except Exception:
        c = 0
    t = thr or {}
    p50 = int(t.get("p50", 100) or 100)
    p75 = int(t.get("p75", 200) or 200)
    p90 = int(t.get("p90", 300) or 300)
    if c > p90:
        return "MACET TOTAL"
    if c > p75:
        return "MACET"
    if c > p50:
        return "PADAT LANCAR"
    return "LANCAR"

def _live_camera_snapshot():
    config = load_config() or []
    cfg_by_id = {c.get("id"): c for c in config if c.get("id")}
    thresholds = _load_thresholds()
    sources = {}
    try:
        stats_path = os.path.join(DATA_DIR, "traffic_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f) or {}
            sources = (stats.get("sources") if isinstance(stats, dict) else {}) or {}
    except Exception:
        sources = {}

    rows = []
    if isinstance(sources, dict):
        for cam_id, s in sources.items():
            if not cam_id:
                continue
            cfg = cfg_by_id.get(cam_id) or {}
            lat = cfg.get("lat")
            lng = cfg.get("lng")
            cur = int((s or {}).get("current_count") or 0)
            rows.append(
                {
                    "camera_id": cam_id,
                    "camera_name": cfg.get("name") or cam_id,
                    "lat": lat,
                    "lng": lng,
                    "current_count": cur,
                    "status": (s or {}).get("status") or "-",
                    "congestion": _congestion_label(cur, thresholds.get(cam_id)),
                }
            )
    return rows


# ---- Authentication ----

@bp.route("/login", methods=["GET", "POST"])
def login_page():
    from app.auth import check_login, ensure_demo_user_session, is_demo_auth_bypass
    if is_demo_auth_bypass():
        user = ensure_demo_user_session() or {"role": "admin"}
        return redirect("/dashboard" if user.get("role") == "admin" else "/operator")

    if request.method == "GET":
        if session.get("user"):
            role = session["user"].get("role")
            return redirect("/dashboard" if role == "admin" else "/operator")
        return render_template("login.html")
    # POST
    data = request.form if request.form else (request.get_json(silent=True) or {})
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    user = check_login(username, password)
    if not user:
        if request.is_json:
            return jsonify({"status": "error", "message": "Username atau password salah"}), 401
        return render_template("login.html", error="Username atau password salah")
    session["user"] = user
    if request.is_json:
        return jsonify({"status": "success", "user": user})
    return redirect("/dashboard" if user["role"] == "admin" else "/operator")


@bp.route("/logout")
def logout_page():
    session.pop("user", None)
    return redirect("/login")


@bp.route("/operator")
def operator_page():
    return render_template("operator.html")


@bp.route("/livemap")
def livemap_page():
    """Live map showing all registered CCTV cameras with real-time density."""
    return render_template("livemap.html")


@bp.route("/ocr-test")
def ocr_test_page():
    return render_template("ocr_test.html")


@bp.route("/")
def index():
    return render_template("index.html")

@bp.route("/dashboard")
def dashboard():
    from app.auth import get_current_user
    user = get_current_user()
    if not user or user.get("role") != "admin":
        return redirect("/login")
    return render_template("dashboard.html")

@bp.route("/documentation")
def documentation():
    from app.auth import get_current_user
    if not session.get("user") or session["user"].get("role") != "admin":
        return redirect("/login")
    return render_template("documentation.html")

@bp.route("/analysis")
def analysis():
    if not session.get("user") or session["user"].get("role") != "admin":
        return redirect("/login")
    return render_template("analysis.html")

@bp.route("/video_feed")
@bp.route("/video_feed/<camera_id>")
def video_feed(camera_id=None):
    if camera_id is None:
        # Default to active source if available, else first source
        if CCTV_SOURCES:
            if isinstance(CCTV_SOURCES, list) and len(CCTV_SOURCES) > 0:
                active = next((s for s in CCTV_SOURCES if s.get("active") is True), None)
                camera_id = (active or CCTV_SOURCES[0]).get("id")
            elif isinstance(CCTV_SOURCES, dict):
                camera_id = list(CCTV_SOURCES.keys())[0]
        
        if not camera_id:
              return "No sources configured", 404
             
    return Response(generate_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route("/api/sources")
def get_sources():
    # Helper to return camera config
    return jsonify(CCTV_SOURCES)

@bp.route("/api/add_camera", methods=["POST"])
@admin_required
@csrf_required
@rate_limit(limit_name="camera_add")
def add_camera():
    try:
        data = request.json or {}

        name = str(data.get("name") or "").strip()
        url = str(data.get("url") or "").strip()
        if not name or not url:
            return jsonify({"status": "error", "message": "Missing name or url"}), 400

        lat_raw = str(data.get("lat") or "").strip()
        lng_raw = str(data.get("lng") or "").strip()
        external_id = str(data.get("external_id") or "").strip()
        try:
            lat = float(lat_raw) if lat_raw else None
            lng = float(lng_raw) if lng_raw else None
        except Exception:
            return jsonify({"status": "error", "message": "Invalid lat/lng"}), 400

        config = load_config()
        new_id = f"cam_{uuid.uuid4().hex[:10]}"
        existing_ids = {c.get("id") for c in config}
        while new_id in existing_ids:
            new_id = f"cam_{uuid.uuid4().hex[:10]}"

        cam = {"id": new_id, "name": name, "url": url, "lat": lat, "lng": lng,
               "active": True if not config else False}
        if external_id:
            cam["external_id"] = external_id
        if not config:
            cam["active"] = True
        config.append(cam)
        if not save_config(config):
            return jsonify({"status": "error", "message": "Failed to save config"}), 500

        CCTV_SOURCES[:] = config
        if new_id not in globals_state.global_stats:
            globals_state.global_stats[new_id] = {
                "name": name,
                "current_count": 0,
                "current_class_counts": {"0": 0, "1": 0},
                "accumulated_count": 0,
                "accumulated_class_counts": {"0": 0, "1": 0},
                "history": deque(maxlen=HISTORY_MAX_LEN),
            }
        save_stats()

        # Start camera agent for the new camera immediately
        try:
            from app.services.camera import CameraAgent
            if new_id not in globals_state.camera_agents:
                agent = CameraAgent(cam, globals_state.yolo_model_instance)
                globals_state.camera_agents[new_id] = agent
                agent.start()
        except Exception as e:
            print(f"[WARN] Failed to start agent for {new_id}: {e}")

        return jsonify({"status": "success", "id": new_id})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.route("/api/delete_camera", methods=["POST"])
@admin_required
@csrf_required
@rate_limit(limit_name="camera_delete")
def delete_camera():
    try:
        data = request.json or {}

        target_id = data.get("id")
        if not target_id:
            return jsonify({"status": "error", "message": "Missing id"}), 400

        config = load_config()
        before = len(config)
        was_active = any(c.get("id") == target_id and c.get("active") for c in config)
        config = [c for c in config if c.get("id") != target_id]

        if len(config) == before:
            return jsonify({"status": "error", "message": "Camera not found"}), 404

        if was_active and config:
            for cam in config:
                cam["active"] = False
            config[0]["active"] = True

        if not save_config(config):
            return jsonify({"status": "error", "message": "Failed to save config"}), 500

        CCTV_SOURCES[:] = config

        # Stop the camera agent FIRST before removing stats
        # to avoid KeyError race condition in the running thread
        try:
            from app.services.camera import stop_agent
            stop_agent(target_id)
        except Exception:
            pass

        # Small delay to let the thread fully exit its loop
        import time
        time.sleep(0.3)

        if target_id in globals_state.global_stats:
            del globals_state.global_stats[target_id]
            save_stats()
        sync_stats_with_config()

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ─── External Camera Source (mugnimaestra.dev) ─────────────────────────────────

# Server-side cache so browser doesn't need direct external API access
_EXT_CAMERAS_CACHE = {"data": None, "fetched_at": 0.0}
_EXT_CACHE_TTL = 600  # 10 minutes
_EXT_CACHE_LOCK = threading.Lock()  # Prevent race conditions on concurrent requests

@bp.route("/api/external_cameras", methods=["GET"])
def get_external_cameras():
    """Fetch camera list from external API (mugnimaestra.dev).

    Results are cached server-side for 10 minutes.
    Uses thread lock to prevent race conditions on concurrent requests.
    Only caches successful responses — errors are returned immediately.
    """
    import urllib.request
    import urllib.error as urllib_err

    now = time.time()

    # Fast path: return cached data if valid (within lock for consistency)
    with _EXT_CACHE_LOCK:
        if _EXT_CAMERAS_CACHE["data"] and (now - _EXT_CAMERAS_CACHE["fetched_at"]) < _EXT_CACHE_TTL:
            return jsonify(_EXT_CAMERAS_CACHE["data"])

    # Slow path: fetch from external API (outside lock so other requests can read cache)
    try:
        external_url = os.environ.get(
            "EXTERNAL_CAMERA_API",
            "https://streetside.mugnimaestra.dev/api/cameras"
        )

        req = urllib.request.Request(
            external_url,
            headers={"User-Agent": "Mozilla/5.0 (SmartTraffic-PADI/1.0)", "Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            import json as _json
            data = _json.loads(raw) if raw else {}

        cameras = data.get("cameras", []) or []
        result = []

        for c in cameras:
            # GeoJSON format: {"geometry": {...}, "properties": {...}}
            # OR flat format: {"external_id": ..., "name": ..., "snapshot_url": ...}
            if isinstance(c, dict) and "properties" in c:
                props = c["properties"] or {}
                geom = c.get("geometry") or {}
                coords = geom.get("coordinates") or []
                lat = coords[1] if len(coords) > 1 else None
                lng = coords[0] if len(coords) > 0 else None
            else:
                props = c
                lat = props.get("latitude")
                lng = props.get("longitude")

            ext_id = str(props.get("external_id") or props.get("id") or "")
            name = props.get("name") or props.get("cctv_name") or f"Camera {ext_id}"
            city = props.get("city") or props.get("city_name") or ""
            address = props.get("address") or ""
            # image1 is the snapshot path from mugnimaestra.dev (e.g. "/img/a/Bendungan-Hilir-001/preview.jpg?token=...")
            image_path = props.get("snapshot_url") or props.get("image1") or ""
            stream_url = props.get("stream_url") or ""

            # Build full snapshot URL
            base_url = os.environ.get(
                "EXTERNAL_BASE_URL", "https://compe.f-mc.my.id/hlsproxy"
            )
            snapshot_url = (base_url + image_path) if image_path and not image_path.startswith("http") else image_path

            # Build HLS stream URL from the same token in image1
            # Pattern: /img/a/{NAME}/preview.jpg?token=... → /stream/a/{NAME}/video.m3u8?token=...
            hls_url = ""
            if image_path:
                try:
                    # Extract camera name and token from image1 path
                    # e.g. "/img/a/Paseban-001/preview.jpg?token=abc123" → "/stream/a/Paseban-001/video.m3u8?token=abc123"
                    if "/img/a/" in image_path:
                        parts = image_path.split("/img/a/")
                        if len(parts) > 1:
                            rest = parts[1]
                            cam_name = rest.split("/")[0]
                            token_match = rest.split("?token=")
                            token = token_match[1] if len(token_match) > 1 else ""
                            hls_url = f"{base_url}/stream/a/{cam_name}/video.m3u8?token={token}"
                except Exception:
                    hls_url = ""

            result.append({
                "external_id": ext_id,
                "name": name,
                "city": city,
                "address": address,
                "latitude": lat,
                "longitude": lng,
                "snapshot_url": snapshot_url,
                "stream_url": stream_url or hls_url,
                "hls_url": hls_url,
            })

        resp_data = {
            "status": "success",
            "total": len(result),
            "cameras": result,
        }
        # Cache only successful responses (inside lock)
        with _EXT_CACHE_LOCK:
            _EXT_CAMERAS_CACHE["data"] = resp_data
            _EXT_CAMERAS_CACHE["fetched_at"] = time.time()
        return jsonify(resp_data)
    except urllib_err.HTTPError as e:
        return jsonify({"status": "error", "message": f"HTTP {e.code}: {e.reason}"}), 502
    except urllib_err.URLError as e:
        return jsonify({"status": "error", "message": f"Connection failed: {e.reason}"}), 503
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ─── HLS Proxy (adds CORS headers so the public live map can play streams) ─────

# The external HLS host does not send Access-Control-Allow-Origin, so browsers
# block direct playback from livemap. Restrict the proxy to known hosts only.
_HLS_ALLOWED_HOSTS = {"compe.f-mc.my.id"}


@bp.route("/api/hls_proxy", methods=["GET"])
def hls_proxy():
    """Proxy HLS manifests (.m3u8) and segments (.ts) from the external
    streaming server, adding CORS headers so the public live map can play them.

    For manifests, segment/key URIs are rewritten to route back through this
    proxy so every request carries the proper CORS headers.
    """
    target = request.args.get("url", "")
    if not target:
        return jsonify({"status": "error", "message": "missing url"}), 400

    parsed = urllib.parse.urlparse(target)
    if parsed.scheme not in ("http", "https") or parsed.hostname not in _HLS_ALLOWED_HOSTS:
        return jsonify({"status": "error", "message": "host not allowed"}), 403

    try:
        req = urllib.request.Request(
            target,
            headers={
                "User-Agent": "Mozilla/5.0 (SmartTraffic-PADI/1.0)",
                "Accept": "*/*",
            },
        )
        upstream = urllib.request.urlopen(req, timeout=15)
        ctype = upstream.headers.get("Content-Type", "")
        body = upstream.read()
    except urllib.error.HTTPError as e:
        return jsonify({"status": "error", "message": f"HTTP {e.code}"}), 502
    except urllib.error.URLError as e:
        return jsonify({"status": "error", "message": f"Connection failed: {e.reason}"}), 503
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    is_manifest = target.split("?")[0].endswith(".m3u8") or "mpegurl" in ctype.lower()

    if is_manifest:
        text = body.decode("utf-8", errors="ignore")
        # Carry the manifest token over to segments that lack their own query.
        token = urllib.parse.parse_qs(parsed.query).get("token", [""])[0]

        def rewrite_uri(uri):
            uri = uri.strip()
            if not uri:
                return uri
            absolute = urllib.parse.urljoin(target, uri)
            if not urllib.parse.urlparse(absolute).query and token:
                absolute = absolute + "?token=" + urllib.parse.quote(token)
            return "/api/hls_proxy?url=" + urllib.parse.quote(absolute, safe="")

        out_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                out_lines.append(line)
            elif stripped.startswith("#"):
                # Rewrite URI="..." attributes (EXT-X-KEY, EXT-X-MAP, etc.)
                if 'URI="' in stripped:
                    line = re.sub(
                        r'URI="([^"]+)"',
                        lambda m: 'URI="' + rewrite_uri(m.group(1)) + '"',
                        line,
                    )
                out_lines.append(line)
            else:
                out_lines.append(rewrite_uri(stripped))
        resp = Response("\n".join(out_lines), content_type="application/vnd.apple.mpegurl")
    else:
        resp = Response(body, content_type=ctype or "video/mp2t")

    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@bp.route("/api/external_cameras/test", methods=["GET"])
@login_required
def test_external_camera():
    """Test a single external camera URL — fetch one frame to verify it works."""
    try:
        import urllib.request
        import urllib.error as urllib_err
        import time

        url = request.args.get("url", "")
        if not url:
            return jsonify({"status": "error", "message": "Missing url"}), 400

        start = time.time()
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (SmartTraffic-PADI/1.0)",
                "Range": "bytes=0-8192",  # Only first 8KB to check if image
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                status = resp.status
                content_type = resp.headers.get("Content-Type", "")
                body = resp.read(8192)
        except urllib_err.HTTPError as e:
            return jsonify({
                "status": "error",
                "code": e.code,
                "message": f"HTTP {e.code}: {e.reason}",
            }), e.code

        latency_ms = int((time.time() - start) * 1000)
        is_image = "image" in content_type.lower() or body[:2] in (b"\xff\xd8", b"\x89PNG")

        return jsonify({
            "status": "success" if status == 200 else "error",
            "http_status": status,
            "content_type": content_type,
            "latency_ms": latency_ms,
            "is_image": bool(is_image),
            "preview_size": len(body),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.route("/add_camera", methods=["GET"])
@admin_required
def add_camera_page():
    """Browse and select cameras from external API to add to SmartTraffic."""
    return render_template("add_camera.html")

@bp.route("/api/switch_source", methods=["POST"])
@admin_required
def switch_source():
    try:
        data = request.json
        new_id = data.get("id")
        
        # Update in-memory
        found = False
        for source in CCTV_SOURCES:
            if source["id"] == new_id:
                source["active"] = True
                found = True
            else:
                source["active"] = False
        
        if not found:
             return jsonify({"status": "error", "message": "Source not found"}), 404

        # Persist to config
        config_path = os.path.join(DATA_DIR, 'cctv_config.json')
        with open(config_path, 'w') as f:
            json.dump(CCTV_SOURCES, f, indent=4)
            
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/replay_camera", methods=["POST"])
@admin_required
def replay_camera():
    """Reset a video file camera back to frame 0 (replay from start)."""
    try:
        data = request.json or {}
        target_id = data.get("id")
        if not target_id:
            return jsonify({"status": "error", "message": "Missing camera id"}), 400

        from app.globals import camera_agents
        agent = camera_agents.get(target_id)
        if agent is None:
            return jsonify({"status": "error", "message": "Camera agent not found"}), 404

        if not agent._is_local_video():
            return jsonify({"status": "error", "message": "Replay only available for video files (MP4/AVI/etc)"}), 400

        # Reset video to frame 0
        if agent.cap is not None:
            try:
                import cv2 as _cv2
                agent.cap.set(_cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception:
                pass

        # Also reset enforcement state so violations are re-evaluated fresh
        try:
            agent.enforcement._tracked.clear()
        except Exception:
            pass

        return jsonify({"status": "success", "message": "Video replaying from start"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.route("/api/history")
@rate_limit(limit_name="history")
def get_history_api():
    period = request.args.get("period", "30m")
    # Validate camera_id safely
    camera_id_raw = request.args.get("camera_id")
    camera_id = safe_camera_id(camera_id_raw) if camera_id_raw else None
    start_ts_arg = request.args.get("start_ts")
    end_ts_arg = request.args.get("end_ts")
    
    now = time.time()
    start_ts = now - 1800 # Default 30m
    end_ts = None
    interval = 60
    
    if period == "30m":
        start_ts = now - 1800
        interval = 60 # 1 min
    elif period == "1h":
        start_ts = now - 3600
        interval = 60 # 1 min
    elif period == "6h":
        start_ts = now - (6 * 3600)
        interval = 300 # 5 min
    elif period == "12h":
        start_ts = now - (12 * 3600)
        interval = 900 # 15 min
    elif period == "24h":
        start_ts = now - (24 * 3600)
        interval = 1800 # 30 min
    elif period == "7d":
        start_ts = now - (7 * 24 * 3600)
        interval = 14400 # 4 hours
    elif period == "30d":
        start_ts = now - (30 * 24 * 3600)
        interval = 86400 # 1 day
    elif period == "custom":
        if start_ts_arg:
            start_ts = float(start_ts_arg)
        if end_ts_arg:
            end_ts = float(end_ts_arg)
        elif start_ts_arg:
            end_ts = start_ts + 86400
        interval = 3600

    if start_ts_arg and period != "custom":
        try:
            start_ts = float(start_ts_arg)
        except Exception:
            pass
    if end_ts_arg and period != "custom":
        try:
            end_ts = float(end_ts_arg)
        except Exception:
            pass
        
    rows = get_history_range(camera_id=camera_id, start_ts=start_ts, end_ts=end_ts)
    
    # Fetch actual violations
    viol_rows = []
    try:
        from app.database import get_db_connection, _fetchall
        conn = get_db_connection(timeout_s=2)
        params = []
        conds = []
        if start_ts:
            conds.append("timestamp >= ?")
            params.append(start_ts)
        if end_ts:
            conds.append("timestamp <= ?")
            params.append(end_ts)
        if camera_id:
            conds.append("camera_id = ?")
            params.append(str(camera_id))
        
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        query = f"SELECT timestamp FROM violations {where}"
        viol_rows = _fetchall(conn, query, params)
        conn.close()
    except Exception as e:
        print(f"Error fetching violations for history: {e}")
    
    # Aggregate
    buckets = {}
    for r in rows:
        ts = r["ts"]
        # Align to interval
        bucket_ts = int(ts // interval) * interval
        if bucket_ts not in buckets:
            buckets[bucket_ts] = {
                "count": 0,
                "cars": 0,
                "motors": 0,
                "violations": 0,
                "density_sum": 0,
                "density_n": 0,
                "density_peak": 0,
                "cam_ids": set(),
            }
        cam_id = r.get("camera_id")
        if cam_id:
            buckets[bucket_ts]["cam_ids"].add(cam_id)
        buckets[bucket_ts]["count"] += int(r.get("new_count") or 0)
        buckets[bucket_ts]["cars"] += int(r.get("new_cars") or 0)
        buckets[bucket_ts]["motors"] += int(r.get("new_motors") or 0)

        dens = int(r.get("count") or 0)
        buckets[bucket_ts]["density_sum"] += dens
        buckets[bucket_ts]["density_n"] += 1
        if dens > buckets[bucket_ts]["density_peak"]:
            buckets[bucket_ts]["density_peak"] = dens
            
    for v in viol_rows:
        ts = v["timestamp"]
        bucket_ts = int(ts // interval) * interval
        if bucket_ts not in buckets:
            buckets[bucket_ts] = {
                "count": 0, "cars": 0, "motors": 0, "violations": 0,
                "density_sum": 0, "density_n": 0, "density_peak": 0, "cam_ids": set()
            }
        buckets[bucket_ts]["violations"] = buckets[bucket_ts].get("violations", 0) + 1
        
    # Format for Chart.js
    sorted_ts = sorted(buckets.keys())
    data = []
    for ts in sorted_ts:
        dt = datetime.datetime.fromtimestamp(ts)
        if period in ["30d", "7d"]:
            label = dt.strftime("%d/%m")
        else:
            label = dt.strftime("%H:%M")

        b = buckets[ts]
        density_n = b.get("density_n", 0) or 0
        density_avg = int(round((b.get("density_sum", 0) or 0) / density_n)) if density_n > 0 else 0
        active_cams = len(b.get("cam_ids") or [])
        denom = active_cams if active_cams > 0 else 1
            
        data.append({
            "label": label,
            "count": b["count"],
            "cars": b["cars"],
            "motors": b["motors"],
            "violations": b.get("violations", 0),
            "count_per_cam": int(round((b["count"] or 0) / denom)),
            "cars_per_cam": int(round((b["cars"] or 0) / denom)),
            "motors_per_cam": int(round((b["motors"] or 0) / denom)),
            "active_cams": int(active_cams),
            "density_avg": density_avg,
            "density_peak": int(b.get("density_peak", 0) or 0),
            "ts": ts
        })
        
    return jsonify(data)

@bp.route("/api/export_csv")
def export_csv_api():
    try:
        period = request.args.get("period", "30m")
        camera_id = request.args.get("camera_id")
        start_ts_arg = request.args.get("start_ts")
        end_ts_arg = request.args.get("end_ts")

        now = time.time()
        start_ts = None
        end_ts = None
        periods = {
            "30m": 1800,
            "1h": 3600,
            "6h": 6 * 3600,
            "12h": 12 * 3600,
            "24h": 24 * 3600,
            "7d": 7 * 24 * 3600,
            "30d": 30 * 24 * 3600,
        }

        if period == "custom":
            if start_ts_arg:
                start_ts = float(start_ts_arg)
            if end_ts_arg:
                end_ts = float(end_ts_arg)
            elif start_ts is not None:
                end_ts = start_ts + 86400
        else:
            seconds = periods.get(period, 1800)
            start_ts = now - seconds
            if start_ts_arg:
                try:
                    start_ts = float(start_ts_arg)
                except Exception:
                    pass
            if end_ts_arg:
                try:
                    end_ts = float(end_ts_arg)
                except Exception:
                    pass

        rows = get_history_range(camera_id=camera_id, start_ts=start_ts, end_ts=end_ts)
        name_by_id = {c.get("id"): c.get("name") for c in load_config()}

        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["timestamp", "camera_id", "camera_name", "new_count", "new_cars", "new_motors", "total_count", "car_count", "motorcycle_count"])
        for r in rows:
            ts = r.get("ts")
            w.writerow([
                datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "",
                r.get("camera_id") or "",
                name_by_id.get(r.get("camera_id"), ""),
                int(r.get("new_count") or 0),
                int(r.get("new_cars") or 0),
                int(r.get("new_motors") or 0),
                int(r.get("count") or 0),
                int(r.get("cars") or 0),
                int(r.get("motors") or 0),
            ])

        csv_data = out.getvalue()
        filename = f"traffic_export_{period}_{int(time.time())}.csv"
        resp = Response(csv_data, mimetype="text/csv; charset=utf-8")
        resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return resp
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.route("/export/csv")
def export_csv():
    def parse_dt(s):
        if not s:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.datetime.strptime(s, fmt)
            except Exception:
                continue
        return None

    try:
        camera_id = request.args.get("camera_id")
        start_s = request.args.get("start")
        end_s = request.args.get("end")

        start_dt = parse_dt(start_s)
        end_dt = parse_dt(end_s)
        now = datetime.datetime.now()

        if not start_dt and not end_dt:
            end_dt = now
            start_dt = now - datetime.timedelta(hours=24)
        elif start_dt and not end_dt:
            end_dt = start_dt + datetime.timedelta(hours=24)
        elif end_dt and not start_dt:
            start_dt = end_dt - datetime.timedelta(hours=24)

        start_ts = start_dt.timestamp() if start_dt else None
        end_ts = end_dt.timestamp() if end_dt else None

        rows = get_history_range(camera_id=camera_id, start_ts=start_ts, end_ts=end_ts)
        name_by_id = {c.get("id"): c.get("name") for c in load_config()}

        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["timestamp", "camera_id", "camera_name", "new_count", "new_cars", "new_motors", "total_count", "car_count", "motorcycle_count"])
        for r in rows:
            ts = r.get("ts")
            w.writerow([
                datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "",
                r.get("camera_id") or "",
                name_by_id.get(r.get("camera_id"), ""),
                int(r.get("new_count") or 0),
                int(r.get("new_cars") or 0),
                int(r.get("new_motors") or 0),
                int(r.get("count") or 0),
                int(r.get("cars") or 0),
                int(r.get("motors") or 0),
            ])

        csv_data = out.getvalue()
        filename = f"traffic_export_{start_dt.strftime('%Y%m%d_%H%M')}_{end_dt.strftime('%Y%m%d_%H%M')}.csv"
        resp = Response(csv_data, mimetype="text/csv; charset=utf-8")
        resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return resp
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.route("/metrics")
def metrics():
    lifetime = get_total_lifetime()
    monthly = get_aggregated_stats(days=30)
    cameras_total = len(CCTV_SOURCES) if isinstance(CCTV_SOURCES, list) else 0
    lines = [
        "# TYPE smarttraffic_cameras_total gauge",
        f"smarttraffic_cameras_total {cameras_total}",
        "# TYPE smarttraffic_lifetime_total counter",
        f"smarttraffic_lifetime_total {int(lifetime.get('accumulated_count', 0))}",
        "# TYPE smarttraffic_lifetime_cars counter",
        f"smarttraffic_lifetime_cars {int(lifetime.get('cars', 0))}",
        "# TYPE smarttraffic_lifetime_motorcycles counter",
        f"smarttraffic_lifetime_motorcycles {int(lifetime.get('motorcycles', 0))}",
        "# TYPE smarttraffic_monthly_total counter",
        f"smarttraffic_monthly_total {int(monthly.get('accumulated_count', 0))}",
        "# TYPE smarttraffic_monthly_cars counter",
        f"smarttraffic_monthly_cars {int(monthly.get('cars', 0))}",
        "# TYPE smarttraffic_monthly_motorcycles counter",
        f"smarttraffic_monthly_motorcycles {int(monthly.get('motorcycles', 0))}",
    ]
    return Response("\n".join(lines) + "\n", mimetype="text/plain; version=0.0.4; charset=utf-8")

@bp.route("/api/stats")
@rate_limit(limit_name="stats")
def get_stats():
    # Return traffic stats
    try:
        stats_path = os.path.join(DATA_DIR, 'traffic_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                data = json.load(f)
            
            # Optimization: Remove heavy history arrays from response
            if 'sources' in data:
                for s_id in data['sources']:
                    if 'history' in data['sources'][s_id]:
                        del data['sources'][s_id]['history']
            
            # Add density level per camera
            if isinstance(data.get("sources"), dict):
                for s_id, s_data in data["sources"].items():
                    count = int(s_data.get("current_count") or 0)
                    s_data["density"] = _get_density_level(count)

            # Add Monthly Aggregated Stats
            monthly = get_aggregated_stats(days=30)
            data['global_monthly'] = monthly

            try:
                lifetime = get_total_lifetime()
                if "global_total" not in data or not isinstance(data.get("global_total"), dict):
                    data["global_total"] = {}
                data["global_total"]["accumulated_count"] = int(lifetime.get("accumulated_count", 0) or 0)
                data["global_total"]["cars"] = int(lifetime.get("cars", 0) or 0)
                data["global_total"]["motorcycles"] = int(lifetime.get("motorcycles", 0) or 0)
            except Exception:
                pass

            try:
                cfg = load_config() or []
                all_ids = [c.get("id") for c in cfg if c.get("id")]
                totals_all = get_totals_by_camera(camera_ids=all_ids) if all_ids else get_totals_by_camera()
                sources = data.get("sources") if isinstance(data.get("sources"), dict) else None
                if isinstance(sources, dict) and isinstance(totals_all, dict):
                    for cam_id, t in totals_all.items():
                        if cam_id not in sources or not isinstance(sources.get(cam_id), dict):
                            continue
                        sources[cam_id]["accumulated_count"] = int((t or {}).get("accumulated_count") or 0)
                        acc = sources[cam_id].get("accumulated_class_counts")
                        if not isinstance(acc, dict):
                            acc = {"0": 0, "1": 0}
                            sources[cam_id]["accumulated_class_counts"] = acc
                        acc["0"] = int((t or {}).get("cars") or 0)
                        acc["1"] = int((t or {}).get("motorcycles") or 0)
            except Exception:
                pass

            try:
                cutoff = time.time() - (30 * 24 * 3600)
                cam_ids = []
                config = load_config()
                for cam in (config or []):
                    cid = cam.get("id")
                    if cid:
                        cam_ids.append(cid)
                data["by_camera_30d"] = get_totals_by_camera(camera_ids=cam_ids, start_ts=cutoff)
            except Exception:
                data["by_camera_30d"] = {}
            
            return jsonify(data)
        return jsonify({})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def _get_density_level(vehicle_count):
    """Classify traffic density based on current vehicle count in frame."""
    if vehicle_count <= 3:
        return {"level": "lancar", "label": "Lancar", "color": "#10b981", "score": 1}
    elif vehicle_count <= 8:
        return {"level": "ramai", "label": "Ramai Lancar", "color": "#f59e0b", "score": 2}
    elif vehicle_count <= 15:
        return {"level": "padat", "label": "Padat", "color": "#f97316", "score": 3}
    else:
        return {"level": "macet", "label": "Macet", "color": "#ef4444", "score": 4}


@bp.route("/api/density")
def api_density():
    """Get real-time traffic density for all cameras."""
    try:
        config = load_config() or []
        result = []
        for cam in config:
            cam_id = cam.get("id")
            stats = globals_state.global_stats.get(cam_id, {})
            count = int(stats.get("current_count") or 0)
            density = _get_density_level(count)
            result.append({
                "camera_id": cam_id,
                "camera_name": cam.get("name") or cam_id,
                "lat": cam.get("lat"),
                "lng": cam.get("lng"),
                "current_count": count,
                "density": density,
            })
        return jsonify({"status": "success", "cameras": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.route("/api/edit_camera", methods=["POST"])
@admin_required
@csrf_required
@rate_limit(limit_name="camera_edit")
def edit_camera():
    try:
        data = request.json

        config = load_config()
            
        # Update
        updated = False
        for cam in config:
            if cam["id"] == data["id"]:
                if "name" in data and data["name"] is not None:
                    cam["name"] = data["name"]
                if "url" in data and data["url"] is not None:
                    cam["url"] = data["url"]
                lat_raw = (data.get("lat") or "").strip() if isinstance(data.get("lat"), str) else data.get("lat")
                lng_raw = (data.get("lng") or "").strip() if isinstance(data.get("lng"), str) else data.get("lng")
                try:
                    cam["lat"] = float(lat_raw) if lat_raw not in (None, "") else None
                    cam["lng"] = float(lng_raw) if lng_raw not in (None, "") else None
                except Exception:
                    return jsonify({"status": "error", "message": "Invalid lat/lng"}), 400
                ext_id = str(data.get("external_id") or "").strip()
                if ext_id:
                    cam["external_id"] = ext_id
                updated = True
                break
        
        if updated:
            if not save_config(config):
                return jsonify({"status": "error", "message": "Failed to save config"}), 500
            CCTV_SOURCES[:] = config
            cam_id = data.get("id")
            if cam_id in globals_state.global_stats:
                globals_state.global_stats[cam_id]["name"] = next((c.get("name") for c in config if c.get("id") == cam_id), globals_state.global_stats[cam_id].get("name"))
                save_stats()
            return jsonify({"status": "success", "message": "Camera updated"})
        else:
            return jsonify({"status": "error", "message": "Camera not found"}), 404
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.route("/api/reset_data", methods=["POST"])
@admin_required
@csrf_required
@rate_limit(limit_name="camera_edit")
def reset_data():
    try:
        data = request.json or {}

        clear_all_history()
        for _, stats in globals_state.global_stats.items():
            stats["current_count"] = 0
            stats["current_class_counts"] = {"0": 0, "1": 0}
            stats["accumulated_count"] = 0
            stats["accumulated_class_counts"] = {"0": 0, "1": 0}
            stats["history"] = deque(maxlen=HISTORY_MAX_LEN)
        save_stats()

        return jsonify({"status": "success", "message": "Data reset successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.route("/api/predict_traffic", methods=["POST"])
@rate_limit(limit_name="predict")
def predict_traffic():
    try:
        data = request.json
        target_time_str = data.get("target_time")
        
        # Support legacy single camera request if needed, but priority is full list
        req_camera_id = data.get("camera_id")
        
        if target_time_str:
            from datetime import datetime
            dt = datetime.fromisoformat(target_time_str)
            day_of_week = int(dt.strftime('%w')) # 0-6
            hour = dt.hour
        else:
            # Fallback to manual params
            day_of_week = data.get("day_of_week")
            hour = data.get("hour")

        if day_of_week is None or hour is None:
             return jsonify({"status": "error", "message": "Missing time parameters"}), 400

        # Get list of cameras to predict for
        cameras_to_process = []
        if req_camera_id:
            # Just one
            # We need the name though, so let's load config anyway
            pass 
        
        # Load active cameras
        config_path = os.path.join(DATA_DIR, 'cctv_config.json')
        with open(config_path, 'r') as f:
            all_cameras = json.load(f)
            
        # Load Thresholds (Dynamic Decision Support)
        thresholds_path = os.path.join(DATA_DIR, 'camera_thresholds.json')
        thresholds = {}
        if os.path.exists(thresholds_path):
            with open(thresholds_path, 'r') as f:
                thresholds = json.load(f)
            
        # User requested to update ALL indicators even if a specific camera is selected
        # So we process ALL cameras regardless of active status or req_camera_id
        # This ensures the entire map updates with prediction data
        cameras_to_process = all_cameras
        
        # Note: We trust predict_future_traffic to handle cases with no history gracefully

        # Ensure the requested camera is included (redundant now but kept for safety logic)
        if req_camera_id:
             if not any(c["id"] == req_camera_id for c in cameras_to_process):
                 pass # Already included all


        predictions = []
        
        # Demo/Simulation Mode Check
        force_scenario = data.get("force_scenario")
        
        for cam in cameras_to_process:
            avg_count = predict_future_traffic(cam["id"], int(day_of_week), int(hour))
            
            # --- DEMO SCENARIO INJECTION ---
            if force_scenario == 'high_traffic':
                # Artificially boost traffic for demo purposes to show decision logic
                import random
                avg_count = max(avg_count, random.randint(250, 400))
            elif force_scenario == 'low_traffic':
                avg_count = min(avg_count, 50)
            # -------------------------------
            
            # Decision Logic / Rules Engine
            # Get camera specific thresholds or use defaults
            cam_thresholds = thresholds.get(cam["id"], {"p50": 100, "p75": 200, "p90": 300})
            
            status = "LANCAR"
            recommendation = "Traffic flow is optimal. Continue standard monitoring."
            action_icon = "fas fa-check-circle"
            status_color = "text-green-500" # Tailwind class for UI
            
            if avg_count > cam_thresholds["p90"]: 
                status = "MACET TOTAL"
                recommendation = "CRITICAL ACTION: 1) Deploy Field Unit to intersection. 2) Override traffic light to manual flush. 3) Notify Traffic Command Center."
                action_icon = "fas fa-exclamation-triangle"
                status_color = "text-red-500"
            elif avg_count > cam_thresholds["p75"]: 
                status = "MACET"
                recommendation = "ACTION REQUIRED: 1) Extend Green Light duration by 15s. 2) Display 'Congestion Ahead' on VMS (Variable Message Signs)."
                action_icon = "fas fa-user-shield"
                status_color = "text-orange-500"
            elif avg_count > cam_thresholds["p50"]: 
                status = "PADAT LANCAR"
                recommendation = "ADVISORY: Monitor queue length. Prepare to activate diversion protocols if density increases by 10%."
                action_icon = "fas fa-stopwatch"
                status_color = "text-yellow-500"
            
            predictions.append({
                "camera_id": cam["id"],
                "camera_name": cam["name"],
                "vehicle_count": int(avg_count),
                "traffic_status": status,
                "recommendation": recommendation,
                "action_icon": action_icon,
                "status_color": status_color
            })
        
        return jsonify({
            "status": "success",
            "predictions": predictions,
            "target_time": target_time_str
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.route("/api/backfill_camera", methods=["POST"])
@admin_required
@csrf_required
@rate_limit(limit_name="camera_edit")
def backfill_camera():
    try:
        data = request.json
        target_id = data.get("target_id")
        template_id = data.get("template_id")
        days = data.get("days", 7)
        start_date = data.get("start_date")
        
        if not target_id or not template_id:
            return jsonify({"status": "error", "message": "Missing target_id or template_id"}), 400
            
        result = backfill_camera_history(target_id, template_id, hours=days*24, generate_datalake=True, start_date=start_date)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.route("/api/datalake/stats")
def datalake_stats():
    date_str = request.args.get("date")
    result = get_datalake_stats(date_str)
    return jsonify(result)

@bp.route("/api/chat", methods=["POST"])
@rate_limit(limit_name="chat")
@csrf_required
def chat_api():
    locked = False
    try:
        locked = bool(globals_state.chat_lock.acquire(blocking=False))
    except Exception:
        locked = False
    if not locked:
        return jsonify({"reply": "Chatbot lagi sibuk memproses permintaan lain. Coba lagi dalam beberapa detik ya."})
    try:
        payload = request.get_json(silent=True) or {}
        message = str(payload.get("message") or "").strip()
        page = str(payload.get("page") or "").strip()
        session_id = str(payload.get("session_id") or "").strip()
        ctx_key = _chat_ctx_key(session_id, request)
        prof = get_chat_profile(ctx_key) or {}
        chat_ctx = {
            "last_intent": prof.get("last_intent"),
            "last_camera_id": prof.get("last_camera_id"),
            "last_camera_name": prof.get("last_camera_name"),
            "last_destination": prof.get("last_destination"),
            "prefs": (prof.get("prefs") if isinstance(prof.get("prefs"), dict) else {}),
        }
        if not any(v for v in chat_ctx.values() if v):
            try:
                chat_ctx = _chat_ctx_get(ctx_key, ttl_s=3600)
            except Exception:
                pass
        chat_recent = get_recent_chat_messages(ctx_key, limit=10) if ctx_key else []
        if message and ctx_key:
            try:
                add_chat_message(ctx_key, "user", message, page=page)
            except Exception:
                pass

        def ollama_enabled():
            return bool(_ollama_resolve_model())

        def build_context_snapshot():
            from app.utils import load_config as _load_config
            config = _load_config() or []
            cam_ids = [c.get("id") for c in config if c.get("id")]
            total_cams = len(cam_ids)

            online = 0
            offline = 0
            try:
                sources = globals_state.global_stats if isinstance(globals_state.global_stats, dict) else {}
                for _, s in sources.items():
                    st = (s or {}).get("status")
                    if st == "online":
                        online += 1
                    elif st == "offline":
                        offline += 1
            except Exception:
                online = 0
                offline = 0
            if online == 0 and offline == 0 and total_cams > 0:
                online = total_cams

            monthly = get_aggregated_stats(days=30) or {}
            lifetime = get_total_lifetime() or {}

            cutoff = time.time() - (30 * 24 * 3600)
            totals = get_totals_by_camera(camera_ids=cam_ids, start_ts=cutoff) if cam_ids else {}
            rows = []
            for cam in config:
                cid = cam.get("id")
                if not cid:
                    continue
                t = totals.get(cid) or {}
                rows.append((cam.get("name") or cid, int(t.get("accumulated_count") or 0)))
            rows.sort(key=lambda x: x[1], reverse=True)
            top = rows[:5]

            top_lines = []
            for i, (name, total) in enumerate(top, 1):
                top_lines.append(f"{i}. {name}: {fmt_int(total)}")

            parts = [
                f"Page: {page or '-'}",
                f"Total cameras (config): {total_cams}",
                f"Online: {online} | Offline: {offline}",
                f"Monthly 30d total: {fmt_int(monthly.get('accumulated_count'))} (Cars {fmt_int(monthly.get('cars'))} | Motors {fmt_int(monthly.get('motorcycles'))})",
                f"Lifetime total: {fmt_int(lifetime.get('accumulated_count'))} (Cars {fmt_int(lifetime.get('cars'))} | Motors {fmt_int(lifetime.get('motorcycles'))})",
            ]
            if top_lines:
                parts.append("Top cameras (30d):\n" + "\n".join(top_lines))
            return "\n".join(parts)

        def call_ollama(user_text):
            global _ollama_last_success_at, _ollama_cached_model, _ollama_cached_at
            base_url = _ollama_base_url()
            model = _ollama_resolve_model()
            if not model:
                return None, "Model Ollama tidak terdeteksi."

            query_data = build_query_data(user_text)
            intent = str(query_data.get("intent") or "general")

            def minify_for_llm(data):
                d = data or {}
                it = str(d.get("intent") or "general")
                base = {"intent": it}
                for k in ("now_local", "target_time_local", "timezone_name", "utc_offset", "model_granularity", "bucket_hour_start", "bucket_hour_end"):
                    if k in d:
                        base[k] = d.get(k)

                if it == "predict_traffic":
                    base["camera"] = d.get("camera")
                    base["prediction"] = d.get("prediction")
                    if d.get("location_match"):
                        base["location_match"] = d.get("location_match")
                    if d.get("alternatives"):
                        base["alternatives"] = d.get("alternatives")
                    if d.get("minutes_ahead") is not None:
                        base["minutes_ahead"] = d.get("minutes_ahead")
                    return base

                if it == "route_advice":
                    base["origin"] = d.get("origin")
                    base["destination"] = d.get("destination")
                    base["avoid_points"] = d.get("avoid_points") or []
                    base["recommended_points"] = d.get("recommended_points") or []
                    base["suggested_waypoints"] = d.get("suggested_waypoints") or []
                    base["live_cameras_count"] = d.get("live_cameras_count")
                    base["rules"] = d.get("rules")
                    return base

                if it == "top_cameras_30d":
                    base["top_cameras_30d"] = d.get("top_cameras_30d") or []
                    return base

                if it == "top_live_density":
                    base["top_live_density"] = d.get("top_live_density") or []
                    return base

                if it == "predict_missing_camera":
                    for k in ("minutes_ahead", "camera_keyword", "message", "location"):
                        if k in d:
                            base[k] = d.get(k)
                    return base

                return d

            llm_query_data = minify_for_llm(query_data)

            system_prompt = (
                "Kamu adalah asisten SmartTraffic AI. "
                "Jawab dalam Bahasa Indonesia yang ramah, jelas, dan mudah dipahami. "
                "Jangan tampilkan reasoning/thinking. "
                "Jika pertanyaan butuh angka/fakta lalu lintas dari sistem ini, gunakan DATA_QUERY sebagai sumber utama jawaban dan jangan mengarang angka/fakta. "
                "Jika pertanyaan bersifat umum (di luar data SmartTraffic), kamu boleh jawab pakai pengetahuan umum, tapi jangan mengklaim itu berasal dari data sistem."
                "Jika TOOL_RESULTS tersedia, gunakan itu untuk menjawab pertanyaan berbasis data sistem, dan jangan mengarang data di luar TOOL_RESULTS/DATA_QUERY. "
                "Untuk prediksi, pakai timezone di DATA_QUERY dan sebutkan bahwa prediksi per jam bila model_granularity=hourly. "
                "Untuk route_advice, gunakan titik kamera (avoid/recommended/waypoints) dan jangan mengarang nama jalan (jangan sebut Jl/Jalan/nama ruas). "
                "Jika DATA_QUERY tidak punya data yang dibutuhkan, katakan datanya belum tersedia dan sebutkan apa yang kurang."
            )

            messages = [{"role": "system", "content": system_prompt}]

            ctx = build_context_snapshot()
            messages.append({"role": "system", "content": f"KONTEKS SISTEM (ringkas):\n{ctx}"})

            mem_lines = []
            try:
                if isinstance(chat_ctx, dict):
                    if chat_ctx.get("last_destination"):
                        mem_lines.append(f"Tujuan terakhir: {chat_ctx.get('last_destination')}")
                    if chat_ctx.get("last_camera_name") or chat_ctx.get("last_camera_id"):
                        nm = chat_ctx.get("last_camera_name") or chat_ctx.get("last_camera_id")
                        mem_lines.append(f"Kamera terakhir: {nm}")
                if isinstance(chat_recent, list) and chat_recent:
                    mem_lines.append("Riwayat singkat:")
                    for it in chat_recent[-8:]:
                        role = str((it or {}).get("role") or "").strip()
                        txt = str((it or {}).get("content") or "").strip()
                        if not role or not txt:
                            continue
                        if len(txt) > 180:
                            txt = txt[:180] + "…"
                        mem_lines.append(f"- {role}: {txt}")
            except Exception:
                mem_lines = []
            if mem_lines:
                messages.append({"role": "system", "content": "MEMORY (untuk konteks percakapan):\n" + "\n".join(mem_lines)})

            messages.append({"role": "system", "content": f"DATA_QUERY (JSON):\n{json.dumps(llm_query_data, ensure_ascii=False)}"})
            tool_ctx = None
            try:
                tool_ctx = _ai_collect_tool_context(user_text, chat_ctx=chat_ctx)
            except Exception:
                tool_ctx = None
            if tool_ctx:
                messages.append({"role": "system", "content": f"TOOL_RESULTS (JSON):\n{json.dumps(tool_ctx, ensure_ascii=False)}"})
            messages.append({"role": "user", "content": user_text})

            num_predict = 420 if intent != "general" else 1200
            temperature = 0.1 if intent != "general" else 0.2

            # ── Use AI Provider (OpenAI-compatible) if configured ──
            import app.config as _ai_cfg
            if _ai_cfg.AI_USE_FOR_CHAT and _ai_cfg.AI_API_KEY and _ai_cfg.AI_BASE_URL:
                try:
                    ai_body = {
                        "model": _ai_cfg.AI_CHAT_MODEL or _ai_cfg.AI_MODEL,
                        "messages": messages,
                        "max_tokens": num_predict,
                        "temperature": temperature,
                    }
                    ai_url = f"{_ai_cfg.AI_BASE_URL.rstrip('/')}/chat/completions"
                    ai_headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {_ai_cfg.AI_API_KEY}",
                    }
                    ai_data = json.dumps(ai_body).encode("utf-8")
                    ai_req = urllib.request.Request(ai_url, data=ai_data, headers=ai_headers, method="POST")
                    
                    with urllib.request.urlopen(ai_req, timeout=30) as ai_resp:
                        ai_result = json.loads(ai_resp.read().decode("utf-8"))
                    
                    ai_content = ai_result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if ai_content:
                        provider = _ai_cfg.AI_PROVIDER
                        model = _ai_cfg.AI_CHAT_MODEL or _ai_cfg.AI_MODEL
                        return reply(ai_content, provider=provider, model=model, ui_actions=ui_actions)
                except Exception as ai_err:
                    print(f"[CHAT] AI provider error, falling back to Ollama: {ai_err}")
                    # Fall through to Ollama

            # ── Ollama (local) fallback ──
            num_ctx = 4096 if intent != "general" else 4096
            body = {
                "model": model,
                "stream": False,
                "think": False,
                "keep_alive": "10m",
                "messages": messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                    "num_ctx": num_ctx,
                    "top_k": 20,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                },
            }

            url = f"{base_url}/api/chat"
            timeout_s = 25.0 if intent != "general" else 140.0

            def is_model_not_found(text):
                low = str(text or "").lower()
                return ("model" in low) and ("not found" in low)

            last_err = None
            for attempt in range(2):
                try:
                    body["model"] = model
                    data = json.dumps(body).encode("utf-8")
                    req = urllib.request.Request(
                        url,
                        data=data,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                        raw = resp.read()
                    raw_text = raw.decode("utf-8", errors="ignore") if raw else ""
                    try:
                        parsed = json.loads(raw_text) if raw_text else {}
                    except Exception:
                        parsed = {}
                    if not isinstance(parsed, dict):
                        return None, "Respons Ollama tidak valid (bukan JSON object)."

                    if parsed.get("error"):
                        err = str(parsed.get("error"))
                        if attempt == 0 and is_model_not_found(err):
                            try:
                                models = _ollama_get_models_cached(cache_seconds=0, timeout_s=4, force=True)
                            except Exception:
                                models = []
                            alt = None
                            if models:
                                prefer_model = _ollama_sanitize_model_name("qwen2.5:3b")
                                if prefer_model in models:
                                    alt = prefer_model
                                else:
                                    alt = models[0]
                            if alt and alt != model:
                                model = alt
                                _ollama_cached_model = model
                                _ollama_cached_at = time.time()
                                continue
                        return None, err

                    msg = parsed.get("message") or {}
                    content = str((msg.get("content") or "")).strip()
                    if not content:
                        content = str(parsed.get("response") or "").strip()

                    if content:
                        def _should_continue(done_reason_value, txt):
                            dr = str(done_reason_value or "").strip().lower()
                            if dr in ("length", "max_tokens", "limit"):
                                return True
                            t = str(txt or "").strip()
                            if not t:
                                return False
                            if len(t) < 200:
                                return False
                            if t.endswith(("...", "…")):
                                return True
                            if re.search(r"\n\\s*\\d+\\.$", t):
                                return True
                            if re.search(r"\\b(jika|karena|dan|atau|yang|untuk)$", t.lower()):
                                return True
                            last = t[-1]
                            if last in (":", ",", ";", "(", "["):
                                return True
                            return False

                        done_reason0 = str(parsed.get("done_reason") or "").strip()
                        def format_route_from_query(d):
                            dest = (d or {}).get("destination") if isinstance((d or {}).get("destination"), dict) else {}
                            dest_name = str(dest.get("name") or dest.get("query") or "tujuan")
                            rec = (d or {}).get("recommended_points") or []
                            avoid = (d or {}).get("avoid_points") or []
                            wp = (d or {}).get("suggested_waypoints") or []

                            lines = [f"Saran area berbasis CCTV sekitar {dest_name}:"]
                            if avoid:
                                lines.append("Hindari area (padat/macet):")
                                for r in avoid[:6]:
                                    lines.append(
                                        f"- {r.get('camera_name')} — {r.get('congestion')} ({fmt_int(r.get('current_count'))}), {r.get('distance_to_destination_km')} km"
                                    )
                            if rec:
                                lines.append("Lebih aman (lebih lancar):")
                                for r in rec[:6]:
                                    lines.append(
                                        f"- {r.get('camera_name')} — {r.get('congestion')} ({fmt_int(r.get('current_count'))}), {r.get('distance_to_destination_km')} km"
                                    )
                            if wp:
                                names = [str(x.get("camera_name") or "").strip() for x in wp if x]
                                names = [n for n in names if n]
                                if names:
                                    lines.append("Titik pantau (waypoints): " + ", ".join(names[:6]))
                            if len(lines) == 1:
                                lines.append("Data CCTV terdekat belum cukup (koordinat tujuan/kamera atau live snapshot belum lengkap).")
                            return "\n".join(lines)

                        if intent == "route_advice":
                            low = content.lower()
                            has_roads = bool(re.search(r"\b(jl\\.?|jalan)\\b", low))
                            too_generic = ("langkah" in low and "memeriksa" in low) or ("berikut adalah beberapa langkah" in low)
                            if has_roads or too_generic:
                                content = format_route_from_query(query_data)

                        if _should_continue(done_reason0, content):
                            full = content
                            for _ in range(2):
                                cont_messages = list(messages) + [
                                    {"role": "assistant", "content": full},
                                    {"role": "user", "content": "Lanjutkan jawabanmu dari bagian terakhir. Jangan mengulang dari awal. Teruskan sampai tuntas."},
                                ]
                                cont_body = dict(body)
                                cont_body["messages"] = cont_messages
                                cont_body["options"] = dict(body.get("options") or {})
                                cont_body["options"]["num_predict"] = max(int(num_predict or 0), 1200)
                                cont_data = json.dumps(cont_body).encode("utf-8")
                                cont_req = urllib.request.Request(
                                    url,
                                    data=cont_data,
                                    headers={"Content-Type": "application/json"},
                                    method="POST",
                                )
                                try:
                                    with urllib.request.urlopen(cont_req, timeout=timeout_s) as resp2:
                                        raw2 = resp2.read()
                                except Exception:
                                    break
                                raw2_text = raw2.decode("utf-8", errors="ignore") if raw2 else ""
                                try:
                                    parsed2 = json.loads(raw2_text) if raw2_text else {}
                                except Exception:
                                    parsed2 = {}
                                if not isinstance(parsed2, dict) or parsed2.get("error"):
                                    break
                                msg2 = parsed2.get("message") or {}
                                content2 = str((msg2.get("content") or "")).strip()
                                if not content2:
                                    content2 = str(parsed2.get("response") or "").strip()
                                if not content2:
                                    break
                                full = (full.rstrip() + "\n" + content2.lstrip()).strip()
                                done_reason2 = str(parsed2.get("done_reason") or "").strip()
                                if not _should_continue(done_reason2, content2):
                                    break
                            content = full

                        _ollama_last_success_at = time.time()
                        return content, None

                    done_reason = str(parsed.get("done_reason") or "").strip()
                    has_thinking = bool((msg.get("thinking") or "").strip())
                    if has_thinking and done_reason:
                        return None, f"Qwen mengembalikan content kosong (done_reason={done_reason})."
                    if has_thinking:
                        return None, "Qwen mengembalikan content kosong (hanya reasoning)."
                    return None, "Qwen mengembalikan content kosong."
                except urllib.error.HTTPError as e:
                    try:
                        raw_text = e.read().decode("utf-8", errors="ignore")
                    except Exception:
                        raw_text = ""
                    detail = raw_text.strip() or str(e)
                    low = detail.lower()
                    if attempt == 0 and is_model_not_found(detail):
                        try:
                            models = _ollama_get_models_cached(cache_seconds=0, timeout_s=4, force=True)
                        except Exception:
                            models = []
                        alt = None
                        if models:
                            prefer_model = _ollama_sanitize_model_name("qwen2.5:3b")
                            if prefer_model in models:
                                alt = prefer_model
                            else:
                                alt = models[0]
                        if alt and alt != model:
                            model = alt
                            _ollama_cached_model = model
                            _ollama_cached_at = time.time()
                            continue
                    if "403" in low and "requires a subscription" in low:
                        return None, (
                            "Model Ollama yang dipakai membutuhkan subscription (403 Forbidden). "
                            "Pakai model lokal lain yang gratis, lalu set OLLAMA_MODEL ke model tersebut."
                        )
                    return None, detail
                except (urllib.error.URLError, TimeoutError) as e:
                    last_err = str(e)
                except Exception as e:
                    last_err = str(e)

            return None, last_err or "Unknown error"

        def call_ollama_planner(user_text, query_data, ui_actions_base):
            base_url = _ollama_base_url()
            model = _ollama_resolve_model()
            if not model:
                return None, None, "Model Ollama tidak terdeteksi."

            def minify_for_llm(data):
                d = data or {}
                it = str(d.get("intent") or "general")
                base = {"intent": it}
                for k in ("now_local", "target_time_local", "timezone_name", "utc_offset", "model_granularity", "bucket_hour_start", "bucket_hour_end"):
                    if k in d:
                        base[k] = d.get(k)
                for k in ("camera", "prediction", "location_match", "alternatives", "origin", "destination", "avoid_points", "recommended_points", "suggested_waypoints"):
                    if k in d:
                        base[k] = d.get(k)
                return base

            ctx = build_context_snapshot()
            qd = minify_for_llm(query_data)
            base_actions = ui_actions_base or []

            system_prompt = (
                "Kamu adalah PLANNER untuk SmartTraffic AI.\n"
                "Tugasmu: ubah permintaan user menjadi (1) jawaban singkat yang ramah untuk orang awam, dan (2) daftar aksi UI yang aman.\n"
                "ATURAN KETAT:\n"
                "- Jangan mengarang data.\n"
                "- Kalau data kurang, jelaskan apa yang kurang dan tawarkan 1 langkah lanjutan yang bisa dieksekusi.\n"
                "- Keluaran HARUS JSON valid, tanpa teks lain.\n"
                "SKEMA OUTPUT:\n"
                "{ \"reply\": string, \"actions\": [ {\"type\": \"navigate\", \"path\": \"/analysis\"} | {\"type\":\"select_camera\",\"camera_id\":string,\"camera_name\":string} | {\"type\":\"set_period\",\"period\":string} | {\"type\":\"show_prediction_popup\", ... } ] }\n"
                "CATATAN:\n"
                "- Aksi yang tidak ada di daftar di atas dilarang.\n"
                "- Prefer pakai actions dari UI_ACTIONS_BASE jika sudah sesuai.\n"
            )

            mem_lines = []
            try:
                if isinstance(chat_ctx, dict):
                    if chat_ctx.get("last_destination"):
                        mem_lines.append(f"Tujuan terakhir: {chat_ctx.get('last_destination')}")
                    if chat_ctx.get("last_camera_name") or chat_ctx.get("last_camera_id"):
                        nm = chat_ctx.get("last_camera_name") or chat_ctx.get("last_camera_id")
                        mem_lines.append(f"Kamera terakhir: {nm}")
                if isinstance(chat_recent, list) and chat_recent:
                    mem_lines.append("Riwayat singkat:")
                    for it in chat_recent[-6:]:
                        role = str((it or {}).get("role") or "").strip()
                        txt = str((it or {}).get("content") or "").strip()
                        if not role or not txt:
                            continue
                        if len(txt) > 140:
                            txt = txt[:140] + "…"
                        mem_lines.append(f"- {role}: {txt}")
            except Exception:
                mem_lines = []

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"KONTEKS SISTEM (ringkas):\n{ctx}"},
            ]
            if mem_lines:
                messages.append({"role": "system", "content": "MEMORY:\n" + "\n".join(mem_lines)})
            messages.append({"role": "system", "content": f"DATA_QUERY (JSON):\n{json.dumps(qd, ensure_ascii=False)}"})
            messages.append({"role": "system", "content": f"UI_ACTIONS_BASE (JSON):\n{json.dumps(base_actions, ensure_ascii=False)}"})
            messages.append({"role": "user", "content": user_text})

            body = {
                "model": model,
                "stream": False,
                "think": False,
                "messages": messages,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 220,
                    "num_ctx": 1024,
                    "top_k": 20,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                },
            }

            try:
                data = json.dumps(body).encode("utf-8")
                req = urllib.request.Request(
                    f"{base_url}/api/chat",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=25) as resp:
                    raw = resp.read()
                raw_text = raw.decode("utf-8", errors="ignore") if raw else ""
                parsed = json.loads(raw_text) if raw_text else {}
                if not isinstance(parsed, dict):
                    return None, None, "Planner response bukan JSON object."
                if parsed.get("error"):
                    return None, None, str(parsed.get("error"))
                msg = parsed.get("message") or {}
                content = str((msg.get("content") or "")).strip()
                if not content:
                    content = str(parsed.get("response") or "").strip()
                if not content:
                    return None, None, "Planner mengembalikan content kosong."
                plan = json.loads(content)
                if not isinstance(plan, dict):
                    return None, None, "Planner output tidak sesuai (bukan object)."
                reply = str(plan.get("reply") or "").strip()
                acts = _planner_validate_actions(plan.get("actions") or [])
                return reply or None, acts or None, None
            except Exception as e:
                return None, None, str(e)

        def build_query_data(user_text):
            q = str(user_text or "").lower()
            data = {"intent": "general"}

            def parse_target_dt(q_text, now_dt):
                ql = str(q_text or "").lower()
                day_add = 1 if any(k in ql for k in ["besok"]) else 0

                hh = None
                mm = 0
                m1 = re.search(r"\b([01]?\d|2[0-3])[:. ]([0-5]\d)\b", ql)
                if m1:
                    try:
                        hh = int(m1.group(1))
                        mm = int(m1.group(2))
                    except Exception:
                        hh = None
                        mm = 0
                if hh is None:
                    m2 = re.search(r"\b(jam|pukul)\s*(\d{1,2})(?:[:.](\d{2}))?\b", ql)
                    if m2:
                        try:
                            hh = int(m2.group(2))
                        except Exception:
                            hh = None
                        try:
                            mm = int(m2.group(3) or 0)
                        except Exception:
                            mm = 0

                if hh is None:
                    return None

                if hh == 24:
                    hh = 0

                if hh < 12 and any(k in ql for k in ["siang", "sore", "malam", "petang", "pm"]):
                    hh += 12
                if hh == 12 and any(k in ql for k in ["malam", "tengah malam"]):
                    hh = 0

                try:
                    target = now_dt.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0) + datetime.timedelta(days=day_add)
                except Exception:
                    return None

                if day_add == 0 and target <= now_dt and any(k in ql for k in ["nanti", "sore", "malam", "siang"]):
                    target = target + datetime.timedelta(days=1)
                return target

            def build_predict_for_camera(cam, target_dt, mins):
                day_of_week = int(target_dt.strftime("%w"))
                hour = int(target_dt.hour)
                bucket_start = target_dt.replace(minute=0, second=0, microsecond=0)
                bucket_end = bucket_start + datetime.timedelta(minutes=59)

                cam_id = cam.get("id")
                cam_name = cam.get("name") or cam_id
                avg_count = int(predict_future_traffic(cam_id, int(day_of_week), int(hour), target_dt_local=target_dt) or 0)

                thresholds_path = os.path.join(DATA_DIR, "camera_thresholds.json")
                thresholds = {}
                if os.path.exists(thresholds_path):
                    try:
                        with open(thresholds_path, "r") as f:
                            thresholds = json.load(f) or {}
                    except Exception:
                        thresholds = {}
                cam_thr = thresholds.get(cam_id, {"p50": 100, "p75": 200, "p90": 300})
                status = "LANCAR"
                if avg_count > int(cam_thr.get("p90", 300)):
                    status = "MACET TOTAL"
                elif avg_count > int(cam_thr.get("p75", 200)):
                    status = "MACET"
                elif avg_count > int(cam_thr.get("p50", 100)):
                    status = "PADAT LANCAR"

                now_local = datetime.datetime.now().astimezone()
                out = {"intent": "predict_traffic"}
                out["minutes_ahead"] = mins
                out["now_local"] = now_local.strftime("%Y-%m-%d %H:%M")
                out["target_time_local"] = target_dt.strftime("%Y-%m-%d %H:%M")
                out["timezone_name"] = str(target_dt.tzname() or "")
                out["utc_offset"] = target_dt.strftime("%z")
                out["model_granularity"] = "hourly"
                out["bucket_hour_start"] = bucket_start.strftime("%Y-%m-%d %H:%M")
                out["bucket_hour_end"] = bucket_end.strftime("%Y-%m-%d %H:%M")
                out["target_hour"] = f"{hour:02d}:00"
                out["camera"] = {"id": cam_id, "name": cam_name, "lat": cam.get("lat"), "lng": cam.get("lng")}
                out["prediction"] = {"vehicles_per_hour": avg_count, "status": status, "thresholds": cam_thr}
                return out

            has_time_hint_only = bool(re.search(r"\b(jam|pukul)\s*\d{1,2}\b", q) or re.search(r"\b([01]?\d|2[0-3])[:. ]([0-5]\d)\b", q))
            has_place_hint = bool(re.search(r"\bke\s+\w+", q)) or any(k in q for k in ["braga", "asia afrika", "dago", "pasteur"])
            if has_time_hint_only and not has_place_hint:
                base_cam_id = str((chat_ctx or {}).get("last_camera_id") or "").strip()
                base_place = str((chat_ctx or {}).get("last_destination") or "").strip()
                now_local = datetime.datetime.now().astimezone()
                target_dt = parse_target_dt(q, now_local)
                if target_dt and (base_cam_id or base_place):
                    mins = int(max(1, min(24 * 60, round((target_dt - now_local).total_seconds() / 60.0))))
                    config = load_config() or []
                    cam = next((c for c in config if str(c.get("id") or "") == base_cam_id), None) if base_cam_id else None
                    if cam is None and base_place:
                        matches = _find_best_cameras(base_place, limit=1)
                        cam = matches[0] if matches else None
                    if cam:
                        return build_predict_for_camera(cam, target_dt, mins)

            route_kw = any(
                k in q
                for k in [
                    "rute",
                    "jalur",
                    "jalan",
                    "lewat mana",
                    "alternatif",
                    "hindari",
                    "macet",
                    "kemacetan",
                    "arah",
                    "ke braga",
                    "ke asia afrika",
                    "ke dago",
                    "ke pasteur",
                ]
            )
            has_dest_phrase = (" ke " in f" {q} ") and any(k in q for k in ["ingin", "mau", "tujuan", "pergi", "berangkat", "arah"])
            want_route = route_kw or has_dest_phrase

            want_top = any(k in q for k in ["paling banyak", "terbanyak", "top", "ranking", "urutan", "ramai", "terramai"])
            want_place = any(k in q for k in ["titik", "lokasi", "tempat", "area", "persimpangan", "simpang"])
            want_live = any(k in q for k in ["live", "sekarang", "saat ini", "hari ini sekarang"])
            want_30d = any(k in q for k in ["30 hari", "30d", "bulan ini", "bulanan", "monthly"])

            if want_route:
                origin_text, dest_text = _extract_origin_destination(user_text)
                if dest_text:
                    wants_time_congestion = any(k in q for k in ["macet", "kemacetan", "padat", "lancar"])
                    has_time_hint = bool(re.search(r"\b(jam|pukul)\s*\d{1,2}\b", q) or re.search(r"\b([01]?\d|2[0-3])[:. ]([0-5]\d)\b", q))
                    if wants_time_congestion and has_time_hint and not origin_text:
                        now_local = datetime.datetime.now().astimezone()
                        target_dt = parse_target_dt(q, now_local)
                        if target_dt:
                            mins = int(max(1, min(24 * 60, round((target_dt - now_local).total_seconds() / 60.0))))

                            matches = _find_best_cameras(dest_text, limit=5)
                            if not matches:
                                place_q = _clean_place_phrase(dest_text)
                                geo = _geocode_place(place_q) if place_q else None
                                if geo and geo.get("lat") is not None and geo.get("lng") is not None:
                                    config = load_config() or []
                                    closest = None
                                    best_d = None
                                    for cam in config:
                                        lat = cam.get("lat")
                                        lng = cam.get("lng")
                                        if lat is None or lng is None:
                                            continue
                                        d = _haversine_km(geo.get("lat"), geo.get("lng"), lat, lng)
                                        if d is None:
                                            continue
                                        if best_d is None or float(d) < float(best_d):
                                            best_d = float(d)
                                            closest = cam
                                    if closest and best_d is not None and float(best_d) <= 6.0:
                                        matches = [closest]
                                        data["location_match"] = {
                                            "query": place_q,
                                            "geocoded": {"name": geo.get("name") or place_q, "lat": geo.get("lat"), "lng": geo.get("lng")},
                                            "nearest_camera": {"id": closest.get("id"), "name": closest.get("name") or closest.get("id"), "distance_km": round(float(best_d), 2)},
                                        }
                                    else:
                                        data["intent"] = "predict_missing_camera"
                                        data["minutes_ahead"] = mins
                                        data["camera_keyword"] = dest_text
                                        data["message"] = "Lokasi terdeteksi, tapi tidak ada kamera terdekat yang punya koordinat (<= 6 km)."
                                        if geo:
                                            data["location"] = {"name": geo.get("name") or place_q, "lat": geo.get("lat"), "lng": geo.get("lng")}
                                        return data
                                else:
                                    data["intent"] = "predict_missing_camera"
                                    data["minutes_ahead"] = mins
                                    data["camera_keyword"] = dest_text
                                    data["message"] = "Kamera/lokasi tidak ditemukan dari kata kunci."
                                    return data

                            day_of_week = int(target_dt.strftime("%w"))
                            hour = int(target_dt.hour)
                            bucket_start = target_dt.replace(minute=0, second=0, microsecond=0)
                            bucket_end = bucket_start + datetime.timedelta(minutes=59)

                            cam = matches[0]
                            cam_id = cam.get("id")
                            cam_name = cam.get("name") or cam_id
                            avg_count = int(predict_future_traffic(cam_id, int(day_of_week), int(hour), target_dt_local=target_dt) or 0)

                            thresholds_path = os.path.join(DATA_DIR, "camera_thresholds.json")
                            thresholds = {}
                            if os.path.exists(thresholds_path):
                                try:
                                    with open(thresholds_path, "r") as f:
                                        thresholds = json.load(f) or {}
                                except Exception:
                                    thresholds = {}
                            cam_thr = thresholds.get(cam_id, {"p50": 100, "p75": 200, "p90": 300})
                            status = "LANCAR"
                            if avg_count > int(cam_thr.get("p90", 300)):
                                status = "MACET TOTAL"
                            elif avg_count > int(cam_thr.get("p75", 200)):
                                status = "MACET"
                            elif avg_count > int(cam_thr.get("p50", 100)):
                                status = "PADAT LANCAR"

                            data["intent"] = "predict_traffic"
                            data["minutes_ahead"] = mins
                            data["now_local"] = now_local.strftime("%Y-%m-%d %H:%M")
                            data["target_time_local"] = target_dt.strftime("%Y-%m-%d %H:%M")
                            data["timezone_name"] = str(target_dt.tzname() or "")
                            data["utc_offset"] = target_dt.strftime("%z")
                            data["model_granularity"] = "hourly"
                            data["bucket_hour_start"] = bucket_start.strftime("%Y-%m-%d %H:%M")
                            data["bucket_hour_end"] = bucket_end.strftime("%Y-%m-%d %H:%M")
                            data["target_hour"] = f"{hour:02d}:00"
                            data["camera"] = {"id": cam_id, "name": cam_name, "lat": cam.get("lat"), "lng": cam.get("lng")}
                            data["prediction"] = {"vehicles_per_hour": avg_count, "status": status, "thresholds": cam_thr}
                            if len(matches) > 1:
                                data["alternatives"] = [{"id": c.get("id"), "name": c.get("name") or c.get("id")} for c in matches[1:4]]
                            return data

                    data["intent"] = "route_advice"
                    now_local = datetime.datetime.now().astimezone()
                    data["now_local"] = now_local.strftime("%Y-%m-%d %H:%M")
                    data["timezone_name"] = str(now_local.tzname() or "")
                    data["utc_offset"] = now_local.strftime("%z")

                    snapshot = _live_camera_snapshot() or []
                    cams = []
                    for r in snapshot:
                        if r is None:
                            continue
                        lat = r.get("lat")
                        lng = r.get("lng")
                        if lat in (None, "") or lng in (None, ""):
                            continue
                        try:
                            _ = float(lat)
                            _ = float(lng)
                        except Exception:
                            continue
                        cams.append(r)
                    data["live_cameras_count"] = len(cams)

                    dest = {"query": dest_text}
                    origin = {"query": origin_text} if origin_text else None

                    dest_match = _find_best_cameras(dest_text, limit=3)
                    dest_point = None
                    if dest_match:
                        m = dest_match[0]
                        if m.get("lat") is not None and m.get("lng") is not None:
                            dest_point = {"lat": m.get("lat"), "lng": m.get("lng"), "name": m.get("name") or m.get("id"), "via": "camera_match"}
                    if not dest_point:
                        g = _geocode_place(dest_text)
                        if g:
                            dest_point = {"lat": g.get("lat"), "lng": g.get("lng"), "name": g.get("name") or dest_text, "via": "geocode"}
                    if not dest_point:
                        dest_point = {"lat": None, "lng": None, "name": dest_text, "via": "unknown"}
                    dest.update(dest_point)
                    if dest_match:
                        dest["camera_candidates"] = [{"id": c.get("id"), "name": c.get("name") or c.get("id")} for c in dest_match]

                    origin_point = None
                    if origin_text:
                        origin_match = _find_best_cameras(origin_text, limit=3)
                        if origin_match:
                            m = origin_match[0]
                            if m.get("lat") is not None and m.get("lng") is not None:
                                origin_point = {"lat": m.get("lat"), "lng": m.get("lng"), "name": m.get("name") or m.get("id"), "via": "camera_match"}
                        if not origin_point:
                            g = _geocode_place(origin_text)
                            if g:
                                origin_point = {"lat": g.get("lat"), "lng": g.get("lng"), "name": g.get("name") or origin_text, "via": "geocode"}
                        if not origin_point:
                            origin_point = {"lat": None, "lng": None, "name": origin_text, "via": "unknown"}
                        origin.update(origin_point)
                        if origin_match:
                            origin["camera_candidates"] = [{"id": c.get("id"), "name": c.get("name") or c.get("id")} for c in origin_match]

                    data["origin"] = origin
                    data["destination"] = dest

                    dest_lat = dest.get("lat")
                    dest_lng = dest.get("lng")
                    origin_lat = (origin or {}).get("lat") if origin else None
                    origin_lng = (origin or {}).get("lng") if origin else None

                    near = []
                    if dest_lat is not None and dest_lng is not None:
                        for r in cams:
                            d = _haversine_km(dest_lat, dest_lng, r.get("lat"), r.get("lng"))
                            if d is None:
                                continue
                            near.append({**r, "distance_to_destination_km": round(float(d), 2)})
                        near.sort(key=lambda x: x.get("distance_to_destination_km", 999999))
                        radius = 3.0
                        in_radius = [x for x in near if float(x.get("distance_to_destination_km") or 999999) <= radius]
                        if len(in_radius) < 4:
                            radius = 8.0
                            in_radius = [x for x in near if float(x.get("distance_to_destination_km") or 999999) <= radius]
                        near = in_radius[:15]

                    def cong_rank(label):
                        if label == "MACET TOTAL":
                            return 3
                        if label == "MACET":
                            return 2
                        if label == "PADAT LANCAR":
                            return 1
                        return 0

                    avoid = [x for x in near if cong_rank(x.get("congestion")) >= 2]
                    avoid.sort(key=lambda x: (cong_rank(x.get("congestion")), int(x.get("current_count") or 0)), reverse=True)
                    recommend = [x for x in near if cong_rank(x.get("congestion")) <= 1]
                    recommend.sort(key=lambda x: (cong_rank(x.get("congestion")), float(x.get("distance_to_destination_km") or 999999)))

                    if origin_lat is not None and origin_lng is not None and dest_lat is not None and dest_lng is not None:
                        start_candidates = []
                        for r in cams:
                            d = _haversine_km(origin_lat, origin_lng, r.get("lat"), r.get("lng"))
                            if d is None:
                                continue
                            start_candidates.append({**r, "distance_to_origin_km": round(float(d), 2)})
                        start_candidates.sort(key=lambda x: x.get("distance_to_origin_km", 999999))
                        data["near_origin"] = start_candidates[:8]

                    data["near_destination"] = near
                    data["avoid_points"] = [
                        {
                            "camera_id": x.get("camera_id"),
                            "camera_name": x.get("camera_name"),
                            "congestion": x.get("congestion"),
                            "current_count": int(x.get("current_count") or 0),
                            "distance_to_destination_km": x.get("distance_to_destination_km"),
                        }
                        for x in avoid[:6]
                    ]
                    data["recommended_points"] = [
                        {
                            "camera_id": x.get("camera_id"),
                            "camera_name": x.get("camera_name"),
                            "congestion": x.get("congestion"),
                            "current_count": int(x.get("current_count") or 0),
                            "distance_to_destination_km": x.get("distance_to_destination_km"),
                        }
                        for x in recommend[:6]
                    ]

                    waypoints = []
                    if recommend:
                        for x in recommend[:4]:
                            waypoints.append({"camera_id": x.get("camera_id"), "camera_name": x.get("camera_name")})
                    data["suggested_waypoints"] = waypoints

                    data["rules"] = {
                        "dont_invent_roads": True,
                        "use_camera_points_as_area_labels": True,
                    }
                    return data

            if (want_top and (want_place or "kamera" in q)) or ("titik mana" in q) or ("lokasi mana" in q):
                if want_live:
                    data["intent"] = "top_live_density"
                    data["top_live_density"] = _top_cameras_live_density(limit=5)
                else:
                    data["intent"] = "top_cameras_30d" if want_30d or not want_live else "top_live_density"
                    data["top_cameras_30d"] = _top_cameras_30d(limit=5)
                return data

            want_cause = any(k in q for k in ["penyebab", "kenapa", "mengapa", "alasan", "why", "cause"]) and any(
                k in q for k in ["macet", "padat", "ramai", "lancar", "kemacetan"]
            )
            if want_cause:
                period = "1h"
                if re.search(r"\b30\s*(menit|mnt|min|m)\b", q) or "30m" in q:
                    period = "30m"
                elif re.search(r"\b(2|dua)\s*(jam|hour|hrs|hr|h)\b", q) or "2h" in q:
                    period = "2h"
                elif re.search(r"\b6\s*(jam|hour|hrs|hr|h)\b", q) or "6h" in q:
                    period = "6h"

                origin_text, dest_text = _extract_origin_destination(user_text)
                cam = None
                matches = []
                if dest_text:
                    matches = _find_best_cameras(dest_text, limit=3)
                    cam = matches[0] if matches else None
                    data["destination"] = {"query": dest_text}
                if cam is None:
                    tokens = _tokenize_query(q)
                    keyword = " ".join(tokens).strip()
                    matches = _find_best_cameras(keyword or q, limit=3)
                    cam = matches[0] if matches else None
                if cam is None:
                    base_cam_id = str((chat_ctx or {}).get("last_camera_id") or "").strip()
                    base_cam_name = str((chat_ctx or {}).get("last_camera_name") or "").strip()
                    base_place = str((chat_ctx or {}).get("last_destination") or "").strip()
                    config = load_config() or []
                    if base_cam_id:
                        cam = next((c for c in config if str(c.get("id") or "") == base_cam_id), None)
                    if cam is None and base_cam_name:
                        m2 = _find_best_cameras(base_cam_name, limit=1)
                        cam = m2[0] if m2 else None
                    if cam is None and base_place:
                        m3 = _find_best_cameras(base_place, limit=1)
                        cam = m3[0] if m3 else None

                if cam:
                    cam_id = cam.get("id")
                    cam_name = cam.get("name") or cam_id
                    data["intent"] = "cause_analysis"
                    data["camera"] = {"id": cam_id, "name": cam_name, "lat": cam.get("lat"), "lng": cam.get("lng")}
                    data["period"] = period
                    if len(matches) > 1:
                        data["alternatives"] = [{"id": c.get("id"), "name": c.get("name") or c.get("id")} for c in matches[1:4]]
                    return data

            want_analysis = any(k in q for k in ["analisa", "analisis", "analysis", "grafik", "graph", "chart", "trend", "statistik", "ringkasan"])
            if want_analysis:
                period = "1h"
                if re.search(r"\b30\s*(menit|mnt|min|m)\b", q) or "30m" in q:
                    period = "30m"
                elif re.search(r"\b(1|satu)\s*(jam|hour|hrs|hr|h)\b", q) or "1h" in q:
                    period = "1h"
                elif re.search(r"\b6\s*(jam|hour|hrs|hr|h)\b", q) or "6h" in q:
                    period = "6h"
                elif re.search(r"\b12\s*(jam|hour|hrs|hr|h)\b", q) or "12h" in q:
                    period = "12h"
                elif any(k in q for k in ["hari ini", "24 jam", "24h", "today"]):
                    period = "24h"
                elif re.search(r"\b7\s*(hari|day|d)\b", q) or "7d" in q:
                    period = "7d"
                elif re.search(r"\b30\s*(hari|day|d)\b", q) or "30d" in q or "30 hari" in q:
                    period = "30d"

                tokens = _tokenize_query(q)
                keyword = " ".join(tokens).strip()
                matches = _find_best_cameras(keyword or q, limit=3)
                cam = matches[0] if matches else None

                if cam is None:
                    base_cam_id = str((chat_ctx or {}).get("last_camera_id") or "").strip()
                    base_cam_name = str((chat_ctx or {}).get("last_camera_name") or "").strip()
                    base_place = str((chat_ctx or {}).get("last_destination") or "").strip()
                    config = load_config() or []
                    if base_cam_id:
                        cam = next((c for c in config if str(c.get("id") or "") == base_cam_id), None)
                    if cam is None and base_cam_name:
                        m2 = _find_best_cameras(base_cam_name, limit=1)
                        cam = m2[0] if m2 else None
                    if cam is None and base_place:
                        m3 = _find_best_cameras(base_place, limit=1)
                        cam = m3[0] if m3 else None

                if cam:
                    cam_id = cam.get("id")
                    cam_name = cam.get("name") or cam_id
                    data["intent"] = "analysis_summary"
                    data["camera"] = {"id": cam_id, "name": cam_name, "lat": cam.get("lat"), "lng": cam.get("lng")}
                    data["period"] = period
                    if len(matches) > 1:
                        data["alternatives"] = [{"id": c.get("id"), "name": c.get("name") or c.get("id")} for c in matches[1:4]]
                    return data

            want_show_cam = any(k in q for k in ["tampilkan", "lihat", "buka", "pilih", "fokus", "switch", "ganti"])
            has_cam_kw = any(k in q for k in ["kamera", "cctv", "cam"])
            if want_show_cam and has_cam_kw:
                tokens = _tokenize_query(q)
                keyword = " ".join(tokens).strip()
                matches = _find_best_cameras(keyword or q, limit=5)
                if matches:
                    cam = matches[0]
                    cam_id = cam.get("id")
                    cam_name = cam.get("name") or cam_id
                    data["intent"] = "select_camera"
                    data["camera"] = {"id": cam_id, "name": cam_name, "lat": cam.get("lat"), "lng": cam.get("lng")}
                    if len(matches) > 1:
                        data["alternatives"] = [{"id": c.get("id"), "name": c.get("name") or c.get("id")} for c in matches[1:4]]
                    return data

            if any(k in q for k in ["prediksi", "predict", "forecast"]):
                horizon_days = None
                dm = re.search(r"(\d+)\s*(hari|day|d)\b", q)
                if dm:
                    try:
                        horizon_days = int(dm.group(1))
                    except Exception:
                        horizon_days = None
                if horizon_days is not None:
                    horizon_days = max(1, min(30, int(horizon_days)))
                mins = 15
                if horizon_days is None:
                    m = re.search(r"(\d+)\s*(menit|mnt|min|m)\b", q)
                    if m:
                        try:
                            mins = int(m.group(1))
                        except Exception:
                            mins = 15
                    h = re.search(r"(\d+)\s*(jam|hour|hrs|hr|h)\b", q)
                    if h:
                        try:
                            mins = int(h.group(1)) * 60
                        except Exception:
                            pass
                    mins = max(1, min(24 * 60, int(mins or 15)))

                tokens = _tokenize_query(q)
                keyword = " ".join(tokens).strip()

                matches = _find_best_cameras(keyword or q, limit=5)
                if not matches:
                    place_q = _clean_place_phrase(keyword or q)
                    geo = _geocode_place(place_q) if place_q else None
                    if geo and geo.get("lat") is not None and geo.get("lng") is not None:
                        config = load_config() or []
                        closest = None
                        best_d = None
                        for cam in config:
                            lat = cam.get("lat")
                            lng = cam.get("lng")
                            if lat is None or lng is None:
                                continue
                            d = _haversine_km(geo.get("lat"), geo.get("lng"), lat, lng)
                            if d is None:
                                continue
                            if best_d is None or float(d) < float(best_d):
                                best_d = float(d)
                                closest = cam
                        if closest and best_d is not None and float(best_d) <= 6.0:
                            matches = [closest]
                            data["location_match"] = {
                                "query": place_q,
                                "geocoded": {"name": geo.get("name") or place_q, "lat": geo.get("lat"), "lng": geo.get("lng")},
                                "nearest_camera": {"id": closest.get("id"), "name": closest.get("name") or closest.get("id"), "distance_km": round(float(best_d), 2)},
                            }
                        else:
                            data["intent"] = "predict_missing_camera"
                            data["minutes_ahead"] = mins
                            data["camera_keyword"] = keyword
                            data["message"] = "Lokasi terdeteksi, tapi tidak ada kamera terdekat yang punya koordinat (<= 6 km)."
                            if geo:
                                data["location"] = {"name": geo.get("name") or place_q, "lat": geo.get("lat"), "lng": geo.get("lng")}
                            return data
                    else:
                        data["intent"] = "predict_missing_camera"
                        data["minutes_ahead"] = mins
                        data["camera_keyword"] = keyword
                        data["message"] = "Kamera/lokasi tidak ditemukan dari kata kunci."
                        return data

                now_local = datetime.datetime.now().astimezone()
                cam = matches[0]
                cam_id = cam.get("id")
                cam_name = cam.get("name") or cam_id
                thresholds_path = os.path.join(DATA_DIR, "camera_thresholds.json")
                thresholds = {}
                if os.path.exists(thresholds_path):
                    try:
                        with open(thresholds_path, "r") as f:
                            thresholds = json.load(f) or {}
                    except Exception:
                        thresholds = {}
                cam_thr = thresholds.get(cam_id, {"p50": 100, "p75": 200, "p90": 300})
                if horizon_days is not None and int(horizon_days) >= 2:
                    tzname = str(now_local.tzname() or "")
                    start_date = now_local.date()
                    end_date = (now_local + datetime.timedelta(days=int(horizon_days) - 1)).date()
                    daily = []
                    for i in range(int(horizon_days)):
                        dt_day = now_local + datetime.timedelta(days=i)
                        dow = int(dt_day.strftime("%w"))
                        vals = []
                        peak_v = None
                        peak_h = 0
                        for hh in range(24):
                            dt_h = dt_day.replace(hour=hh, minute=0, second=0, microsecond=0)
                            v = int(predict_future_traffic(cam_id, int(dow), int(hh), target_dt_local=dt_h) or 0)
                            vals.append(v)
                            if peak_v is None or v > int(peak_v):
                                peak_v = v
                                peak_h = hh
                        avg_v = int(round(sum(vals) / 24.0)) if vals else 0
                        peak_v = int(peak_v or 0)
                        status = "LANCAR"
                        if peak_v > int(cam_thr.get("p90", 300)):
                            status = "MACET TOTAL"
                        elif peak_v > int(cam_thr.get("p75", 200)):
                            status = "MACET"
                        elif peak_v > int(cam_thr.get("p50", 100)):
                            status = "PADAT LANCAR"
                        daily.append(
                            {
                                "date_local": dt_day.strftime("%Y-%m-%d"),
                                "weekday": dt_day.strftime("%a"),
                                "avg_vph": avg_v,
                                "peak_vph": peak_v,
                                "peak_hour": f"{int(peak_h):02d}:00",
                                "status": status,
                                "thresholds": cam_thr,
                            }
                        )
                    data["intent"] = "forecast_days"
                    data["days"] = int(horizon_days)
                    data["start_date_local"] = str(start_date)
                    data["end_date_local"] = str(end_date)
                    data["timezone_name"] = tzname
                    data["model_granularity"] = "hourly"
                    data["camera"] = {"id": cam_id, "name": cam_name, "lat": cam.get("lat"), "lng": cam.get("lng")}
                    data["daily_forecast"] = daily
                    if len(matches) > 1:
                        data["alternatives"] = [{"id": c.get("id"), "name": c.get("name") or c.get("id")} for c in matches[1:4]]
                    return data

                target_dt = now_local + datetime.timedelta(minutes=mins)
                day_of_week = int(target_dt.strftime("%w"))
                hour = int(target_dt.hour)
                bucket_start = target_dt.replace(minute=0, second=0, microsecond=0)
                bucket_end = bucket_start + datetime.timedelta(minutes=59)
                avg_count = int(predict_future_traffic(cam_id, int(day_of_week), int(hour), target_dt_local=target_dt) or 0)
                status = "LANCAR"
                if avg_count > int(cam_thr.get("p90", 300)):
                    status = "MACET TOTAL"
                elif avg_count > int(cam_thr.get("p75", 200)):
                    status = "MACET"
                elif avg_count > int(cam_thr.get("p50", 100)):
                    status = "PADAT LANCAR"

                data["intent"] = "predict_traffic"
                data["minutes_ahead"] = mins
                data["now_local"] = now_local.strftime("%Y-%m-%d %H:%M")
                data["target_time_local"] = target_dt.strftime("%Y-%m-%d %H:%M")
                data["timezone_name"] = str(target_dt.tzname() or "")
                data["utc_offset"] = target_dt.strftime("%z")
                data["model_granularity"] = "hourly"
                data["bucket_hour_start"] = bucket_start.strftime("%Y-%m-%d %H:%M")
                data["bucket_hour_end"] = bucket_end.strftime("%Y-%m-%d %H:%M")
                data["target_hour"] = f"{hour:02d}:00"
                data["camera"] = {"id": cam_id, "name": cam_name, "lat": cam.get("lat"), "lng": cam.get("lng")}
                data["prediction"] = {"vehicles_per_hour": avg_count, "status": status, "thresholds": cam_thr}
                if len(matches) > 1:
                    data["alternatives"] = [{"id": c.get("id"), "name": c.get("name") or c.get("id")} for c in matches[1:4]]
                return data

            return data

        def fmt_int(n):
            try:
                v = int(n or 0)
            except Exception:
                v = 0
            return f"{v:,}".replace(",", ".")

        def _period_to_seconds(p):
            m = {
                "30m": 30 * 60,
                "1h": 60 * 60,
                "2h": 2 * 60 * 60,
                "6h": 6 * 60 * 60,
                "12h": 12 * 60 * 60,
                "24h": 24 * 60 * 60,
                "7d": 7 * 24 * 60 * 60,
                "30d": 30 * 24 * 60 * 60,
            }
            return int(m.get(str(p or "").strip(), 60 * 60))

        def _cause_analysis_from_system(qd):
            d = qd or {}
            cam = d.get("camera") if isinstance(d.get("camera"), dict) else {}
            cam_id = str(cam.get("id") or "").strip()
            cam_name = str(cam.get("name") or cam_id or "").strip()
            period = str(d.get("period") or "1h").strip()

            now = time.time()
            start_ts = now - float(_period_to_seconds(period))
            rows = get_history_range(camera_id=cam_id, start_ts=start_ts, end_ts=now) if cam_id else []

            last = rows[-1] if rows else {}
            last_density = int((last or {}).get("count") or 0)
            last_new = int((last or {}).get("new_count") or 0)

            def safe_int(x):
                try:
                    return int(x or 0)
                except Exception:
                    return 0

            sum_new = sum(safe_int((r or {}).get("new_count")) for r in rows) if rows else 0
            dens = [safe_int((r or {}).get("count")) for r in rows] if rows else []
            avg_den = int(round(sum(dens) / float(len(dens)))) if dens else 0
            max_den = max(dens) if dens else 0

            # Trend sederhana: bandingkan 1/3 terakhir vs 1/3 awal
            trend = None
            if len(dens) >= 9:
                k = max(1, len(dens) // 3)
                early = dens[:k]
                late = dens[-k:]
                early_avg = int(round(sum(early) / float(len(early)))) if early else 0
                late_avg = int(round(sum(late) / float(len(late)))) if late else 0
                if late_avg > early_avg * 1.25:
                    trend = "naik"
                elif late_avg < early_avg * 0.8:
                    trend = "turun"
                else:
                    trend = "stabil"

            live = None
            try:
                snap = _live_camera_snapshot() or []
                live = next((x for x in snap if isinstance(x, dict) and str(x.get("camera_id") or "") == cam_id), None)
            except Exception:
                live = None

            live_cong = str((live or {}).get("congestion") or "-").strip()
            live_status = str((live or {}).get("status") or "-").strip()
            live_cnt = safe_int((live or {}).get("current_count"))

            # Titik macet sekitar kamera (radius 2.5km)
            nearby = []
            try:
                lat0 = float(cam.get("lat")) if cam.get("lat") not in (None, "") else None
                lng0 = float(cam.get("lng")) if cam.get("lng") not in (None, "") else None
            except Exception:
                lat0 = None
                lng0 = None
            if lat0 is not None and lng0 is not None:
                try:
                    snap = _live_camera_snapshot() or []
                except Exception:
                    snap = []
                for r in snap:
                    if not isinstance(r, dict):
                        continue
                    if r.get("status") != "online":
                        continue
                    cong = str(r.get("congestion") or "")
                    if cong not in ("MACET", "MACET TOTAL"):
                        continue
                    try:
                        rlat = float(r.get("lat")) if r.get("lat") not in (None, "") else None
                        rlng = float(r.get("lng")) if r.get("lng") not in (None, "") else None
                    except Exception:
                        rlat = None
                        rlng = None
                    if rlat is None or rlng is None:
                        continue
                    dist = _haversine_km(lat0, lng0, rlat, rlng)
                    if dist is None or float(dist) > 2.5:
                        continue
                    nearby.append(
                        {
                            "camera_name": r.get("camera_name"),
                            "congestion": cong,
                            "current_count": safe_int(r.get("current_count")),
                            "distance_km": round(float(dist), 2),
                        }
                    )
                def cong_rank(label):
                    if label == "MACET TOTAL":
                        return 3
                    if label == "MACET":
                        return 2
                    return 0
                nearby.sort(key=lambda x: (cong_rank(x.get("congestion")), float(x.get("distance_km") or 999999)), reverse=True)
                nearby = nearby[:6]

            lines = []
            lines.append(f"Ini penjelasan paling masuk akal dari data sistem untuk {cam_name}.")
            lines.append(f"Periode yang aku lihat: {period}.")
            if live_status != "-":
                lines.append(f"Kondisi live sekarang: {live_cong} | status kamera: {live_status} | hitungan live: {fmt_int(live_cnt)}")
            if rows:
                lines.append(f"Yang terlihat dari data: total kendaraan (akumulasi) {fmt_int(sum_new)} di periode ini.")
                lines.append(f"Rata-rata density {fmt_int(avg_den)} (puncak {fmt_int(max_den)}).")
                if trend:
                    lines.append(f"Arah trennya: {trend} (dibanding awal periode).")
                lines.append(f"Update terakhir: density {fmt_int(last_density)}, kenaikan terakhir {fmt_int(last_new)}.")
            else:
                lines.append("Catatan: untuk periode ini, data historis belum kebaca, jadi aku hanya bisa pakai data live.")
            if nearby:
                lines.append("Titik macet di sekitar (live):")
                for r in nearby:
                    lines.append(f"- {r.get('camera_name')} — {r.get('congestion')} ({fmt_int(r.get('current_count'))}), {r.get('distance_km')} km")
                lines.append("Kalau titik sekitar banyak yang macet, biasanya itu efek 'imbas area' (arus saling tarik).")
            else:
                lines.append("Sekitar lokasi ini belum ada titik macet (live) yang terdeteksi dekat, jadi kemungkinan macetnya lebih lokal (mis. simpang ini saja).")
            lines.append("Catatan: sistem ini tidak punya data kejadian (kecelakaan/penutupan jalan), jadi penyebab yang aku jelaskan murni dari pola volume & kepadatan CCTV.")
            return "\n".join([x for x in lines if x]).strip()

        def _analysis_summary_from_system(qd):
            d = qd or {}
            cam = d.get("camera") if isinstance(d.get("camera"), dict) else {}
            cam_id = str(cam.get("id") or "").strip()
            cam_name = str(cam.get("name") or cam_id or "").strip()
            period = str(d.get("period") or "1h").strip()

            now = time.time()
            start_ts = now - float(_period_to_seconds(period))
            rows = get_history_range(camera_id=cam_id, start_ts=start_ts, end_ts=now) if cam_id else []

            n = len(rows)
            sum_new = 0
            sum_density = 0
            max_density = None
            last_ts = None
            last_new = None
            for r in rows:
                try:
                    sum_new += int((r or {}).get("new_count") or 0)
                except Exception:
                    pass
                try:
                    den = int((r or {}).get("count") or 0)
                except Exception:
                    den = 0
                sum_density += den
                max_density = den if max_density is None else max(max_density, den)
                last_ts = (r or {}).get("ts") if (r or {}).get("ts") is not None else last_ts
                last_new = (r or {}).get("new_count") if (r or {}).get("new_count") is not None else last_new

            avg_density = int(round(sum_density / float(n))) if n else 0
            max_density = int(max_density or 0)
            sum_new = int(sum_new)

            live = None
            try:
                snap = _live_camera_snapshot() or []
                live = next((x for x in snap if isinstance(x, dict) and str(x.get("camera_id") or "") == cam_id), None)
            except Exception:
                live = None

            live_status = str((live or {}).get("status") or "-")
            live_cong = str((live or {}).get("congestion") or "-")
            live_count = int((live or {}).get("current_count") or 0) if isinstance(live, dict) else 0

            def explain_cong(s):
                k = str(s or "").upper()
                if k == "LANCAR":
                    return "Lancar: arus masih nyaman."
                if k == "PADAT LANCAR":
                    return "Padat lancar: ramai tapi masih bergerak."
                if k == "MACET":
                    return "Macet: laju cenderung tersendat."
                if k == "MACET TOTAL":
                    return "Macet total: kemungkinan antrean panjang."
                return ""

            lines = []
            lines.append(f"Oke, ini ringkasan untuk {cam_name} berdasarkan data sistem.")
            lines.append(f"Periode yang aku pakai: {period}.")
            if live_status != "-":
                lines.append(f"Kondisi live sekarang: {live_cong} ({explain_cong(live_cong)}) | status kamera: {live_status} | hitungan live: {fmt_int(live_count)}")
            if n:
                lines.append(f"Di periode ini, total kendaraan (akumulasi) yang tercatat: {fmt_int(sum_new)}.")
                lines.append(f"Rata-rata density (keramaian frame) di periode ini: {fmt_int(avg_density)} (puncak: {fmt_int(max_density)}).")
                if last_ts is not None:
                    try:
                        dt = datetime.datetime.fromtimestamp(float(last_ts)).strftime("%H:%M")
                        lines.append(f"Update terakhir sekitar: {dt} (kenaikan terakhir: {fmt_int(last_new)}).")
                    except Exception:
                        pass
            else:
                lines.append("Untuk periode ini, data historisnya belum kebaca (mungkin belum ada rekaman DB / periode terlalu sempit).")

            lines.append("Kalau kamu mau, sebutkan: 'periode 6 jam' atau 'periode 7 hari' biar aku ringkas ulang.")
            return "\n".join([x for x in lines if x]).strip()

        def fallback_from_query(query_data):
            data = query_data or {}
            intent = str(data.get("intent") or "general")

            if intent == "predict_traffic":
                cam = data.get("camera") or {}
                pred = data.get("prediction") or {}
                name = str(cam.get("name") or cam.get("id") or "-")
                status = str(pred.get("status") or "-")
                vph = int(pred.get("vehicles_per_hour") or 0)
                target = str(data.get("target_time_local") or "")
                tz = str(data.get("timezone_name") or "")
                bucket = str(data.get("bucket_hour_start") or "")
                bucket_end = str(data.get("bucket_hour_end") or "")
                mins = int(data.get("minutes_ahead") or 0)
                return (
                    "Qwen sedang sibuk/timeout, jadi aku jawab cepat dari data sistem.\n"
                    f"- Kamera: {name}\n"
                    f"- Prediksi +{mins} menit (target {target} {tz}): {fmt_int(vph)} kendaraan/jam\n"
                    f"- Status: {status}\n"
                    f"- Catatan: model prediksi per jam (bucket {bucket}–{bucket_end})"
                )

            if intent == "top_cameras_30d":
                rows = data.get("top_cameras_30d") or []
                if not rows:
                    return "Qwen sedang sibuk (timeout). Data top kamera 30 hari belum tersedia."
                lines = []
                for i, r in enumerate(rows[:5], 1):
                    lines.append(f"{i}. {r.get('camera_name')}: {fmt_int(r.get('total_30d'))}")
                return "Qwen sedang sibuk (timeout), ini top kamera 30 hari (berdasarkan sistem):\n" + "\n".join(lines)

            if intent == "top_live_density":
                rows = data.get("top_live_density") or []
                if not rows:
                    return "Qwen sedang sibuk (timeout). Data live density belum tersedia."
                lines = []
                for i, r in enumerate(rows[:5], 1):
                    lines.append(f"{i}. {r.get('camera_name')}: {fmt_int(r.get('current_count'))} (live)")
                return "Qwen sedang sibuk (timeout), ini titik terpadat live (berdasarkan sistem):\n" + "\n".join(lines)

            if intent == "route_advice":
                dest = data.get("destination") or {}
                dest_name = str(dest.get("name") or dest.get("query") or "tujuan")
                rec = data.get("recommended_points") or []
                avoid = data.get("avoid_points") or []

                out = ["Qwen sedang sibuk (timeout), tapi ini saran berbasis titik kamera (data live sistem):"]
                out.append(f"- Tujuan: {dest_name}")
                if avoid:
                    out.append("- Hindari (macet):")
                    for r in avoid[:6]:
                        out.append(
                            f"  • {r.get('camera_name')} — {r.get('congestion')} ({fmt_int(r.get('current_count'))}), {r.get('distance_to_destination_km')} km"
                        )
                if rec:
                    out.append("- Rekomendasi (lebih aman):")
                    for r in rec[:6]:
                        out.append(
                            f"  • {r.get('camera_name')} — {r.get('congestion')} ({fmt_int(r.get('current_count'))}), {r.get('distance_to_destination_km')} km"
                        )
                if not rec and not avoid:
                    out.append("- Info: belum ada data kamera terdekat (koordinat tujuan/kamera mungkin belum lengkap).")
                return "\n".join(out)

            if intent == "predict_missing_camera":
                kw = str(data.get("camera_keyword") or "")
                msg = str(data.get("message") or "Kamera tidak ditemukan dari kata kunci.")
                return f"Qwen sedang sibuk (timeout). {msg} (kata kunci: {kw})"

            return None

        def reply(text, suggestions=None, provider=None, model=None, ui_actions=None):
            out = {"reply": text}
            if provider:
                out["provider"] = provider
            if model:
                out["model"] = model
            if ui_actions:
                out["ui_actions"] = ui_actions
            if suggestions:
                out["suggestions"] = suggestions
            if page:
                out["page"] = page
            if ctx_key and text:
                try:
                    add_chat_message(ctx_key, "assistant", str(text), page=page, meta={"provider": provider, "model": model})
                except Exception:
                    pass
            return jsonify(out)

        if not message:
            return reply("Ketik pertanyaan dulu ya.", provider="system")

        qd = build_query_data(message)
        try:
            intent0 = str((qd or {}).get("intent") or "")
            chat_ctx["last_intent"] = intent0
            if intent0 == "predict_traffic":
                cam = (qd or {}).get("camera") if isinstance((qd or {}).get("camera"), dict) else {}
                cid = str(cam.get("id") or "").strip()
                cname = str(cam.get("name") or "").strip()
                if cid:
                    chat_ctx["last_camera_id"] = cid
                if cname:
                    chat_ctx["last_camera_name"] = cname
                tgt = str((qd or {}).get("target_time_local") or "").strip()
                tz = str((qd or {}).get("timezone_name") or "").strip()
                if tgt:
                    chat_ctx["last_target_time_local"] = tgt
                if tz:
                    chat_ctx["last_timezone_name"] = tz
            if intent0 == "forecast_days":
                cam = (qd or {}).get("camera") if isinstance((qd or {}).get("camera"), dict) else {}
                cid = str(cam.get("id") or "").strip()
                cname = str(cam.get("name") or "").strip()
                if cid:
                    chat_ctx["last_camera_id"] = cid
                if cname:
                    chat_ctx["last_camera_name"] = cname
                tz = str((qd or {}).get("timezone_name") or "").strip()
                if tz:
                    chat_ctx["last_timezone_name"] = tz
            if intent0 == "route_advice":
                dest = (qd or {}).get("destination") if isinstance((qd or {}).get("destination"), dict) else {}
                dname = str(dest.get("query") or dest.get("name") or "").strip()
                if dname:
                    chat_ctx["last_destination"] = dname
            if ctx_key:
                try:
                    upsert_chat_profile(
                        ctx_key,
                        {
                            "last_intent": chat_ctx.get("last_intent"),
                            "last_camera_id": chat_ctx.get("last_camera_id"),
                            "last_camera_name": chat_ctx.get("last_camera_name"),
                            "last_destination": chat_ctx.get("last_destination"),
                            "last_target_time_local": chat_ctx.get("last_target_time_local"),
                            "last_timezone_name": chat_ctx.get("last_timezone_name"),
                            "prefs": chat_ctx.get("prefs") if isinstance(chat_ctx.get("prefs"), dict) else None,
                        },
                    )
                except Exception:
                    pass
                try:
                    _chat_ctx_set(ctx_key, chat_ctx)
                except Exception:
                    pass
        except Exception:
            pass
        def build_ui_actions(query_data):
            d = query_data or {}
            intent = str(d.get("intent") or "general")
            cam = d.get("camera") if isinstance(d.get("camera"), dict) else {}
            cam_id = str((cam or {}).get("id") or "").strip()
            cam_name = str((cam or {}).get("name") or cam_id or "").strip()
            acts = []
            if intent in ("predict_traffic", "forecast_days", "select_camera", "analysis_summary") and cam_id:
                if not str(page or "").startswith("/analysis"):
                    acts.append({"type": "navigate", "path": "/analysis"})
                acts.append({"type": "select_camera", "camera_id": cam_id, "camera_name": cam_name})
            if intent == "analysis_summary" and cam_id:
                period = str(d.get("period") or "1h").strip()
                acts.append({"type": "set_period", "period": period})
            if intent == "predict_traffic" and cam_id:
                pred = d.get("prediction") if isinstance(d.get("prediction"), dict) else {}
                lat = cam.get("lat")
                lng = cam.get("lng")
                nearby = []
                try:
                    lat_f = float(lat) if lat not in (None, "") else None
                    lng_f = float(lng) if lng not in (None, "") else None
                except Exception:
                    lat_f = None
                    lng_f = None
                if lat_f is not None and lng_f is not None:
                    try:
                        snapshot = _live_camera_snapshot() or []
                    except Exception:
                        snapshot = []
                    rows = []
                    for r in snapshot:
                        if not isinstance(r, dict):
                            continue
                        if r.get("status") != "online":
                            continue
                        cong = str(r.get("congestion") or "")
                        if cong not in ("MACET", "MACET TOTAL"):
                            continue
                        rlat = r.get("lat")
                        rlng = r.get("lng")
                        try:
                            rlat_f = float(rlat) if rlat not in (None, "") else None
                            rlng_f = float(rlng) if rlng not in (None, "") else None
                        except Exception:
                            rlat_f = None
                            rlng_f = None
                        if rlat_f is None or rlng_f is None:
                            continue
                        dist = _haversine_km(lat_f, lng_f, rlat_f, rlng_f)
                        if dist is None:
                            continue
                        if float(dist) > 2.5:
                            continue
                        rows.append(
                            {
                                "camera_id": r.get("camera_id"),
                                "camera_name": r.get("camera_name"),
                                "lat": rlat_f,
                                "lng": rlng_f,
                                "congestion": cong,
                                "current_count": int(r.get("current_count") or 0),
                                "distance_km": round(float(dist), 2),
                            }
                        )
                    def cong_rank(label):
                        if label == "MACET TOTAL":
                            return 3
                        if label == "MACET":
                            return 2
                        return 0
                    rows.sort(key=lambda x: (cong_rank(x.get("congestion")), float(x.get("distance_km") or 999999)), reverse=True)
                    nearby = rows[:8]
                acts.append(
                    {
                        "type": "show_prediction_popup",
                        "camera_id": cam_id,
                        "camera_name": cam_name,
                        "lat": lat,
                        "lng": lng,
                        "target_time_local": d.get("target_time_local"),
                        "timezone_name": d.get("timezone_name"),
                        "minutes_ahead": d.get("minutes_ahead"),
                        "vehicles_per_hour": pred.get("vehicles_per_hour"),
                        "status": pred.get("status"),
                        "thresholds": pred.get("thresholds"),
                        "nearby_congested": nearby,
                    }
                )
            if intent == "forecast_days" and cam_id:
                lat = cam.get("lat")
                lng = cam.get("lng")
                nearby = []
                try:
                    lat_f = float(lat) if lat not in (None, "") else None
                    lng_f = float(lng) if lng not in (None, "") else None
                except Exception:
                    lat_f = None
                    lng_f = None
                if lat_f is not None and lng_f is not None:
                    try:
                        snapshot = _live_camera_snapshot() or []
                    except Exception:
                        snapshot = []
                    rows = []
                    for r in snapshot:
                        if not isinstance(r, dict):
                            continue
                        if r.get("status") != "online":
                            continue
                        cong = str(r.get("congestion") or "")
                        if cong not in ("MACET", "MACET TOTAL"):
                            continue
                        rlat = r.get("lat")
                        rlng = r.get("lng")
                        try:
                            rlat_f = float(rlat) if rlat not in (None, "") else None
                            rlng_f = float(rlng) if rlng not in (None, "") else None
                        except Exception:
                            rlat_f = None
                            rlng_f = None
                        if rlat_f is None or rlng_f is None:
                            continue
                        dist = _haversine_km(lat_f, lng_f, rlat_f, rlng_f)
                        if dist is None:
                            continue
                        if float(dist) > 2.5:
                            continue
                        rows.append(
                            {
                                "camera_id": r.get("camera_id"),
                                "camera_name": r.get("camera_name"),
                                "lat": rlat_f,
                                "lng": rlng_f,
                                "congestion": cong,
                                "current_count": int(r.get("current_count") or 0),
                                "distance_km": round(float(dist), 2),
                            }
                        )
                    def cong_rank(label):
                        if label == "MACET TOTAL":
                            return 3
                        if label == "MACET":
                            return 2
                        return 0
                    rows.sort(key=lambda x: (cong_rank(x.get("congestion")), float(x.get("distance_km") or 999999)), reverse=True)
                    nearby = rows[:10]

                acts.append(
                    {
                        "type": "show_forecast_popup",
                        "camera_id": cam_id,
                        "camera_name": cam_name,
                        "lat": lat,
                        "lng": lng,
                        "timezone_name": d.get("timezone_name"),
                        "days": d.get("days"),
                        "start_date_local": d.get("start_date_local"),
                        "end_date_local": d.get("end_date_local"),
                        "daily_forecast": d.get("daily_forecast"),
                        "nearby_congested": nearby,
                    }
                )
            return acts or None

        ui_actions = build_ui_actions(qd)
        if str((qd or {}).get("intent") or "") == "select_camera" and ui_actions:
            cam = (qd or {}).get("camera") if isinstance((qd or {}).get("camera"), dict) else {}
            cam_name = str((cam or {}).get("name") or (cam or {}).get("id") or "").strip()
            return reply(f"Oke, aku ganti tampilan ke kamera: {cam_name or '-'}", provider="system", ui_actions=ui_actions)

        if str((qd or {}).get("intent") or "") == "analysis_summary":
            return reply(_analysis_summary_from_system(qd), provider="system", ui_actions=ui_actions)

        if str((qd or {}).get("intent") or "") == "cause_analysis":
            return reply(_cause_analysis_from_system(qd), provider="system", ui_actions=ui_actions)

        if str((qd or {}).get("intent") or "general") == "general":
            try:
                area = _area_live_summary_from_system(message, page=page)
            except Exception:
                area = None
            if area and area.get("reply"):
                return reply(str(area.get("reply")), provider="system", ui_actions=area.get("ui_actions"))

        def _predict_reply_from_system(qd, ui_actions):
            d = qd or {}
            cam = d.get("camera") if isinstance(d.get("camera"), dict) else {}
            pred = d.get("prediction") if isinstance(d.get("prediction"), dict) else {}
            cam_name = str(cam.get("name") or cam.get("id") or "-").strip()

            vph = pred.get("vehicles_per_hour")
            status = str(pred.get("status") or "-").strip()
            mins = int(d.get("minutes_ahead") or 0)
            target = str(d.get("target_time_local") or "").strip()
            tz = str(d.get("timezone_name") or "").strip()
            thr = pred.get("thresholds") if isinstance(pred.get("thresholds"), dict) else {}

            popup = None
            try:
                for a in (ui_actions or []):
                    if isinstance(a, dict) and a.get("type") == "show_prediction_popup":
                        popup = a
                        break
            except Exception:
                popup = None

            nearby = (popup or {}).get("nearby_congested") if isinstance((popup or {}).get("nearby_congested"), list) else []

            def explain_status(s):
                k = str(s or "").upper()
                if k == "LANCAR":
                    return "Artinya arus masih nyaman."
                if k == "PADAT LANCAR":
                    return "Artinya ramai, tapi masih bergerak."
                if k == "MACET":
                    return "Artinya laju cenderung tersendat."
                if k == "MACET TOTAL":
                    return "Artinya kemungkinan antrean panjang."
                return ""

            if vph in (None, "", 0) and status in ("-", ""):
                return f"Untuk {cam_name}, prediksi untuk {target} {tz} belum tersedia dari data sistem."

            lines = []
            lines.append(f"Prediksi untuk {cam_name}:")
            if target or tz:
                lines.append(f"- Waktu target: {target} {tz}".strip())
            if mins:
                lines.append(f"- Kira-kira: +{mins} menit dari sekarang")
            if vph not in (None, ""):
                try:
                    lines.append(f"- Estimasi volume: {fmt_int(int(vph))} kendaraan/jam")
                except Exception:
                    lines.append(f"- Estimasi volume: {vph} kendaraan/jam")
            if status and status != "-":
                extra = explain_status(status)
                lines.append(f"- Kondisi: {status}{(' — ' + extra) if extra else ''}")
            if thr:
                p50 = thr.get("p50")
                p75 = thr.get("p75")
                p90 = thr.get("p90")
                try:
                    lines.append(f"- Patokan sistem: p50={fmt_int(p50)} • p75={fmt_int(p75)} • p90={fmt_int(p90)} (makin besar makin padat)")
                except Exception:
                    pass
            if nearby:
                lines.append("- Titik macet di sekitar (live):")
                for r in nearby[:6]:
                    name = str((r or {}).get("camera_name") or "-")
                    cong = str((r or {}).get("congestion") or "-")
                    cnt = fmt_int((r or {}).get("current_count"))
                    dist = (r or {}).get("distance_km")
                    dist_s = f"{dist} km" if dist is not None else "- km"
                    lines.append(f"  • {name} — {cong} ({cnt}), {dist_s}")
            else:
                lines.append("- Info sekitar: belum ada titik macet (live) yang terdeteksi dekat lokasi ini.")
            lines.append("Grafik dan peta sudah aku tampilkan di halaman Analytics.")
            return "\n".join([x for x in lines if x]).strip()

        if str((qd or {}).get("intent") or "") == "predict_traffic":
            return reply(_predict_reply_from_system(qd, ui_actions), provider="system", ui_actions=ui_actions)

        def _forecast_reply_from_system(qd, ui_actions):
            d = qd or {}
            cam = d.get("camera") if isinstance(d.get("camera"), dict) else {}
            cam_name = str(cam.get("name") or cam.get("id") or "-").strip()
            tz = str(d.get("timezone_name") or "").strip()
            days = int(d.get("days") or 0)
            start_date = str(d.get("start_date_local") or "").strip()
            end_date = str(d.get("end_date_local") or "").strip()
            items = d.get("daily_forecast") if isinstance(d.get("daily_forecast"), list) else []

            popup = None
            try:
                for a in (ui_actions or []):
                    if isinstance(a, dict) and a.get("type") == "show_forecast_popup":
                        popup = a
                        break
            except Exception:
                popup = None
            nearby = (popup or {}).get("nearby_congested") if isinstance((popup or {}).get("nearby_congested"), list) else []

            lines = []
            lines.append(f"Prediksi {days} hari ke depan untuk {cam_name} (berdasarkan pola historis per jam).")
            if start_date or end_date or tz:
                lines.append(f"Rentang: {start_date} s/d {end_date} {tz}".strip())
            lines.append("Catatan: ini perkiraan rata-rata per jam per hari, bukan kepastian kejadian di lapangan.")
            lines.append("")
            if items:
                lines.append("Ringkasan harian (avg/peak per jam):")
                for it in items[: max(1, min(31, len(items)))]:
                    dt = str((it or {}).get("date_local") or "")
                    wd = str((it or {}).get("weekday") or "").strip()
                    avg_v = fmt_int((it or {}).get("avg_vph"))
                    peak_v = fmt_int((it or {}).get("peak_vph"))
                    peak_h = str((it or {}).get("peak_hour") or "").strip()
                    st = str((it or {}).get("status") or "-").strip()
                    tail = f" (puncak {peak_h} ~{peak_v}/jam)" if peak_h else f" (puncak ~{peak_v}/jam)"
                    lines.append(f"- {dt} {wd}: {st} • rata-rata ~{avg_v}/jam{tail}")
            else:
                lines.append("Data prediksi harian belum tersedia untuk kamera ini.")

            if nearby:
                lines.append("")
                lines.append("Titik macet live di sekitar kamera (saat ini):")
                for r in nearby[:6]:
                    name = str((r or {}).get("camera_name") or "-")
                    cong = str((r or {}).get("congestion") or "-")
                    cnt = fmt_int((r or {}).get("current_count"))
                    dist = (r or {}).get("distance_km")
                    dist_s = f"{dist} km" if dist is not None else "- km"
                    lines.append(f"  • {name} — {cong} ({cnt}), {dist_s}")

            lines.append("")
            lines.append("Aku sudah buka popup forecast di halaman Analytics.")
            return "\n".join([x for x in lines if x]).strip()

        if str((qd or {}).get("intent") or "") == "forecast_days":
            return reply(_forecast_reply_from_system(qd, ui_actions), provider="system", ui_actions=ui_actions)

        def _top_30d_reply_from_system(qd):
            rows = (qd or {}).get("top_cameras_30d") or []
            if not rows:
                return "Data top kamera 30 hari belum tersedia di sistem."
            lines = ["Top CCTV (30 hari terakhir) berdasarkan data sistem:"]
            for i, r in enumerate(rows[:8], 1):
                nm = str((r or {}).get("camera_name") or "-")
                tot = fmt_int((r or {}).get("total_30d"))
                lines.append(f"{i}. {nm}: {tot}")
            return "\n".join(lines).strip()

        def _top_live_reply_from_system(qd):
            rows = (qd or {}).get("top_live_density") or []
            if not rows:
                return "Data live density belum tersedia di sistem (kamera live belum kebaca)."
            lines = ["Titik terpadat (live) berdasarkan data sistem:"]
            for i, r in enumerate(rows[:8], 1):
                nm = str((r or {}).get("camera_name") or "-")
                cnt = fmt_int((r or {}).get("current_count"))
                cong = str((r or {}).get("congestion") or "").strip()
                tail = f" — {cong}" if cong else ""
                lines.append(f"{i}. {nm}: {cnt}{tail}")
            return "\n".join(lines).strip()

        def _ack_from_actions(actions, qd):
            acts = actions or []
            cam_id = ""
            cam_name = ""
            period = ""
            opened_analysis = False
            showed_popup = False
            for a in acts:
                if not isinstance(a, dict):
                    continue
                if a.get("type") == "navigate" and str(a.get("path") or "") == "/analysis":
                    opened_analysis = True
                if a.get("type") == "select_camera":
                    cam_id = str(a.get("camera_id") or cam_id or "")
                    cam_name = str(a.get("camera_name") or cam_name or "")
                if a.get("type") == "set_period":
                    period = str(a.get("period") or period or "")
                if a.get("type") in ("show_prediction_popup", "show_forecast_popup"):
                    showed_popup = True

            nm = cam_name or cam_id
            bits = []
            if opened_analysis:
                bits.append("Aku sudah buka halaman Analytics.")
            if nm:
                bits.append(f"Aku sudah pilih kamera: {nm}.")
            if period:
                bits.append(f"Periode grafik: {period}.")
            if showed_popup:
                bits.append("Aku juga munculkan popup peta + detailnya.")
            if bits:
                return "Sip. " + " ".join(bits).strip()

            it = str((qd or {}).get("intent") or "general")
            if it == "general":
                return "Siap. Sebutkan nama kamera/titik atau tujuan lokasinya, nanti aku ambil datanya dari sistem dan tampilkan di UI."
            return "Oke."

        intent_now = str((qd or {}).get("intent") or "")
        if intent_now == "top_cameras_30d":
            return reply(_top_30d_reply_from_system(qd), provider="system", ui_actions=ui_actions)
        if intent_now == "top_live_density":
            return reply(_top_live_reply_from_system(qd), provider="system", ui_actions=ui_actions)
        if intent_now == "predict_missing_camera":
            msg = str((qd or {}).get("message") or "Kamera tidak ditemukan dari kata kunci.").strip()
            alts = (qd or {}).get("alternatives") or []
            lines = [msg]
            if alts:
                lines.append("Coba pilih salah satu ini:")
                for i, a in enumerate(alts[:6], 1):
                    lines.append(f"{i}. {a.get('name') or a.get('id')}")
            return reply("\n".join([x for x in lines if x]).strip(), provider="system", ui_actions=ui_actions)

        if intent_now == "general":
            q0 = str(message or "").strip().lower()
            is_greet = bool(re.search(r"\b(hai|halo|hello|hi|selamat pagi|selamat siang|selamat sore|selamat malam|pagi|siang|sore|malam)\b", q0))
            is_thanks = bool(re.search(r"\b(makasih|terima kasih|thx|thanks)\b", q0))
            # Greetings handled by AI provider (no hardcoded response)
            if is_thanks:
                return reply("Sama-sama! Silakan tanya lagi kalau butuh bantuan.", provider="system", ui_actions=ui_actions)

        def format_route_advice(data):
            d = data or {}
            dest = d.get("destination") if isinstance(d.get("destination"), dict) else {}
            dest_name = str(dest.get("name") or dest.get("query") or "tujuan")
            dest_via = str(dest.get("via") or "").strip()
            dest_lat = dest.get("lat")
            dest_lng = dest.get("lng")
            tz = str(d.get("timezone_name") or "").strip()
            now_local = str(d.get("now_local") or "").strip()
            live_cams = d.get("live_cameras_count")
            avoid = d.get("avoid_points") or []
            rec = d.get("recommended_points") or []
            wp = d.get("suggested_waypoints") or []

            def fmt_point(r):
                name = str((r or {}).get("camera_name") or "-")
                cong = str((r or {}).get("congestion") or "-")
                cnt = fmt_int((r or {}).get("current_count"))
                dist = (r or {}).get("distance_to_destination_km")
                dist_s = f"{dist} km" if dist is not None else "- km"
                return f"{name} — {cong} ({cnt}), {dist_s}"

            lines = [f"Oke, aku bantu arah ke {dest_name} ya."]
            lines.append("Aku tidak menebak nama jalan. Aku pakai CCTV yang ada di sekitar tujuan untuk lihat area mana yang lagi padat/macet.")
            if now_local or tz:
                lines.append(f"Patokan waktunya: {now_local or '-'} {tz}".strip())
            if live_cams is not None:
                try:
                    lines.append(f"Jumlah CCTV live yang kebaca sistem: {int(live_cams)}")
                except Exception:
                    pass
            if dest_via:
                lines.append(f"Tujuan dikenali lewat: {dest_via}")
            if dest_lat is None or dest_lng is None:
                lines.append("Catatan: koordinat tujuan belum lengkap, jadi pemilihan CCTV terdekat bisa kurang akurat.")

            lines.append("")
            lines.append("1) Area yang sebaiknya dihindari (padat/macet):")
            if avoid:
                for i, r in enumerate(avoid[:6], 1):
                    lines.append(f"   {i}. {fmt_point(r)}")
            else:
                lines.append("   - Untuk saat ini, sistem tidak melihat titik macet di sekitar tujuan.")

            lines.append("")
            lines.append("2) Area yang relatif lebih aman (lebih lancar):")
            if rec:
                for i, r in enumerate(rec[:6], 1):
                    lines.append(f"   {i}. {fmt_point(r)}")
            else:
                lines.append("   - Belum ada rekomendasi titik yang jelas lebih lancar (data live/koordinat CCTV mungkin belum lengkap).")

            if wp:
                names = [str(x.get("camera_name") or "").strip() for x in wp if isinstance(x, dict)]
                names = [n for n in names if n]
                if names:
                    lines.append("")
                    lines.append("3) Titik pantau yang disarankan:")
                    for i, n in enumerate(names[:6], 1):
                        lines.append(f"   {i}. {n}")

            return "\n".join(lines).strip()

        if str((qd or {}).get("intent") or "") == "route_advice":
            return reply(format_route_advice(qd), provider="system")

        # Skip planner for simple greetings — let AI provider handle naturally
        if not is_greet and not (_ai_cfg_main.AI_USE_FOR_CHAT and _ai_cfg_main.AI_API_KEY) and _planner_enabled() and ollama_enabled() and _ollama_reachable(timeout_s=4):
            plan_reply, plan_actions, plan_err = call_ollama_planner(message, qd, ui_actions)
            merged_actions = []
            if ui_actions:
                merged_actions.extend(ui_actions)
            if plan_actions:
                merged_actions.extend(plan_actions)
            if merged_actions:
                seen = set()
                dedup = []
                for a in merged_actions:
                    if not isinstance(a, dict):
                        continue
                    key = json.dumps(a, sort_keys=True, ensure_ascii=False)
                    if key in seen:
                        continue
                    seen.add(key)
                    dedup.append(a)
                merged_actions = dedup
            if merged_actions and _chat_grounded_mode_enabled():
                return reply(_ack_from_actions(merged_actions, qd), provider="system", ui_actions=merged_actions)
            if plan_err:
                try:
                    add_chat_message(ctx_key, "assistant", f"[planner_error] {plan_err}", page=page, meta={"provider": "planner"})
                except Exception:
                    pass

        if _chat_grounded_mode_enabled():
            if intent_now != "general":
                fb = fallback_from_query(qd)
                if fb:
                    return reply(fb, provider="system", ui_actions=ui_actions)
            if intent_now == "general" and not (ollama_enabled() and _ollama_reachable(timeout_s=4)):
                # Check if AI provider is available as alternative
                import app.config as _ai_cfg3
                if _ai_cfg3.AI_USE_FOR_CHAT and _ai_cfg3.AI_API_KEY:
                    pass  # Don't block — AI provider will handle it below
                else:
                    return reply(
                        "LLM belum tersambung. Konfigurasi AI provider di Settings atau jalankan Ollama.",
                        provider="system",
                        ui_actions=ui_actions,
                    )

        # Try AI provider FIRST (before Ollama)
        import app.config as _ai_cfg_main
        if _ai_cfg_main.AI_USE_FOR_CHAT and _ai_cfg_main.AI_API_KEY and _ai_cfg_main.AI_BASE_URL:
            try:
                # Build context with real system data
                _sys_context = ""
                try:
                    from app.utils import load_config
                    from app.database import get_violations_summary, get_history_range
                    import app.globals as _g_chat
                    
                    cameras = load_config() or []
                    cam_names = [c.get("name", "") for c in cameras]
                    
                    # Get violation stats
                    violations_info = ""
                    try:
                        v_summary = get_violations_summary(hours=24) or {}
                        violations_info = f"Pelanggaran 24 jam terakhir: total={v_summary.get('total', 0)}, by_type={v_summary.get('by_type', {})}, by_camera={v_summary.get('by_camera', [])[:3]}"
                    except Exception:
                        violations_info = "Data pelanggaran tersedia via API /api/violations/executive_summary"
                    
                    # Get live traffic stats
                    live_stats = ""
                    try:
                        stats = _g_chat.global_stats or {}
                        live_info = []
                        for cam_id, s in stats.items():
                            if isinstance(s, dict):
                                live_info.append(f"{s.get('name','?')}: {s.get('current_count',0)} kendaraan")
                        live_stats = "Live traffic: " + ", ".join(live_info[:5]) if live_info else ""
                    except Exception:
                        pass
                    
                    _sys_context = f"""
DATA SISTEM SAAT INI:
- Kamera aktif: {len(cameras)} ({', '.join(cam_names[:5])})
- {violations_info}
- {live_stats}
- Enforcement: parkir liar (60s), busway (5s), jalur sepeda (5s), lawan arah (8s)
"""
                except Exception:
                    _sys_context = ""
                
                system_prompt = f"""Kamu adalah PADI Assistant — asisten AI untuk Platform Adaptive Deteksi Indonesia.
Sistem ini adalah platform monitoring lalu lintas real-time berbasis CCTV + AI (YOLO detection + enforcement otomatis).

KEMAMPUAN SISTEM:
- Deteksi kendaraan real-time (mobil, motor, bus) di multiple CCTV
- Enforcement otomatis: parkir liar, busway, jalur sepeda, lawan arah
- ANPR (baca plat nomor) dengan PaddleOCR + AI correction
- Executive Summary / laporan pelanggaran
- CRM monitoring social media (@DishubDKI di X.com)
- Analytics & export data CSV

{_sys_context}

ATURAN:
- Jawab dalam Bahasa Indonesia, ringkas dan informatif
- Jika ditanya data spesifik yang kamu punya di context, berikan jawabannya
- Jika ditanya hal yang tidak ada di context, arahkan user ke halaman yang relevan (Enforcement, Analytics, Executive Summary)
- Jangan bilang "tidak memiliki data" jika informasinya ada di context di atas"""

                ai_messages = [{"role": "system", "content": system_prompt}]
                ai_messages.append({"role": "user", "content": message})
                ai_body = {
                    "model": _ai_cfg_main.AI_CHAT_MODEL or _ai_cfg_main.AI_MODEL,
                    "messages": ai_messages,
                    "max_tokens": 1000,
                    "temperature": 0.3,
                }
                ai_url = f"{_ai_cfg_main.AI_BASE_URL.rstrip('/')}/chat/completions"
                ai_headers = {"Content-Type": "application/json", "Authorization": f"Bearer {_ai_cfg_main.AI_API_KEY}"}
                ai_data = json.dumps(ai_body).encode("utf-8")
                ai_req = urllib.request.Request(ai_url, data=ai_data, headers=ai_headers, method="POST")
                with urllib.request.urlopen(ai_req, timeout=30) as ai_resp:
                    ai_result = json.loads(ai_resp.read().decode("utf-8"))
                ai_content = ai_result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if ai_content:
                    return reply(ai_content, provider=_ai_cfg_main.AI_PROVIDER, model=_ai_cfg_main.AI_CHAT_MODEL or _ai_cfg_main.AI_MODEL, ui_actions=ui_actions)
            except Exception as ai_err:
                print(f"[CHAT] AI provider failed: {ai_err}")

        if ollama_enabled() and _ollama_reachable(timeout_s=4):
            if llm:
                return reply(llm, provider="ollama", model=_ollama_resolve_model() or "", ui_actions=ui_actions)
            return reply(
                f"Aku gagal mengambil jawaban dari Qwen (Ollama).\nDetail: {llm_err or '-'}",
                provider="ollama",
                ui_actions=ui_actions,
                suggestions=["status kamera", "total 30 hari", "top kamera 30 hari", "export csv"],
            )

        fb = fallback_from_query(qd)
        if fb:
            return reply(fb, provider="system", model=_ollama_resolve_model() or "", ui_actions=ui_actions)
        return reply(
            "LLM masih belum siap (Ollama belum reachable atau model tidak terdeteksi). Cek /api/chat/health dulu ya.",
            suggestions=["status kamera", "total 30 hari", "top kamera 30 hari", "export csv"],
            provider="system",
        )
    finally:
        if locked:
            try:
                globals_state.chat_lock.release()
            except Exception:
                pass

@bp.route("/api/chat/health")
def chat_health_api():
    base_url = _ollama_base_url()
    env_model = _ollama_env_model()
    resolved_model = _ollama_resolve_model()
    enabled = bool(resolved_model)

    reachable = False
    error = None
    models = []
    try:
        models = _ollama_fetch_models(timeout_s=4)
        reachable = True
    except Exception as e:
        reachable = False
        error = str(e)

    test_mode = str(request.args.get("test") or "").strip() in ("1", "true", "yes")
    chat_test_ok = None
    chat_test_error = None
    if test_mode and enabled and reachable:
        try:
            body = {
                "model": resolved_model,
                "stream": False,
                "think": False,
                "messages": [
                    {"role": "system", "content": "Jawab hanya satu kata: pong. Jangan tampilkan reasoning/thinking."},
                    {"role": "user", "content": "ping"},
                ],
                "options": {"temperature": 0.0, "num_predict": 32},
            }
            data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/api/chat",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            parsed = json.loads(raw) if raw else {}
            msg = (parsed.get("message") or {}) if isinstance(parsed, dict) else {}
            content = str((msg.get("content") or "")).strip()
            done_reason = str((parsed.get("done_reason") or "")).strip() if isinstance(parsed, dict) else ""
            has_thinking = bool(str((msg.get("thinking") or "")).strip())
            chat_test_ok = bool(content)
            if not chat_test_ok:
                if has_thinking:
                    chat_test_error = f"content kosong; thinking terisi; done_reason={done_reason or '-'}"
                else:
                    chat_test_error = f"content kosong; done_reason={done_reason or '-'}"
        except Exception as e:
            chat_test_ok = False
            chat_test_error = str(e)

    # Check if AI provider is configured (takes priority over Ollama)
    import app.config as _ai_health
    ai_active = bool(_ai_health.AI_USE_FOR_CHAT and _ai_health.AI_API_KEY and _ai_health.AI_BASE_URL)
    
    if ai_active:
        # Report AI provider as the active LLM
        return jsonify({
            "enabled": True,
            "url": _ai_health.AI_BASE_URL,
            "model": _ai_health.AI_CHAT_MODEL or _ai_health.AI_MODEL,
            "env_model": "",
            "available_models": [_ai_health.AI_CHAT_MODEL or _ai_health.AI_MODEL],
            "reachable": True,
            "error": None,
            "chat_test_ok": None,
            "chat_test_error": None,
            "provider": _ai_health.AI_PROVIDER,
        })

    return jsonify(
        {
            "enabled": enabled,
            "url": base_url,
            "model": resolved_model or "",
            "env_model": env_model or "",
            "available_models": models,
            "reachable": reachable,
            "error": error,
            "chat_test_ok": chat_test_ok,
            "chat_test_error": chat_test_error,
            "provider": "ollama",
        }
    )


def _ai_tools_spec():
    return [
        {
            "name": "cameras",
            "description": "Daftar kamera dari config (id, name, lat, lng) + status live jika tersedia.",
            "args": {"include_live": "bool (default true)"},
        },
        {
            "name": "live",
            "description": "Snapshot live semua kamera (status, congestion, current_count, lat/lng jika ada).",
            "args": {"limit": "int (default 200)"},
        },
        {
            "name": "history",
            "description": "Ambil history traffic dari SQLite untuk kamera tertentu pada rentang waktu.",
            "args": {"camera_id": "string", "start_ts": "float", "end_ts": "float", "limit": "int (default 5000)"},
        },
        {
            "name": "aggregates",
            "description": "Agregasi global untuk N hari terakhir (accumulated_count, cars, motorcycles).",
            "args": {"days": "int (default 30)"},
        },
        {
            "name": "nearest_cameras",
            "description": "Cari kamera terdekat dari koordinat atau nama lokasi (geocode).",
            "args": {"place": "string (optional)", "lat": "float (optional)", "lng": "float (optional)", "limit": "int (default 8)", "max_km": "float (default 8.0)"},
        },
    ]


def _ai_exec_tool(name, args):
    tool = str(name or "").strip()
    a = args if isinstance(args, dict) else {}

    if tool == "cameras":
        include_live = True
        if "include_live" in a:
            include_live = bool(a.get("include_live"))
        config = load_config() or []
        live_map = {}
        if include_live:
            try:
                snap = _live_camera_snapshot() or []
            except Exception:
                snap = []
            for r in snap:
                if not isinstance(r, dict):
                    continue
                cid = str(r.get("camera_id") or "").strip()
                if not cid:
                    continue
                live_map[cid] = {
                    "status": r.get("status"),
                    "congestion": r.get("congestion"),
                    "current_count": r.get("current_count"),
                }
        out = []
        for cam in config:
            cid = str(cam.get("id") or "").strip()
            if not cid:
                continue
            row = {
                "id": cid,
                "name": cam.get("name") or cid,
                "lat": cam.get("lat"),
                "lng": cam.get("lng"),
            }
            if include_live and cid in live_map:
                row.update(live_map[cid])
            out.append(row)
        return {"items": out}

    if tool == "live":
        try:
            limit = int(a.get("limit") or 200)
        except Exception:
            limit = 200
        limit = max(1, min(500, limit))
        try:
            snap = _live_camera_snapshot() or []
        except Exception:
            snap = []
        items = []
        for r in snap:
            if not isinstance(r, dict):
                continue
            items.append(
                {
                    "camera_id": r.get("camera_id"),
                    "camera_name": r.get("camera_name"),
                    "status": r.get("status"),
                    "congestion": r.get("congestion"),
                    "current_count": r.get("current_count"),
                    "lat": r.get("lat"),
                    "lng": r.get("lng"),
                }
            )
        return {"items": items[:limit]}

    if tool == "history":
        cam_id = str(a.get("camera_id") or "").strip()
        if not cam_id:
            return {"error": "camera_id wajib diisi."}
        try:
            start_ts = float(a.get("start_ts")) if a.get("start_ts") is not None else None
        except Exception:
            start_ts = None
        try:
            end_ts = float(a.get("end_ts")) if a.get("end_ts") is not None else None
        except Exception:
            end_ts = None
        try:
            limit = int(a.get("limit") or 5000)
        except Exception:
            limit = 5000
        limit = max(1, min(20000, limit))

        if start_ts is None or end_ts is None:
            now = time.time()
            end_ts = now if end_ts is None else end_ts
            start_ts = (end_ts - 3600.0) if start_ts is None else start_ts

        if end_ts < start_ts:
            start_ts, end_ts = end_ts, start_ts

        rows = get_history_range(camera_id=cam_id, start_ts=start_ts, end_ts=end_ts) or []
        if len(rows) > limit:
            rows = rows[-limit:]
        return {"camera_id": cam_id, "start_ts": start_ts, "end_ts": end_ts, "rows": rows, "count": len(rows)}

    if tool == "aggregates":
        try:
            days = int(a.get("days") or 30)
        except Exception:
            days = 30
        days = max(1, min(365, days))
        return {"days": days, "stats": get_aggregated_stats(days=days) or {}}

    if tool == "nearest_cameras":
        place = str(a.get("place") or "").strip()
        lat = a.get("lat")
        lng = a.get("lng")
        try:
            limit = int(a.get("limit") or 8)
        except Exception:
            limit = 8
        limit = max(1, min(20, limit))
        try:
            max_km = float(a.get("max_km") or 8.0)
        except Exception:
            max_km = 8.0
        max_km = max(0.2, min(50.0, max_km))

        geo = None
        if place:
            try:
                geo = _geocode_place(place)
            except Exception:
                geo = None
        lat0 = None
        lng0 = None
        if geo and geo.get("lat") is not None and geo.get("lng") is not None:
            lat0 = geo.get("lat")
            lng0 = geo.get("lng")
        else:
            try:
                lat0 = float(lat) if lat not in (None, "") else None
                lng0 = float(lng) if lng not in (None, "") else None
            except Exception:
                lat0 = None
                lng0 = None
        if lat0 is None or lng0 is None:
            return {"error": "Berikan place atau lat/lng."}

        config = load_config() or []
        scored = []
        for cam in config:
            clat = cam.get("lat")
            clng = cam.get("lng")
            if clat in (None, "") or clng in (None, ""):
                continue
            try:
                clat_f = float(clat)
                clng_f = float(clng)
            except Exception:
                continue
            d = _haversine_km(lat0, lng0, clat_f, clng_f)
            if d is None:
                continue
            if float(d) > max_km:
                continue
            scored.append((float(d), cam))
        scored.sort(key=lambda x: x[0])
        items = []
        for dist, cam in scored[:limit]:
            cid = cam.get("id")
            items.append(
                {
                    "id": cid,
                    "name": cam.get("name") or cid,
                    "lat": cam.get("lat"),
                    "lng": cam.get("lng"),
                    "distance_km": round(float(dist), 2),
                }
            )
        return {
            "query": {"place": place or None, "lat": lat0, "lng": lng0, "max_km": max_km, "limit": limit},
            "geocode": geo,
            "items": items,
        }

    return {"error": f"Tool tidak dikenal: {tool}"}


def _ai_autotool_plan(user_text, chat_ctx=None):
    q_raw = str(user_text or "").strip()
    q = _normalize_text(q_raw).lower()
    calls = []

    def add(tool, args):
        calls.append({"tool": tool, "args": args if isinstance(args, dict) else {}})

    want_live = any(k in q for k in ["live", "status kamera", "status cctv", "kamera online", "kondisi kamera", "kondisi live", "sekarang"])
    want_cameras = any(k in q for k in ["daftar kamera", "list kamera", "kamera apa saja", "cctv apa saja", "daftar cctv", "list cctv"])
    want_agg = any(k in q for k in ["total", "akumulasi", "rekap"]) and any(k in q for k in ["hari", "bulan", "30", "7", "24h", "30d", "7d"])

    if want_cameras:
        add("cameras", {"include_live": True})
        want_live = True

    if want_agg:
        days = 30
        if "7 hari" in q or "7d" in q:
            days = 7
        elif "30 hari" in q or "30d" in q or "bulan ini" in q:
            days = 30
        add("aggregates", {"days": days})

    cam = None
    try:
        tokens = _tokenize_query(q)
        keyword = " ".join(tokens).strip()
        matches = _find_best_cameras(keyword or q, limit=1)
        cam = matches[0] if matches else None
    except Exception:
        cam = None

    if cam:
        try:
            now = time.time()
            add("history", {"camera_id": cam.get("id"), "start_ts": now - 3600.0, "end_ts": now, "limit": 1500})
        except Exception:
            pass
        want_live = True
    else:
        try:
            _, dest = _extract_origin_destination(q_raw)
        except Exception:
            dest = None
        if dest:
            add("nearest_cameras", {"place": _clean_place_phrase(dest), "limit": 6, "max_km": 8.0})
            want_live = True

    if want_live:
        add("live", {"limit": 200})

    uniq = []
    seen = set()
    for c in calls:
        key = json.dumps(c, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def _area_live_summary_from_system(user_text, page=None):
    q_raw = str(user_text or "").strip()
    if not q_raw:
        return None
    q = _normalize_text(q_raw).lower()
    wants_area = any(k in q for k in ["kondisi", "macet", "padat", "lancar", "ramai", "sepi"]) and any(
        k in q for k in ["sekarang", "saat ini", "hari ini"]
    )
    if not wants_area:
        return None

    try:
        _, dest = _extract_origin_destination(q_raw)
    except Exception:
        dest = None
    if not dest:
        m = re.search(r"\bdi\s+(.+)$", q)
        if m:
            dest = _clean_place_phrase(m.group(1))
    dest = _clean_place_phrase(dest) if dest else ""
    if not dest:
        return None

    near = _ai_exec_tool("nearest_cameras", {"place": dest, "limit": 8, "max_km": 10.0}) or {}
    near_items = near.get("items") if isinstance(near.get("items"), list) else []

    live = _ai_exec_tool("live", {"limit": 500}) or {}
    live_items = live.get("items") if isinstance(live.get("items"), list) else []
    live_map = {}
    for r in live_items:
        if not isinstance(r, dict):
            continue
        cid = str(r.get("camera_id") or "").strip()
        if cid:
            live_map[cid] = r

    if not near_items:
        return {
            "reply": f"Aku belum nemu CCTV terdekat dari kata kunci '{dest}'. Kalau kamu tahu nama titik/CCTV-nya, sebutkan ya (contoh: 'Simpang Braga').",
            "ui_actions": None,
        }

    merged = []
    for it in near_items:
        cid = str(it.get("id") or "").strip()
        lm = live_map.get(cid) or {}
        merged.append(
            {
                "id": cid,
                "name": it.get("name") or cid,
                "distance_km": it.get("distance_km"),
                "status": lm.get("status"),
                "congestion": lm.get("congestion"),
                "current_count": lm.get("current_count"),
            }
        )

    def cong_rank(label):
        k = str(label or "").upper()
        if k == "MACET TOTAL":
            return 4
        if k == "MACET":
            return 3
        if k == "PADAT LANCAR":
            return 2
        if k == "LANCAR":
            return 1
        return 0

    merged.sort(
        key=lambda x: (
            cong_rank(x.get("congestion")),
            0 if str(x.get("status") or "") == "online" else -1,
            -float(x.get("distance_km") or 999999),
        ),
        reverse=True,
    )

    best = merged[0] if merged else None
    best_id = str((best or {}).get("id") or "").strip()
    best_name = str((best or {}).get("name") or best_id or "").strip()

    lines = []
    lines.append(f"Ini kondisi sekitar {dest} berdasarkan CCTV terdekat yang kebaca sistem:")
    lines.append("")
    for i, r in enumerate(merged[:6], 1):
        nm = str(r.get("name") or "-")
        st = str(r.get("status") or "-")
        cg = str(r.get("congestion") or "-")
        cnt = fmt_int(r.get("current_count"))
        dk = r.get("distance_km")
        dk_s = f"{dk} km" if dk is not None else "- km"
        lines.append(f"{i}. {nm} — {cg} (live {cnt}) • {st} • {dk_s}")
    lines.append("")
    if best_name:
        lines.append(f"Aku fokuskan tampilan ke titik terdekat yang paling relevan: {best_name}.")

    acts = []
    if best_id:
        if not str(page or "").startswith("/analysis"):
            acts.append({"type": "navigate", "path": "/analysis"})
        acts.append({"type": "select_camera", "camera_id": best_id, "camera_name": best_name})
        acts.append({"type": "set_period", "period": "1h"})

    return {"reply": "\n".join(lines).strip(), "ui_actions": acts or None}


def _ai_collect_tool_context(user_text, chat_ctx=None):
    plan = _ai_autotool_plan(user_text, chat_ctx=chat_ctx)
    if not plan:
        return None
    results = []
    for c in plan[:6]:
        tool = c.get("tool")
        args = c.get("args") if isinstance(c.get("args"), dict) else {}
        results.append({"tool": tool, "args": args, "result": _ai_exec_tool(tool, args)})
    return {"plan": plan[:6], "results": results}


@bp.route("/api/ai/tools")
def ai_tools_api():
    return jsonify({"tools": _ai_tools_spec()})


@bp.route("/api/ai/query", methods=["POST"])
@csrf_required
@rate_limit(limit_name="chat")
def ai_query_api():
    payload = request.get_json(silent=True) or {}
    if isinstance(payload, dict) and isinstance(payload.get("batch"), list):
        out = []
        for it in payload.get("batch") or []:
            if not isinstance(it, dict):
                continue
            name = it.get("tool") or it.get("name")
            args = it.get("args") if isinstance(it.get("args"), dict) else {}
            out.append({"tool": name, "result": _ai_exec_tool(name, args)})
        return jsonify({"results": out})

    tool = (payload.get("tool") or payload.get("name")) if isinstance(payload, dict) else None
    args = payload.get("args") if isinstance(payload, dict) and isinstance(payload.get("args"), dict) else {}
    return jsonify({"tool": tool, "result": _ai_exec_tool(tool, args)})


# ================================================================
# CASE 1 - Intelligent Traffic Enforcement & Behaviour Analysis
# ================================================================

def _parse_ts(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except Exception:
        pass
    try:
        # Accept ISO 8601
        dt = datetime.datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def _count_violations(camera_id=None, violation_type=None, start_ts=None, end_ts=None,
                      plate_contains=None, status=None):
    """Count total violations matching filters (for pagination)."""
    try:
        where = []
        params = []
        if camera_id:
            where.append("camera_id = ?")
            params.append(str(camera_id))
        if violation_type:
            where.append("violation_type = ?")
            params.append(str(violation_type))
        if start_ts is not None:
            where.append("timestamp >= ?")
            params.append(float(start_ts))
        if end_ts is not None:
            where.append("timestamp <= ?")
            params.append(float(end_ts))
        if plate_contains:
            where.append("plate_text LIKE ?")
            params.append(f"%{plate_contains}%")
        if status:
            where.append("status = ?")
            params.append(str(status))
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        conn = get_db_connection()
        try:
            from app.database import _execute
            c = _execute(conn, f"SELECT COUNT(*) as cnt FROM violations{where_sql}", params)
            row = c.fetchone()
            if hasattr(row, 'keys'):
                r_dict = dict(row)
                return int(r_dict.get("cnt", 0) or r_dict.get("count", 0) or 0)
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except Exception:
        return 0


def _camera_by_id(cam_id):
    try:
        sources = globals_state.CCTV_SOURCES or []
        for s in sources:
            if str(s.get("id")) == str(cam_id):
                return s
    except Exception:
        pass
    return None


# ---- Pages ----

@bp.route("/enforcement")
def enforcement_page():
    if not session.get("user") or session["user"].get("role") != "admin":
        return redirect("/login")
    return render_template("enforcement.html")


@bp.route("/zones")
def zones_page():
    if not session.get("user") or session["user"].get("role") != "admin":
        return redirect("/login")
    return render_template("zones.html")


@bp.route("/cameras")
def cameras_page():
    if not session.get("user") or session["user"].get("role") != "admin":
        return redirect("/login")
    return render_template("cameras.html")


@bp.route("/crm")
def crm_page():
    if not session.get("user") or session["user"].get("role") != "admin":
        return redirect("/login")
    return render_template("crm.html")


# ---- Evidence serving (static files under data/violations_evidence) ----

@bp.route("/evidence/<path:relpath>")
def evidence_file(relpath):
    from flask import send_from_directory
    base = os.path.dirname(EVIDENCE_DIR)
    try:
        return send_from_directory(base, relpath)
    except Exception:
        return Response("Not Found", status=404)


@bp.route("/evidence_crop/<int:violation_id>")
def evidence_crop(violation_id):
    """Serve a cropped vehicle thumbnail from evidence image using stored bbox."""
    import cv2
    import numpy as np
    try:
        from app.database import get_violation
        v = get_violation(violation_id)
        if not v or not v.get("evidence_path") or not v.get("bbox"):
            return Response("Not Found", status=404)
        
        # Load evidence image - try both path resolutions
        from app.config import EVIDENCE_DIR
        img_path = os.path.join(EVIDENCE_DIR, v["evidence_path"].replace("/", os.sep))
        if not os.path.isfile(img_path):
            # Try from parent dir (DB stores path relative to data/ dir)
            img_path = os.path.join(os.path.dirname(EVIDENCE_DIR), v["evidence_path"].replace("/", os.sep))
        if not os.path.isfile(img_path):
            return Response("Not Found", status=404)
        
        img = cv2.imread(img_path)
        if img is None:
            return Response("Not Found", status=404)
        
        # Crop bbox area with padding
        bbox = v["bbox"]
        if isinstance(bbox, str):
            bbox = json.loads(bbox)
        x1, y1, x2, y2 = [int(b) for b in bbox]
        h, w = img.shape[:2]
        pad = 30
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return Response("Not Found", status=404)
        
        # Resize to thumbnail (max 200px)
        ch, cw = crop.shape[:2]
        if cw > 200:
            scale = 200 / cw
            crop = cv2.resize(crop, (200, int(ch * scale)))
        
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(buf.tobytes(), mimetype="image/jpeg")
    except Exception:
        return Response("Not Found", status=404)


# ---- Zones API ----

@bp.route("/api/zones", methods=["GET"])
@admin_required
def api_zones_list():
    cam = request.args.get("camera_id")
    if cam:
        rows = get_zones_for_camera(cam, only_active=False)
    else:
        rows = get_all_zones()
    return jsonify({"status": "success", "zones": rows})


@bp.route("/api/zones", methods=["POST"])
@admin_required
@csrf_required
@rate_limit(limit_name="camera_edit")
def api_zones_create():
    try:
        p = request.get_json(silent=True) or {}
        camera_id = p.get("camera_id")
        zone_type = p.get("zone_type")
        geometry = p.get("geometry")
        name = p.get("name") or ""
        notes = p.get("notes") or ""
        active = bool(p.get("active", True))
        frame_width = int(p.get("frame_width") or 0)
        frame_height = int(p.get("frame_height") or 0)
        if not camera_id or not zone_type or geometry is None:
            return jsonify({"status": "error", "message": "camera_id, zone_type, geometry required"}), 400
        if zone_type not in ZONE_TYPES:
            return jsonify({"status": "error", "message": f"zone_type must be one of {ZONE_TYPES}"}), 400
        zid = insert_zone(camera_id, zone_type, geometry, name=name, notes=notes, active=active,
                          frame_width=frame_width, frame_height=frame_height)
        # Force the camera's enforcement engine to refresh zones next loop
        try:
            agent = globals_state.camera_agents.get(camera_id)
            if agent and hasattr(agent, "enforcement"):
                agent.enforcement.load_zones(force=True)
        except Exception:
            pass
        return jsonify({"status": "success", "id": zid})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/zones/<int:zone_id>", methods=["PUT", "PATCH"])
@admin_required
@csrf_required
def api_zones_update(zone_id):
    try:
        p = request.get_json(silent=True) or {}
        fields = {}
        for k in ("name", "zone_type", "geometry", "active", "notes"):
            if k in p:
                if k == "geometry":
                    fields["geometry_json"] = json.dumps(p[k])
                else:
                    fields[k] = p[k]
        ok = update_zone(zone_id, **fields)
        return jsonify({"status": "success" if ok else "not_modified"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/zones/<int:zone_id>", methods=["DELETE"])
@admin_required
@csrf_required
def api_zones_delete(zone_id):
    try:
        ok = delete_zone(zone_id)
        return jsonify({"status": "success" if ok else "not_found"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ---- Violations API ----

@bp.route("/api/violations")
@rate_limit(limit_name="violations_list")
def api_violations_list():
    try:
        limit = max(1, min(1000, int(request.args.get("limit") or 50)))
        offset = max(0, int(request.args.get("offset") or 0))
        # Safe validation for all string params
        cam = safe_camera_id(request.args.get("camera_id")) if request.args.get("camera_id") else None
        vtype = safe_string(request.args.get("violation_type"), max_length=32)
        plate = safe_string(request.args.get("plate"), max_length=20)
        status = safe_string(request.args.get("status"), max_length=16)
        start_ts = _parse_ts(request.args.get("start"))
        end_ts = _parse_ts(request.args.get("end"))
        rows = list_violations(
            limit=limit, offset=offset,
            camera_id=cam, violation_type=vtype,
            start_ts=start_ts, end_ts=end_ts,
            plate_contains=plate, status=status,
        )
        # Get total count for pagination
        total = _count_violations(cam, vtype, start_ts, end_ts, plate, status)
        return jsonify({"status": "success", "count": len(rows), "total": total, "violations": rows})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/violations/<int:vid>")
def api_violations_get(vid):
    row = get_violation(vid)
    if not row:
        return jsonify({"status": "not_found"}), 404
    return jsonify({"status": "success", "violation": row})


@bp.route("/api/violations/<int:vid>", methods=["PATCH", "PUT"])
@admin_required
def api_violations_update(vid):
    try:
        p = request.get_json(silent=True) or {}
        fields = {k: v for k, v in p.items() if k in ("status", "dispatched_unit", "notes", "plate_text", "plate_confidence", "vehicle_class")}
        ok = update_violation(vid, **fields)
        return jsonify({"status": "success" if ok else "not_modified"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/violations/summary")
def api_violations_summary():
    try:
        start_ts = _parse_ts(request.args.get("start"))
        end_ts = _parse_ts(request.args.get("end"))
        period = request.args.get("period")
        if period and not start_ts:
            now = time.time()
            mult = {
                "1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800, "30d": 2592000,
            }.get(period)
            if mult:
                start_ts = now - mult
                end_ts = now
        summary = violation_summary(start_ts=start_ts, end_ts=end_ts)
        return jsonify({"status": "success", "summary": summary,
                        "start_ts": start_ts, "end_ts": end_ts})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/violations/heatmap")
def api_violations_heatmap():
    try:
        start_ts = _parse_ts(request.args.get("start"))
        end_ts = _parse_ts(request.args.get("end"))
        period = request.args.get("period")
        if period and not start_ts:
            now = time.time()
            mult = {
                "1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800, "30d": 2592000,
            }.get(period)
            if mult:
                start_ts = now - mult
                end_ts = now
        rows = violation_heatmap_by_camera(start_ts=start_ts, end_ts=end_ts)
        return jsonify({"status": "success", "points": rows})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/violations/recommendations")
def api_violations_recommendations():
    try:
        top_n = max(1, min(50, int(request.args.get("top_n") or 10)))
        start_ts = _parse_ts(request.args.get("start"))
        end_ts = _parse_ts(request.args.get("end"))
        recs = recommend_enforcement_points(top_n=top_n, start_ts=start_ts, end_ts=end_ts)
        return jsonify({"status": "success", "recommendations": recs})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/violations/export_csv")
def api_violations_export_csv():
    try:
        start_ts = _parse_ts(request.args.get("start"))
        end_ts = _parse_ts(request.args.get("end"))
        cam = request.args.get("camera_id") or None
        vtype = request.args.get("violation_type") or None
        rows = list_violations(
            limit=10000, offset=0,
            camera_id=cam, violation_type=vtype,
            start_ts=start_ts, end_ts=end_ts,
        )
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow([
            "id", "timestamp", "iso_time", "camera_id", "camera_name",
            "violation_type", "zone_type", "duration_s", "vehicle_class",
            "plate_text", "plate_confidence", "lat", "lng", "status",
            "dispatched_unit", "evidence_path",
        ])
        for r in rows:
            ts = float(r.get("timestamp") or 0)
            iso = datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds")
            writer.writerow([
                r.get("id"), ts, iso,
                r.get("camera_id"), r.get("camera_name"),
                r.get("violation_type"), r.get("zone_type"),
                r.get("duration_s"), r.get("vehicle_class"),
                r.get("plate_text"), r.get("plate_confidence"),
                r.get("lat"), r.get("lng"),
                r.get("status"), r.get("dispatched_unit"),
                r.get("evidence_path"),
            ])
        resp = Response(out.getvalue(), mimetype="text/csv")
        resp.headers["Content-Disposition"] = "attachment; filename=violations_export.csv"
        return resp
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ---- CRM / Public complaints API ----

@bp.route("/api/crm/reports", methods=["GET"])
def api_crm_list():
    try:
        limit = max(1, min(500, int(request.args.get("limit") or 50)))
        offset = max(0, int(request.args.get("offset") or 0))
        status = request.args.get("status") or None
        rows = list_crm_reports(limit=limit, offset=offset, status=status)
        return jsonify({"status": "success", "reports": rows})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/crm/reports", methods=["POST"])
@admin_required
@csrf_required
@rate_limit(limit_name="crm")
def api_crm_create():
    try:
        p = request.get_json(silent=True) or {}
        description = (p.get("description") or "").strip()
        if not description:
            return jsonify({"status": "error", "message": "description required"}), 400
        category = p.get("category") or ""
        name = p.get("reporter_name") or ""
        contact = p.get("reporter_contact") or ""
        lat = p.get("lat")
        lng = p.get("lng")
        camera_id = p.get("camera_id")
        # Auto-classify description into a violation type
        auto_type = auto_classify_crm(description)
        # Simple priority heuristic
        prio = "normal"
        urgent_kw = ("kecelakaan", "accident", "urgent", "segera", "darurat")
        if any(k in description.lower() for k in urgent_kw):
            prio = "high"
        rid = insert_crm_report(
            reporter_name=name,
            reporter_contact=contact,
            category=category,
            description=description,
            lat=lat, lng=lng,
            camera_id=camera_id,
            auto_classified_type=auto_type,
            priority=prio,
        )
        return jsonify({"status": "success", "id": rid, "auto_classified_type": auto_type, "priority": prio})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/crm/reports/<int:rid>", methods=["PATCH", "PUT"])
@admin_required
@csrf_required
def api_crm_update(rid):
    try:
        p = request.get_json(silent=True) or {}
        fields = {k: v for k, v in p.items() if k in ("status", "priority", "auto_classified_type", "camera_id")}
        ok = update_crm_report(rid, **fields)
        return jsonify({"status": "success" if ok else "not_modified"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/crm/summary")
def api_crm_summary():
    try:
        return jsonify({"status": "success", "summary": crm_summary()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/crm/social_mentions")
def api_crm_social_mentions():
    """Monitor social media mentions of @DishubDKI using Playwright scraper.
    Scrapes Twitter/X, Threads, Instagram, Facebook via headless Chromium.
    NO news articles — only social media posts.
    """
    all_mentions = []
    
    # Playwright-based scraping (all social platforms)
    try:
        from app.services.social_scraper import scrape_twitter_mentions
        playwright_results = scrape_twitter_mentions(max_results=20)
        # Filter: only social media, no news
        for m in playwright_results:
            if m.get("platform") in ("twitter", "threads", "instagram", "facebook"):
                all_mentions.append(m)
    except Exception as e:
        print(f"[CRM] Playwright scraper error: {e}")
    
    # Sort: high priority first, then by timestamp
    all_mentions.sort(key=lambda x: (0 if x.get("priority") == "high" else 1, -x.get("timestamp", 0)))
    
    return jsonify({
        "status": "success",
        "mentions": all_mentions[:20],
        "count": len(all_mentions),
        "sources": ["twitter/x", "threads", "instagram", "facebook"],
    })


# ---- Helpers & metadata ----

@bp.route("/api/ai/test")
def api_ai_test():
    """Test AI provider connection."""
    try:
        from app.services.ai_ocr import ai_test_connection
        result = ai_test_connection()
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@bp.route("/api/ai/ollama_models")
@admin_required
def api_ai_ollama_models():
    """List locally available Ollama models (for AI Engine switching)."""
    try:
        models = _ollama_get_models_cached(cache_seconds=10, timeout_s=4, force=True)
        return jsonify({
            "status": "success",
            "reachable": True,
            "models": models or [],
            "url": _ollama_base_url(),
        })
    except Exception as e:
        return jsonify({
            "status": "success",
            "reachable": False,
            "models": [],
            "url": _ollama_base_url(),
            "error": str(e),
        })


@bp.route("/api/enforcement/meta")
def api_enforcement_meta():
    return jsonify({
        "status": "success",
        "violation_types": VIOLATION_TYPES,
        "zone_types": ZONE_TYPES,
    })


@bp.route("/api/ocr/test", methods=["POST"])
@csrf_required
@rate_limit(limit_name="ocr_test")
def api_ocr_test():
    """Test OCR on an uploaded image. Single-pass: PaddleOCR + AI correction + Vehicle ID."""
    try:
        import cv2
        import numpy as np
        from app.services.anpr import recognize_plate
        from app.services.ai_ocr import ai_identify_vehicle

        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No image uploaded"}), 400

        file = request.files['image']
        if not file.filename:
            return jsonify({"status": "error", "message": "Empty filename"}), 400

        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"status": "error", "message": "Cannot decode image"}), 400

        h, w = img.shape[:2]

        # Single pass: use full ANPR pipeline (PaddleOCR + AI)
        plate, conf, engine = recognize_plate(img, (0, 0, w, h))
        
        results = []
        if plate:
            results.append({
                "text": plate,
                "confidence": round(float(conf), 3),
                "method": engine,
            })

        # Vehicle identification using AI vision model
        vehicle_identity = None
        try:
            info = ai_identify_vehicle(img)
            if info:
                vehicle_identity = info
        except Exception:
            pass

        # Pick best
        best = results[0] if results else None

        return jsonify({
            "status": "success",
            "plate": best["text"] if best else None,
            "confidence": best["confidence"] if best else 0,
            "engine": best["method"] if best else "none",
            "all_results": results,
            "image_size": f"{w}x{h}",
            "vehicle_identity": vehicle_identity,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def _find_vehicle_region(img):
    """Find the red bounding box in evidence image and crop the vehicle inside it."""
    import cv2
    import numpy as np
    try:
        # Detect red color (the violation bbox is drawn in red)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Red in HSV has hue around 0-10 and 170-180
        mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        red_mask = mask1 | mask2
        # Thicken thin bbox lines so contour detection sees one rectangle
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)
        
        # Find contours of red regions
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Find the most plausible red rectangle:
        # - allow small boxes (main violation bbox is often < 2% of full frame)
        # - prefer boxes away from image borders to avoid the top-right inset
        # - prefer moderate aspect ratio like a vehicle bbox
        best_box = None
        best_score = -1.0
        h, w = img.shape[:2]
        frame_area = float(max(1, w * h))
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            rel_area = area / frame_area
            if rel_area < 0.0003 or rel_area > 0.15:
                continue

            aspect = bw / float(max(1, bh))
            if aspect < 0.35 or aspect > 2.5:
                continue

            touches_border = x <= 4 or y <= 4 or (x + bw) >= (w - 4) or (y + bh) >= (h - 4)
            in_top_right_corner = x > int(w * 0.72) and y < int(h * 0.25)

            score = area
            if touches_border:
                score *= 0.35
            if in_top_right_corner:
                score *= 0.20

            if score > best_score:
                best_box = (x, y, x + bw, y + bh)
                best_score = score
        
        if best_box is None:
            return None
        
        # Crop inside the red box (shrink slightly to avoid the red border itself)
        x1, y1, x2, y2 = best_box
        pad = 6
        x1 = min(w - 2, x1 + pad)
        y1 = min(h - 2, y1 + pad)
        x2 = max(x1 + 1, x2 - pad)
        y2 = max(y1 + 1, y2 - pad)
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
            return None
        return crop
    except Exception:
        return None


@bp.route("/api/twitter/search")
def api_twitter_search():
    """Search for traffic-related news/social media posts.
    Uses Google News RSS as primary source (free, no API key needed).
    Falls back to Twitter API v2 if Bearer Token has credits.
    Query is configurable from Settings page and persists across restarts.
    """
    import app.config as cfg
    query = request.args.get("q") or cfg.TWITTER_SEARCH_QUERY
    max_results = min(int(request.args.get("max_results") or cfg.TWITTER_MAX_RESULTS), 30)

    # Primary: Google News RSS (always works, free)
    try:
        import urllib.request
        import urllib.error
        import xml.etree.ElementTree as ET
        import html
        import re

        encoded_q = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=id&gl=ID&ceid=ID:id"
        req = urllib.request.Request(rss_url)
        req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SmartTrafficAI/1.0")

        with urllib.request.urlopen(req, timeout=10) as resp:
            xml_data = resp.read().decode("utf-8")

        root = ET.fromstring(xml_data)
        channel = root.find("channel")
        items = channel.findall("item") if channel is not None else []

        tweets = []
        for item in items[:max_results]:
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            pub_date = item.findtext("pubDate") or ""
            source = item.findtext("source") or ""
            description = item.findtext("description") or ""
            # Clean HTML from description
            description = re.sub(r"<[^>]+>", "", html.unescape(description))[:200]

            tweets.append({
                "id": link,
                "text": title,
                "description": description,
                "created_at": pub_date,
                "author_name": source,
                "author_username": "news",
                "author_avatar": "",
                "link": link,
                "metrics": {},
                "source": "google_news",
            })

        if tweets:
            # Sort by date — newest first
            from email.utils import parsedate_to_datetime
            def _parse_date(t):
                try:
                    return parsedate_to_datetime(t.get("created_at") or "")
                except Exception:
                    return datetime.datetime(2000, 1, 1)
            import datetime
            tweets.sort(key=_parse_date, reverse=True)
            return jsonify({"status": "success", "tweets": tweets, "query": query, "count": len(tweets), "source": "google_news"})
    except Exception as e:
        pass  # Fall through to Twitter API

    # Fallback: Twitter API v2 (if credits available)
    if cfg.TWITTER_BEARER_TOKEN:
        try:
            import urllib.request
            import urllib.error
            url = f"https://api.twitter.com/2/tweets/search/recent?query={urllib.parse.quote(cfg.TWITTER_SEARCH_QUERY)}&max_results={max(10, max_results)}&tweet.fields=created_at,author_id,public_metrics,text&expansions=author_id&user.fields=name,username,profile_image_url"
            req = urllib.request.Request(url)
            req.add_header("Authorization", f"Bearer {cfg.TWITTER_BEARER_TOKEN}")
            req.add_header("User-Agent", "SmartTrafficAI/1.0")

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            tweets = []
            users_map = {}
            if "includes" in data and "users" in data["includes"]:
                for u in data["includes"]["users"]:
                    users_map[u["id"]] = u

            for t in (data.get("data") or []):
                author = users_map.get(t.get("author_id"), {})
                tweets.append({
                    "id": t.get("id"),
                    "text": t.get("text"),
                    "created_at": t.get("created_at"),
                    "author_name": author.get("name", ""),
                    "author_username": author.get("username", ""),
                    "author_avatar": author.get("profile_image_url", ""),
                    "metrics": t.get("public_metrics", {}),
                    "source": "twitter",
                })

            return jsonify({"status": "success", "tweets": tweets, "query": cfg.TWITTER_SEARCH_QUERY, "count": len(tweets), "source": "twitter"})
        except Exception:
            pass

    return jsonify({"status": "error", "message": "Tidak dapat memuat berita. Coba lagi nanti."}), 503



# ---- Executive Summary (Case 1 deliverable) ----

@bp.route("/api/violations/executive_summary")
def api_executive_summary():
    """
    Generate a daily / weekly executive summary combining enforcement, CRM,
    and operational metrics. Intended for DISHUB stakeholder reports.
    """
    try:
        period = request.args.get("period", "24h")
        now = time.time()
        secs = {"24h": 86400, "7d": 604800, "30d": 2592000}.get(period, 86400)
        start_ts = now - secs

        summary = violation_summary(start_ts=start_ts, end_ts=now)
        prev_summary = violation_summary(start_ts=start_ts - secs, end_ts=start_ts)
        all_time_summary = violation_summary()  # No time filter = all time
        heatmap = violation_heatmap_by_camera(start_ts=start_ts, end_ts=now)
        top_recs = recommend_enforcement_points(top_n=5, start_ts=start_ts, end_ts=now)
        crm = crm_summary()
        
        # Get recent violations for dashboard display
        recent = list_violations(limit=20, offset=0, start_ts=start_ts, end_ts=now)
        # Fallback: if no violations in period, get most recent overall
        recent_all = []
        if not recent:
            recent_all = list_violations(limit=20, offset=0)

        total = int(summary.get("total") or 0)
        prev_total = int(prev_summary.get("total") or 0)
        total_all_time = int(all_time_summary.get("total") or 0)
        summary["total_all_time"] = total_all_time
        summary["by_type_all_time"] = all_time_summary.get("by_type", {})
        delta_pct = None
        if prev_total > 0:
            delta_pct = round(((total - prev_total) / prev_total) * 100.0, 1)

        # Build narrative bullets
        bullets = []
        if total == 0:
            bullets.append("No violations recorded in the selected period. Confirm zones are defined and cameras online.")
        else:
            bullets.append(f"Total of {total} violations detected in the last {period}.")
            if delta_pct is not None:
                trend = "increased" if delta_pct > 0 else ("decreased" if delta_pct < 0 else "unchanged")
                bullets.append(f"Violations {trend} by {abs(delta_pct)}% vs the previous {period}.")
            by_type = summary.get("by_type") or {}
            if by_type:
                dominant = max(by_type.items(), key=lambda kv: kv[1])
                bullets.append(f"Most common violation: {dominant[0].replace('_', ' ')} ({dominant[1]} cases).")
            by_hour = summary.get("by_hour") or []
            if any(by_hour):
                peak_hour = max(range(len(by_hour)), key=lambda i: by_hour[i])
                bullets.append(f"Peak hour of violations: {peak_hour:02d}:00 ({by_hour[peak_hour]} cases).")
            by_cam = summary.get("by_camera") or []
            if by_cam:
                hot = by_cam[0]
                bullets.append(f"Hotspot location: {hot['camera_name']} ({hot['count']} violations).")
        if crm.get("total"):
            bullets.append(f"{crm['total']} public complaints received; {crm.get('by_status', {}).get('open', 0)} still open.")
        if top_recs:
            bullets.append(
                f"Recommended to install E-TLE cameras or deploy officers at top {len(top_recs)} locations based on vulnerability score."
            )

        return jsonify({
            "status": "success",
            "period": period,
            "start_ts": start_ts,
            "end_ts": now,
            "delta_pct": delta_pct,
            "summary": summary,
            "prev_summary": prev_summary,
            "heatmap": heatmap,
            "top_recommendations": top_recs,
            "crm": crm,
            "recent": recent,
            "recent_all": recent_all,
            "narrative": bullets,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/executive_summary")
def executive_summary_page():
    if not session.get("user") or session["user"].get("role") != "admin":
        return redirect("/login")
    return render_template("executive_summary.html")



# ================================================================
# SETTINGS PAGE
# ================================================================

@bp.route("/settings")
def settings_page():
    if not session.get("user") or session["user"].get("role") != "admin":
        return redirect("/login")
    return render_template("settings.html")


@bp.route("/api/settings", methods=["GET"])
@admin_required
def api_settings_get():
    """Return current runtime settings."""
    import app.config as cfg
    from app.globals import camera_agents, CCTV_SOURCES, yolo_model_instance

    # Detect available models
    models_dir = os.path.join(cfg.BASE_DIR, "models")
    available_models = []
    if os.path.isdir(models_dir):
        for f in sorted(os.listdir(models_dir)):
            if f.endswith(".pt") and "plate" not in f.lower():
                fpath = os.path.join(models_dir, f)
                try:
                    size_mb = os.path.getsize(fpath) / 1024 / 1024
                    available_models.append({"name": f, "path": fpath, "size_mb": round(size_mb, 1)})
                except Exception:
                    pass

    # Current active model
    active_model = None
    if yolo_model_instance is not None:
        if hasattr(yolo_model_instance, "ckpt_path"):
            active_model = os.path.basename(str(yolo_model_instance.ckpt_path))
        elif hasattr(yolo_model_instance, "onnx_path"):
            active_model = os.path.basename(str(yolo_model_instance.onnx_path))

    # GPU info
    gpu_info = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
                "cuda_version": torch.version.cuda,
            }
    except Exception:
        pass

    return jsonify({
        "status": "success",
        "settings": {
            "detection": {
                "conf_threshold": cfg.CONF_THRESHOLD,
                "iou_threshold": cfg.IOU_THRESHOLD,
                "infer_imgsz": cfg.INFER_IMGSZ,
                "stream_fps": cfg.STREAM_FPS,
                "stream_max_width": cfg.STREAM_MAX_WIDTH,
                "stream_jpeg_quality": cfg.STREAM_JPEG_QUALITY,
            },
            "model": {
                "active_model": active_model,
                "custom_model_path": cfg.YOLO_CUSTOM_PATH,
                "use_custom_yolo": cfg.USE_CUSTOM_YOLO,
                "available_models": available_models,
            },
            "enforcement": {
                "violations_enabled": cfg.VIOLATIONS_ENABLED,
                "anpr_enabled": cfg.ANPR_ENABLED,
                "anpr_fallback_simulate": cfg.ANPR_FALLBACK_SIMULATE,
                "illegal_parking_min_seconds": cfg.ILLEGAL_PARKING_MIN_SECONDS,
                "static_movement_px": cfg.STATIC_MOVEMENT_PX,
                "dynamic_lane_min_seconds": cfg.DYNAMIC_LANE_MIN_SECONDS,
                "violation_cooldown_seconds": cfg.VIOLATION_COOLDOWN_SECONDS,
            },
            "social_media": {
                "search_query": cfg.TWITTER_SEARCH_QUERY,
                "max_results": cfg.TWITTER_MAX_RESULTS,
                "x_search_query": cfg.X_SEARCH_QUERY,
                "x_cookies_loaded": os.path.exists(cfg.X_COOKIES_FILE),
            },
            "ai_provider": {
                "provider": cfg.AI_PROVIDER,
                "base_url": cfg.AI_BASE_URL,
                "api_key_set": bool(cfg.AI_API_KEY),
                "model": cfg.AI_MODEL,
                "use_ai_for_anpr": cfg.AI_USE_FOR_ANPR,
                "use_ai_for_chat": cfg.AI_USE_FOR_CHAT,
                "chat_model": cfg.AI_CHAT_MODEL,
            },
            "general": {
                "timezone": cfg.TIMEZONE,
            },
            "server": {
                "host": cfg.HOST_IP,
                "port": cfg.HOST_PORT,
                "cameras_count": len(CCTV_SOURCES) if CCTV_SOURCES else 0,
                "agents_running": len(camera_agents),
            },
            "gpu": gpu_info,
        },
    })


@bp.route("/api/settings", methods=["POST", "PATCH"])
@admin_required
@csrf_required
@rate_limit(limit_name="camera_edit")
def api_settings_update():
    """Update runtime settings used by the active live pipeline."""
    import app.config as cfg

    p = request.get_json(silent=True) or {}
    updated = []

    # Detection settings
    det = p.get("detection") or {}
    if "conf_threshold" in det:
        cfg.CONF_THRESHOLD = max(0.01, min(1.0, float(det["conf_threshold"])))
        updated.append("conf_threshold")
    if "iou_threshold" in det:
        cfg.IOU_THRESHOLD = max(0.01, min(1.0, float(det["iou_threshold"])))
        updated.append("iou_threshold")
    if "infer_imgsz" in det:
        cfg.INFER_IMGSZ = int(det["infer_imgsz"])
        updated.append("infer_imgsz")
    if "stream_fps" in det:
        cfg.STREAM_FPS = max(1.0, float(det["stream_fps"]))
        updated.append("stream_fps")
    if "stream_max_width" in det:
        cfg.STREAM_MAX_WIDTH = int(det["stream_max_width"])
        updated.append("stream_max_width")
    if "stream_jpeg_quality" in det:
        cfg.STREAM_JPEG_QUALITY = max(30, min(95, int(det["stream_jpeg_quality"])))
        updated.append("stream_jpeg_quality")

    # Enforcement settings
    enf = p.get("enforcement") or {}
    if "violations_enabled" in enf:
        cfg.VIOLATIONS_ENABLED = bool(enf["violations_enabled"])
        updated.append("violations_enabled")
    if "anpr_enabled" in enf:
        cfg.ANPR_ENABLED = bool(enf["anpr_enabled"])
        updated.append("anpr_enabled")
    if "anpr_fallback_simulate" in enf:
        cfg.ANPR_FALLBACK_SIMULATE = bool(enf["anpr_fallback_simulate"])
        updated.append("anpr_fallback_simulate")
    if "illegal_parking_min_seconds" in enf:
        cfg.ILLEGAL_PARKING_MIN_SECONDS = max(1.0, float(enf["illegal_parking_min_seconds"]))
        updated.append("illegal_parking_min_seconds")
    if "static_movement_px" in enf:
        cfg.STATIC_MOVEMENT_PX = max(1.0, float(enf["static_movement_px"]))
        updated.append("static_movement_px")
    if "dynamic_lane_min_seconds" in enf:
        cfg.DYNAMIC_LANE_MIN_SECONDS = max(0.5, float(enf["dynamic_lane_min_seconds"]))
        updated.append("dynamic_lane_min_seconds")
    if "violation_cooldown_seconds" in enf:
        cfg.VIOLATION_COOLDOWN_SECONDS = max(5.0, float(enf["violation_cooldown_seconds"]))
        updated.append("violation_cooldown_seconds")

    # Social media settings
    social = p.get("social_media") or {}
    if "search_query" in social:
        cfg.TWITTER_SEARCH_QUERY = str(social["search_query"]).strip()
        updated.append("search_query")
    if "max_results" in social:
        cfg.TWITTER_MAX_RESULTS = max(5, min(50, int(social["max_results"])))
        updated.append("max_results")
    if "x_search_query" in social:
        cfg.X_SEARCH_QUERY = str(social["x_search_query"]).strip()
        updated.append("x_search_query")
        # Clear scraper cache so next request uses new query
        try:
            from app.services.social_scraper import _cache_lock, _cached_mentions
            with _cache_lock:
                _cached_mentions.clear()
        except Exception:
            pass
    if "x_cookies_json" in social:
        # Save cookies to file
        cookies_str = str(social["x_cookies_json"]).strip()
        if cookies_str:
            try:
                # Validate it's valid JSON
                import json as _json_val
                _json_val.loads(cookies_str)
                with open(cfg.X_COOKIES_FILE, 'w', encoding='utf-8') as f:
                    f.write(cookies_str)
                updated.append("x_cookies")
                # Clear scraper cache so next request uses new cookies
                try:
                    from app.services.social_scraper import _cache_lock, _cached_mentions
                    with _cache_lock:
                        _cached_mentions.clear()
                except Exception:
                    pass
            except Exception as e:
                return jsonify({"status": "error", "message": f"Invalid cookies JSON: {e}"}), 400

    # AI Provider settings
    ai = p.get("ai_provider") or {}
    if "provider" in ai:
        cfg.AI_PROVIDER = str(ai["provider"]).strip()
        updated.append("ai_provider")
    if "base_url" in ai:
        cfg.AI_BASE_URL = str(ai["base_url"]).strip()
        updated.append("ai_base_url")
    if "api_key" in ai:
        cfg.AI_API_KEY = str(ai["api_key"]).strip()
        updated.append("ai_api_key")
    if "model" in ai:
        cfg.AI_MODEL = str(ai["model"]).strip()
        updated.append("ai_model")
    if "use_ai_for_anpr" in ai:
        cfg.AI_USE_FOR_ANPR = bool(ai["use_ai_for_anpr"])
        updated.append("ai_use_for_anpr")
    if "use_ai_for_chat" in ai:
        cfg.AI_USE_FOR_CHAT = bool(ai["use_ai_for_chat"])
        updated.append("ai_use_for_chat")
    if "chat_model" in ai:
        cfg.AI_CHAT_MODEL = str(ai["chat_model"]).strip()
        updated.append("ai_chat_model")

    # General settings
    general = p.get("general") or {}
    if "timezone" in general:
        cfg.TIMEZONE = str(general["timezone"]).strip()
        updated.append("timezone")

    # Model switch (requires reload)
    model = p.get("model") or {}
    model_switched = False
    if "active_model" in model:
        new_model_name = str(model["active_model"]).strip()
        models_dir = os.path.join(cfg.BASE_DIR, "models")
        new_path = os.path.join(models_dir, new_model_name)
        if os.path.isfile(new_path):
            try:
                from ultralytics import YOLO
                import app.globals as g_state
                from app.config import VEHICLE_CLASSES_CUSTOM, CLASS_MAPPING_CUSTOM, VEHICLE_CLASSES_COCO, CLASS_MAPPING_COCO

                new_model = YOLO(new_path)
                g_state.yolo_model_instance = new_model

                # Switch class mapping based on model
                if "vehicle_v3" in new_model_name or "custom" in new_model_name:
                    cfg.VEHICLE_CLASSES = VEHICLE_CLASSES_CUSTOM
                    cfg.CLASS_MAPPING = CLASS_MAPPING_CUSTOM
                    cfg.USE_CUSTOM_YOLO = True
                else:
                    cfg.VEHICLE_CLASSES = VEHICLE_CLASSES_COCO
                    cfg.CLASS_MAPPING = CLASS_MAPPING_COCO
                    cfg.USE_CUSTOM_YOLO = False

                # Update all running agents to use new model
                for agent in g_state.camera_agents.values():
                    agent.model = new_model

                model_switched = True
                updated.append(f"active_model -> {new_model_name}")
            except Exception as e:
                return jsonify({"status": "error", "message": f"Failed to load model: {e}"}), 500
        else:
            return jsonify({"status": "error", "message": f"Model file not found: {new_model_name}"}), 404

    # Persist all settings to disk (survives restart)
    cfg.save_persisted_settings()

    return jsonify({
        "status": "success",
        "updated": updated,
        "model_switched": model_switched,
        "message": f"{len(updated)} setting(s) updated" + (" (model hot-swapped)" if model_switched else ""),
    })


@bp.route("/api/settings/restart_agents", methods=["POST"])
@admin_required
@csrf_required
@rate_limit(limit_name="camera_edit")
def api_restart_agents():
    """Restart all camera agents (useful after model switch)."""
    import app.globals as g_state
    from app.services.camera import start_camera_agents, stop_agent

    # Stop all
    ids = list(g_state.camera_agents.keys())
    for cid in ids:
        try:
            stop_agent(cid)
        except Exception:
            pass

    # Restart
    start_camera_agents()
    return jsonify({
        "status": "success",
        "agents_running": len(g_state.camera_agents),
    })


# ==============================================================
# AGENT TOOLS (scripts in /tools folder)
# ==============================================================
import sys
import subprocess as _subprocess

# Registry of available agent tools (scripts in the /tools folder).
# Each tool maps to metadata used by the Settings UI for grouping,
# labeling, argument rendering, and destructive-action warnings.
AGENT_TOOLS = {
    # ---- DIAGNOSTIC (read-only) ----
    "check_status": {
        "name": "Check Status",
        "script": "check_status.py",
        "category": "diagnostic",
        "description": "Cek statistik pelanggaran (total, ada plat, ada evidence, ada notes).",
        "icon": "fa-chart-bar",
        "destructive": False,
        "args": [],
    },
    "check_busway": {
        "name": "Check Busway",
        "script": "check_busway.py",
        "category": "diagnostic",
        "description": "Hitung pelanggaran berdasarkan tipe, khusus busway.",
        "icon": "fa-road",
        "destructive": False,
        "args": [],
    },
    "check_notes": {
        "name": "Check Notes",
        "script": "check_notes.py",
        "category": "diagnostic",
        "description": "Distribusi notes pada pelanggaran (top 20).",
        "icon": "fa-sticky-note",
        "destructive": False,
        "args": [],
    },
    "check_zones": {
        "name": "Check Zones",
        "script": "check_zones.py",
        "category": "diagnostic",
        "description": "Daftar semua zone pelanggaran terdaftar.",
        "icon": "fa-map-location-dot",
        "destructive": False,
        "args": [],
    },
    # ---- MAINTENANCE (modifies data) ----
    "dedup_violations": {
        "name": "Dedup Violations",
        "script": "dedup_violations.py",
        "category": "maintenance",
        "description": "Hapus pelanggaran duplikat (plat sama + kamera sama dalam window). Juga hapus yang evidence-nya hilang.",
        "icon": "fa-clone",
        "destructive": True,
        "args": [
            {"name": "dry_run", "flag": "--dry-run", "type": "bool", "default": True, "label": "Dry Run (preview saja)"},
            {"name": "window", "flag": "--window", "type": "number", "default": 300, "label": "Window (detik)", "min": 10, "max": 3600},
        ],
    },
    "remove_no_evidence": {
        "name": "Remove No Evidence",
        "script": "remove_no_evidence.py",
        "category": "maintenance",
        "description": "Hapus pelanggaran yang file evidence-nya tidak ada di disk.",
        "icon": "fa-trash-can",
        "destructive": True,
        "args": [],
    },
    "cleanup_bad_violations": {
        "name": "Cleanup Bad Violations",
        "script": "cleanup_bad_violations.py",
        "category": "maintenance",
        "description": "Hapus pelanggaran dengan evidence buruk (TIDAK ADA KENDARAAN, kendaraan terpotong, atau tidak ada plat) via AI vision.",
        "icon": "fa-broom",
        "destructive": True,
        "args": [
            {"name": "dry_run", "flag": "--dry-run", "type": "bool", "default": True, "label": "Dry Run (preview saja)"},
            {"name": "limit", "flag": "--limit", "type": "number", "default": 50, "label": "Limit", "min": 1, "max": 500},
        ],
    },
    "fill_plates_now": {
        "name": "Fill Plates Now",
        "script": "fill_plates_now.py",
        "category": "maintenance",
        "description": "Isi plat nomor / merek / warna yang kosong via AI vision dari gambar evidence.",
        "icon": "fa-id-card",
        "destructive": False,
        "args": [
            {"name": "limit", "flag": "--limit", "type": "number", "default": 50, "label": "Limit", "min": 1, "max": 500},
        ],
    },
    # ---- TEST ----
    "test_ai_api": {
        "name": "Test AI API",
        "script": "test_ai_api.py",
        "category": "test",
        "description": "Tes koneksi AI API dan ketersediaan model (text + vision).",
        "icon": "fa-robot",
        "destructive": False,
        "args": [],
    },
    "test_dashboard_api": {
        "name": "Test Dashboard API",
        "script": "test_dashboard_api.py",
        "category": "test",
        "description": "Tes endpoint dashboard API untuk diagnose data kosong.",
        "icon": "fa-gauge-high",
        "destructive": False,
        "args": [],
    },
    "test_ocr_pipeline": {
        "name": "Test OCR Pipeline",
        "script": "test_ocr_pipeline.py",
        "category": "test",
        "description": "Tes pipeline OCR penuh dengan gambar sintetis (PaddleOCR + AI).",
        "icon": "fa-font",
        "destructive": False,
        "args": [],
    },
}


@bp.route("/api/agent_tools", methods=["GET"])
@admin_required
def api_agent_tools_list():
    """Return list of available agent tools (scripts in /tools folder)."""
    return jsonify({"status": "success", "tools": AGENT_TOOLS})


@bp.route("/api/agent_tools/run", methods=["POST"])
@admin_required
@csrf_required
@rate_limit(limit_name="ai_test")
def api_agent_tools_run():
    """Run a specific agent tool script and return captured stdout/stderr."""
    import app.config as cfg

    data = request.get_json(silent=True) or {}
    tool_id = data.get("tool")
    if tool_id not in AGENT_TOOLS:
        return jsonify({"status": "error", "message": "Unknown tool"}), 400

    tool = AGENT_TOOLS[tool_id]
    script_path = os.path.join(cfg.BASE_DIR, "tools", tool["script"])
    if not os.path.isfile(script_path):
        return jsonify({"status": "error", "message": f"Script not found: {tool['script']}"}), 404

    # Build command: <python> <script> [args...]
    cmd = [sys.executable, script_path]

    # Parse args from request based on tool's arg config
    req_args = data.get("args") or {}
    for arg_cfg in tool.get("args", []):
        arg_name = arg_cfg["name"]
        arg_val = req_args.get(arg_name, arg_cfg.get("default"))
        if arg_cfg["type"] == "bool":
            if arg_val:
                cmd.append(arg_cfg["flag"])
        elif arg_cfg["type"] == "number":
            if arg_val is not None:
                try:
                    num_val = int(arg_val)
                    if "min" in arg_cfg:
                        num_val = max(arg_cfg["min"], num_val)
                    if "max" in arg_cfg:
                        num_val = min(arg_cfg["max"], num_val)
                    cmd.extend([arg_cfg["flag"], str(num_val)])
                except (TypeError, ValueError):
                    pass

    # Run synchronously with a generous timeout (some tools call AI vision)
    try:
        proc = _subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=cfg.BASE_DIR,
        )
        output = proc.stdout or ""
        if proc.stderr:
            output += ("\n--- STDERR ---\n" + proc.stderr) if output else proc.stderr

        # Truncate very long outputs so we don't blow up the JSON response
        if len(output) > 20000:
            output = output[-20000:] + "\n... (truncated)"

        return jsonify({
            "status": "success",
            "tool": tool_id,
            "exit_code": proc.returncode,
            "command": " ".join([os.path.basename(sys.executable), tool["script"]] + cmd[2:]),
            "output": output if output.strip() else "(no output)",
        })
    except _subprocess.TimeoutExpired:
        return jsonify({"status": "error", "message": "Tool timed out (180s limit)"}), 408
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ==============================================================
# Vehicle Identity Agent Control (pause/resume — no API calls = no token usage)
# ==============================================================

@bp.route("/api/agent/vehicle_identity", methods=["GET"])
@admin_required
def api_agent_vehicle_identity_status():
    """Return current status of the Vehicle Identity Agent."""
    try:
        from app.services.vehicle_agent import is_agent_paused, _agent_running
        return jsonify({
            "status": "success",
            "running": _agent_running,
            "paused": is_agent_paused(),
            "description": (
                "Vehicle Identity Agent fills missing plate numbers, vehicle make/model, and color "
                "from evidence images using AI vision. When PAUSED, it does NOT call the AI API — "
                "no token is consumed."
            ),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/agent/vehicle_identity", methods=["POST"])
@admin_required
def api_agent_vehicle_identity_control():
    """Pause or resume the Vehicle Identity Agent to control AI token usage."""
    data = request.get_json(silent=True) or {}
    action = str(data.get("action") or "").strip().lower()

    try:
        from app.services.vehicle_agent import pause_agent, resume_agent, is_agent_paused, _agent_running

        if action == "pause":
            if not _agent_running:
                return jsonify({"status": "error", "message": "Agent is not running"}), 400
            if is_agent_paused():
                return jsonify({"status": "success", "message": "Already paused", "paused": True})
            pause_agent()
            return jsonify({"status": "success", "message": "Agent paused — AI API calls suspended", "paused": True})

        elif action == "resume":
            if not _agent_running:
                return jsonify({"status": "error", "message": "Agent is not running"}), 400
            if not is_agent_paused():
                return jsonify({"status": "success", "message": "Already running", "paused": False})
            resume_agent()
            return jsonify({"status": "success", "message": "Agent resumed — AI API calls active", "paused": False})

        else:
            return jsonify({"status": "error", "message": "Unknown action. Use 'pause' or 'resume'."}), 400

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ==============================================================
# Hikvision ISAPI Integration
# ==============================================================

@bp.route("/api/hikvision/test", methods=["POST"])
@admin_required
def api_hikvision_test():
    """Test connection to a Hikvision camera via ISAPI.
    
    Body: {"host": "192.168.1.100", "username": "admin", "password": "xxx", "port": 80}
    """
    from app.services.hikvision import HikvisionISAPI
    try:
        data = request.get_json(force=True) or {}
        host = str(data.get("host") or "").strip()
        username = str(data.get("username") or "admin").strip()
        password = str(data.get("password") or "").strip()
        port = int(data.get("port") or 80)

        if not host:
            return jsonify({"status": "error", "message": "Host is required"}), 400

        cam = HikvisionISAPI(host, username, password, port)
        result = cam.test_connection()

        if result.get("success"):
            # Also try to get streaming channels
            channels = cam.get_streaming_channels()
            rtsp_main = cam.get_rtsp_url(channel=1, stream=1)
            rtsp_sub = cam.get_rtsp_url(channel=1, stream=2)
            return jsonify({
                "status": "success",
                "device": result,
                "channels": channels[:4],  # Limit to first 4
                "rtsp_urls": {
                    "mainstream": rtsp_main,
                    "substream": rtsp_sub,
                },
            })
        else:
            return jsonify({"status": "error", "message": result.get("error", "Connection failed")}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/hikvision/snapshot", methods=["POST"])
@admin_required
def api_hikvision_snapshot():
    """Capture a high-resolution snapshot from Hikvision camera via ISAPI.
    
    Body: {"host": "192.168.1.100", "username": "admin", "password": "xxx", "port": 80, "channel": 1}
    Returns: JPEG image
    """
    from app.services.hikvision import HikvisionISAPI
    try:
        data = request.get_json(force=True) or {}
        host = str(data.get("host") or "").strip()
        username = str(data.get("username") or "admin").strip()
        password = str(data.get("password") or "").strip()
        port = int(data.get("port") or 80)
        channel = int(data.get("channel") or 1)

        if not host:
            return jsonify({"status": "error", "message": "Host is required"}), 400

        cam = HikvisionISAPI(host, username, password, port)
        jpg_bytes = cam.capture_snapshot(channel)

        if jpg_bytes:
            return Response(jpg_bytes, mimetype="image/jpeg",
                            headers={"Cache-Control": "no-cache"})
        else:
            return jsonify({"status": "error", "message": "Failed to capture snapshot"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/hikvision/device_info", methods=["POST"])
@admin_required
def api_hikvision_device_info():
    """Get full device info from Hikvision camera.
    
    Body: {"host": "...", "username": "admin", "password": "...", "port": 80}
    """
    from app.services.hikvision import HikvisionISAPI
    try:
        data = request.get_json(force=True) or {}
        host = str(data.get("host") or "").strip()
        username = str(data.get("username") or "admin").strip()
        password = str(data.get("password") or "").strip()
        port = int(data.get("port") or 80)

        if not host:
            return jsonify({"status": "error", "message": "Host is required"}), 400

        cam = HikvisionISAPI(host, username, password, port)
        info = cam.get_device_info()
        time_info = cam.get_time()
        channels = cam.get_streaming_channels()

        return jsonify({
            "status": "success",
            "device_info": info,
            "time": time_info,
            "channels": channels,
            "rtsp_urls": {
                "mainstream": cam.get_rtsp_url(1, 1),
                "substream": cam.get_rtsp_url(1, 2),
            },
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/hikvision/anpr/check", methods=["POST"])
@admin_required
def api_hikvision_anpr_check():
    """Check if a Hikvision camera supports built-in ANPR.
    
    Body: {"host": "...", "username": "admin", "password": "...", "port": 80}
    """
    from app.services.hikvision import HikvisionISAPI
    try:
        data = request.get_json(force=True) or {}
        host = str(data.get("host") or "").strip()
        username = str(data.get("username") or "admin").strip()
        password = str(data.get("password") or "").strip()
        port = int(data.get("port") or 80)

        if not host:
            return jsonify({"status": "error", "message": "Host is required"}), 400

        cam = HikvisionISAPI(host, username, password, port)
        supported = cam.check_anpr_support()

        return jsonify({
            "status": "success",
            "anpr_supported": supported,
            "message": "Kamera mendukung ANPR built-in (DeepinView/iDS series)" if supported
                       else "Kamera tidak mendukung ANPR built-in. Sistem akan menggunakan OCR server-side (EasyOCR/PaddleOCR).",
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/hikvision/anpr/trigger", methods=["POST"])
@admin_required
def api_hikvision_anpr_trigger():
    """Trigger on-demand plate recognition on a Hikvision ANPR camera.
    
    Body: {"host": "...", "username": "admin", "password": "...", "port": 80, "channel": 1}
    """
    from app.services.hikvision import HikvisionISAPI
    try:
        data = request.get_json(force=True) or {}
        host = str(data.get("host") or "").strip()
        username = str(data.get("username") or "admin").strip()
        password = str(data.get("password") or "").strip()
        port = int(data.get("port") or 80)
        channel = int(data.get("channel") or 1)

        if not host:
            return jsonify({"status": "error", "message": "Host is required"}), 400

        cam = HikvisionISAPI(host, username, password, port)
        result = cam.trigger_plate_recognition(channel)

        if result:
            return jsonify({"status": "success", "result": result})
        else:
            return jsonify({
                "status": "error",
                "message": "Tidak ada plat terdeteksi atau kamera tidak mendukung fitur ini."
            }), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/hikvision/anpr/plates", methods=["POST"])
@admin_required
def api_hikvision_anpr_plates():
    """Get list of plates detected by Hikvision camera's built-in ANPR.
    
    Body: {"host": "...", "username": "admin", "password": "...", "port": 80,
           "channel": 1, "since": "2026-05-28T00:00:00Z", "max_results": 20}
    """
    from app.services.hikvision import HikvisionISAPI
    try:
        data = request.get_json(force=True) or {}
        host = str(data.get("host") or "").strip()
        username = str(data.get("username") or "admin").strip()
        password = str(data.get("password") or "").strip()
        port = int(data.get("port") or 80)
        channel = int(data.get("channel") or 1)
        since = data.get("since")
        max_results = int(data.get("max_results") or 20)

        if not host:
            return jsonify({"status": "error", "message": "Host is required"}), 400

        cam = HikvisionISAPI(host, username, password, port)
        plates = cam.get_detected_plates(channel, since=since, max_results=max_results)

        return jsonify({
            "status": "success",
            "count": len(plates),
            "plates": plates,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



# ---- Snapshot endpoint for zone editor ----

@bp.route("/api/snapshot/<camera_id>")
def api_snapshot(camera_id):
    """Return a single JPEG frame from the camera (not MJPEG stream).
    Used by zone editor for accurate coordinate mapping."""
    import app.globals as g_state
    import cv2

    # Wait briefly for the latest frame of this camera without hijacking
    # the globally selected live view.
    frame = None
    for _ in range(30):
        with g_state.lock:
            frame = g_state.outputFrames.get(camera_id)
        if frame is not None:
            break
        time.sleep(0.1)

    if frame is None:
        # Return a placeholder
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "NO SIGNAL", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Encode as JPEG
    _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return Response(buf.tobytes(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-cache", "X-Frame-Width": str(frame.shape[1]),
                             "X-Frame-Height": str(frame.shape[0])})



@bp.route("/api/enforcement/live_status")
def api_enforcement_live_status():
    """Get real-time enforcement status: active zones + recent violations for each camera."""
    import app.globals as g_state
    
    cameras = []
    for src in (g_state.CCTV_SOURCES or []):
        cam_id = src.get("id")
        agent = g_state.camera_agents.get(cam_id)
        recent_violations = []
        if agent and hasattr(agent, 'enforcement'):
            recent_violations = agent.enforcement.get_recent_local_violations(limit=5)
        
        zones = get_zones_for_camera(cam_id, only_active=True) if cam_id else []
        
        cameras.append({
            "id": cam_id,
            "name": src.get("name"),
            "zones_count": len(zones),
            "zones": [{"id": z["id"], "zone_type": z["zone_type"], "name": z.get("name")} for z in zones],
            "recent_violations": recent_violations,
            "is_active": cam_id in g_state.camera_agents,
        })
    
    return jsonify({"status": "success", "cameras": cameras})



@bp.route("/api/violations/clear", methods=["POST"])
@admin_required
def api_violations_clear():
    """Clear all violations from database."""
    try:
        conn = get_db_connection()
        conn.execute("DELETE FROM violations")
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": "All violations cleared"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/violations/backfill_vehicle_id", methods=["POST"])
@admin_required
def api_violations_backfill():
    """Backfill vehicle identity (plate, make/model, color) for violations missing this info.
    
    Reads the evidence image from disk and sends to AI vision model.
    Processes up to 20 violations per call to avoid timeouts.
    """
    import threading

    limit = int(request.args.get("limit") or 20)

    def _backfill_worker(violation_ids_and_paths):
        import cv2
        from app.services.ai_ocr import ai_identify_vehicle, ai_read_plate_from_image
        from app.database import update_violation_plate, update_violation_field

        for vid, evidence_path, camera_id in violation_ids_and_paths:
            try:
                img_path = os.path.join(EVIDENCE_DIR, evidence_path.replace("/", os.sep))
                if not os.path.isfile(img_path):
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Try plate reading
                plate_result = ai_read_plate_from_image(img)
                if plate_result and plate_result[0]:
                    plate_text, plate_conf = plate_result
                    try:
                        conn = get_db_connection(timeout_s=5)
                        from app.database import _execute
                        _execute(conn, "UPDATE violations SET plate_text=?, plate_confidence=? WHERE id=?",
                                 (plate_text, float(plate_conf), vid))
                        conn.commit()
                        conn.close()
                    except Exception:
                        pass

                # Try vehicle identification
                info = ai_identify_vehicle(img)
                if info:
                    notes_parts = []
                    if info.get('vehicle_type') and info['vehicle_type'] not in ('Unknown', 'N/A', ''):
                        notes_parts.append(f"Jenis: {info['vehicle_type']}")
                    if info.get('company') and info['company'] not in ('Private', 'Unknown', 'N/A', ''):
                        notes_parts.append(f"Perusahaan: {info['company']}")
                    if info.get('make_model') and info['make_model'] not in ('Unknown', 'N/A', ''):
                        notes_parts.append(f"Merek/Model: {info['make_model']}")
                    if info.get('color') and info['color'] not in ('Unknown', 'N/A', ''):
                        notes_parts.append(f"Warna: {info['color']}")
                    if info.get('registration_area') and info['registration_area'] not in ('Unknown', 'N/A', ''):
                        notes_parts.append(f"Daerah: {info['registration_area']}")

                    # Update vehicle_class
                    vtype = info.get('vehicle_type', '')
                    if vtype and vtype not in ('Unknown', 'N/A', ''):
                        try:
                            conn = get_db_connection(timeout_s=5)
                            from app.database import _execute
                            _execute(conn, "UPDATE violations SET vehicle_class=? WHERE id=?", (vtype, vid))
                            conn.commit()
                            conn.close()
                        except Exception:
                            pass

                    # Update notes
                    notes_str = " | ".join(notes_parts)
                    if notes_str:
                        try:
                            conn = get_db_connection(timeout_s=5)
                            from app.database import _execute
                            _execute(conn, "UPDATE violations SET notes=? WHERE id=?", (notes_str, vid))
                            conn.commit()
                            conn.close()
                        except Exception:
                            pass

                    # If AI found plate and violation doesn't have one yet
                    ai_plate = info.get('plate', '')
                    if ai_plate and ai_plate not in ('N/A', 'Unknown', '') and not plate_result[0]:
                        try:
                            conn = get_db_connection(timeout_s=5)
                            from app.database import _execute
                            _execute(conn, "UPDATE violations SET plate_text=?, plate_confidence=? WHERE id=? AND (plate_text IS NULL OR plate_text='')",
                                     (ai_plate, 0.7, vid))
                            conn.commit()
                            conn.close()
                        except Exception:
                            pass

                    print(f"[BACKFILL] id={vid} | {notes_str}")

            except Exception as e:
                print(f"[BACKFILL] id={vid} | ERROR: {e}")

    try:
        # Find violations without notes (no vehicle identity yet)
        conn = get_db_connection()
        from app.database import _execute
        c = _execute(conn,
            "SELECT id, evidence_path, camera_id FROM violations WHERE (notes IS NULL OR notes = '') AND evidence_path IS NOT NULL ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = c.fetchall()
        conn.close()

        if hasattr(rows[0], 'keys') if rows else False:
            violations = [(r['id'], r['evidence_path'], r['camera_id']) for r in rows]
        else:
            violations = [(dict(r)['id'], dict(r)['evidence_path'], dict(r)['camera_id']) for r in rows]

        if not violations:
            return jsonify({"status": "success", "message": "No violations need backfilling", "count": 0})

        # Run in background thread
        t = threading.Thread(target=_backfill_worker, args=(violations,), daemon=True)
        t.start()

        return jsonify({
            "status": "success",
            "message": f"Backfilling {len(violations)} violations in background",
            "count": len(violations),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
