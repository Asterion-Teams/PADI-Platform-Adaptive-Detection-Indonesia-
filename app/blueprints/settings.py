"""
Settings page and runtime config API.
Extracted from routes.py for modularity.
"""
import os
import json
from flask import Blueprint, render_template, jsonify, request
from app.auth import admin_required, get_current_user

bp = Blueprint('settings', __name__)


@bp.route("/settings")
def settings_page():
    user = get_current_user()
    if not user or user.get("role") != "admin":
        from flask import redirect
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
                "finetuned_11m_path": cfg.YOLO11M_FINETUNED_PATH,
                "generic_11m_path": cfg.YOLO11M_GENERIC_PATH,
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
                "busway_fast_min_seconds": cfg.BUSWAY_FAST_MIN_SECONDS,
                "busway_medium_min_seconds": cfg.BUSWAY_MEDIUM_MIN_SECONDS,
                "busway_slow_min_seconds": cfg.BUSWAY_SLOW_MIN_SECONDS,
                "busway_min_zone_hits": cfg.BUSWAY_MIN_ZONE_HITS,
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
def api_settings_update():
    """Update runtime settings used by the active live pipeline."""
    import app.config as cfg
    from app.globals import CCTV_SOURCES

    p = request.get_json(silent=True) or {}
    updated = []

    # Detection settings
    det = p.get("detection") or {}
    if "conf_threshold" in det:
        cfg.CONF_THRESHOLD = max(0.01, min(1.0, float(det["conf_threshold"]))); updated.append("conf_threshold")
    if "iou_threshold" in det:
        cfg.IOU_THRESHOLD = max(0.01, min(1.0, float(det["iou_threshold"]))); updated.append("iou_threshold")
    if "infer_imgsz" in det:
        cfg.INFER_IMGSZ = int(det["infer_imgsz"]); updated.append("infer_imgsz")
    if "stream_fps" in det:
        cfg.STREAM_FPS = max(1.0, float(det["stream_fps"])); updated.append("stream_fps")
    if "stream_max_width" in det:
        cfg.STREAM_MAX_WIDTH = int(det["stream_max_width"]); updated.append("stream_max_width")
    if "stream_jpeg_quality" in det:
        cfg.STREAM_JPEG_QUALITY = max(30, min(95, int(det["stream_jpeg_quality"]))); updated.append("stream_jpeg_quality")

    # Enforcement settings
    enf = p.get("enforcement") or {}
    if "violations_enabled" in enf:
        cfg.VIOLATIONS_ENABLED = bool(enf["violations_enabled"]); updated.append("violations_enabled")
    if "anpr_enabled" in enf:
        cfg.ANPR_ENABLED = bool(enf["anpr_enabled"]); updated.append("anpr_enabled")
    if "anpr_fallback_simulate" in enf:
        cfg.ANPR_FALLBACK_SIMULATE = bool(enf["anpr_fallback_simulate"]); updated.append("anpr_fallback_simulate")
    if "illegal_parking_min_seconds" in enf:
        cfg.ILLEGAL_PARKING_MIN_SECONDS = max(1.0, float(enf["illegal_parking_min_seconds"])); updated.append("illegal_parking_min_seconds")
    if "static_movement_px" in enf:
        cfg.STATIC_MOVEMENT_PX = max(1.0, float(enf["static_movement_px"])); updated.append("static_movement_px")
    if "dynamic_lane_min_seconds" in enf:
        cfg.DYNAMIC_LANE_MIN_SECONDS = max(0.5, float(enf["dynamic_lane_min_seconds"])); updated.append("dynamic_lane_min_seconds")
    if "violation_cooldown_seconds" in enf:
        cfg.VIOLATION_COOLDOWN_SECONDS = max(5.0, float(enf["violation_cooldown_seconds"])); updated.append("violation_cooldown_seconds")
    if "busway_fast_min_seconds" in enf:
        cfg.BUSWAY_FAST_MIN_SECONDS = max(0.5, float(enf["busway_fast_min_seconds"])); updated.append("busway_fast_min_seconds")
    if "busway_medium_min_seconds" in enf:
        cfg.BUSWAY_MEDIUM_MIN_SECONDS = max(0.5, float(enf["busway_medium_min_seconds"])); updated.append("busway_medium_min_seconds")
    if "busway_slow_min_seconds" in enf:
        cfg.BUSWAY_SLOW_MIN_SECONDS = max(0.5, float(enf["busway_slow_min_seconds"])); updated.append("busway_slow_min_seconds")
    if "busway_min_zone_hits" in enf:
        cfg.BUSWAY_MIN_ZONE_HITS = max(1, int(enf["busway_min_zone_hits"])); updated.append("busway_min_zone_hits")

    # Social media settings
    social = p.get("social_media") or {}
    if "search_query" in social:
        cfg.TWITTER_SEARCH_QUERY = str(social["search_query"]).strip(); updated.append("search_query")
    if "max_results" in social:
        cfg.TWITTER_MAX_RESULTS = max(5, min(50, int(social["max_results"]))); updated.append("max_results")
    if "x_search_query" in social:
        cfg.X_SEARCH_QUERY = str(social["x_search_query"]).strip(); updated.append("x_search_query")
        try:
            from app.services.social_scraper import _cache_lock, _cached_mentions
            with _cache_lock:
                _cached_mentions.clear()
        except Exception:
            pass
    if "x_cookies_json" in social:
        cookies_str = str(social["x_cookies_json"]).strip()
        if cookies_str:
            try:
                json.loads(cookies_str)
                with open(cfg.X_COOKIES_FILE, 'w', encoding='utf-8') as f:
                    f.write(cookies_str)
                updated.append("x_cookies")
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
        cfg.AI_PROVIDER = str(ai["provider"]).strip(); updated.append("ai_provider")
    if "base_url" in ai:
        cfg.AI_BASE_URL = str(ai["base_url"]).strip(); updated.append("ai_base_url")
    if "api_key" in ai:
        cfg.AI_API_KEY = str(ai["api_key"]).strip(); updated.append("ai_api_key")
    if "model" in ai:
        cfg.AI_MODEL = str(ai["model"]).strip(); updated.append("ai_model")
    if "use_ai_for_anpr" in ai:
        cfg.AI_USE_FOR_ANPR = bool(ai["use_ai_for_anpr"]); updated.append("ai_use_for_anpr")
    if "use_ai_for_chat" in ai:
        cfg.AI_USE_FOR_CHAT = bool(ai["use_ai_for_chat"]); updated.append("ai_use_for_chat")
    if "chat_model" in ai:
        cfg.AI_CHAT_MODEL = str(ai["chat_model"]).strip(); updated.append("ai_chat_model")

    # General settings
    general = p.get("general") or {}
    if "timezone" in general:
        cfg.TIMEZONE = str(general["timezone"]).strip(); updated.append("timezone")

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

                # Detect if it's a fine-tuned/custom model (uses 6-class vehicle mapping)
                if any(k in new_model_name for k in ["vehicle_v3", "custom", "finetuned", "yolo11m"]):
                    from app.config import VEHICLE_CLASSES_CUSTOM, CLASS_MAPPING_CUSTOM, VEHICLE_CLASSES_COCO, CLASS_MAPPING_COCO
                    cfg.VEHICLE_CLASSES = VEHICLE_CLASSES_CUSTOM
                    cfg.CLASS_MAPPING = CLASS_MAPPING_CUSTOM
                    cfg.USE_CUSTOM_YOLO = True
                else:
                    from app.config import VEHICLE_CLASSES_COCO, CLASS_MAPPING_COCO
                    cfg.VEHICLE_CLASSES = VEHICLE_CLASSES_COCO
                    cfg.CLASS_MAPPING = CLASS_MAPPING_COCO
                    cfg.USE_CUSTOM_YOLO = False

                for agent in g_state.camera_agents.values():
                    agent.model = new_model

                model_switched = True
                updated.append(f"active_model -> {new_model_name}")
            except Exception as e:
                return jsonify({"status": "error", "message": f"Failed to load model: {e}"}), 500
        else:
            return jsonify({"status": "error", "message": f"Model file not found: {new_model_name}"}), 404

    # Persist all settings to disk
    cfg.save_persisted_settings()

    return jsonify({
        "status": "success",
        "updated": updated,
        "model_switched": model_switched,
        "message": f"{len(updated)} setting(s) updated" + (" (model hot-swapped)" if model_switched else ""),
    })


@bp.route("/api/settings/restart_agents", methods=["POST"])
@admin_required
def api_restart_agents():
    """Restart all camera agents (useful after model switch)."""
    import app.globals as g_state
    from app.services.camera import start_camera_agents, stop_agent

    ids = list(g_state.camera_agents.keys())
    for cid in ids:
        try:
            stop_agent(cid)
        except Exception:
            pass

    start_camera_agents()

    return jsonify({
        "status": "success",
        "message": f"{len(ids)} agent(s) restarted",
        "agents_running": len(g_state.camera_agents),
    })
