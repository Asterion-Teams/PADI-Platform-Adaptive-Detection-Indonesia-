"""
Camera management, stats, history, and prediction routes.
Separated from main routes.py for modularity.
"""
import os
import json
import time
import io
import csv
import uuid
import datetime
from collections import deque
from flask import Blueprint, render_template, Response, jsonify, request, session, redirect
from app.config import DATA_DIR, HISTORY_MAX_LEN
from app.auth import admin_required, get_current_user
from app.database import (
    predict_future_traffic,
    get_history_range,
    get_aggregated_stats,
    clear_all_history,
    get_total_lifetime,
    get_totals_by_camera,
)
from app.utils import load_config, save_config, save_stats, sync_stats_with_config, backfill_camera_history
import app.globals as globals_state

bp = Blueprint('cameras', __name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── Page routes ───────────────────────────────────────────────────────────────

@bp.route("/dashboard")
def dashboard_page():
    user = get_current_user()
    if not user:
        return redirect("/login")
    return render_template("dashboard.html")


@bp.route("/analysis")
def analysis_page():
    user = get_current_user()
    if not user:
        return redirect("/login")
    return render_template("analysis.html")


@bp.route("/cameras")
def cameras_page():
    user = get_current_user()
    if not user:
        return redirect("/login")
    return render_template("cameras.html")


# ── Video stream ──────────────────────────────────────────────────────────────

@bp.route("/stream")
def stream_page():
    user = get_current_user()
    if not user:
        return redirect("/login")
    return render_template("stream.html")


@bp.route("/video_feed")
def video_feed():
    from app.services.camera import generate_frames
    from flask import Response, request as flask_request

    def generate():
        for frame_data in generate_frames():
            if frame_data is None:
                continue
            if isinstance(frame_data, bytes):
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n")
            elif isinstance(frame_data, dict):
                yield (b"data: " + json.dumps(frame_data).encode() + b"\n")
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Camera management API ─────────────────────────────────────────────────────

@bp.route("/api/cameras", methods=["GET"])
def api_cameras_list():
    config = load_config()
    return jsonify({"status": "success", "sources": config or []})


@bp.route("/api/add_camera", methods=["POST"])
@admin_required
def add_camera():
    try:
        data = request.json or {}
        name = str(data.get("name") or "").strip()
        url = str(data.get("url") or "").strip()
        lat_raw = str(data.get("lat") or "").strip()
        lng_raw = str(data.get("lng") or "").strip()
        try:
            lat = float(lat_raw) if lat_raw else None
            lng = float(lng_raw) if lng_raw else None
        except Exception:
            return jsonify({"status": "error", "message": "Invalid lat/lng"}), 400

        if not name or not url:
            return jsonify({"status": "error", "message": "Name and URL required"}), 400

        config = load_config()
        new_id = f"cam_{uuid.uuid4().hex[:10]}"
        existing_ids = {c.get("id") for c in config}
        while new_id in existing_ids:
            new_id = f"cam_{uuid.uuid4().hex[:10]}"

        cam = {"id": new_id, "name": name, "url": url, "lat": lat, "lng": lng, "active": True if not config else False}
        if not config:
            cam["active"] = True
        config.append(cam)
        if not save_config(config):
            return jsonify({"status": "error", "message": "Failed to save config"}), 500

        from app.globals import CCTV_SOURCES
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

        from app.globals import CCTV_SOURCES
        CCTV_SOURCES[:] = config

        # Stop the camera agent FIRST before removing stats
        try:
            from app.services.camera import stop_agent
            stop_agent(target_id)
        except Exception:
            pass

        # Small delay to let the thread fully exit its loop
        import time as time_module
        time_module.sleep(0.3)

        if target_id in globals_state.global_stats:
            del globals_state.global_stats[target_id]
            save_stats()
        sync_stats_with_config()

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/switch_source", methods=["POST"])
@admin_required
def switch_source():
    try:
        from app.globals import CCTV_SOURCES
        data = request.json
        new_id = data.get("id")

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
            return jsonify({"status": "error", "message": "Replay only available for video files"}), 400

        if agent.cap is not None:
            try:
                import cv2 as _cv2
                agent.cap.set(_cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception:
                pass

        try:
            agent.enforcement._tracked.clear()
        except Exception:
            pass

        return jsonify({"status": "success", "message": "Video replaying from start"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/edit_camera", methods=["POST"])
@admin_required
def edit_camera():
    try:
        from app.globals import CCTV_SOURCES
        data = request.json
        config = load_config()

        updated = False
        for cam in config:
            if cam["id"] == data["id"]:
                if "name" in data and data["name"] is not None:
                    cam["name"] = data["name"]
                if "url" in data and data["url"] is not None:
                    cam["url"] = data["url"]
                lat_raw = str(data.get("lat") or "").strip()
                lng_raw = str(data.get("lng") or "").strip()
                try:
                    cam["lat"] = float(lat_raw) if lat_raw not in (None, "") else None
                    cam["lng"] = float(lng_raw) if lng_raw not in (None, "") else None
                except Exception:
                    return jsonify({"status": "error", "message": "Invalid lat/lng"}), 400
                updated = True
                break

        if updated:
            if not save_config(config):
                return jsonify({"status": "error", "message": "Failed to save config"}), 500
            CCTV_SOURCES[:] = config
            cam_id = data.get("id")
            if cam_id in globals_state.global_stats:
                globals_state.global_stats[cam_id]["name"] = next(
                    (c.get("name") for c in config if c.get("id") == cam_id),
                    globals_state.global_stats[cam_id].get("name")
                )
                save_stats()
            return jsonify({"status": "success", "message": "Camera updated"})
        else:
            return jsonify({"status": "error", "message": "Camera not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/reset_data", methods=["POST"])
@admin_required
def reset_data():
    try:
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


# ── Stats & History API ───────────────────────────────────────────────────────

@bp.route("/api/history")
def get_history_api():
    period = request.args.get("period", "30m")
    camera_id = request.args.get("camera_id")
    start_ts_arg = request.args.get("start_ts")
    end_ts_arg = request.args.get("end_ts")

    now = time.time()
    interval = 60

    if period == "30m":
        start_ts = now - 1800
        interval = 60
    elif period == "1h":
        start_ts = now - 3600
        interval = 60
    elif period == "6h":
        start_ts = now - (6 * 3600)
        interval = 300
    elif period == "12h":
        start_ts = now - (12 * 3600)
        interval = 900
    elif period == "24h":
        start_ts = now - (24 * 3600)
        interval = 1800
    elif period == "7d":
        start_ts = now - (7 * 24 * 3600)
        interval = 14400
    elif period == "30d":
        start_ts = now - (30 * 24 * 3600)
        interval = 86400
    elif period == "custom":
        start_ts = float(start_ts_arg) if start_ts_arg else (now - 86400)
        end_ts = float(end_ts_arg) if end_ts_arg else (start_ts + 86400)
        interval = 3600
    else:
        start_ts = now - 1800
        interval = 60

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

    # Aggregate
    buckets = {}
    for r in rows:
        ts = r["ts"]
        bucket_ts = int(ts // interval) * interval
        if bucket_ts not in buckets:
            buckets[bucket_ts] = {
                "count": 0, "cars": 0, "motors": 0,
                "density_sum": 0, "density_n": 0, "density_peak": 0,
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
            "count_per_cam": int(round((b["count"] or 0) / denom)),
            "cars_per_cam": int(round((b["cars"] or 0) / denom)),
            "motors_per_cam": int(round((b["motors"] or 0) / denom)),
            "active_cams": int(active_cams),
            "density_avg": density_avg,
            "density_peak": int(b.get("density_peak", 0) or 0),
            "ts": ts
        })

    return jsonify(data)


@bp.route("/api/stats")
def get_stats():
    try:
        stats_path = os.path.join(DATA_DIR, 'traffic_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                data = json.load(f)

            # Remove heavy history arrays from response
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


# ── CSV Export ─────────────────────────────────────────────────────────────────

@bp.route("/api/export_csv")
def export_csv_api():
    try:
        period = request.args.get("period", "30m")
        camera_id = request.args.get("camera_id")
        start_ts_arg = request.args.get("start_ts")
        end_ts_arg = request.args.get("end_ts")

        now = time.time()
        periods = {
            "30m": 1800, "1h": 3600, "6h": 6 * 3600,
            "12h": 12 * 3600, "24h": 24 * 3600,
            "7d": 7 * 24 * 3600, "30d": 30 * 24 * 3600,
        }

        if period == "custom":
            start_ts = float(start_ts_arg) if start_ts_arg else (now - 86400)
            end_ts = float(end_ts_arg) if end_ts_arg else (start_ts + 86400)
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
                int(r.get("motorcycles") or 0),
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
                int(r.get("motorcycles") or 0),
            ])

        csv_data = out.getvalue()
        filename = f"traffic_export_{start_dt.strftime('%Y%m%d_%H%M')}_{end_dt.strftime('%Y%m%d_%H%M')}.csv"
        resp = Response(csv_data, mimetype="text/csv; charset=utf-8")
        resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return resp
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Prometheus Metrics ─────────────────────────────────────────────────────────

@bp.route("/metrics")
def metrics():
    lifetime = get_total_lifetime()
    monthly = get_aggregated_stats(days=30)
    from app.globals import CCTV_SOURCES
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


# ── Traffic Prediction ─────────────────────────────────────────────────────────

@bp.route("/api/predict_traffic", methods=["POST"])
def predict_traffic():
    try:
        data = request.json
        target_time_str = data.get("target_time")
        req_camera_id = data.get("camera_id")
        force_scenario = data.get("force_scenario")

        if target_time_str:
            from datetime import datetime
            dt = datetime.fromisoformat(target_time_str)
            day_of_week = int(dt.strftime('%w'))
            hour = dt.hour
        else:
            day_of_week = data.get("day_of_week")
            hour = data.get("hour")

        if day_of_week is None or hour is None:
            return jsonify({"status": "error", "message": "Missing time parameters"}), 400

        config = load_config()
        thresholds_path = os.path.join(DATA_DIR, 'camera_thresholds.json')
        thresholds = {}
        if os.path.exists(thresholds_path):
            with open(thresholds_path, 'r') as f:
                thresholds = json.load(f)

        predictions = []
        for cam in config:
            avg_count = predict_future_traffic(cam["id"], int(day_of_week), int(hour))

            # Demo scenario injection
            if force_scenario == 'high_traffic':
                import random
                avg_count = max(avg_count, random.randint(250, 400))
            elif force_scenario == 'low_traffic':
                avg_count = min(avg_count, 50)

            # Decision logic
            cam_thresholds = thresholds.get(cam["id"], {"p50": 100, "p75": 200, "p90": 300})

            status = "LANCAR"
            recommendation = "Traffic flow is optimal. Continue standard monitoring."
            action_icon = "fas fa-check-circle"
            status_color = "text-green-500"

            if avg_count > cam_thresholds["p90"]:
                status = "MACET TOTAL"
                recommendation = "CRITICAL ACTION: 1) Deploy Field Unit to intersection. 2) Override traffic light to manual flush. 3) Notify Traffic Command Center."
                action_icon = "fas fa-exclamation-triangle"
                status_color = "text-red-500"
            elif avg_count > cam_thresholds["p75"]:
                status = "MACET"
                recommendation = "ACTION REQUIRED: 1) Extend Green Light duration by 15s. 2) Display 'Congestion Ahead' on VMS."
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
    from app.utils import get_datalake_stats
    date_str = request.args.get("date")
    result = get_datalake_stats(date_str)
    return jsonify(result)
