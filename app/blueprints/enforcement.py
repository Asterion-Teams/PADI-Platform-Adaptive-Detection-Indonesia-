"""
Enforcement, violations, zones, CRM, evidence serving, and executive summary routes.
Separated from main routes.py for modularity.
"""
import os
import json
import time
import io
import csv
import datetime
from flask import Blueprint, render_template, Response, jsonify, request, redirect, session
from app.config import EVIDENCE_DIR, VIOLATION_TYPES, ZONE_TYPES
from app.auth import admin_required, get_current_user
from app.database import (
    insert_zone, update_zone, delete_zone, get_zones_for_camera, get_all_zones,
    insert_violation, list_violations, get_violation, update_violation,
    violation_summary, violation_heatmap_by_camera,
    insert_crm_report, list_crm_reports, update_crm_report, crm_summary,
    recommend_enforcement_points,
)
from app.services.enforcement import auto_classify_crm
import app.globals as globals_state

bp = Blueprint('enforcement', __name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_ts(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except Exception:
        pass
    try:
        dt = datetime.datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def _count_violations(camera_id=None, violation_type=None, start_ts=None, end_ts=None,
                      plate_contains=None, status=None):
    """Count total violations matching filters (for pagination)."""
    try:
        from app.database import get_db_connection, _execute
        where = []
        params = []
        if camera_id:
            where.append("camera_id = ?"); params.append(str(camera_id))
        if violation_type:
            where.append("violation_type = ?"); params.append(str(violation_type))
        if start_ts is not None:
            where.append("timestamp >= ?"); params.append(float(start_ts))
        if end_ts is not None:
            where.append("timestamp <= ?"); params.append(float(end_ts))
        if plate_contains:
            where.append("plate_text LIKE ?"); params.append(f"%{plate_contains}%")
        if status:
            where.append("status = ?"); params.append(str(status))
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        conn = get_db_connection()
        try:
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


# ── Page routes ───────────────────────────────────────────────────────────────

@bp.route("/enforcement")
def enforcement_page():
    user = get_current_user()
    if not user or user.get("role") != "admin":
        return redirect("/login")
    return render_template("enforcement.html")


@bp.route("/zones")
def zones_page():
    user = get_current_user()
    if not user or user.get("role") != "admin":
        return redirect("/login")
    return render_template("zones.html")


@bp.route("/crm")
def crm_page():
    user = get_current_user()
    if not user or user.get("role") != "admin":
        return redirect("/login")
    return render_template("crm.html")


@bp.route("/executive_summary")
def executive_summary_page():
    user = get_current_user()
    if not user or user.get("role") != "admin":
        return redirect("/login")
    return render_template("executive_summary.html")


# ── Evidence serving ───────────────────────────────────────────────────────────

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
        v = get_violation(violation_id)
        if not v or not v.get("evidence_path") or not v.get("bbox"):
            return Response("Not Found", status=404)

        img_path = os.path.join(EVIDENCE_DIR, v["evidence_path"].replace("/", os.sep))
        if not os.path.isfile(img_path):
            img_path = os.path.join(os.path.dirname(EVIDENCE_DIR), v["evidence_path"].replace("/", os.sep))
        if not os.path.isfile(img_path):
            return Response("Not Found", status=404)

        img = cv2.imread(img_path)
        if img is None:
            return Response("Not Found", status=404)

        bbox = v["bbox"]
        if isinstance(bbox, str):
            bbox = json.loads(bbox)
        x1, y1, x2, y2 = [int(b) for b in bbox]
        h, w = img.shape[:2]
        pad = 30
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return Response("Not Found", status=404)

        ch, cw = crop.shape[:2]
        if cw > 200:
            scale = 200 / cw
            crop = cv2.resize(crop, (200, int(ch * scale)))

        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(buf.tobytes(), mimetype="image/jpeg")
    except Exception:
        return Response("Not Found", status=404)


# ── Zones API ─────────────────────────────────────────────────────────────────

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
def api_zones_delete(zone_id):
    try:
        ok = delete_zone(zone_id)
        return jsonify({"status": "success" if ok else "not_found"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Violations API ─────────────────────────────────────────────────────────────

@bp.route("/api/violations")
def api_violations_list():
    try:
        limit = max(1, min(1000, int(request.args.get("limit") or 50)))
        offset = max(0, int(request.args.get("offset") or 0))
        cam = request.args.get("camera_id") or None
        vtype = request.args.get("violation_type") or None
        plate = request.args.get("plate") or None
        status = request.args.get("status") or None
        start_ts = _parse_ts(request.args.get("start"))
        end_ts = _parse_ts(request.args.get("end"))
        rows = list_violations(
            limit=limit, offset=offset,
            camera_id=cam, violation_type=vtype,
            start_ts=start_ts, end_ts=end_ts,
            plate_contains=plate, status=status,
        )
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
            mult = {"1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800, "30d": 2592000}.get(period)
            if mult:
                start_ts = now - mult
                end_ts = now
        summary = violation_summary(start_ts=start_ts, end_ts=end_ts)
        return jsonify({"status": "success", "summary": summary, "start_ts": start_ts, "end_ts": end_ts})
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
            mult = {"1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800, "30d": 2592000}.get(period)
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
        rows = list_violations(limit=10000, offset=0, camera_id=cam, violation_type=vtype, start_ts=start_ts, end_ts=end_ts)
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


@bp.route("/api/violations/executive_summary")
def api_executive_summary():
    """Generate daily/weekly executive summary for DISHUB stakeholder reports."""
    try:
        period = request.args.get("period", "24h")
        now = time.time()
        secs = {"24h": 86400, "7d": 604800, "30d": 2592000}.get(period, 86400)
        start_ts = now - secs

        summary = violation_summary(start_ts=start_ts, end_ts=now)
        prev_summary = violation_summary(start_ts=start_ts - secs, end_ts=start_ts)
        all_time_summary = violation_summary()
        heatmap = violation_heatmap_by_camera(start_ts=start_ts, end_ts=now)
        top_recs = recommend_enforcement_points(top_n=5, start_ts=start_ts, end_ts=now)
        crm = crm_summary()

        recent = list_violations(limit=20, offset=0, start_ts=start_ts, end_ts=now)
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
            bullets.append(f"Recommended to install E-TLE cameras or deploy officers at top {len(top_recs)} locations based on vulnerability score.")

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


# ── CRM API ────────────────────────────────────────────────────────────────────

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
        auto_type = auto_classify_crm(description)
        prio = "normal"
        urgent_kw = ("kecelakaan", "accident", "urgent", "segera", "darurat")
        if any(k in description.lower() for k in urgent_kw):
            prio = "high"
        rid = insert_crm_report(
            reporter_name=name, reporter_contact=contact, category=category,
            description=description, lat=lat, lng=lng, camera_id=camera_id,
            auto_classified_type=auto_type, priority=prio,
        )
        return jsonify({"status": "success", "id": rid, "auto_classified_type": auto_type, "priority": prio})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/crm/reports/<int:rid>", methods=["PATCH", "PUT"])
@admin_required
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
    """Monitor social media mentions (@DishubDKI) via Playwright scraper."""
    all_mentions = []
    try:
        from app.services.social_scraper import scrape_twitter_mentions
        playwright_results = scrape_twitter_mentions(max_results=20)
        for m in playwright_results:
            if m.get("platform") in ("twitter", "threads", "instagram", "facebook"):
                all_mentions.append(m)
    except Exception as e:
        print(f"[CRM] Playwright scraper error: {e}")

    all_mentions.sort(key=lambda x: (0 if x.get("priority") == "high" else 1, -x.get("timestamp", 0)))

    return jsonify({
        "status": "success",
        "mentions": all_mentions[:20],
        "count": len(all_mentions),
        "sources": ["twitter/x", "threads", "instagram", "facebook"],
    })


# ── Meta & OCR Test ───────────────────────────────────────────────────────────

@bp.route("/api/enforcement/meta")
def api_enforcement_meta():
    return jsonify({
        "status": "success",
        "violation_types": VIOLATION_TYPES,
        "zone_types": ZONE_TYPES,
    })


@bp.route("/api/ocr/test", methods=["POST"])
def api_ocr_test():
    """Test OCR on an uploaded image."""
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
        plate, conf, engine = recognize_plate(img, (0, 0, w, h))

        results = []
        if plate:
            results.append({"text": plate, "confidence": round(float(conf), 3), "method": engine})

        vehicle_identity = None
        try:
            info = ai_identify_vehicle(img)
            if info:
                vehicle_identity = info
        except Exception:
            pass

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


@bp.route("/api/ai/test")
def api_ai_test():
    """Test AI provider connection."""
    try:
        from app.services.ai_ocr import ai_test_connection
        result = ai_test_connection()
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


def _find_vehicle_region(img):
    """Find red bounding box in evidence image and crop vehicle inside it."""
    import cv2
    import numpy as np
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        red_mask = mask1 | mask2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

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

        x1, y1, x2, y2 = best_box
        pad = 6
        x1 = min(w - 2, x1 + pad); y1 = min(h - 2, y1 + pad)
        x2 = max(x1 + 1, x2 - pad); y2 = max(y1 + 1, y2 - pad)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
            return None
        return crop
    except Exception:
        return None
