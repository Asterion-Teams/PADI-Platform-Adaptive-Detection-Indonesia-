"""
Vehicle Identity Agent
======================
Background agent that continuously maintains violation data quality:
1. Fills missing plate numbers, vehicle make/model, color from evidence images
2. Deduplicates violations (same plate at same camera within window)
3. Removes violations with corrupted/missing evidence

Runs as a daemon thread, processing in cycles every 30 seconds.
"""
import os
import time
import json
import base64
import urllib.request
import urllib.error
import threading

import cv2

from app.config import EVIDENCE_DIR
import app.config as cfg


_agent_running = False
_agent_paused = False  # When True, agent loop sleeps but stays alive (no API calls, no token usage)


def _resolve_evidence_path(evidence_path):
    """Find the actual file path for an evidence_path from database."""
    if not evidence_path:
        return None
    p1 = os.path.join(EVIDENCE_DIR, evidence_path.replace("/", os.sep))
    if os.path.isfile(p1):
        return p1
    p2 = os.path.join(os.path.dirname(EVIDENCE_DIR), evidence_path.replace("/", os.sep))
    if os.path.isfile(p2):
        return p2
    return None


def _ai_analyze_vehicle(img):
    """Send vehicle image to AI and get plate + details.
    
    Returns dict: {plate, vehicle_type, make_model, color, registration_area}
    """
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    img_b64 = base64.b64encode(buf.tobytes()).decode()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.AI_API_KEY}",
    }

    prompt = """Dari gambar CCTV ini, identifikasi kendaraan di dalam kotak merah:
1. Plat nomor (format Indonesia: [HURUF 1-2] [ANGKA 1-4] [HURUF 0-3]). Baca dari bumper depan/belakang.
2. Jenis kendaraan: Sedan/MPV/SUV/Hatchback/Motorcycle/Pickup/Truck/Bus/Van/Minibus
3. Merek dan Model spesifik (Toyota Avanza, Honda Jazz, Suzuki Carry, Mitsubishi Xpander, dll)
4. Warna dominan (Putih/Hitam/Silver/Abu-abu/Merah/Biru/Kuning/Hijau/Coklat/Emas)
5. Daerah registrasi dari prefix plat (B=Jakarta, D=Bandung, F=Bogor, H=Semarang, L=Surabaya)

PENTING: JANGAN jawab "Unknown" atau "N/A" untuk jenis, merek, atau warna. Selalu estimasi.
Jika plat tidak terlihat, plate = "N/A".

Jawab JSON SAJA:
{"plate":"B 1234 ABC","vehicle_type":"MPV","make_model":"Toyota Avanza","color":"Putih","registration_area":"Jakarta"}"""

    vehicle_model = getattr(cfg, 'AI_VEHICLE_MODEL', None) or cfg.AI_MODEL

    body = {
        "model": vehicle_model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]}],
        "max_tokens": 150,
        "temperature": 0.1,
    }

    url = f"{cfg.AI_BASE_URL.rstrip('/')}/chat/completions"
    req = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")

    with urllib.request.urlopen(req, timeout=15) as resp:
        result = json.loads(resp.read())

    content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    return json.loads(content)


def _cycle_fill_details():
    """Fill vehicle details for violations missing them."""
    from app.database import get_db_connection, _execute

    conn = get_db_connection(timeout_s=5)
    try:
        c = _execute(conn,
            "SELECT id, evidence_path, plate_text, notes FROM violations WHERE evidence_path IS NOT NULL ORDER BY id DESC",
            ())
        rows = c.fetchall()
    finally:
        conn.close()

    # Filter: missing Merek/Model or Warna
    to_process = []
    for row in rows:
        r = dict(row) if hasattr(row, 'keys') else row
        notes = r.get('notes') or ''
        if 'Merek/Model' not in notes or 'Warna' not in notes:
            to_process.append(r)
        if len(to_process) >= 5:  # Process 5 per cycle
            break

    filled = 0
    for r in to_process:
        vid = r.get('id')
        evidence_path = r.get('evidence_path')
        existing_plate = r.get('plate_text') or ''

        img_path = _resolve_evidence_path(evidence_path)
        if not img_path:
            # Mark as no file to avoid retrying
            try:
                conn = get_db_connection(timeout_s=3)
                _execute(conn, "UPDATE violations SET notes=? WHERE id=?", ("no_file", vid))
                conn.commit()
                conn.close()
            except Exception:
                pass
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        try:
            info = _ai_analyze_vehicle(img)
            plate = info.get("plate", "N/A")
            vtype = info.get("vehicle_type", "")
            make_model = info.get("make_model", "")
            color = info.get("color", "")
            reg_area = info.get("registration_area", "")

            # Build notes
            notes_parts = []
            if vtype and vtype not in ("N/A", "Unknown", ""):
                notes_parts.append(f"Jenis: {vtype}")
            if make_model and make_model not in ("N/A", "Unknown", ""):
                notes_parts.append(f"Merek/Model: {make_model}")
            if color and color not in ("N/A", "Unknown", ""):
                notes_parts.append(f"Warna: {color}")
            if reg_area and reg_area not in ("N/A", "Unknown", ""):
                notes_parts.append(f"Daerah: {reg_area}")

            notes_str = " | ".join(notes_parts) if notes_parts else "analyzed"

            conn = get_db_connection(timeout_s=5)
            try:
                if plate and plate != "N/A" and "unknown" not in plate.lower() and not existing_plate:
                    _execute(conn, "UPDATE violations SET plate_text=?, plate_confidence=? WHERE id=?",
                             (plate, 0.85, vid))
                if vtype and vtype not in ("N/A", "Unknown", ""):
                    _execute(conn, "UPDATE violations SET vehicle_class=? WHERE id=?", (vtype, vid))
                _execute(conn, "UPDATE violations SET notes=? WHERE id=?", (notes_str, vid))
                conn.commit()
            finally:
                conn.close()

            filled += 1
            print(f"[AGENT] Fill id={vid} | {plate} | {make_model} | {color}")

        except Exception as e:
            # Mark as processed to avoid infinite retry
            try:
                conn = get_db_connection(timeout_s=3)
                _execute(conn, "UPDATE violations SET notes=? WHERE id=?", ("ai_error", vid))
                conn.commit()
                conn.close()
            except Exception:
                pass

        time.sleep(2)  # Rate limit

    return filled


def _cycle_dedup():
    """Remove duplicate violations (same plate + camera within 5 min window)."""
    from app.database import get_db_connection, _execute

    conn = get_db_connection(timeout_s=5)
    try:
        c = _execute(conn,
            "SELECT id, camera_id, plate_text, vehicle_class, violation_type, timestamp FROM violations ORDER BY timestamp ASC",
            ())
        rows = c.fetchall()
    finally:
        conn.close()

    seen = {}
    to_delete = []
    window = 300  # 5 minutes

    for row in rows:
        r = dict(row) if hasattr(row, 'keys') else row
        vid = r.get('id')
        cam_id = r.get('camera_id', '')
        plate = (r.get('plate_text') or '').strip()
        vtype = r.get('violation_type', '')
        ts = float(r.get('timestamp') or 0)
        vehicle_class = r.get('vehicle_class') or ''

        # NEVER dedup busway violations by class — many different cars pass quickly
        # Only dedup busway by exact same PLATE (same car re-entering within window)
        is_busway = 'busway' in vtype.lower()

        # Plate-based dedup (applies to all violation types)
        if plate and plate != 'N/A':
            key = (cam_id, plate, vtype)
            if key in seen and ts - seen[key] < window:
                to_delete.append(vid)
                continue
            seen[key] = ts

        # Class-based dedup (within 10s) — SKIP for busway violations
        if not plate and vehicle_class and not is_busway:
            key2 = (cam_id, vehicle_class, vtype)
            if key2 in seen and ts - seen[key2] < 10:
                to_delete.append(vid)
                continue
            seen[key2] = ts

    if to_delete:
        conn = get_db_connection(timeout_s=10)
        try:
            for vid in to_delete:
                _execute(conn, "DELETE FROM violations WHERE id=?", (vid,))
            conn.commit()
        finally:
            conn.close()
        print(f"[AGENT] Dedup: removed {len(to_delete)} duplicates")

    return len(to_delete)


def _agent_loop():
    """Main agent loop."""
    global _agent_running, _agent_paused
    _agent_running = True

    time.sleep(20)  # Wait for app to fully start
    print("[AGENT] Vehicle Identity Agent started")

    cycle = 0
    while _agent_running:
        try:
            # If paused, skip work and sleep (no API calls = no token usage)
            if _agent_paused:
                time.sleep(5)
                continue

            # Every cycle: fill details (5 violations per cycle)
            filled = _cycle_fill_details()

            # Every 10 cycles: run dedup
            cycle += 1
            if cycle % 10 == 0:
                _cycle_dedup()

        except Exception as e:
            print(f"[AGENT] Error: {e}")

        time.sleep(30)


def start_agent():
    """Start the vehicle identity agent as a daemon thread."""
    if not cfg.AI_API_KEY or not cfg.AI_BASE_URL:
        print("[AGENT] Skipped: AI API not configured")
        return

    t = threading.Thread(target=_agent_loop, daemon=True, name="VehicleAgent")
    t.start()
    return t


def stop_agent():
    """Signal the agent to stop."""
    global _agent_running
    _agent_running = False


def pause_agent():
    """Pause the agent — skips all work cycles (no AI API calls = no token usage)."""
    global _agent_paused
    _agent_paused = True
    print("[AGENT] Paused — AI API calls suspended")


def resume_agent():
    """Resume the agent from paused state."""
    global _agent_paused
    _agent_paused = False
    print("[AGENT] Resumed — AI API calls active")


def is_agent_paused():
    """Return True if agent is currently paused."""
    return _agent_paused
