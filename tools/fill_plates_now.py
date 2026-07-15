"""
Force-fill plates for all violations that are missing plate_text.
Reads evidence image and uses AI vision to read the plate directly.
Updates the database immediately.

Usage: python tools/fill_plates_now.py [--limit 100]
"""
import sys, os, time, argparse, json, base64
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import urllib.request

os.environ['ANPR_ENABLED'] = '1'
import app.config as cfg
cfg.load_persisted_settings()

from app.database import get_db_connection, _execute
from app.config import EVIDENCE_DIR


def read_plate_from_image(img):
    """Send image to AI and get plate + vehicle details."""
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    img_b64 = base64.b64encode(buf.tobytes()).decode()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.AI_API_KEY}",
    }

    prompt = """Dari gambar CCTV ini, identifikasi kendaraan di dalam kotak merah:
1. Plat nomor (format Indonesia: [HURUF 1-2] [ANGKA 1-4] [HURUF 0-3]). Baca dari bumper depan/belakang.
2. Jenis kendaraan: Sedan/MPV/SUV/Hatchback/Motorcycle/Pickup/Truck/Bus/Van/Minibus
3. Merek dan Model spesifik (contoh: Toyota Avanza, Honda Jazz, Suzuki Carry, Daihatsu Xenia, Mitsubishi Xpander)
4. Warna dominan kendaraan (Putih/Hitam/Silver/Abu-abu/Merah/Biru/Kuning/Hijau/Coklat/Emas)

PENTING: 
- JANGAN jawab "N/A" untuk jenis, merek, atau warna. Selalu berikan estimasi terbaik.
- Jika tidak yakin merek, estimasi dari bentuk body (contoh: mobil box = "Daihatsu Gran Max", sedan kecil = "Toyota Vios")
- Jika plat tidak terlihat sama sekali, plate = "N/A"

Jawab JSON SAJA:
{"plate":"B 1234 ABC","vehicle_type":"MPV","make_model":"Toyota Avanza","color":"Putih"}"""

    body = {
        "model": cfg.AI_VEHICLE_MODEL or cfg.AI_MODEL,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    print(f"[FILL] Reading violations without complete details (limit={args.limit})...")

    conn = get_db_connection()
    try:
        c = _execute(conn,
            "SELECT id, evidence_path, camera_name, plate_text, notes FROM violations WHERE evidence_path IS NOT NULL ORDER BY id DESC LIMIT ?",
            (args.limit * 3,))
        all_rows = c.fetchall()
    finally:
        conn.close()

    # Filter: keep only those missing Merek/Model or Warna in notes
    rows = []
    for row in all_rows:
        r = dict(row) if hasattr(row, 'keys') else row
        notes = r.get('notes') or ''
        if 'Merek/Model' not in notes or 'Warna' not in notes:
            rows.append(r)
        if len(rows) >= args.limit:
            break

    print(f"Found {len(rows)} violations to process\n")

    updated = 0
    failed = 0

    for row in rows:
        r = row  # Already a dict from filtering above
        vid = r.get('id')
        evidence_path = r.get('evidence_path')
        cam_name = r.get('camera_name', '')
        existing_plate = r.get('plate_text', '')

        if not evidence_path:
            continue

        # Try both paths
        img_path = os.path.join(EVIDENCE_DIR, evidence_path.replace("/", os.sep))
        if not os.path.isfile(img_path):
            img_path = os.path.join(os.path.dirname(EVIDENCE_DIR), evidence_path.replace("/", os.sep))
        if not os.path.isfile(img_path):
            print(f"  [{vid}] SKIP - file not found")
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        try:
            info = read_plate_from_image(img)
            plate = info.get("plate", "N/A")
            vtype = info.get("vehicle_type", "")
            make_model = info.get("make_model", "")
            color = info.get("color", "")

            # Build notes
            notes_parts = []
            if vtype and vtype not in ("N/A", "Unknown", ""):
                notes_parts.append(f"Jenis: {vtype}")
            if make_model and make_model not in ("N/A", "Unknown", ""):
                notes_parts.append(f"Merek/Model: {make_model}")
            if color and color not in ("N/A", "Unknown", ""):
                notes_parts.append(f"Warna: {color}")

            notes_str = " | ".join(notes_parts) if notes_parts else None

            # Update database
            conn = get_db_connection(timeout_s=5)
            try:
                if plate and plate != "N/A" and "unknown" not in plate.lower() and not existing_plate:
                    _execute(conn, "UPDATE violations SET plate_text=?, plate_confidence=? WHERE id=?",
                             (plate, 0.85, vid))
                if vtype and vtype not in ("N/A", "Unknown", ""):
                    _execute(conn, "UPDATE violations SET vehicle_class=? WHERE id=?", (vtype, vid))
                if notes_str:
                    _execute(conn, "UPDATE violations SET notes=? WHERE id=?", (notes_str, vid))
                conn.commit()
            finally:
                conn.close()

            plate_display = plate if plate != "N/A" else "—"
            print(f"  [{vid}] ✓ plate={plate_display} | {vtype} | {make_model} | {color}")
            updated += 1

        except Exception as e:
            print(f"  [{vid}] ✗ ERROR: {e}")
            failed += 1

        # Rate limit
        time.sleep(1.5)

    print(f"\n[DONE] Updated: {updated} | Failed: {failed} | Total: {len(rows)}")


if __name__ == "__main__":
    main()
