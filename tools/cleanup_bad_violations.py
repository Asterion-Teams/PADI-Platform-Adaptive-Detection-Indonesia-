"""
Cleanup bad violations: remove entries where the evidence image shows 
a cropped/partial vehicle (no plate visible, vehicle cut off).

Uses AI vision to check if the evidence image is usable.
Deletes both the DB record and the evidence file.

Usage:
  python tools/cleanup_bad_violations.py [--dry-run] [--limit 50]
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import json
import urllib.request
import base64

os.environ['ANPR_ENABLED'] = '1'
import app.config as cfg
cfg.load_persisted_settings()

from app.database import get_db_connection, _execute
from app.config import EVIDENCE_DIR


def check_evidence_quality(img_path):
    """Use AI to check if evidence image shows a complete vehicle with visible plate.
    
    Returns: (is_good, reason)
    """
    img = cv2.imread(img_path)
    if img is None:
        return False, "cannot_read_image"
    
    h, w = img.shape[:2]
    if h < 100 or w < 100:
        return False, "image_too_small"
    
    # Check if image is mostly black
    if float(img.mean()) < 15:
        return False, "image_is_black"
    
    # Use AI to assess
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    img_b64 = base64.b64encode(buf.tobytes()).decode()
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.AI_API_KEY}",
    }
    
    prompt = """Lihat gambar evidence CCTV ini. Jawab dalam format JSON:
{"has_vehicle": true/false, "complete": true/false, "plate_visible": true/false, "reason": "..."}

Rules:
- "has_vehicle": true jika ADA kendaraan (mobil/motor/bus/truk) di dalam gambar/kotak
- "complete": true jika kendaraan terlihat UTUH (tidak terpotong di tepi frame)
- "plate_visible": true jika plat nomor TERLIHAT di gambar (meski blur)
- "reason": alasan singkat

Jawab JSON saja, tanpa markdown."""

    body = {
        "model": cfg.AI_VEHICLE_MODEL or cfg.AI_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]}],
        "max_tokens": 100,
        "temperature": 0,
    }
    
    try:
        url = f"{cfg.AI_BASE_URL.rstrip('/')}/chat/completions"
        req = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        
        data = json.loads(content)
        has_vehicle = data.get("has_vehicle", True)
        is_complete = data.get("complete", False)
        plate_visible = data.get("plate_visible", False)
        reason = data.get("reason", "")
        
        if not has_vehicle:
            return False, f"no_vehicle: {reason}"
            
        # Good if vehicle is complete OR plate is visible
        if is_complete or plate_visible:
            return True, reason
        else:
            return False, f"bad_quality: {reason}"
            
    except Exception as e:
        # If AI fails, keep the violation (don't delete on uncertainty)
        return True, f"ai_error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Cleanup bad violation evidence")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually delete, just report")
    parser.add_argument("--limit", type=int, default=50, help="Max violations to check")
    args = parser.parse_args()
    
    print(f"[CLEANUP] Scanning violations (limit={args.limit}, dry_run={args.dry_run})")
    print(f"[CLEANUP] Evidence dir: {EVIDENCE_DIR}")
    print()
    
    # Get all violations with evidence
    conn = get_db_connection()
    try:
        c = _execute(conn, 
            "SELECT id, evidence_path, plate_text, camera_name, violation_type FROM violations WHERE evidence_path IS NOT NULL ORDER BY id DESC LIMIT ?",
            (args.limit,))
        rows = c.fetchall()
    finally:
        conn.close()
    
    print(f"Found {len(rows)} violations to check")
    print("-" * 70)
    
    to_delete = []
    checked = 0
    
    for row in rows:
        r = dict(row) if hasattr(row, 'keys') else row
        vid = r.get('id')
        evidence_path = r.get('evidence_path')
        plate = r.get('plate_text')
        cam_name = r.get('camera_name', '')
        vtype = r.get('violation_type', '')
        
        if not evidence_path:
            continue
        
        img_path = os.path.join(EVIDENCE_DIR, evidence_path.replace("/", os.sep))
        
        # Also try from parent dir (DB sometimes stores relative to data/ dir)
        if not os.path.isfile(img_path):
            parent = os.path.dirname(EVIDENCE_DIR)
            img_path = os.path.join(parent, evidence_path.replace("/", os.sep))
        
        if not os.path.isfile(img_path):
            print(f"  [{vid}] MISSING FILE: {evidence_path}")
            to_delete.append((vid, img_path, "missing_file"))
            continue
        
        # Check quality
        is_good, reason = check_evidence_quality(img_path)
        checked += 1
        
        status = "[KEEP]" if is_good else "[DELETE]"
        print(f"  [{vid}] {status} | {cam_name} | {vtype} | plate={plate or '—'} | {reason}")
        
        if not is_good:
            to_delete.append((vid, img_path, reason))
        
        # Rate limit AI calls
        time.sleep(1)
    
    print()
    print(f"[RESULT] Checked: {checked} | To delete: {len(to_delete)} | To keep: {checked - len(to_delete)}")
    
    if not to_delete:
        print("Nothing to clean up!")
        return
    
    if args.dry_run:
        print("\n[DRY RUN] Would delete these violations:")
        for vid, path, reason in to_delete:
            print(f"  - id={vid} | {reason}")
        return
    
    # Actually delete
    print(f"\nDeleting {len(to_delete)} bad violations...")
    conn = get_db_connection()
    try:
        for vid, img_path, reason in to_delete:
            try:
                _execute(conn, "DELETE FROM violations WHERE id=?", (vid,))
                # Delete evidence file
                if os.path.isfile(img_path):
                    os.remove(img_path)
                print(f"  Deleted id={vid} ({reason})")
            except Exception as e:
                print(f"  Error deleting id={vid}: {e}")
        conn.commit()
    finally:
        conn.close()
    
    print(f"\nDone! Removed {len(to_delete)} bad violations.")


if __name__ == "__main__":
    main()
