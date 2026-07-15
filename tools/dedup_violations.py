"""
Deduplicate violations: remove duplicate entries where the same vehicle
(same plate number) is recorded multiple times at the same camera.

Keeps only the FIRST (oldest) violation per plate+camera+zone combination 
within a configurable time window (default: 5 minutes).

Also removes violations without evidence files.

Usage:
  python tools/dedup_violations.py [--dry-run] [--window 300]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.config as cfg
cfg.load_persisted_settings()

from app.database import get_db_connection, _execute
from app.config import EVIDENCE_DIR


def main():
    parser = argparse.ArgumentParser(description="Deduplicate violations")
    parser.add_argument("--dry-run", action="store_true", help="Don't delete, just report")
    parser.add_argument("--window", type=int, default=300, help="Time window in seconds to consider duplicates (default: 300 = 5 min)")
    args = parser.parse_args()

    print(f"[DEDUP] Window: {args.window}s | Dry run: {args.dry_run}")
    print()

    conn = get_db_connection()
    try:
        c = _execute(conn,
            "SELECT id, camera_id, plate_text, vehicle_class, violation_type, timestamp, evidence_path, notes FROM violations ORDER BY timestamp ASC",
            ())
        rows = c.fetchall()
    finally:
        conn.close()

    print(f"Total violations: {len(rows)}")

    to_delete = []
    seen = {}  # key: (camera_id, plate_text) -> last_timestamp

    for row in rows:
        r = dict(row) if hasattr(row, 'keys') else row
        vid = r.get('id')
        cam_id = r.get('camera_id', '')
        plate = (r.get('plate_text') or '').strip()
        vtype = r.get('violation_type', '')
        ts = float(r.get('timestamp') or 0)
        evidence_path = r.get('evidence_path', '')
        notes = r.get('notes') or ''
        vehicle_class = r.get('vehicle_class') or ''

        # Rule 1: Remove violations without evidence file
        if evidence_path:
            img_path = os.path.join(EVIDENCE_DIR, evidence_path.replace("/", os.sep))
            if not os.path.isfile(img_path):
                img_path = os.path.join(os.path.dirname(EVIDENCE_DIR), evidence_path.replace("/", os.sep))
            if not os.path.isfile(img_path):
                to_delete.append((vid, "missing_evidence_file"))
                continue

        # Rule 2: Plate-based dedup (same plate + same camera within window)
        if plate and plate != 'N/A':
            dedup_key = (cam_id, plate, vtype)
            if dedup_key in seen:
                last_ts = seen[dedup_key]
                if ts - last_ts < args.window:
                    to_delete.append((vid, f"dup_plate:{plate} (delta={ts-last_ts:.0f}s)"))
                    continue
            seen[dedup_key] = ts

        # Rule 3: Same vehicle class + same camera + very close timestamp (within 10s) = likely same vehicle
        # SKIP for busway — many different cars pass quickly
        is_busway = 'busway' in vtype.lower()
        if not plate and vehicle_class and not is_busway:
            nearby_key = (cam_id, vehicle_class, vtype)
            if nearby_key in seen:
                last_ts = seen[nearby_key]
                if ts - last_ts < 10:  # Within 10 seconds, same class = likely duplicate
                    to_delete.append((vid, f"dup_class:{vehicle_class} (delta={ts-last_ts:.0f}s)"))
                    continue
            seen[nearby_key] = ts

    print(f"Duplicates found: {len(to_delete)}")
    print()

    if not to_delete:
        print("No duplicates!")
        return

    # Breakdown
    reasons = {}
    for vid, reason in to_delete:
        key = reason.split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    print("Breakdown:")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print()

    if args.dry_run:
        print("[DRY RUN] Would delete:")
        for vid, reason in to_delete[:20]:
            print(f"  id={vid} | {reason}")
        if len(to_delete) > 20:
            print(f"  ... and {len(to_delete) - 20} more")
        return

    # Delete
    print(f"Deleting {len(to_delete)} duplicates...")
    conn = get_db_connection()
    try:
        deleted = 0
        for vid, reason in to_delete:
            try:
                _execute(conn, "DELETE FROM violations WHERE id=?", (vid,))
                deleted += 1
            except Exception as e:
                print(f"  Error deleting id={vid}: {e}")
        conn.commit()
    finally:
        conn.close()

    # Also delete orphaned evidence files
    evidence_deleted = 0
    for vid, reason in to_delete:
        if "missing_evidence" not in reason:
            # Find and delete evidence file if exists
            # (We don't have the path stored after deletion, skip this)
            pass

    print(f"\n[DONE] Deleted: {deleted} violations")


if __name__ == "__main__":
    main()
