"""Test the dashboard API endpoints to diagnose empty data."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.config as cfg
cfg.load_persisted_settings()
from app.database import get_db_connection, _execute, violation_summary, list_violations

now = time.time()
start_24h = now - 86400

print("=== violation_summary (24h) ===")
try:
    s = violation_summary(start_ts=start_24h, end_ts=now)
    print(f"  total: {s.get('total')}")
    print(f"  by_type: {s.get('by_type')}")
    print(f"  by_camera (first 2): {s.get('by_camera', [])[:2]}")
    print(f"  pending: {s.get('pending')}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n=== violation_summary (all time) ===")
try:
    s2 = violation_summary()
    print(f"  total: {s2.get('total')}")
    print(f"  by_type: {s2.get('by_type')}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n=== list_violations (recent 5) ===")
try:
    rows = list_violations(limit=5)
    print(f"  count: {len(rows)}")
    for r in rows[:3]:
        print(f"    id={r.get('id')} | ts={r.get('timestamp')} | type={r.get('violation_type')} | plate={r.get('plate_text')}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n=== Check timestamps ===")
conn = get_db_connection()
c = _execute(conn, "SELECT MIN(timestamp) as mn, MAX(timestamp) as mx, COUNT(*) as cnt FROM violations", ())
row = dict(c.fetchone())
conn.close()
print(f"  count: {row['cnt']}")
print(f"  min_ts: {row['mn']} ({time.ctime(row['mn']) if row['mn'] else 'N/A'})")
print(f"  max_ts: {row['mx']} ({time.ctime(row['mx']) if row['mx'] else 'N/A'})")
print(f"  now: {now} ({time.ctime(now)})")
print(f"  24h ago: {start_24h} ({time.ctime(start_24h)})")

# Check if violations are within 24h window
if row['mx']:
    age_hours = (now - row['mx']) / 3600
    print(f"  newest violation age: {age_hours:.1f} hours ago")
