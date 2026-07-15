import os
os.environ.setdefault('FLASK_ENV', 'production')
from dotenv import load_dotenv
load_dotenv('.env')
import sys
sys.path.insert(0, '.')

from app.database import get_db_connection, _fetchall

conn = get_db_connection()

# Count total
rows = _fetchall(conn, "SELECT COUNT(*) as cnt FROM violations")
total = rows[0]["cnt"]
print(f"Total violations: {total}")

# Count no plate
rows2 = _fetchall(conn, "SELECT COUNT(*) as cnt FROM violations WHERE plate_text IS NULL OR plate_text = ''")
no_plate = rows2[0]["cnt"]
print(f"Without plate: {no_plate}")

# Show sample
samples = _fetchall(conn,
    "SELECT id, camera_name, violation_type, plate_text, timestamp "
    "FROM violations WHERE plate_text IS NULL OR plate_text = '' "
    "ORDER BY timestamp DESC LIMIT 10")
if samples:
    print("\nSample violations to delete:")
    for r in samples:
        print(f"  id={r['id']} cam={r['camera_name']} type={r['violation_type']} plate={repr(r['plate_text'])} ts={r['timestamp']}")

    # Delete using cursor
    cur = conn.cursor()
    cur.execute("DELETE FROM violations WHERE plate_text IS NULL OR plate_text = ''")
    conn.commit()
    cur.close()
    rows3 = _fetchall(conn, "SELECT COUNT(*) as cnt FROM violations")
    print(f"\nDeleted {no_plate} violations. Remaining: {rows3[0]['cnt']}")
else:
    print("No violations without plate found.")

conn.close()
