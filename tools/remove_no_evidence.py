"""Remove violations whose evidence file doesn't exist on disk."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.config as cfg
cfg.load_persisted_settings()
from app.database import get_db_connection, _execute
from app.config import EVIDENCE_DIR

conn = get_db_connection()
c = _execute(conn, "SELECT id, evidence_path FROM violations WHERE evidence_path IS NOT NULL ORDER BY id", ())
rows = c.fetchall()
conn.close()

to_delete = []
for row in rows:
    r = dict(row) if hasattr(row, 'keys') else row
    vid = r.get('id')
    ep = r.get('evidence_path', '')
    if not ep:
        to_delete.append(vid)
        continue
    path1 = os.path.join(EVIDENCE_DIR, ep.replace("/", os.sep))
    path2 = os.path.join(os.path.dirname(EVIDENCE_DIR), ep.replace("/", os.sep))
    if not os.path.isfile(path1) and not os.path.isfile(path2):
        to_delete.append(vid)

print(f"Total violations: {len(rows)}")
print(f"Missing evidence files: {len(to_delete)}")

if to_delete:
    conn = get_db_connection()
    for vid in to_delete:
        _execute(conn, "DELETE FROM violations WHERE id=?", (vid,))
    conn.commit()
    conn.close()
    print(f"Deleted {len(to_delete)} violations without evidence")
else:
    print("All violations have evidence files!")
