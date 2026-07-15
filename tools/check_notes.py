import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.config as cfg
cfg.load_persisted_settings()
from app.database import get_db_connection, _execute

conn = get_db_connection()
c = _execute(conn, "SELECT notes, COUNT(*) as cnt FROM violations GROUP BY notes ORDER BY cnt DESC LIMIT 20", ())
rows = c.fetchall()
conn.close()

print("Notes distribution:")
for r in rows:
    d = dict(r) if hasattr(r, 'keys') else r
    notes = d.get('notes', '') or ''
    cnt = d.get('cnt', 0)
    display = notes[:80] if notes else '(NULL)'
    print(f"  [{cnt:4d}] {display}")
