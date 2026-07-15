import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.config as cfg
cfg.load_persisted_settings()
from app.database import get_db_connection, _execute

conn = get_db_connection()
c = _execute(conn, "SELECT violation_type, COUNT(*) as cnt FROM violations GROUP BY violation_type", ())
rows = c.fetchall()
print("Violations by type:")
for r in rows:
    d = dict(r) if hasattr(r, 'keys') else r
    print(f"  {d}")

print()
c2 = _execute(conn, "SELECT COUNT(*) as cnt FROM violations WHERE violation_type = ?", ('busway_occupancy',))
row = dict(c2.fetchone())
print(f"Busway violations: {row['cnt']}")
conn.close()
