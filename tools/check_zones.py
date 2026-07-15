import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.config as cfg
cfg.load_persisted_settings()
from app.database import get_db_connection, _execute

conn = get_db_connection()
c = _execute(conn, "SELECT id, camera_id, name, zone_type, active FROM violation_zones ORDER BY id", ())
rows = c.fetchall()
conn.close()

print(f"Total zones: {len(rows)}")
for r in rows:
    d = dict(r) if hasattr(r, 'keys') else r
    zid = d.get("id")
    cam = d.get("camera_id")
    name = d.get("name")
    ztype = d.get("zone_type")
    active = d.get("active")
    print(f"  id={zid} | cam={cam} | name={name} | type={ztype} | active={active}")
