import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.config as cfg
cfg.load_persisted_settings()
from app.database import get_db_connection, _execute

conn = get_db_connection()
c = _execute(conn, "SELECT COUNT(*) as cnt FROM violations", ())
print(f"Total violations: {dict(c.fetchone())['cnt']}")

c2 = _execute(conn, "SELECT COUNT(*) as cnt FROM violations WHERE notes LIKE ?", ('%Merek/Model%',))
print(f"With Merek/Model: {dict(c2.fetchone())['cnt']}")

c3 = _execute(conn, "SELECT COUNT(*) as cnt FROM violations WHERE notes LIKE ?", ('%Warna%',))
print(f"With Warna: {dict(c3.fetchone())['cnt']}")

c4 = _execute(conn, "SELECT COUNT(*) as cnt FROM violations WHERE plate_text IS NOT NULL AND plate_text != ?", ('',))
print(f"With plate: {dict(c4.fetchone())['cnt']}")

c5 = _execute(conn, "SELECT COUNT(*) as cnt FROM violations WHERE notes IS NULL OR notes = ? OR notes = ?", ('', 'processed'))
print(f"Missing notes: {dict(c5.fetchone())['cnt']}")

c6 = _execute(conn, "SELECT COUNT(*) as cnt FROM violations WHERE evidence_path IS NOT NULL", ())
print(f"With evidence: {dict(c6.fetchone())['cnt']}")

conn.close()
