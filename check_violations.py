import sys
sys.path.insert(0, '.')
from app.database import DB_PATH
import sqlite3
import os

if os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print("Tables:", [t[0] for t in tables])

    if ('violations',) in tables:
        c = conn.execute('SELECT COUNT(*) FROM violations')
        total = c.fetchone()[0]
        c2 = conn.execute('SELECT COUNT(*) FROM violations WHERE plate_text IS NULL OR plate_text = ""')
        no_plate = c2.fetchone()[0]
        c3 = conn.execute('SELECT id, camera_name, violation_type, plate_text, timestamp FROM violations ORDER BY timestamp DESC LIMIT 20')
        recent = c3.fetchall()
        print(f'Total violations: {total}')
        print(f'Without plate: {no_plate}')
        print('Recent violations:')
        for r in recent:
            print(f'  id={r[0]} | cam={r[1]} | type={r[2]} | plate={repr(r[3])} | ts={r[4]}')
    else:
        print('No violations table')
    conn.close()
else:
    print('DB not found at', DB_PATH)
