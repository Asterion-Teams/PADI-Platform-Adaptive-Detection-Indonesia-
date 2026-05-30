import sqlite3
import os
import time
import datetime
import json
import threading
from app.config import DATA_DIR

DB_PATH = os.path.join(DATA_DIR, "traffic_data.db")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

_transformer_models = {}
_transformer_training = set()
_transformer_training_lock = threading.Lock()

def _apply_sqlite_pragmas(conn, busy_timeout_ms=30000):
    try:
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    try:
        conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms or 0)};")
    except Exception:
        pass
    try:
        conn.execute("PRAGMA temp_store=MEMORY;")
    except Exception:
        pass

def get_db_connection(timeout_s=30, busy_timeout_ms=None):
    timeout_s = float(timeout_s or 0)
    if timeout_s <= 0:
        timeout_s = 1.0
    if busy_timeout_ms is None:
        busy_timeout_ms = int(timeout_s * 1000)
    else:
        busy_timeout_ms = int(busy_timeout_ms or 0)

    conn = sqlite3.connect(DB_PATH, timeout=timeout_s)
    conn.row_factory = sqlite3.Row
    _apply_sqlite_pragmas(conn, busy_timeout_ms=busy_timeout_ms)
    return conn

def _local_tzinfo():
    return datetime.datetime.now().astimezone().tzinfo

def _hour_bucket_ts(ts, tzinfo):
    try:
        dt = datetime.datetime.fromtimestamp(float(ts), tz=tzinfo)
    except Exception:
        return None
    dt0 = dt.replace(minute=0, second=0, microsecond=0)
    return int(dt0.timestamp())

def _build_hourly_series(camera_id, days=60):
    tzinfo = _local_tzinfo()
    cutoff = time.time() - (float(days or 60) * 24 * 3600)
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(
            """
            SELECT timestamp, new_count
            FROM traffic_history
            WHERE camera_id = ?
              AND timestamp >= ?
            ORDER BY timestamp ASC
            """,
            (camera_id, cutoff),
        )
        rows = c.fetchall()
    finally:
        conn.close()

    buckets = {}
    last_bucket = None
    for r in rows:
        ts = r["timestamp"]
        b = _hour_bucket_ts(ts, tzinfo)
        if b is None:
            continue
        last_bucket = b if last_bucket is None else max(last_bucket, b)
        try:
            v = int(r["new_count"] or 0)
        except Exception:
            v = 0
        buckets[b] = int(buckets.get(b, 0) + v)

    if not buckets:
        return {"series": [], "tzinfo": tzinfo, "max_bucket_ts": None}

    first = min(buckets.keys())
    last = max(buckets.keys())
    out = []
    cur = first
    while cur <= last:
        out.append((cur, int(buckets.get(cur, 0))))
        cur += 3600
    return {"series": out, "tzinfo": tzinfo, "max_bucket_ts": last}

def _time_features_from_bucket_ts(bucket_ts, tzinfo):
    dt = datetime.datetime.fromtimestamp(int(bucket_ts), tz=tzinfo)
    dow = int(dt.strftime("%w"))
    hour = int(dt.strftime("%H"))
    return dow, hour

if torch is not None and nn is not None:
    class _TinyTransformerForecaster(nn.Module):
        def __init__(self, d_model=32, nhead=4, num_layers=2, dropout=0.1, max_len=256):
            super().__init__()
            self.d_model = int(d_model)
            self.value_proj = nn.Linear(1, self.d_model)
            self.hour_emb = nn.Embedding(24, self.d_model)
            self.dow_emb = nn.Embedding(7, self.d_model)
            self.pos_emb = nn.Embedding(int(max_len), self.d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=int(nhead), dropout=float(dropout), batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
            self.head = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, 1))

        def forward(self, x_val, x_hour, x_dow):
            b, t, _ = x_val.shape
            pos = torch.arange(t, device=x_val.device).unsqueeze(0).expand(b, t)
            h = self.value_proj(x_val) + self.hour_emb(x_hour) + self.dow_emb(x_dow) + self.pos_emb(pos)
            z = self.encoder(h)
            y = self.head(z[:, -1, :])
            return y
else:
    _TinyTransformerForecaster = None

def _get_or_train_transformer(camera_id, context_len=48, max_days=60):
    if torch is None or nn is None or optim is None or _TinyTransformerForecaster is None:
        return None
    cam_id = str(camera_id or "").strip()
    if not cam_id:
        return None

    info = _build_hourly_series(cam_id, days=max_days)
    series = info.get("series") or []
    tzinfo = info.get("tzinfo")
    max_bucket_ts = info.get("max_bucket_ts")
    if len(series) < max(context_len + 24, 96):
        return None

    cache = _transformer_models.get(cam_id)
    if cache and cache.get("max_bucket_ts") == max_bucket_ts:
        return cache

    values = [float(v) for _, v in series]
    mean = sum(values) / float(len(values) or 1)
    var = sum((v - mean) ** 2 for v in values) / float(max(1, len(values) - 1))
    std = (var ** 0.5) if var > 1e-8 else 1.0

    xs_val = []
    xs_hour = []
    xs_dow = []
    ys = []

    for i in range(context_len, len(series)):
        window = series[i - context_len : i]
        target = series[i][1]
        xw = []
        xh = []
        xd = []
        for bucket_ts, v in window:
            dow, hour = _time_features_from_bucket_ts(bucket_ts, tzinfo)
            xw.append([(float(v) - mean) / std])
            xh.append(hour)
            xd.append(dow)
        xs_val.append(xw)
        xs_hour.append(xh)
        xs_dow.append(xd)
        ys.append([(float(target) - mean) / std])

    device = torch.device("cpu")
    x_val = torch.tensor(xs_val, dtype=torch.float32, device=device)
    x_hour = torch.tensor(xs_hour, dtype=torch.long, device=device)
    x_dow = torch.tensor(xs_dow, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.float32, device=device)

    model = _TinyTransformerForecaster(d_model=32, nhead=4, num_layers=2, dropout=0.1, max_len=max(256, int(context_len) + 8)).to(device)
    model.train()

    opt = optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.MSELoss()

    n = x_val.shape[0]
    batch = 64 if n >= 256 else 32
    epochs = 6
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    for _ in range(epochs):
        idx = torch.randperm(n, generator=gen)
        for s in range(0, n, batch):
            j = idx[s : s + batch]
            pred = model(x_val[j], x_hour[j], x_dow[j])
            loss = loss_fn(pred, y[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    cache = {
        "model": model,
        "context_len": int(context_len),
        "mean": float(mean),
        "std": float(std),
        "tzinfo": tzinfo,
        "series": series,
        "max_bucket_ts": max_bucket_ts,
        "trained_at": time.time(),
    }
    _transformer_models[cam_id] = cache
    return cache

def _get_transformer_cache(camera_id):
    cam_id = str(camera_id or "").strip()
    if not cam_id:
        return None
    return _transformer_models.get(cam_id)

def _predict_with_transformer(camera_id, target_dt_local, context_len=48):
    if torch is None:
        return None
    if not isinstance(target_dt_local, datetime.datetime):
        return None

    cache = _get_transformer_cache(camera_id)
    if not cache:
        return None

    model = cache.get("model")
    tzinfo = cache.get("tzinfo")
    mean = float(cache.get("mean") or 0.0)
    std = float(cache.get("std") or 1.0)
    series = cache.get("series") or []
    if not model or not series:
        return None

    target_dt_local = target_dt_local.astimezone(tzinfo)
    target_bucket = int(target_dt_local.replace(minute=0, second=0, microsecond=0).timestamp())

    last_bucket = int(series[-1][0])
    if target_bucket <= last_bucket:
        i = None
        for k, (b, _) in enumerate(series):
            if int(b) == int(target_bucket):
                i = k
                break
        if i is None or i < context_len:
            return None
        window = series[i - context_len : i]
        device = torch.device("cpu")
        xw, xh, xd = [], [], []
        for bucket_ts, v in window:
            dow, hour = _time_features_from_bucket_ts(bucket_ts, tzinfo)
            xw.append([(float(v) - mean) / std])
            xh.append(hour)
            xd.append(dow)
        x_val = torch.tensor([xw], dtype=torch.float32, device=device)
        x_hour = torch.tensor([xh], dtype=torch.long, device=device)
        x_dow = torch.tensor([xd], dtype=torch.long, device=device)
        with torch.no_grad():
            y = model(x_val, x_hour, x_dow).cpu().numpy().reshape(-1)[0]
        pred = float(y) * std + mean
        return max(0, int(round(pred)))

    steps = int((target_bucket - last_bucket) // 3600)
    if steps <= 0:
        return None
    if steps > 48:
        steps = 48

    window = list(series[-context_len:])
    device = torch.device("cpu")
    cur_last = last_bucket
    pred_val = None
    for _ in range(steps):
        xw, xh, xd = [], [], []
        for bucket_ts, v in window:
            dow, hour = _time_features_from_bucket_ts(bucket_ts, tzinfo)
            xw.append([(float(v) - mean) / std])
            xh.append(hour)
            xd.append(dow)
        x_val = torch.tensor([xw], dtype=torch.float32, device=device)
        x_hour = torch.tensor([xh], dtype=torch.long, device=device)
        x_dow = torch.tensor([xd], dtype=torch.long, device=device)
        with torch.no_grad():
            y = model(x_val, x_hour, x_dow).cpu().numpy().reshape(-1)[0]
        pred = float(y) * std + mean
        pred_val = max(0, int(round(pred)))
        cur_last += 3600
        window.append((cur_last, pred_val))
        if len(window) > context_len:
            window = window[-context_len:]
    return pred_val

def init_db():
    conn = get_db_connection(timeout_s=10)
    c = conn.cursor()
    try:
        c.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    # Create table for traffic history
    # Using specific types for efficiency
    c.execute('''
        CREATE TABLE IF NOT EXISTS traffic_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            total_count INTEGER DEFAULT 0,
            car_count INTEGER DEFAULT 0,
            motorcycle_count INTEGER DEFAULT 0,
            new_count INTEGER DEFAULT 0,
            new_cars INTEGER DEFAULT 0,
            new_motors INTEGER DEFAULT 0
        )
    ''')
    
    # Create index for fast time-range queries
    c.execute('''
        CREATE INDEX IF NOT EXISTS idx_camera_timestamp 
        ON traffic_history (camera_id, timestamp)
    ''')

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_profile (
            session_id TEXT PRIMARY KEY,
            updated_ts REAL NOT NULL,
            last_intent TEXT,
            last_camera_id TEXT,
            last_camera_name TEXT,
            last_destination TEXT,
            prefs_json TEXT
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            ts REAL NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            page TEXT,
            meta_json TEXT
        )
        """
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session_ts ON chat_messages (session_id, ts)")

    # =====================================================================
    # E-TLE / Violation Detection tables (Case 1)
    # =====================================================================

    # Zones of interest (no-parking, busway, bicycle lane, bus stop) per camera.
    # Geometry is stored as JSON: either a polygon list of (x, y) in image coords,
    # or a simple bbox [x1,y1,x2,y2]. The engine handles both.
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS violation_zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL,
            name TEXT,
            zone_type TEXT NOT NULL,
            geometry_json TEXT NOT NULL,
            active INTEGER DEFAULT 1,
            notes TEXT,
            created_ts REAL NOT NULL
        )
        """
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_zones_camera ON violation_zones (camera_id, active)")

    # Safe migration: add frame dimension columns if schema pre-dates them.
    # Stored geometry is always in the reference frame size (frame_width x frame_height);
    # the enforcement engine scales to the current frame size at inference time.
    for col, ddl in (
        ("frame_width", "ALTER TABLE violation_zones ADD COLUMN frame_width INTEGER DEFAULT 0"),
        ("frame_height", "ALTER TABLE violation_zones ADD COLUMN frame_height INTEGER DEFAULT 0"),
    ):
        try:
            c.execute(ddl)
        except Exception:
            pass  # Column already exists

    # Violation events captured by the enforcement engine
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL,
            camera_name TEXT,
            zone_id INTEGER,
            zone_type TEXT NOT NULL,
            violation_type TEXT NOT NULL,
            timestamp REAL NOT NULL,
            duration_s REAL DEFAULT 0,
            vehicle_class TEXT,
            plate_text TEXT,
            plate_confidence REAL DEFAULT 0,
            bbox_json TEXT,
            evidence_path TEXT,
            lat REAL,
            lng REAL,
            status TEXT DEFAULT 'pending',
            dispatched_unit TEXT,
            notes TEXT
        )
        """
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_violations_ts ON violations (timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_violations_cam ON violations (camera_id, timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_violations_type ON violations (violation_type, timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_violations_plate ON violations (plate_text)")

    # Citizen / CRM complaints (public reports - for Case 1 integration)
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS crm_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            reporter_name TEXT,
            reporter_contact TEXT,
            category TEXT,
            description TEXT,
            lat REAL,
            lng REAL,
            camera_id TEXT,
            status TEXT DEFAULT 'open',
            auto_classified_type TEXT,
            priority TEXT DEFAULT 'normal'
        )
        """
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_crm_ts ON crm_reports (timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_crm_status ON crm_reports (status, timestamp)")

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

def get_chat_profile(session_id):
    sid = str(session_id or "").strip()
    if not sid:
        return {}
    conn = get_db_connection(timeout_s=2)
    c = conn.cursor()
    try:
        c.execute("SELECT * FROM chat_profile WHERE session_id = ?", (sid,))
        row = c.fetchone()
        if not row:
            return {}
        out = dict(row)
        try:
            prefs = json.loads(out.get("prefs_json") or "{}")
        except Exception:
            prefs = {}
        out["prefs"] = prefs if isinstance(prefs, dict) else {}
        return out
    finally:
        conn.close()

def upsert_chat_profile(session_id, fields):
    sid = str(session_id or "").strip()
    if not sid:
        return False
    f = fields or {}
    now = time.time()
    prefs_json = None
    if "prefs" in f:
        try:
            prefs_json = json.dumps(f.get("prefs") or {}, ensure_ascii=False)
        except Exception:
            prefs_json = "{}"
    conn = get_db_connection(timeout_s=5)
    c = conn.cursor()
    try:
        c.execute(
            """
            INSERT INTO chat_profile (session_id, updated_ts, last_intent, last_camera_id, last_camera_name, last_destination, prefs_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                updated_ts=excluded.updated_ts,
                last_intent=COALESCE(excluded.last_intent, chat_profile.last_intent),
                last_camera_id=COALESCE(excluded.last_camera_id, chat_profile.last_camera_id),
                last_camera_name=COALESCE(excluded.last_camera_name, chat_profile.last_camera_name),
                last_destination=COALESCE(excluded.last_destination, chat_profile.last_destination),
                prefs_json=COALESCE(excluded.prefs_json, chat_profile.prefs_json)
            """,
            (
                sid,
                now,
                f.get("last_intent"),
                f.get("last_camera_id"),
                f.get("last_camera_name"),
                f.get("last_destination"),
                prefs_json,
            ),
        )
        conn.commit()
        return True
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        conn.close()

def add_chat_message(session_id, role, content, page=None, meta=None):
    sid = str(session_id or "").strip()
    if not sid:
        return False
    r = str(role or "").strip()
    txt = str(content or "").strip()
    if not r or not txt:
        return False
    try:
        meta_json = json.dumps(meta or {}, ensure_ascii=False) if meta is not None else None
    except Exception:
        meta_json = None
    conn = get_db_connection(timeout_s=5)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO chat_messages (session_id, ts, role, content, page, meta_json) VALUES (?, ?, ?, ?, ?, ?)",
            (sid, time.time(), r, txt, str(page or "") or None, meta_json),
        )
        conn.commit()
        return True
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        conn.close()

def get_recent_chat_messages(session_id, limit=12):
    sid = str(session_id or "").strip()
    if not sid:
        return []
    lim = int(limit or 0)
    if lim <= 0:
        lim = 12
    lim = min(50, lim)
    conn = get_db_connection(timeout_s=3)
    c = conn.cursor()
    try:
        c.execute(
            "SELECT ts, role, content FROM chat_messages WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
            (sid, lim),
        )
        rows = c.fetchall() or []
        out = []
        for r in reversed(rows):
            out.append({"ts": r["ts"], "role": r["role"], "content": r["content"]})
        return out
    finally:
        conn.close()

def insert_history_batch(records):
    """
    Batch insert records.
    records: list of tuples (camera_id, timestamp, total, cars, motors)
    """
    if not records:
        return

    insert_sql = '''
        INSERT INTO traffic_history (camera_id, timestamp, total_count, car_count, motorcycle_count, new_count, new_cars, new_motors)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    '''

    last_err = None
    for attempt in range(5):
        conn = get_db_connection(timeout_s=30)
        c = conn.cursor()
        try:
            c.executemany(insert_sql, records)
            conn.commit()
            conn.close()
            return
        except sqlite3.OperationalError as e:
            last_err = e
            try:
                conn.rollback()
            except Exception:
                pass
            conn.close()
            if "locked" in str(e).lower():
                time.sleep(0.05 * (2 ** attempt))
                continue
            break
        except Exception as e:
            last_err = e
            try:
                conn.rollback()
            except Exception:
                pass
            conn.close()
            break
    if last_err is not None:
        print(f"Error inserting batch: {last_err}")

def clear_all_history():
    conn = get_db_connection(timeout_s=10)
    c = conn.cursor()
    try:
        c.execute("DELETE FROM traffic_history")
        conn.commit()
    except Exception as e:
        print(f"Error clearing history: {e}")
    finally:
        conn.close()

def get_camera_history(camera_id, start_ts=None, end_ts=None):
    conn = get_db_connection(timeout_s=2)
    c = conn.cursor()
    
    query = "SELECT timestamp, total_count, car_count, motorcycle_count, new_count, new_cars, new_motors FROM traffic_history WHERE camera_id = ?"
    params = [camera_id]
    
    if start_ts:
        query += " AND timestamp >= ?"
        params.append(start_ts)
        
    if end_ts:
        query += " AND timestamp <= ?"
        params.append(end_ts)
        
    query += " ORDER BY timestamp ASC"
    
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    
    # Convert to list of dicts to match existing API format
    return [
        {
            "ts": row["timestamp"],
            "count": row["total_count"],
            "cars": row["car_count"],
            "motors": row["motorcycle_count"],
            "new_count": row["new_count"],
            "new_cars": row["new_cars"],
            "new_motors": row["new_motors"]
        }
        for row in rows
    ]

def predict_future_traffic(camera_id, day_of_week, hour_of_day, target_dt_local=None):
    """
    Predict traffic volume for a specific day of week and hour.
    day_of_week: 0 (Sunday) to 6 (Saturday) - SQLite format
    hour_of_day: 0-23
    Returns: Average vehicles per hour
    """
    if target_dt_local is not None and torch is not None:
        try:
            pred = _predict_with_transformer(camera_id, target_dt_local, context_len=48)
            if pred is not None:
                return float(pred)
        except Exception:
            pass
        cam_id = str(camera_id or "").strip()
        if cam_id:
            should_start = False
            with _transformer_training_lock:
                if cam_id not in _transformer_models and cam_id not in _transformer_training:
                    _transformer_training.add(cam_id)
                    should_start = True
            if should_start:
                def _train_bg():
                    try:
                        _ = _get_or_train_transformer(cam_id, context_len=48, max_days=60)
                    finally:
                        with _transformer_training_lock:
                            _transformer_training.discard(cam_id)
                t = threading.Thread(target=_train_bg, daemon=True)
                t.start()

    conn = get_db_connection(timeout_s=2)
    c = conn.cursor()
    
    # Calculate average hourly volume for this specific time slot across all historical data
    query = '''
        WITH HourlySums AS (
            SELECT 
                date(timestamp, 'unixepoch', 'localtime') as date_str,
                SUM(new_count) as hourly_total
            FROM traffic_history
            WHERE camera_id = ?
              AND cast(strftime('%w', datetime(timestamp, 'unixepoch', 'localtime')) as int) = ?
              AND cast(strftime('%H', datetime(timestamp, 'unixepoch', 'localtime')) as int) = ?
            GROUP BY date_str
        )
        SELECT AVG(hourly_total) as avg_hourly_traffic
        FROM HourlySums
    '''
    
    try:
        c.execute(query, (camera_id, day_of_week, hour_of_day))
        result = c.fetchone()
        avg_traffic = result['avg_hourly_traffic'] if result and result['avg_hourly_traffic'] is not None else 0
    except Exception as e:
        print(f"Prediction Error: {e}")
        avg_traffic = 0
    finally:
        conn.close()
    
    return avg_traffic

def get_total_lifetime():
    conn = get_db_connection(timeout_s=2)
    c = conn.cursor()
    try:
        c.execute("""
            SELECT 
                COALESCE(SUM(new_cars), 0) as cars,
                COALESCE(SUM(new_motors), 0) as motors
            FROM traffic_history
        """)
        row = c.fetchone()
        total = 0
        if row:
            total = int((row["cars"] or 0) + (row["motors"] or 0))
        return {
            "accumulated_count": total,
            "cars": int(row["cars"] or 0) if row and "cars" in row.keys() else 0,
            "motorcycles": int(row["motors"] or 0) if row and "motors" in row.keys() else 0,
        }
    except Exception:
        return {"accumulated_count": 0, "cars": 0, "motorcycles": 0}
    finally:
        conn.close()

def get_totals_by_camera(camera_ids=None, start_ts=None, end_ts=None):
    conn = get_db_connection(timeout_s=2)
    c = conn.cursor()
    try:
        params = []
        conditions = []
        if camera_ids:
            placeholders = ",".join(["?"] * len(camera_ids))
            conditions.append(f"camera_id IN ({placeholders})")
            params.extend(list(camera_ids))
        if start_ts:
            conditions.append("timestamp >= ?")
            params.append(start_ts)
        if end_ts:
            conditions.append("timestamp <= ?")
            params.append(end_ts)
        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        c.execute(
            f"""
            SELECT
                camera_id,
                COALESCE(SUM(new_cars), 0) as cars,
                COALESCE(SUM(new_motors), 0) as motors
            FROM traffic_history
            {where_clause}
            GROUP BY camera_id
            """,
            params,
        )
        rows = c.fetchall()
        out = {}
        for row in rows:
            total = int((row["cars"] or 0) + (row["motors"] or 0))
            out[row["camera_id"]] = {
                "accumulated_count": total,
                "cars": int(row["cars"] or 0),
                "motorcycles": int(row["motors"] or 0),
            }
        return out
    except Exception:
        return {}
    finally:
        conn.close()

def get_aggregated_stats(days=30):
    """
    Get aggregated stats for the last N days.
    """
    conn = get_db_connection(timeout_s=2)
    c = conn.cursor()
    try:
        cutoff = time.time() - (days * 24 * 3600)
        c.execute("""
            SELECT 
                COALESCE(SUM(new_cars), 0) as cars,
                COALESCE(SUM(new_motors), 0) as motors
            FROM traffic_history
            WHERE timestamp >= ?
        """, (cutoff,))
        row = c.fetchone()
        total = 0
        if row:
            total = int((row["cars"] or 0) + (row["motors"] or 0))
        return {
            "accumulated_count": total,
            "cars": int(row["cars"] or 0) if row else 0,
            "motorcycles": int(row["motors"] or 0) if row else 0,
        }
    except Exception as e:
        print(f"Error getting aggregated stats: {e}")
        return {"accumulated_count": 0, "cars": 0, "motorcycles": 0}
    finally:
        conn.close()

def get_history_range(camera_id=None, start_ts=None, end_ts=None):
    """
    Fetch history rows across cameras within optional time range.
    Returns list of dicts including camera_id.
    """
    conn = get_db_connection(timeout_s=2)
    c = conn.cursor()
    try:
        conditions = []
        params = []
        if camera_id:
            conditions.append("camera_id = ?")
            params.append(camera_id)
        if start_ts:
            conditions.append("timestamp >= ?")
            params.append(start_ts)
        if end_ts:
            conditions.append("timestamp <= ?")
            params.append(end_ts)
        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        query = f"""
            SELECT camera_id, timestamp, total_count, car_count, motorcycle_count,
                   new_count, new_cars, new_motors
            FROM traffic_history
            {where_clause}
            ORDER BY camera_id, timestamp ASC
        """
        c.execute(query, params)
        rows = c.fetchall()
        return [
            {
                "camera_id": row["camera_id"],
                "ts": row["timestamp"],
                "count": row["total_count"],
                "cars": row["car_count"],
                "motors": row["motorcycle_count"],
                "new_count": row["new_count"],
                "new_cars": row["new_cars"],
                "new_motors": row["new_motors"],
            }
            for row in rows
        ]
    except Exception:
        return []
    finally:
        conn.close()

def get_last_history_row(camera_id):
    conn = get_db_connection(timeout_s=2)
    c = conn.cursor()
    try:
        c.execute(
            """
            SELECT timestamp, total_count, car_count, motorcycle_count, new_count, new_cars, new_motors
            FROM traffic_history
            WHERE camera_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (camera_id,),
        )
        row = c.fetchone()
        if not row:
            return None
        return {
            "ts": row["timestamp"],
            "count": row["total_count"],
            "cars": row["car_count"],
            "motors": row["motorcycle_count"],
            "new_count": row["new_count"],
            "new_cars": row["new_cars"],
            "new_motors": row["new_motors"],
        }
    finally:
        conn.close()

def get_recent_history_averages(camera_id, start_ts, end_ts):
    conn = get_db_connection(timeout_s=2)
    c = conn.cursor()
    try:
        c.execute(
            """
            SELECT
                AVG(total_count) as avg_total,
                AVG(car_count) as avg_cars,
                AVG(motorcycle_count) as avg_motors,
                AVG(new_count) as avg_new,
                AVG(new_cars) as avg_new_cars,
                AVG(new_motors) as avg_new_motors,
                COUNT(*) as n
            FROM traffic_history
            WHERE camera_id = ?
              AND timestamp >= ?
              AND timestamp <= ?
            """,
            (camera_id, start_ts, end_ts),
        )
        row = c.fetchone()
        if not row or row["n"] == 0:
            return None
        return {
            "avg_total": float(row["avg_total"] or 0.0),
            "avg_cars": float(row["avg_cars"] or 0.0),
            "avg_motors": float(row["avg_motors"] or 0.0),
            "avg_new": float(row["avg_new"] or 0.0),
            "avg_new_cars": float(row["avg_new_cars"] or 0.0),
            "avg_new_motors": float(row["avg_new_motors"] or 0.0),
            "n": int(row["n"] or 0),
        }
    finally:
        conn.close()

# =====================================================================
# Violation / Zone / CRM helpers (Case 1 - E-TLE support)
# =====================================================================

def insert_zone(camera_id, zone_type, geometry, name=None, notes=None, active=True,
                frame_width=0, frame_height=0):
    """Insert a violation zone for a given camera. geometry is a dict/list serialized to JSON.

    frame_width/frame_height are the dimensions of the reference frame in which the
    polygon coordinates were drawn; the enforcement engine uses them to scale at runtime.
    """
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(
            """
            INSERT INTO violation_zones (camera_id, name, zone_type, geometry_json,
                                         active, notes, created_ts, frame_width, frame_height)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(camera_id),
                name or "",
                str(zone_type),
                json.dumps(geometry),
                1 if active else 0,
                notes or "",
                time.time(),
                int(frame_width or 0),
                int(frame_height or 0),
            ),
        )
        conn.commit()
        return int(c.lastrowid)
    finally:
        conn.close()


def update_zone(zone_id, **fields):
    if not fields:
        return False
    allowed = {"name", "zone_type", "geometry_json", "active", "notes"}
    sets = []
    params = []
    for k, v in fields.items():
        if k not in allowed:
            continue
        if k == "geometry_json" and not isinstance(v, str):
            v = json.dumps(v)
        if k == "active":
            v = 1 if v else 0
        sets.append(f"{k} = ?")
        params.append(v)
    if not sets:
        return False
    params.append(int(zone_id))
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(f"UPDATE violation_zones SET {', '.join(sets)} WHERE id = ?", params)
        conn.commit()
        return c.rowcount > 0
    finally:
        conn.close()


def delete_zone(zone_id):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("DELETE FROM violation_zones WHERE id = ?", (int(zone_id),))
        conn.commit()
        return c.rowcount > 0
    finally:
        conn.close()


def get_zones_for_camera(camera_id, only_active=True):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        if only_active:
            c.execute(
                "SELECT * FROM violation_zones WHERE camera_id = ? AND active = 1 ORDER BY id",
                (str(camera_id),),
            )
        else:
            c.execute(
                "SELECT * FROM violation_zones WHERE camera_id = ? ORDER BY id",
                (str(camera_id),),
            )
        rows = c.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["geometry"] = json.loads(d.get("geometry_json") or "null")
            except Exception:
                d["geometry"] = None
            out.append(d)
        return out
    finally:
        conn.close()


def get_all_zones():
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("SELECT * FROM violation_zones ORDER BY camera_id, id")
        rows = c.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["geometry"] = json.loads(d.get("geometry_json") or "null")
            except Exception:
                d["geometry"] = None
            out.append(d)
        return out
    finally:
        conn.close()


def insert_violation(
    camera_id,
    camera_name,
    violation_type,
    zone_type,
    timestamp,
    duration_s=0.0,
    zone_id=None,
    vehicle_class=None,
    plate_text=None,
    plate_confidence=0.0,
    bbox=None,
    evidence_path=None,
    lat=None,
    lng=None,
    status="pending",
    dispatched_unit=None,
    notes=None,
):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(
            """
            INSERT INTO violations (
                camera_id, camera_name, zone_id, zone_type, violation_type,
                timestamp, duration_s, vehicle_class, plate_text, plate_confidence,
                bbox_json, evidence_path, lat, lng, status, dispatched_unit, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(camera_id),
                camera_name or "",
                int(zone_id) if zone_id is not None else None,
                str(zone_type),
                str(violation_type),
                float(timestamp),
                float(duration_s or 0.0),
                str(vehicle_class) if vehicle_class is not None else None,
                plate_text,
                float(plate_confidence or 0.0),
                json.dumps(bbox) if bbox is not None else None,
                evidence_path,
                float(lat) if lat is not None else None,
                float(lng) if lng is not None else None,
                status or "pending",
                dispatched_unit,
                notes,
            ),
        )
        conn.commit()
        return int(c.lastrowid)
    finally:
        conn.close()


def list_violations(
    limit=100,
    offset=0,
    camera_id=None,
    violation_type=None,
    start_ts=None,
    end_ts=None,
    plate_contains=None,
    status=None,
):
    where = []
    params = []
    if camera_id:
        where.append("camera_id = ?")
        params.append(str(camera_id))
    if violation_type:
        where.append("violation_type = ?")
        params.append(str(violation_type))
    if start_ts is not None:
        where.append("timestamp >= ?")
        params.append(float(start_ts))
    if end_ts is not None:
        where.append("timestamp <= ?")
        params.append(float(end_ts))
    if plate_contains:
        where.append("plate_text LIKE ?")
        params.append(f"%{plate_contains}%")
    if status:
        where.append("status = ?")
        params.append(str(status))
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    sql = f"SELECT * FROM violations{where_sql} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([int(limit), int(offset)])
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(sql, params)
        rows = [dict(r) for r in c.fetchall()]
        # Parse bbox JSON
        for r in rows:
            try:
                r["bbox"] = json.loads(r.get("bbox_json") or "null")
            except Exception:
                r["bbox"] = None
        return rows
    finally:
        conn.close()


def get_violation(violation_id):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("SELECT * FROM violations WHERE id = ?", (int(violation_id),))
        r = c.fetchone()
        if not r:
            return None
        d = dict(r)
        try:
            d["bbox"] = json.loads(d.get("bbox_json") or "null")
        except Exception:
            d["bbox"] = None
        return d
    finally:
        conn.close()


def update_violation(violation_id, **fields):
    if not fields:
        return False
    allowed = {"status", "dispatched_unit", "notes", "plate_text"}
    sets = []
    params = []
    for k, v in fields.items():
        if k not in allowed:
            continue
        sets.append(f"{k} = ?")
        params.append(v)
    if not sets:
        return False
    params.append(int(violation_id))
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(f"UPDATE violations SET {', '.join(sets)} WHERE id = ?", params)
        conn.commit()
        return c.rowcount > 0
    finally:
        conn.close()


def violation_summary(start_ts=None, end_ts=None):
    """Aggregate counts by type, by hour, by day-of-week, and overall."""
    where = []
    params = []
    if start_ts is not None:
        where.append("timestamp >= ?")
        params.append(float(start_ts))
    if end_ts is not None:
        where.append("timestamp <= ?")
        params.append(float(end_ts))
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    conn = get_db_connection()
    c = conn.cursor()
    try:
        # Total
        c.execute(f"SELECT COUNT(*) AS n FROM violations{where_sql}", params)
        total = int((c.fetchone() or {"n": 0})["n"] or 0)

        # By type
        c.execute(
            f"""
            SELECT violation_type, COUNT(*) AS n
            FROM violations{where_sql}
            GROUP BY violation_type
            ORDER BY n DESC
            """,
            params,
        )
        by_type = {r["violation_type"]: int(r["n"]) for r in c.fetchall()}

        # By camera
        c.execute(
            f"""
            SELECT camera_id, camera_name, COUNT(*) AS n
            FROM violations{where_sql}
            GROUP BY camera_id, camera_name
            ORDER BY n DESC
            LIMIT 50
            """,
            params,
        )
        by_camera = [
            {"camera_id": r["camera_id"], "camera_name": r["camera_name"], "count": int(r["n"])}
            for r in c.fetchall()
        ]

        # By hour (0-23) - use local tz
        c.execute(f"SELECT timestamp, violation_type FROM violations{where_sql}", params)
        by_hour = [0] * 24
        by_dow = [0] * 7
        tzinfo = datetime.datetime.now().astimezone().tzinfo
        for r in c.fetchall():
            try:
                dt = datetime.datetime.fromtimestamp(float(r["timestamp"]), tz=tzinfo)
                by_hour[dt.hour] += 1
                by_dow[(dt.weekday() + 1) % 7] += 1  # 0=Sun,...6=Sat
            except Exception:
                pass

        return {
            "total": total,
            "by_type": by_type,
            "by_camera": by_camera,
            "by_hour": by_hour,
            "by_day_of_week": by_dow,
        }
    finally:
        conn.close()


def violation_heatmap_by_camera(start_ts=None, end_ts=None):
    """Return list of {camera_id, camera_name, lat, lng, count, by_type}."""
    where = ["lat IS NOT NULL", "lng IS NOT NULL"]
    params = []
    if start_ts is not None:
        where.append("timestamp >= ?")
        params.append(float(start_ts))
    if end_ts is not None:
        where.append("timestamp <= ?")
        params.append(float(end_ts))
    where_sql = " WHERE " + " AND ".join(where)
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(
            f"""
            SELECT camera_id, camera_name, lat, lng, violation_type, COUNT(*) AS n
            FROM violations{where_sql}
            GROUP BY camera_id, camera_name, lat, lng, violation_type
            """,
            params,
        )
        agg = {}
        for r in c.fetchall():
            key = r["camera_id"]
            if key not in agg:
                agg[key] = {
                    "camera_id": r["camera_id"],
                    "camera_name": r["camera_name"],
                    "lat": float(r["lat"]),
                    "lng": float(r["lng"]),
                    "count": 0,
                    "by_type": {},
                }
            n = int(r["n"])
            agg[key]["count"] += n
            agg[key]["by_type"][r["violation_type"]] = n
        return list(agg.values())
    finally:
        conn.close()


def insert_crm_report(
    reporter_name,
    reporter_contact,
    category,
    description,
    lat=None,
    lng=None,
    camera_id=None,
    auto_classified_type=None,
    priority="normal",
    status="open",
):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(
            """
            INSERT INTO crm_reports (
                timestamp, reporter_name, reporter_contact, category, description,
                lat, lng, camera_id, status, auto_classified_type, priority
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                reporter_name or "",
                reporter_contact or "",
                category or "",
                description or "",
                float(lat) if lat is not None else None,
                float(lng) if lng is not None else None,
                str(camera_id) if camera_id else None,
                status,
                auto_classified_type,
                priority,
            ),
        )
        conn.commit()
        return int(c.lastrowid)
    finally:
        conn.close()


def list_crm_reports(limit=100, offset=0, status=None):
    where = []
    params = []
    if status:
        where.append("status = ?")
        params.append(str(status))
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    sql = f"SELECT * FROM crm_reports{where_sql} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([int(limit), int(offset)])
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(sql, params)
        return [dict(r) for r in c.fetchall()]
    finally:
        conn.close()


def update_crm_report(report_id, **fields):
    if not fields:
        return False
    allowed = {"status", "priority", "auto_classified_type", "camera_id"}
    sets = []
    params = []
    for k, v in fields.items():
        if k not in allowed:
            continue
        sets.append(f"{k} = ?")
        params.append(v)
    if not sets:
        return False
    params.append(int(report_id))
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(f"UPDATE crm_reports SET {', '.join(sets)} WHERE id = ?", params)
        conn.commit()
        return c.rowcount > 0
    finally:
        conn.close()


def crm_summary():
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("SELECT COUNT(*) AS n FROM crm_reports")
        total = int((c.fetchone() or {"n": 0})["n"] or 0)
        c.execute("SELECT status, COUNT(*) AS n FROM crm_reports GROUP BY status")
        by_status = {r["status"]: int(r["n"]) for r in c.fetchall()}
        c.execute("SELECT auto_classified_type, COUNT(*) AS n FROM crm_reports GROUP BY auto_classified_type")
        by_type = {(r["auto_classified_type"] or "unclassified"): int(r["n"]) for r in c.fetchall()}
        return {"total": total, "by_status": by_status, "by_type": by_type}
    finally:
        conn.close()


def recommend_enforcement_points(top_n=10, start_ts=None, end_ts=None):
    """
    Recommend the top N camera locations for enforcement (officer / E-TLE camera
    placement) based on violation density and vulnerability score.

    Score =  (violations_per_day) * 0.6
          +  (distinct violation types * 2) * 0.2
          +  (recency_weight, recent_count / total) * 0.2
    """
    if end_ts is None:
        end_ts = time.time()
    if start_ts is None:
        start_ts = end_ts - (30.0 * 24.0 * 3600.0)  # 30 days default
    span_days = max(1.0, (end_ts - start_ts) / 86400.0)
    recent_cutoff = end_ts - (7.0 * 24.0 * 3600.0)

    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(
            """
            SELECT camera_id, camera_name, violation_type, lat, lng,
                   SUM(CASE WHEN timestamp >= ? THEN 1 ELSE 0 END) AS recent_n,
                   COUNT(*) AS n
            FROM violations
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY camera_id, camera_name, violation_type, lat, lng
            """,
            (recent_cutoff, start_ts, end_ts),
        )
        rows = c.fetchall()
    finally:
        conn.close()

    agg = {}
    for r in rows:
        cam = r["camera_id"]
        if cam not in agg:
            agg[cam] = {
                "camera_id": cam,
                "camera_name": r["camera_name"],
                "lat": r["lat"],
                "lng": r["lng"],
                "count": 0,
                "recent": 0,
                "types": {},
            }
        agg[cam]["count"] += int(r["n"])
        agg[cam]["recent"] += int(r["recent_n"] or 0)
        agg[cam]["types"][r["violation_type"]] = int(r["n"])

    out = []
    for v in agg.values():
        vpd = v["count"] / span_days
        type_diversity = len(v["types"])
        recency_ratio = (v["recent"] / v["count"]) if v["count"] > 0 else 0.0
        score = (vpd * 0.6) + (type_diversity * 2.0 * 0.2) + (recency_ratio * 10.0 * 0.2)
        # Recommend primary violation types to target
        primary = sorted(v["types"].items(), key=lambda kv: -kv[1])
        top_types = [k for k, _ in primary[:3]]
        v_out = dict(v)
        v_out["violations_per_day"] = round(vpd, 2)
        v_out["type_diversity"] = type_diversity
        v_out["recency_ratio"] = round(recency_ratio, 2)
        v_out["score"] = round(score, 2)
        v_out["recommended_target_types"] = top_types
        # Recommendation: high score → fixed camera; medium → officer patrol
        if score >= 3.0:
            v_out["recommendation"] = "install_etle_camera"
        elif score >= 1.0:
            v_out["recommendation"] = "officer_patrol"
        else:
            v_out["recommendation"] = "monitor"
        out.append(v_out)

    out.sort(key=lambda x: -x["score"])
    return out[: int(top_n)]
