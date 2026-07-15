# PADI — Engineering Documentation

## Project Architecture

```
big-data-traffict-competitiom/
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── auth.py                  # Authentication (demo mode, login)
│   ├── config.py                # All configuration + settings persistence
│   ├── database.py              # SQLite schema, CRUD operations
│   ├── globals.py               # Shared state (global_stats, locks, frames)
│   ├── routes.py                # All API endpoints + page routes
│   ├── utils.py                 # Config file I/O, stats helpers
│   ├── services/
│   │   ├── camera.py            # Camera agents (capture, inference, stream)
│   │   ├── enforcement.py       # Violation detection engine
│   │   ├── anpr.py              # Plate recognition (PaddleOCR pipeline)
│   │   ├── ai_ocr.py            # AI-enhanced OCR correction
│   │   ├── gpu_batch.py         # GPU batch inference worker
│   │   ├── social_scraper.py    # X.com/social media scraping (Playwright)
│   │   └── hikvision.py         # Hikvision ISAPI integration
│   └── templates/               # Jinja2 HTML templates
│       ├── base.html            # Layout (sidebar, header, chatbot)
│       ├── dashboard.html       # Main command center
│       ├── enforcement.html     # E-TLE monitoring
│       ├── cameras.html         # Camera management
│       ├── zones.html           # Zone polygon editor
│       ├── settings.html        # System configuration
│       ├── executive_summary.html  # PDF-ready report
│       ├── crm.html             # Citizen reports + social mentions
│       ├── documentation.html   # System docs
│       └── ...
├── data/
│   ├── cctv_config.json         # Camera list (id, name, url, lat, lng)
│   ├── app_settings.json        # Persisted settings (survives restart)
│   ├── x_cookies.json           # X.com session cookies
│   ├── traffic_stats.json       # In-memory stats snapshot
│   ├── traffic.db               # SQLite database
│   └── violations_evidence/     # Evidence JPEG files (by date)
├── models/                      # YOLO model weights (.pt files)
├── run.py                       # Entry point
└── docs/                        # Documentation
```

## System Flow

```
┌─────────────────────────────────────────────────────────┐
│                    PADI Architecture                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  CCTV Stream (HLS/RTSP)                                 │
│       │                                                  │
│       ▼                                                  │
│  Camera Agent (1 per camera, daemon thread)              │
│       │                                                  │
│       ├─► Frame Capture (OpenCV VideoCapture)            │
│       │                                                  │
│       ├─► GPU Batch Inference Worker (shared)            │
│       │       └─► YOLO detection → bounding boxes       │
│       │                                                  │
│       ├─► Vehicle Tracker (IoU matching)                 │
│       │                                                  │
│       ├─► Enforcement Engine (per camera)                │
│       │       ├─► Zone check (point-in-polygon)         │
│       │       ├─► Dwell timer + stationary check        │
│       │       ├─► Violation trigger → Evidence capture   │
│       │       └─► ANPR (PaddleOCR + AI correction)      │
│       │                                                  │
│       └─► Stream Output (720p, JPEG 45, 15 FPS)         │
│               └─► Browser (MJPEG)                        │
│                                                          │
├─────────────────────────────────────────────────────────┤
│  Flask Server (single process, multi-thread)             │
│       ├─► Page routes (dashboard, enforcement, etc.)     │
│       ├─► REST API (/api/stats, /api/violations, etc.)  │
│       ├─► Video feed (/video_feed/<camera_id>)          │
│       └─► PADI Assistant (OpenAI-compatible API)         │
└─────────────────────────────────────────────────────────┘
```

## Database Schema (SQLite)

### traffic_history
Stores per-camera vehicle count data at 2-second intervals.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| camera_id | TEXT | Camera identifier |
| timestamp | REAL | Unix timestamp |
| total_count | INTEGER | Current vehicles in frame |
| car_count | INTEGER | Cars detected |
| motorcycle_count | INTEGER | Motorcycles detected |
| new_count | INTEGER | New vehicles this interval |
| new_cars | INTEGER | New cars |
| new_motors | INTEGER | New motorcycles |

**Indexes:** `(camera_id, timestamp)`

### violation_zones
Enforcement zones drawn by operator on camera frames.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| camera_id | TEXT | Which camera |
| name | TEXT | Zone label |
| zone_type | TEXT | `no_parking`, `busway`, `bicycle`, `bus_stop`, `wrong_way` |
| geometry_json | TEXT | Polygon points `[[x,y],...]` or bbox `[x1,y1,x2,y2]` |
| active | INTEGER | 1=enabled, 0=disabled |
| notes | TEXT | Extra info (e.g. `direction:90` for wrong_way) |
| created_ts | REAL | Creation timestamp |
| frame_width | INTEGER | Reference frame width for scaling |
| frame_height | INTEGER | Reference frame height for scaling |

**Indexes:** `(camera_id, active)`

### violations
Detected violations (E-TLE evidence records).

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| camera_id | TEXT | Camera that detected |
| camera_name | TEXT | Human-readable name |
| zone_id | INTEGER | FK to violation_zones |
| zone_type | TEXT | Zone type at time of violation |
| violation_type | TEXT | `illegal_parking`, `busway_occupancy`, `bicycle_lane_occupancy`, `wrong_way` |
| timestamp | REAL | Unix timestamp |
| duration_s | REAL | How long vehicle was in zone |
| vehicle_class | TEXT | `car`, `motorcycle`, `bus` |
| plate_text | TEXT | Recognized plate (nullable) |
| plate_confidence | REAL | OCR confidence 0-1 |
| bbox_json | TEXT | `[x1,y1,x2,y2]` bounding box |
| evidence_path | TEXT | Relative path to evidence JPEG |
| lat | REAL | Camera latitude |
| lng | REAL | Camera longitude |
| status | TEXT | `pending`, `confirmed`, `dismissed` |
| dispatched_unit | TEXT | Assigned officer (nullable) |
| notes | TEXT | Operator notes |

**Indexes:** `(timestamp)`, `(camera_id, timestamp)`, `(violation_type, timestamp)`, `(plate_text)`

### crm_reports
Citizen complaints / social media reports.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| timestamp | REAL | Report time |
| reporter_name | TEXT | Who reported |
| reporter_contact | TEXT | Email/phone |
| category | TEXT | `traffic`, `illegal_parking`, `public_transport`, `other` |
| description | TEXT | Free-text description |
| lat | REAL | Location latitude |
| lng | REAL | Location longitude |
| camera_id | TEXT | Linked camera (nullable) |
| status | TEXT | `open`, `investigating`, `resolved`, `closed` |
| auto_classified_type | TEXT | AI-classified violation type |
| priority | TEXT | `normal`, `high` |

### chat_messages
PADI Assistant conversation history.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| session_id | TEXT | Browser session |
| ts | REAL | Message timestamp |
| role | TEXT | `user`, `assistant`, `system` |
| content | TEXT | Message text |
| page | TEXT | Which page user was on |
| meta_json | TEXT | Extra metadata (provider, model) |

## Configuration Files

### data/cctv_config.json
```json
[
  {
    "id": "cam_xxxxx",
    "name": "Camera Name",
    "url": "https://cctv.example.com/stream.m3u8",
    "lat": -6.2146,
    "lng": 106.8020,
    "active": true
  }
]
```

### data/app_settings.json
Persisted settings that survive server restart.
```json
{
  "detection": { "conf_threshold", "iou_threshold", "infer_imgsz", ... },
  "enforcement": { "violations_enabled", "illegal_parking_min_seconds", ... },
  "social_media": { "search_query", "x_search_query" },
  "ai_provider": { "provider", "base_url", "api_key", "model", "use_ai_for_anpr", "use_ai_for_chat", "chat_model" },
  "general": { "timezone" }
}
```

## Coding Rules

### Python
- **Single file for routes** — all endpoints in `routes.py` (large but searchable)
- **Services are independent** — each service in `app/services/` handles one concern
- **Config is centralized** — all constants in `config.py`, loaded from env vars + `app_settings.json`
- **Thread safety** — use `g.lock` for shared state, `g.model_lock` for YOLO model (deprecated in favor of batch worker)
- **No external task queue** — everything runs in-process with daemon threads
- **Error handling** — all camera/enforcement errors are caught and logged, never crash the server

### Frontend
- **Jinja2 templates** extending `base.html`
- **Tailwind CSS** via CDN (no build step)
- **Vanilla JavaScript** — no React/Vue, just plain JS in `<script>` blocks
- **Dark theme only** — all colors from slate/sky/emerald palette
- **MJPEG streaming** — `<img src="/video_feed/cam_id">` for live video
- **Auto-refresh** — dashboard polls APIs every 30s, enforcement every 20s

### Performance Rules
- Stream: 720p, JPEG 45, 15 FPS (lightweight for browser)
- Evidence: full resolution, JPEG 90+ (high quality for proof)
- Background cameras: 2 FPS capture, inference every 10s
- Active camera: 30 FPS capture, inference every 5 frames
- GPU batch: all cameras share 1 YOLO model, inference batched
- HLS reconnect: 8s timeout, auto-retry with backoff

### Enforcement Rules
- Illegal parking: vehicle stationary >60s in no_parking zone
- Busway: non-bus in busway zone >5s (buses exempt via class_id + size heuristic)
- Bicycle lane: motorized vehicle >5s
- Wrong way: >8s, >150px movement, >20px/s speed, >160° angle difference, ALL segments confirm
- Cooldown: 120s between same vehicle same zone re-violation

### Security
- Demo mode: pages open, actions require password (`@sterion1`)
- Anti-inspect: right-click disabled, F12/Ctrl+Shift+I blocked
- Session cookies for X.com stored in `data/x_cookies.json` (gitignored)
- AI API keys stored in `data/app_settings.json` (gitignored)
