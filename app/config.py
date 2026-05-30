import os

# Base Directories
# Moved inside app/, so go up two levels to reach root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Files
CONFIG_FILE = os.path.join(DATA_DIR, "cctv_config.json")
STATS_FILE = os.path.join(DATA_DIR, "traffic_stats.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "app_settings.json")
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolo11m.pt")
YOLO_ONNX_PATH = os.path.join(MODELS_DIR, "yolov5n.onnx")
YOLO_ONNX_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"
DATA_LAKE_PATH = os.environ.get("DATA_LAKE_PATH") or os.path.join(BASE_DIR, "data_lake", "raw")

# Custom YOLO weights trained on Roboflow 'vehicle-detection v3' (6 classes
# of Indonesian vehicles: bus, car, microbus, motorbike, pickup-van, truck).
# If present, the camera pipeline will prefer this over the generic COCO weights.
# Produced by scripts/train_vehicle_yolo.py
YOLO_CUSTOM_PATH = os.environ.get("YOLO_CUSTOM_PATH") or os.path.join(MODELS_DIR, "vehicle_v3_best.pt")
# Whether to use the custom model (auto-disabled if file missing)
USE_CUSTOM_YOLO = str(os.environ.get("USE_CUSTOM_YOLO") or "0").strip().lower() in {"1", "true", "yes", "on"}

# Server
HOST_IP = "0.0.0.0"
HOST_PORT = int(os.environ.get("HOST_PORT") or os.environ.get("PORT") or 5002)

# Timezone (WIB = Asia/Jakarta for Bogor/Jakarta area)
TIMEZONE = os.environ.get("TIMEZONE") or "Asia/Jakarta"

# YOLO & Detection Config
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.50
# PROCESS_INTERVAL: seconds between INFERENCE runs (not frame capture).
# Frame capture runs at full speed for smooth streaming; inference is decoupled.
PROCESS_INTERVAL = float(os.environ.get("PROCESS_INTERVAL") or 0.15)
STREAM_FPS = float(os.environ.get("STREAM_FPS") or 30)
STREAM_JPEG_QUALITY = int(os.environ.get("STREAM_JPEG_QUALITY") or 85)
STREAM_MAX_WIDTH = int(os.environ.get("STREAM_MAX_WIDTH") or 1920)
INFER_IMGSZ = int(os.environ.get("INFER_IMGSZ") or 640)
CAPTURE_DROP_FRAMES = int(os.environ.get("CAPTURE_DROP_FRAMES") or 0)
# How many frames to skip between inferences (0 = every frame, 4 = every 5th frame)
INFER_SKIP_FRAMES = int(os.environ.get("INFER_SKIP_FRAMES") or 4)
# Increase history length to support up to ~24h in memory (Hot Data)
# 24h * 60m * 30 (2s intervals) = ~43,200 points
HISTORY_MAX_LEN = 50000

# Vehicle Classes
# For COCO model (yolov8l.pt): class IDs from COCO dataset
CLASS_CAR = 0
CLASS_MOTORCYCLE = 1
CLASS_BUS = 2

VEHICLE_CLASSES_COCO = [1, 2, 3, 5, 7]
CLASS_MAPPING_COCO = {
    1: CLASS_MOTORCYCLE, # Bicycle -> Motorcycle
    2: CLASS_CAR,        # Car -> Car
    3: CLASS_MOTORCYCLE, # Motorcycle -> Motorcycle
    5: CLASS_BUS,        # Bus -> Bus
    7: CLASS_CAR         # Truck -> Car
}

# For custom model (vehicle_v3_best.pt): 6 classes
# Index: 0=bus, 1=car, 2=microbus, 3=motorbike, 4=pickup-van, 5=truck
VEHICLE_CLASSES_CUSTOM = [0, 1, 2, 3, 4, 5]
CLASS_MAPPING_CUSTOM = {
    0: CLASS_BUS,        # Bus -> Bus
    1: CLASS_CAR,        # Car -> Car
    2: CLASS_BUS,        # Microbus -> Bus
    3: CLASS_MOTORCYCLE, # Motorbike -> Motorcycle
    4: CLASS_CAR,        # Pickup-van -> Car
    5: CLASS_CAR,        # Truck -> Car (large vehicle)
}
CUSTOM_CLASS_NAMES = ['bus', 'car', 'microbus', 'motorbike', 'pickup-van', 'truck']

# Active mapping (selected at runtime based on which model is loaded)
# Default to COCO; camera.py overrides if custom model is used
VEHICLE_CLASSES = VEHICLE_CLASSES_COCO
CLASS_MAPPING = CLASS_MAPPING_COCO

# ==============================================================
# E-TLE / Violation Detection Config (Case 1)
# ==============================================================
# Enforcement features (Intelligent Traffic Enforcement & Behaviour Analysis)
VIOLATIONS_ENABLED = str(os.environ.get("VIOLATIONS_ENABLED") or "1").strip().lower() in {"1", "true", "yes", "on"}

# Zone types recognized by the enforcement engine
ZONE_TYPE_NO_PARKING = "no_parking"   # Static violation (illegal parking)
ZONE_TYPE_BUSWAY = "busway"           # Dynamic violation (busway lane occupancy)
ZONE_TYPE_BICYCLE = "bicycle"         # Dynamic violation (bicycle lane occupancy)
ZONE_TYPE_BUS_STOP = "bus_stop"       # Valid bus stop (inverse rule; for PT pickup/drop)
ZONE_TYPE_WRONG_WAY = "wrong_way"     # Directional violation (counter-flow / lawan arah)

ZONE_TYPES = [ZONE_TYPE_NO_PARKING, ZONE_TYPE_BUSWAY, ZONE_TYPE_BICYCLE, ZONE_TYPE_BUS_STOP, ZONE_TYPE_WRONG_WAY]

# Mapping of violation types
VIOLATION_ILLEGAL_PARKING = "illegal_parking"
VIOLATION_BUSWAY = "busway_occupancy"
VIOLATION_BICYCLE_LANE = "bicycle_lane_occupancy"
VIOLATION_PICKUP_DROPOFF = "illegal_pickup_dropoff"
VIOLATION_WRONG_WAY = "wrong_way"

VIOLATION_TYPES = [
    VIOLATION_ILLEGAL_PARKING,
    VIOLATION_BUSWAY,
    VIOLATION_BICYCLE_LANE,
    VIOLATION_PICKUP_DROPOFF,
    VIOLATION_WRONG_WAY,
]

# How long a vehicle must remain (low movement) in a no-parking zone to trigger violation
ILLEGAL_PARKING_MIN_SECONDS = float(os.environ.get("ILLEGAL_PARKING_MIN_SECONDS") or 60.0)
# Grace period: extra time before recording (allows emergency stops to move away)
ILLEGAL_PARKING_GRACE_SECONDS = float(os.environ.get("ILLEGAL_PARKING_GRACE_SECONDS") or 0.0)
# Pixel movement threshold to consider a vehicle "static"
# This is MAX DISPLACEMENT from initial position (not accumulated movement)
# YOLO bounding boxes jitter ~5-15px per frame even for stationary vehicles
# At 1920x1080, vehicles in queue may shift 30-60px due to detection jitter
STATIC_MOVEMENT_PX = float(os.environ.get("STATIC_MOVEMENT_PX") or 300.0)
# Seconds required in dynamic lane (busway/bicycle) before flagging (debounce)
DYNAMIC_LANE_MIN_SECONDS = float(os.environ.get("DYNAMIC_LANE_MIN_SECONDS") or 2.0)
# Seconds a vehicle must be moving in wrong direction before flagging
WRONG_WAY_MIN_SECONDS = float(os.environ.get("WRONG_WAY_MIN_SECONDS") or 4.0)
# Cooldown per track to avoid re-logging the same violation too quickly (seconds)
VIOLATION_COOLDOWN_SECONDS = float(os.environ.get("VIOLATION_COOLDOWN_SECONDS") or 120.0)

# Evidence: capture multiple frames for stronger proof
# Number of evidence snapshots to capture (first entry, mid-dwell, violation trigger)
EVIDENCE_MULTI_FRAME = str(os.environ.get("EVIDENCE_MULTI_FRAME") or "1").strip().lower() in {"1", "true", "yes", "on"}

# Where to save violation snapshot JPGs (used for E-TLE evidence)
EVIDENCE_DIR = os.path.join(BASE_DIR, "data", "violations_evidence")
EVIDENCE_JPEG_QUALITY = int(os.environ.get("EVIDENCE_JPEG_QUALITY") or 85)

# ANPR (Automatic Number Plate Recognition)
ANPR_ENABLED = str(os.environ.get("ANPR_ENABLED") or "1").strip().lower() in {"1", "true", "yes", "on"}
# Auto-generate pseudo-plate when OCR is not available (for demo/dev environments)
# DISABLED by default: showing fake plates is misleading for enforcement evidence
ANPR_FALLBACK_SIMULATE = str(os.environ.get("ANPR_FALLBACK_SIMULATE") or "0").strip().lower() in {"1", "true", "yes", "on"}

# Indonesian regional plate prefix by region (auto-picked from camera name)
# Keys are substrings to match in camera name (case-insensitive)
REGIONAL_PLATE_PREFIX = {
    "jakarta": "B",
    "bekasi":  "B",
    "tangerang": "B",
    "depok": "B",
    "bogor": "F",
    "sukabumi": "F",
    "bandung": "D",
    "cianjur": "F",
    "surabaya": "L",
    "malang": "N",
    "yogyakarta": "AB",
    "semarang": "H",
    "medan": "BK",
}
DEFAULT_PLATE_PREFIX = "B"

# ==============================================================
# Twitter/X API Config (for social media monitoring)
# ==============================================================
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN") or "AAAAAAAAAAAAAAAAAAAAANGG9gEAAAAAptGWj9Cx51A32KJ8vwrcMNzVPVI%3DM88yPaXw5uXVOpqNFrV97dtHNRes9IcoLqHihani3lQa0IgdE1"
TWITTER_SEARCH_QUERY = os.environ.get("TWITTER_SEARCH_QUERY") or "(macet OR kemacetan OR parkir liar OR kecelakaan OR lalu lintas) (jakarta OR sudirman OR senayan OR bendungan hilir OR gelora)"
TWITTER_MAX_RESULTS = int(os.environ.get("TWITTER_MAX_RESULTS") or 15)

# X.com (Twitter) CRM Scraper Settings
X_SEARCH_QUERY = "@DishubDKI OR #DishubDKI OR to:DishubDKI"
X_COOKIES_FILE = os.path.join(DATA_DIR, "x_cookies.json")


# ==============================================================
# Settings Persistence
# ==============================================================
import json as _json

def load_persisted_settings():
    """Load saved settings from disk and apply to module-level variables."""
    global CONF_THRESHOLD, IOU_THRESHOLD, INFER_IMGSZ, PROCESS_INTERVAL
    global STREAM_FPS, STREAM_JPEG_QUALITY, STREAM_MAX_WIDTH, INFER_SKIP_FRAMES
    global VIOLATIONS_ENABLED, ANPR_ENABLED, ANPR_FALLBACK_SIMULATE
    global ILLEGAL_PARKING_MIN_SECONDS, STATIC_MOVEMENT_PX
    global DYNAMIC_LANE_MIN_SECONDS, WRONG_WAY_MIN_SECONDS, VIOLATION_COOLDOWN_SECONDS
    global TWITTER_SEARCH_QUERY, TWITTER_MAX_RESULTS, TIMEZONE, X_SEARCH_QUERY

    if not os.path.exists(SETTINGS_FILE):
        return
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            data = _json.load(f)
    except Exception:
        return

    det = data.get("detection") or {}
    if "conf_threshold" in det:
        CONF_THRESHOLD = float(det["conf_threshold"])
    if "iou_threshold" in det:
        IOU_THRESHOLD = float(det["iou_threshold"])
    if "infer_imgsz" in det:
        INFER_IMGSZ = int(det["infer_imgsz"])
    if "process_interval" in det:
        # Legacy setting: kept for backward compatibility with old settings files,
        # but no longer exposed as an active live-pipeline control.
        PROCESS_INTERVAL = float(det["process_interval"])
    if "stream_fps" in det:
        STREAM_FPS = float(det["stream_fps"])
    if "stream_jpeg_quality" in det:
        STREAM_JPEG_QUALITY = int(det["stream_jpeg_quality"])
    if "stream_max_width" in det:
        STREAM_MAX_WIDTH = int(det["stream_max_width"])
    if "infer_skip_frames" in det:
        INFER_SKIP_FRAMES = int(det["infer_skip_frames"])

    enf = data.get("enforcement") or {}
    if "violations_enabled" in enf:
        VIOLATIONS_ENABLED = bool(enf["violations_enabled"])
    if "anpr_enabled" in enf:
        ANPR_ENABLED = bool(enf["anpr_enabled"])
    if "anpr_fallback_simulate" in enf:
        ANPR_FALLBACK_SIMULATE = bool(enf["anpr_fallback_simulate"])
    if "illegal_parking_min_seconds" in enf:
        ILLEGAL_PARKING_MIN_SECONDS = float(enf["illegal_parking_min_seconds"])
    if "static_movement_px" in enf:
        STATIC_MOVEMENT_PX = float(enf["static_movement_px"])
    if "dynamic_lane_min_seconds" in enf:
        DYNAMIC_LANE_MIN_SECONDS = float(enf["dynamic_lane_min_seconds"])
    if "wrong_way_min_seconds" in enf:
        WRONG_WAY_MIN_SECONDS = float(enf["wrong_way_min_seconds"])
    if "violation_cooldown_seconds" in enf:
        VIOLATION_COOLDOWN_SECONDS = float(enf["violation_cooldown_seconds"])

    social = data.get("social_media") or {}
    if "search_query" in social:
        TWITTER_SEARCH_QUERY = str(social["search_query"])
    if "max_results" in social:
        TWITTER_MAX_RESULTS = int(social["max_results"])
    if "x_search_query" in social:
        X_SEARCH_QUERY = str(social["x_search_query"])
    # Load X cookies from settings if provided (overwrite file)
    if "x_cookies_json" in social and social["x_cookies_json"]:
        try:
            cookies_data = social["x_cookies_json"]
            if isinstance(cookies_data, str) and cookies_data.strip():
                with open(X_COOKIES_FILE, 'w', encoding='utf-8') as f:
                    f.write(cookies_data)
        except Exception:
            pass

    general = data.get("general") or {}
    if "timezone" in general:
        TIMEZONE = str(general["timezone"])


def save_persisted_settings():
    """Save current settings to disk so they survive restart."""
    data = {
        "detection": {
            "conf_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "infer_imgsz": INFER_IMGSZ,
            "stream_fps": STREAM_FPS,
            "stream_jpeg_quality": STREAM_JPEG_QUALITY,
            "stream_max_width": STREAM_MAX_WIDTH,
            "infer_skip_frames": INFER_SKIP_FRAMES,
        },
        "enforcement": {
            "violations_enabled": VIOLATIONS_ENABLED,
            "anpr_enabled": ANPR_ENABLED,
            "anpr_fallback_simulate": ANPR_FALLBACK_SIMULATE,
            "illegal_parking_min_seconds": ILLEGAL_PARKING_MIN_SECONDS,
            "static_movement_px": STATIC_MOVEMENT_PX,
            "dynamic_lane_min_seconds": DYNAMIC_LANE_MIN_SECONDS,
            "wrong_way_min_seconds": WRONG_WAY_MIN_SECONDS,
            "violation_cooldown_seconds": VIOLATION_COOLDOWN_SECONDS,
        },
        "social_media": {
            "search_query": TWITTER_SEARCH_QUERY,
            "max_results": TWITTER_MAX_RESULTS,
            "x_search_query": X_SEARCH_QUERY,
        },
        "general": {
            "timezone": TIMEZONE,
        },
    }
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            _json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save settings: {e}")
        return False


# Auto-load persisted settings on import
load_persisted_settings()
