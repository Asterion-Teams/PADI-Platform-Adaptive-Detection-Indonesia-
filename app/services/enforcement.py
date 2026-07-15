"""
Enforcement Engine (Case 1)
----------------------------
Detects violations by tracking YOLO-detected vehicles inside violation zones.

Key principle: A violation is ONLY triggered when:
  1. A YOLO-tracked vehicle (with valid track ID and bounding box) is inside a zone
  2. That SAME track ID has been continuously detected for >= 60 seconds
  3. The vehicle has NOT moved significantly (truly stationary)

This engine uses YOLO track IDs directly — the same tracking that draws
green/blue bounding boxes on the live feed. If YOLO doesn't see a vehicle,
the enforcement engine doesn't see it either. No phantom detections.
"""
from __future__ import annotations

import datetime
import math
import os
import threading
import time
import uuid

import cv2
import numpy as np

from app.config import (
    VIOLATIONS_ENABLED,
    ILLEGAL_PARKING_MIN_SECONDS,
    ILLEGAL_PARKING_GRACE_SECONDS,
    DYNAMIC_LANE_MIN_SECONDS,
    WRONG_WAY_MIN_SECONDS,
    VIOLATION_COOLDOWN_SECONDS,
    EVIDENCE_DIR,
    EVIDENCE_JPEG_QUALITY,
    ZONE_TYPE_NO_PARKING,
    ZONE_TYPE_BUSWAY,
    ZONE_TYPE_BICYCLE,
    ZONE_TYPE_BUS_STOP,
    ZONE_TYPE_WRONG_WAY,
    VIOLATION_ILLEGAL_PARKING,
    VIOLATION_BUSWAY,
    VIOLATION_BICYCLE_LANE,
    VIOLATION_PICKUP_DROPOFF,
    VIOLATION_WRONG_WAY,
    CLASS_CAR,
    CLASS_MOTORCYCLE,
    CLASS_BUS,
    CLASS_TRUCK,
)
import app.config as app_config
from app.database import (
    get_zones_for_camera,
    insert_violation,
)
from app.services.anpr import recognize_plate
from app.services.anpr_enhanced import preprocess_sharpen, preprocess_denoise, preprocess_adaptive_clahe, preprocess_combo_heavy_clahe_sharpen


# -------------------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------------------

def _bbox_center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_bottom_center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, float(y2))


def _bbox_bottom_anchors(b):
    x1, y1, x2, y2 = [float(v) for v in b]
    width = max(1.0, x2 - x1)
    y = max(y1, y2 - max(2.0, (y2 - y1) * 0.05))
    return [
        (x1 + width * 0.2, y),
        ((x1 + x2) / 2.0, y),
        (x1 + width * 0.8, y),
    ]


def _point_in_polygon(px, py, polygon):
    if not polygon or len(polygon) < 3:
        return False
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _point_in_bbox(px, py, bbox):
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def _point_in_geometry(px, py, geometry):
    if geometry is None:
        return False
    if isinstance(geometry, list) and geometry and isinstance(geometry[0], (list, tuple)):
        try:
            poly = [(float(p[0]), float(p[1])) for p in geometry]
        except Exception:
            return False
        return _point_in_polygon(px, py, poly)
    if isinstance(geometry, (list, tuple)) and len(geometry) == 4 and all(
        isinstance(v, (int, float)) for v in geometry
    ):
        return _point_in_bbox(px, py, geometry)
    if isinstance(geometry, dict):
        if "polygon" in geometry:
            return _point_in_geometry(px, py, geometry["polygon"])
        if "bbox" in geometry:
            return _point_in_geometry(px, py, geometry["bbox"])
    return False


def _geometry_to_polyline_for_draw(geometry):
    if geometry is None:
        return None
    if isinstance(geometry, list) and geometry and isinstance(geometry[0], (list, tuple)):
        try:
            return np.array([[int(p[0]), int(p[1])] for p in geometry], dtype=np.int32)
        except Exception:
            return None
    if isinstance(geometry, (list, tuple)) and len(geometry) == 4 and all(
        isinstance(v, (int, float)) for v in geometry
    ):
        x1, y1, x2, y2 = [int(v) for v in geometry]
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    if isinstance(geometry, dict):
        if "polygon" in geometry:
            return _geometry_to_polyline_for_draw(geometry["polygon"])
        if "bbox" in geometry:
            return _geometry_to_polyline_for_draw(geometry["bbox"])
    return None


def _count_points_in_geometry(points, geometry):
    return sum(1 for (px, py) in points if _point_in_geometry(px, py, geometry))


def _track_motion_span(st) -> float:
    if len(st.positions) < 2:
        return 0.0
    first_ts, first_cx, first_cy = st.positions[0]
    max_disp = 0.0
    for (_, cx, cy) in st.positions[1:]:
        max_disp = max(max_disp, math.hypot(cx - first_cx, cy - first_cy))
    return max_disp


def _looks_like_vehicle_box(box, class_id) -> bool:
    """Check if a bounding box looks like a vehicle, not a barrier/portal/other object.
    
    Applies strict size, aspect ratio, and shape filters to reject:
    - Portals, barriers, gates (tall and narrow or very wide and flat)
    - Signs, poles, people, etc.
    """
    x1, y1, x2, y2 = box
    bw = abs(x2 - x1)
    bh = abs(y2 - y1)
    if bw <= 0 or bh <= 0:
        return False
    
    area = bw * bh
    aspect = bw / max(1.0, bh)  # width / height
    
    # REJECT: Object is too small to be a vehicle
    if bw < 50 or bh < 40 or area < 4000:
        return False
    
    # REJECT: Object is too TALL and NARROW (likely a portal, barrier, gate, pole)
    # A typical vehicle is wider than tall or roughly square
    if aspect < 0.35:
        return False
    
    # REJECT: Object is too WIDE and FLAT (likely a barrier, fence, or road marking)
    # A typical vehicle has some height
    if aspect > 5.5:
        return False
    
    # REJECT: Very small area even if dimensions pass (likely a sign or pole)
    if area < 5000:
        return False
    
    # Per-class specific checks
    if class_id == CLASS_MOTORCYCLE:
        # Motorcycles are smaller and can be taller than cars
        return 0.28 <= aspect <= 3.5 and area >= 2000 and bw >= 30 and bh >= 25
    if class_id == CLASS_BUS:
        # Buses are large and wide
        return 0.55 <= aspect <= 6.0 and area >= 15000 and bw >= 100
    if class_id == CLASS_CAR:
        # Cars are roughly 1.5-2.5x wider than tall
        return 0.55 <= aspect <= 4.0 and area >= 4000 and bw >= 50 and bh >= 40
    if class_id == CLASS_TRUCK:
        return 0.45 <= aspect <= 5.0 and area >= 8000 and bw >= 60
    
    # Default: standard vehicle shape
    return 0.45 <= aspect <= 4.5 and area >= 5000 and bw >= 50 and bh >= 40


def _has_vehicle_texture(frame_bgr, box) -> bool:
    if frame_bgr is None:
        return True
    fh, fw = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(fw - 1, x1))
    y1 = max(0, min(fh - 1, y1))
    x2 = max(x1 + 1, min(fw, x2))
    y2 = max(y1 + 1, min(fh, y2))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 24 or crop.shape[1] < 24:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size or 1)
    contrast = float(np.std(gray))

    # Real vehicles still show enough structure from body edges, windows,
    # wheel arches, lights, or reflections. Road puddles tend to stay flat.
    return edge_density >= 0.018 or contrast >= 22.0


_VIOLATION_COLORS = {
    ZONE_TYPE_NO_PARKING: (0, 80, 220),
    ZONE_TYPE_BUSWAY:     (80, 180, 255),
    ZONE_TYPE_BICYCLE:    (50, 220, 220),
    ZONE_TYPE_BUS_STOP:   (80, 200, 80),
    ZONE_TYPE_WRONG_WAY:  (255, 0, 255),
}


# -------------------------------------------------------------------
# Per-track violation state
# -------------------------------------------------------------------

class _TrackViolationState:
    """State for a single YOLO track ID inside a violation zone."""
    __slots__ = (
        "track_id", "zone_id", "first_seen_ts", "last_seen_ts",
        "first_box", "last_box", "positions", "logged_until",
        "plate", "plate_conf", "_last_anpr_attempt", "class_id",
        "vehicle_info",
    )

    def __init__(self, track_id, zone_id, timestamp, box, class_id=None):
        self.track_id = track_id
        self.zone_id = zone_id
        self.class_id = class_id  # Store vehicle class at first detection
        self.first_seen_ts = timestamp
        self.last_seen_ts = timestamp
        self.first_box = tuple(box)
        self.last_box = tuple(box)
        # Store (timestamp, center_x, center_y) for movement analysis
        cx, cy = _bbox_center(box)
        self.positions = [(timestamp, cx, cy)]
        self.logged_until = 0.0  # cooldown timestamp
        self.plate = None
        self.plate_conf = 0.0
        self._last_anpr_attempt = 0.0
        self.vehicle_info = None  # Vehicle identity with confidence scores


# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Module-level worker functions (avoid closure issues with Python 3.12)
# -------------------------------------------------------------------

def _anpr_fallback_worker(state, frame, box, cam_id, key_ref, cam_name):
    """Background ANPR worker — runs PaddleOCR as fallback."""
    try:
        from app.services.anpr import recognize_plate
        plate, conf, eng = recognize_plate(
            frame, box,
            region_hint=cam_name,
            seed=f"{cam_id}:{key_ref}:fallback",
        )
        if plate and conf >= 0.15 and eng != "simulated":
            state.plate = plate
            state.plate_conf = float(conf)
            print(f"[ANPR-ASYNC] Track {key_ref} | plate={plate} | conf={conf:.2f} | engine={eng}")
            from app.database import update_violation_plate
            update_violation_plate(cam_id, time.time() - 120, plate, conf, only_if_null=True)
    except Exception as e:
        print(f"[ANPR-ASYNC] Track {key_ref} | ERROR: {e}")


def _vehicle_id_worker(state, frame, box, cam_id, key_ref):
    """Background vehicle identification worker — uses AI vision."""
    try:
        from app.services.ai_ocr import ai_identify_vehicle
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        vehicle_crop = frame[y1:y2, x1:x2]
        if vehicle_crop.size > 0 and vehicle_crop.shape[0] > 30:
            info = ai_identify_vehicle(vehicle_crop)
            if info:
                state.vehicle_info = info
                notes_parts = []
                if info.get('vehicle_type') and info['vehicle_type'] not in ('Unknown', 'N/A', ''):
                    notes_parts.append(f"Jenis: {info['vehicle_type']}")
                if info.get('company') and info['company'] not in ('Private', 'Unknown', 'N/A', ''):
                    notes_parts.append(f"Perusahaan: {info['company']}")
                if info.get('make_model') and info['make_model'] not in ('Unknown', 'N/A', ''):
                    notes_parts.append(f"Merek/Model: {info['make_model']}")
                if info.get('color') and info['color'] not in ('Unknown', 'N/A', ''):
                    notes_parts.append(f"Warna: {info['color']}")
                if info.get('registration_area') and info['registration_area'] not in ('Unknown', 'N/A', ''):
                    notes_parts.append(f"Daerah: {info['registration_area']}")
                ai_plate = info.get('plate', '')
                if ai_plate and ai_plate != 'N/A' and 'unknown' not in ai_plate.lower():
                    notes_parts.append(f"Plat: {ai_plate}")
                    if not state.plate:
                        state.plate = ai_plate
                        state.plate_conf = 0.7
                        from app.database import update_violation_plate
                        update_violation_plate(cam_id, time.time() - 120, ai_plate, 0.7, only_if_null=True)
                if info.get('vehicle_type') and info['vehicle_type'] not in ('Unknown', 'N/A', ''):
                    from app.database import update_violation_field
                    update_violation_field(cam_id, time.time() - 120, "vehicle_class", info['vehicle_type'])
                notes_str = " | ".join(notes_parts)
                if notes_str:
                    from app.database import update_violation_field
                    update_violation_field(cam_id, time.time() - 120, "notes", notes_str)
                    print(f"[VEHICLE-ID] Track {key_ref} | {notes_str}")
    except Exception as e:
        print(f"[VEHICLE-ID] Track {key_ref} | ERROR: {e}")


# -------------------------------------------------------------------
# Enforcement Engine
# -------------------------------------------------------------------

class EnforcementEngine:
    """
    One engine per camera. Uses YOLO track IDs to monitor vehicles in zones.
    
    Logic:
    - On each inference frame, check which tracked vehicles are inside zones
    - Track how long each vehicle (by track ID) stays in a zone
    - If a vehicle is stationary for >= ILLEGAL_PARKING_MIN_SECONDS (60s), flag it
    - "Stationary" = the vehicle's center hasn't moved more than a small threshold
    """

    def __init__(self, camera_id: str, camera_name: str, lat=None, lng=None):
        self.camera_id = str(camera_id)
        self.camera_name = str(camera_name or "")
        self.lat = lat
        self.lng = lng
        self.zones = []
        self._zone_lock = threading.Lock()
        self._last_zone_load = 0.0
        self._zone_refresh_s = 30.0
        # Key: (track_id, zone_id) → _TrackViolationState
        self._tracked = {}
        self._last_violation_ids = []
        self._last_violation_lock = threading.Lock()
        # Frame buffer for smart capture: keep recent frames for best-quality ANPR
        # Stores (timestamp, frame) tuples, keeps last 5 frames
        self._frame_buffer = []
        self._frame_buffer_lock = threading.Lock()
        self._max_buffer_frames = 5

    # ---------- Zone management ----------

    def load_zones(self, force=False):
        now = time.time()
        if (not force) and (now - self._last_zone_load) < self._zone_refresh_s:
            return
        try:
            zones = get_zones_for_camera(self.camera_id, only_active=True) or []
        except Exception:
            zones = []
        with self._zone_lock:
            self.zones = zones
            self._last_zone_load = now

    def has_zones(self):
        with self._zone_lock:
            return bool(self.zones)

    def get_zones_snapshot(self):
        with self._zone_lock:
            return list(self.zones)

    def _scale_geometry(self, geom, zone_row, frame_w, frame_h):
        """Scale zone geometry from reference frame size to current frame size.
        
        Zones are drawn on the /zones page at a specific resolution (stored as
        frame_width/frame_height in DB). If the inference frame is a different
        resolution, we scale the polygon coordinates proportionally.
        """
        if not geom or frame_w <= 0 or frame_h <= 0:
            return geom
        ref_w = int(zone_row.get("frame_width") or 0)
        ref_h = int(zone_row.get("frame_height") or 0)
        # If reference dimensions not stored, assume zone was drawn at current frame size
        # (no scaling needed) — this is the safest default
        if ref_w <= 0 or ref_h <= 0:
            return geom
        if ref_w == frame_w and ref_h == frame_h:
            return geom
        sx = frame_w / float(ref_w)
        sy = frame_h / float(ref_h)
        if isinstance(geom, list) and geom and isinstance(geom[0], (list, tuple)):
            return [[p[0] * sx, p[1] * sy] for p in geom]
        if isinstance(geom, (list, tuple)) and len(geom) == 4:
            return [geom[0] * sx, geom[1] * sy, geom[2] * sx, geom[3] * sy]
        return geom

    # ---------- Core check ----------

    def check_frame(self, frame_bgr, tracks: dict, timestamp: float, is_inference_frame: bool = True):
        """
        Check tracked vehicles against violation zones.
        
        ONLY processes on inference frames (when YOLO actually ran).
        Uses track IDs from the YOLO tracker — same IDs that draw bounding boxes.
        """
        if not bool(app_config.VIOLATIONS_ENABLED):
            return []
        if not is_inference_frame:
            return []
        self.load_zones(force=False)
        if not self.zones:
            return []
        if not tracks:
            return []

        # Store frame in buffer for smart capture (best frame selection)
        if frame_bgr is not None and is_inference_frame:
            with self._frame_buffer_lock:
                # Store timestamp + frame copy (need copy to avoid reference issues)
                try:
                    self._frame_buffer.append((timestamp, frame_bgr.copy()))
                except Exception:
                    pass
                # Keep only last N frames
                if len(self._frame_buffer) > self._max_buffer_frames:
                    self._frame_buffer = self._frame_buffer[-self._max_buffer_frames:]

        frame_h, frame_w = 0, 0
        if frame_bgr is not None:
            frame_h, frame_w = frame_bgr.shape[:2]

        # Step 1: For each active YOLO track, check if it's inside any zone
        active_track_zones = set()
        vehicles_in_zone_count = 0

        for tid, t in tracks.items():
            box = t.get("box")
            if not box:
                continue
            
            # Only process tracks that were detected THIS frame
            track_age = abs(timestamp - float(t.get("last_seen") or 0))
            if track_age > 5.0:
                continue  # Stale track — not detected recently (relaxed from 2.0 for background cameras)
            
            bw = abs(box[2] - box[0])
            bh = abs(box[3] - box[1])
            box_area = bw * bh
            
            # SIZE FILTER: Filter out small/noisy detections
            # Reduced thresholds for distant CCTV cameras (elevated/top-down view)
            min_bw = 25
            min_bh = 20
            min_area = 800
            if bw < min_bw or bh < min_bh or box_area < min_area:
                continue
            
            # Aspect ratio check - reasonable range for vehicles
            aspect = bw / max(1, bh)
            if aspect < 0.2 or aspect > 8.0:  # Tighter range to filter noise
                continue

            cx, cy = _bbox_center(box)

            for zone in self.zones:
                geom = zone.get("geometry")
                if geom is None:
                    continue
                ztype = zone.get("zone_type")
                if ztype == ZONE_TYPE_BUS_STOP:
                    continue
                zone_id = int(zone["id"])
                scaled_geom = self._scale_geometry(geom, zone, frame_w, frame_h)
                
                # Check if center point is inside zone
                if _point_in_geometry(cx, cy, scaled_geom):
                    if ztype == ZONE_TYPE_BUSWAY:
                        # Busway: center is in zone — vehicle detected in busway!
                        # Check if it's a bus (allowed in busway)
                        track_cls = t.get("class_id")
                        box_w_check = abs(box[2] - box[0])
                        box_h_check = abs(box[3] - box[1])
                        is_bus = (track_cls == CLASS_BUS and box_w_check * box_h_check > 80000)
                        
                        if not is_bus:
                            # INSTANT BUSWAY VIOLATION — log immediately on first detection
                            busway_key = (tid, zone_id)
                            # Check cooldown: don't re-log same track
                            if busway_key in self._tracked:
                                st_existing = self._tracked[busway_key]
                                if timestamp < st_existing.logged_until:
                                    # Already logged, skip
                                    pass
                                else:
                                    print(f"[BUSWAY-DETECT] Track {tid} | center=({cx:.0f},{cy:.0f}) IN zone {zone_id} | box={box}")
                            else:
                                print(f"[BUSWAY-DETECT] Track {tid} | center=({cx:.0f},{cy:.0f}) IN zone {zone_id} | box={box}")
                    elif ztype == ZONE_TYPE_BICYCLE:
                        # Bicycle lane: ULTRA-LENIENT - only 1 anchor OR center in zone
                        # For instant detection of any vehicle in bicycle lane
                        anchor_hits = _count_points_in_geometry(_bbox_bottom_anchors(box), scaled_geom)
                        if anchor_hits < 1:
                            # Still allow if center is detected (already true at this point)
                            pass  # Center already checked above
                    key = (tid, zone_id)
                    active_track_zones.add(key)
                    vehicles_in_zone_count += 1

                    if key in self._tracked:
                        st = self._tracked[key]
                        st.last_seen_ts = timestamp
                        st.last_box = tuple(box)
                        bcx, bcy = _bbox_center(box)
                        st.positions.append((timestamp, bcx, bcy))
                        if len(st.positions) > 200:
                            st.positions = st.positions[-200:]
                        
                        # Early ANPR: try to read plate immediately when vehicle detected in zone
                        # SENSITIVE MODE: Start ANPR immediately, not after 3 seconds
                        # Use smart capture with deblurring for clearer OCR
                        if (st.plate is None
                            and frame_bgr is not None
                            and (timestamp - st._last_anpr_attempt) >= 2.0):  # Reduced from 8s to 2s
                            try:
                                # Smart capture: get enhanced crop for better OCR
                                enhanced_crop, _ = _capture_best_frame_for_plate(
                                    frame_bgr, box, timestamp, st.positions, self.camera_id, frame_buffer=self
                                )
                                _plate, _conf, _eng = None, 0.0, None
                                if enhanced_crop is not None:
                                    _plate, _conf, _eng = recognize_plate(
                                        enhanced_crop, None,
                                        region_hint=self.camera_name,
                                        seed=f"{self.camera_id}:{key}:smart",
                                    )
                                if not (_plate and _conf >= 0.15 and _eng != "simulated"):
                                    # Fallback: try original frame
                                    _plate, _conf, _eng = recognize_plate(
                                        frame_bgr, box,
                                        region_hint=self.camera_name,
                                        seed=f"{self.camera_id}:{key}:orig",
                                    )
                                st._last_anpr_attempt = timestamp
                                if _plate and _conf >= 0.15 and _eng != "simulated":
                                    st.plate = _plate
                                    st.plate_conf = float(_conf)
                                    # Update database with plate confidence
                                    try:
                                        from app.database import get_db_connection
                                        conn = get_db_connection(timeout_s=5)
                                        conn.execute(
                                            "UPDATE violations SET plate_text=?, plate_confidence=? WHERE camera_id=? AND timestamp > ? AND plate_text IS NULL ORDER BY id DESC LIMIT 1",
                                            (_plate, float(_conf), self.camera_id, time.time() - 60)
                                        )
                                        conn.commit()
                                        conn.close()
                                    except Exception:
                                        pass
                                    print(f"[ANPR-EARLY] Track {tid} | plate={_plate} | conf={_conf:.2f} | engine={_eng}")
                            except Exception:
                                st._last_anpr_attempt = timestamp
                    else:
                        # --- REJECT NON-VEHICLE CLASSES IMMEDIATELY ---
                        # This prevents portals, barriers, gates, signs, people, etc. from being tracked
                        track_cls = t.get("class_id")
                        
                        # Get all valid vehicle class IDs (from config)
                        import app.config as _cfg
                        valid_cls = set(_cfg.VEHICLE_CLASSES_COCO) | {v for v in _cfg.CLASS_MAPPING_COCO.values()}
                        
                        # If class_id is available, check if it's a valid vehicle class
                        if track_cls is not None and track_cls not in valid_cls:
                            # Class is known but not a vehicle - skip this detection
                            # Common false positives: traffic signs, barriers, gates, poles
                            continue
                        
                        # --- SIZE FILTER: Reject objects that are too small or wrong aspect ratio ---
                        # RAISED thresholds for bicycle lane to reduce false positives
                        bx1, by1, bx2, by2 = [int(float(v)) for v in box]
                        bw = abs(bx2 - bx1)
                        bh = abs(by2 - by1)
                        obj_area = bw * bh
                        obj_aspect = bw / max(1.0, bh)  # width / height
                        
                        # Minimum size threshold - RAISED to filter distant/noisy detections
                        if bw < 45 or bh < 35 or obj_area < 2500:  # Increased from 30x25, 1500
                            continue
                        
                        # Aspect ratio filter - reasonable range for vehicles
                        if obj_aspect < 0.2 or obj_aspect > 8.0:  # Tighter than 0.15-10.0
                            continue
                        
                        # --- PASSED ALL FILTERS: Create track state ---
                        self._tracked[key] = _TrackViolationState(tid, zone_id, timestamp, box, class_id=track_cls)

        # Step 2: Remove stale tracks (vehicle left or tracker lost it)
        to_remove = []

        # Periodic debug: log enforcement status every ~30 seconds
        if not hasattr(self, '_last_debug_ts'):
            self._last_debug_ts = 0.0
        if (timestamp - self._last_debug_ts) >= 30.0:
            self._last_debug_ts = timestamp
            n_tracks = len(tracks)
            n_tracked = len(self._tracked)
            n_zones = len(self.zones)
            
            # Debug: count bicycle zones specifically
            bike_zones = [z for z in self.zones if z.get("zone_type") == ZONE_TYPE_BICYCLE]
            
            print(f"[ENFORCE-STATUS] {self.camera_name} | "
                  f"frame={frame_w}x{frame_h} | zones={n_zones} | bike_zones={len(bike_zones)} | "
                  f"yolo_tracks={n_tracks} | in_zone={vehicles_in_zone_count} | "
                  f"tracked_states={n_tracked}")
            
            # Log all active tracks that are in bicycle zones
            for z in bike_zones:
                zid = int(z["id"])
                zname = z.get("name", zid)
                geom = z.get("geometry")
                if geom:
                    print(f"[ENFORCE-BICYCLE-DEBUG] Zone '{zname}' | geometry_points={len(geom)}")

        for key, st in self._tracked.items():
            if key not in active_track_zones:
                if (timestamp - st.last_seen_ts) > 10.0:
                    to_remove.append(key)
            else:
                # Remove tracks with extremely long dwell (permanent objects like poles)
                # Real illegal parking should be reported even if > 30 min
                # But if something is detected for > 30 min continuously, it's likely
                # a permanent fixture, not a vehicle
                dwell = st.last_seen_ts - st.first_seen_ts
                if dwell > 1800.0:  # > 30 minutes = permanent object
                    to_remove.append(key)
        for k in to_remove:
            del self._tracked[k]

        # Step 3: Check for violations
        created = []
        try:
            for tk, st in list(self._tracked.items()):
                if timestamp < st.logged_until:
                    continue

                try:
                    dwell = st.last_seen_ts - st.first_seen_ts
                    zone_id = st.zone_id
                    zone = next((z for z in self.zones if int(z["id"]) == zone_id), None)
                    if not zone:
                        continue
                    ztype = zone.get("zone_type")
                    geom = zone.get("geometry")
                    scaled_geom = self._scale_geometry(geom, zone, frame_w, frame_h) if geom is not None else None

                    # Final bbox size validation at violation time
                    bx1, by1, bx2, by2 = st.last_box
                    vw = abs(bx2 - bx1)
                    vh = abs(by2 - by1)
                    if vw < 60 or vh < 45 or (vw * vh) < 5000:
                        continue

                    track_data_now = tracks.get(st.track_id)
                    track_cls_id = st.class_id
                    if track_data_now and track_data_now.get("class_id") is not None:
                        track_cls_id = track_data_now.get("class_id")
                    if track_cls_id is None:
                        track_cls_id = CLASS_CAR

                    # Reject clearly non-vehicle boxes before they become persisted
                    # violations. This catches tall plang-like detections and flat puddles.
                    if not _looks_like_vehicle_box(st.last_box, track_cls_id):
                        continue

                    # Check if vehicle is truly stationary
                    is_stationary = self._is_stationary(st)

                    # Debug: log tracking progress for vehicles dwelling in zones
                    if dwell >= 10.0 and len(st.positions) >= 3 and int(dwell) % 15 == 0:
                        print(f"[ENFORCE] Track {tk[0]} in zone '{zone.get('name', zone_id)}' ({ztype}) | "
                              f"dwell={dwell:.0f}s | stationary={is_stationary} | "
                              f"bbox={vw}x{vh} | samples={len(st.positions)}")

                    violation_type = None
                    if ztype == ZONE_TYPE_NO_PARKING:
                        threshold = float(app_config.ILLEGAL_PARKING_MIN_SECONDS) + float(app_config.ILLEGAL_PARKING_GRACE_SECONDS)
                        if dwell >= threshold and is_stationary:
                            violation_type = VIOLATION_ILLEGAL_PARKING
                    elif ztype == ZONE_TYPE_BUSWAY:
                        # Busway violation: motorized vehicles in busway lane for sufficient duration.
                        # EXCEPT buses — buses ARE ALLOWED to use the busway lane.
                        #
                        # DETECTION STRATEGY (balanced responsiveness + accuracy):
                        # - Moving vehicles (speed > 40 px/s, traveled > 50px): 3s minimum
                        # - Slow-moving (20-40 px/s): 5s minimum
                        # - Stationary/nearly stationary (< 20 px/s): 8s minimum
                        # - Must have ≥ 3 consecutive zone detections before triggering
                        #
                        # Rationale: 3s is enough to confirm it's not a brief lane-crossing
                        # or detection jitter. 40 px/s = ~2.3 km/h in a 1920px frame,
                        # which filters out noise while catching real violators.

                        # Check class_id from multiple sources to avoid misclassification
                        if track_data_now:
                            current_cls = track_data_now.get("class_id")
                            if current_cls == CLASS_BUS:
                                track_cls_id = CLASS_BUS

                        # Size-based heuristic: Only VERY large vehicles are buses
                        # TransJakarta buses are typically 300+ px wide in CCTV
                        box_w = abs(st.last_box[2] - st.last_box[0])
                        box_h = abs(st.last_box[3] - st.last_box[1])
                        box_area = box_w * box_h

                        is_bus_sized = (
                            box_area > 80000 or
                            (box_w > 300 and box_h > 150) or
                            (box_w > 400)
                        )
                        if is_bus_sized and track_cls_id == CLASS_BUS:
                            continue  # Bus is allowed in busway — skip
                        if track_cls_id == CLASS_BUS and is_bus_sized:
                            continue

                        # Ghost detection filter: bbox must contain real vehicle texture
                        # Reject tiny or very dark/uniform patches (empty road, noise)
                        if frame_bgr is not None:
                            try:
                                bx1v, by1v, bx2v, by2v = [int(v) for v in st.last_box]
                                fh_v, fw_v = frame_bgr.shape[:2]
                                bx1v = max(0, bx1v); by1v = max(0, by1v)
                                bx2v = min(fw_v, bx2v); by2v = min(fh_v, by2v)
                                crop = frame_bgr[by1v:by2v, bx1v:bx2v]
                                if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                                    continue
                                # Check if there's enough contrast (real object, not empty road)
                                import cv2 as _cv2
                                gray = _cv2.cvtColor(crop, _cv2.COLOR_BGR2GRAY)
                                std_dev = float(gray.std())
                                if std_dev < 8.0:  # Very uniform patch = empty road/noise
                                    continue
                            except Exception:
                                pass

                        # Require minimum consecutive detections before triggering
                        # This filters out detection jitter / brief crossing
                        # Use config value
                        min_zone_hits = int(getattr(app_config, 'BUSWAY_MIN_ZONE_HITS', 3))
                        zone_hit_count = 0
                        check_len = min(len(st.positions), 8)
                        for i in range(check_len):
                            _, tcx, tcy = st.positions[-(i + 1)]
                            if _point_in_geometry(tcx, tcy, scaled_geom):
                                zone_hit_count += 1
                        if zone_hit_count < min_zone_hits:
                            # Not enough consecutive zone detections — skip
                            continue

                        # SPEED & DISTANCE ANALYSIS
                        speed_px_s = 0.0
                        total_distance = 0.0
                        speeds = []
                        if len(st.positions) >= 3:
                            try:
                                # Average speed over last 3 segments for stability (not just 1 frame)
                                for i in range(1, min(4, len(st.positions))):
                                    pts_prev = st.positions[-(i + 1)]
                                    pts_curr = st.positions[-i]
                                    dt = abs(pts_curr[0] - pts_prev[0])
                                    if dt > 0.001:
                                        seg_dist = math.hypot(pts_curr[1] - pts_prev[1], pts_curr[2] - pts_prev[2])
                                        speeds.append(seg_dist / dt)
                                if speeds:
                                    speed_px_s = sum(speeds) / len(speeds)
                                # Total distance traveled in zone
                                first_ts, first_cx, first_cy = st.positions[0]
                                last_ts, last_cx, last_cy = st.positions[-1]
                                total_distance = math.hypot(last_cx - first_cx, last_cy - first_cy)
                            except Exception:
                                pass

                        # Dwell requirement: adaptive based on speed, using config values
                        # - Fast moving (40+ px/s, traveled >40px): FAST threshold
                        # - Medium speed (20-40 px/s): MEDIUM threshold
                        # - Slow/stationary (<20 px/s): SLOW threshold
                        fast_speed = float(getattr(app_config, 'BUSWAY_FAST_SPEED_PX_S', 40.0))
                        min_dist = float(getattr(app_config, 'BUSWAY_MIN_TRAVEL_DIST_PX', 40.0))
                        if speed_px_s > fast_speed and total_distance > min_dist:
                            violation_threshold = float(getattr(app_config, 'BUSWAY_FAST_MIN_SECONDS', 3.0))
                        elif speed_px_s > 20.0:
                            violation_threshold = float(getattr(app_config, 'BUSWAY_MEDIUM_MIN_SECONDS', 5.0))
                        else:
                            violation_threshold = float(getattr(app_config, 'BUSWAY_SLOW_MIN_SECONDS', 8.0))

                        if dwell >= violation_threshold:
                            print(f"[BUSWAY-VIOLATION] Track {tk[0]} | dwell={dwell:.1f}s | "
                                  f"speed={speed_px_s:.0f}px/s | dist={total_distance:.0f}px | "
                                  f"zone_hits={zone_hit_count}/{check_len} | threshold={violation_threshold}s")
                            violation_type = VIOLATION_BUSWAY
                    elif ztype == ZONE_TYPE_BICYCLE:
                        # Bicycle lane: ANY motorized vehicle in bicycle zone
                        # LESS SENSITIVE MODE: Require longer dwell time to reduce false positives
                        # Check if vehicle center is in zone (more lenient than anchor)
                        in_zone = _point_in_geometry(cx, cy, scaled_geom) if scaled_geom else False
                        
                        # Also check bottom anchors (lenient - only 1 anchor needed)
                        anchor_hits = 0
                        if scaled_geom:
                            anchor_hits = _count_points_in_geometry(_bbox_bottom_anchors(box), scaled_geom)
                        
                        # Pass if center OR any anchor is in zone
                        if not in_zone and anchor_hits < 1:
                            continue
                        
                        # Check if we have an existing track for this vehicle
                        key = (tid, zone_id)
                        if key not in self._tracked:
                            continue  # Skip - not an existing tracked vehicle
                        
                        st = self._tracked[key]
                        
                        # Debug: log what we're detecting
                        track_cls = t.get("class_id", 0)
                        print(f"[ENFORCE-BICYCLE] Track {tid} | class={track_cls} | "
                              f"box={box} | center=({cx:.0f},{cy:.0f}) | in_zone={in_zone} | anchors={anchor_hits}")
                        
                        # BICYCLE LANE: Require minimum dwell time to avoid false positives
                        # Use config value (default 1.0s in sensitive mode, 2.0s otherwise)
                        bicycle_dwell_threshold = float(getattr(app_config, 'BICYCLE_LANE_MIN_SECONDS', 1.5))
                        dwell = timestamp - st.first_seen_ts
                        if dwell >= bicycle_dwell_threshold:
                            print(f"[ENFORCE-BICYCLE] Track {tk[0]} | dwell={dwell:.1f}s | "
                                  f"BICYCLE LANE VIOLATION | center_in_zone={in_zone} | anchors={anchor_hits}")
                            violation_type = VIOLATION_BICYCLE_LANE
                    elif ztype == ZONE_TYPE_WRONG_WAY:
                        # Wrong-way / lawan arah: vehicle moving AGAINST the allowed direction
                        if dwell >= float(app_config.WRONG_WAY_MIN_SECONDS) and len(st.positions) >= 8:
                            violation_type = VIOLATION_WRONG_WAY
                    
                    if violation_type is None:
                        continue  # Move to next track
                    
                    # --- Violation confirmed ---
                    print(f"[VIOLATION] {self.camera_name} | Track {tk[0]} | "
                          f"{violation_type} | dwell={dwell:.0f}s | bbox={vw}x{vh}")
                    
                    # Use CURRENT box position (from this inference frame) for evidence
                    # This ensures the box is at the vehicle's actual position
                    current_box = tracks.get(st.track_id, {}).get("box", st.last_box)
                    evidence_bbox = [int(v) for v in current_box]
                    
                    # Use smart capture: deblur the full frame before saving evidence
                    speed_px_s = 0.0
                    if st.positions and len(st.positions) >= 2:
                        try:
                            last_ts, last_cx, last_cy = st.positions[-1]
                            prev_ts, prev_cx, prev_cy = st.positions[-2]
                            dt = abs(last_ts - prev_ts)
                            if dt > 0.001:
                                dist = math.hypot(last_cx - prev_cx, last_cy - prev_cy)
                                speed_px_s = dist / dt
                        except Exception:
                            pass
                    
                    # Apply deblurring to full frame for evidence clarity
                    enhanced_frame = frame_bgr
                    if speed_px_s > 5.0:
                        try:
                            enhanced_frame = _apply_deblur(frame_bgr, evidence_bbox, speed_px_s) or frame_bgr
                        except Exception:
                            pass
                    
                    evidence_rel = _save_evidence(
                        enhanced_frame, evidence_bbox, self.camera_id, timestamp, violation_type
                    )

                    track_data_now = tracks.get(st.track_id)
                    cls_id = track_data_now.get("class_id") if track_data_now else st.class_id
                    if cls_id is None:
                        cls_id = st.class_id
                    vehicle_class = _class_to_str(cls_id)

                    # --- ANPR: Read plate BEFORE inserting violation ---
                    # Run synchronously but ONLY use AI vision (fast, ~1s).
                    # Skip slow PaddleOCR/EasyOCR path in sync mode.
                    if st.plate is None and frame_bgr is not None:
                        try:
                            import app.config as _anpr_cfg
                            if _anpr_cfg.AI_USE_FOR_ANPR and _anpr_cfg.AI_API_KEY:
                                from app.services.ai_ocr import ai_read_plate_from_image
                                x1v, y1v, x2v, y2v = [int(v) for v in current_box]
                                h_f, w_f = frame_bgr.shape[:2]
                                x1v, y1v = max(0, x1v), max(0, y1v)
                                x2v, y2v = min(w_f, x2v), min(h_f, y2v)
                                v_crop = frame_bgr[y1v:y2v, x1v:x2v]
                                if v_crop.size > 0 and v_crop.shape[0] > 30:
                                    ai_plate, ai_conf = ai_read_plate_from_image(v_crop)
                                    if ai_plate and ai_conf > 0.3:
                                        import re as _re_anpr
                                        raw = _re_anpr.sub(r'[^A-Z0-9]', '', ai_plate.upper())
                                        m = _re_anpr.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})$', raw)
                                        if m:
                                            st.plate = ai_plate
                                            st.plate_conf = max(ai_conf, 0.85)
                                            print(f"[ANPR-SYNC] Track {tk} | plate={ai_plate} | conf={st.plate_conf:.2f} | engine=ai_vision")
                        except Exception as e:
                            print(f"[ANPR-SYNC] Track {tk} | AI vision error: {e}")
                    
                    # Fallback: if AI vision failed, launch async PaddleOCR (won't block)
                    if st.plate is None and frame_bgr is not None:
                        import threading
                        _fb_key = f"_anpr_fb_{tk}"
                        if not getattr(self, _fb_key, False):
                            setattr(self, _fb_key, True)
                            _args = (st, frame_bgr.copy(), list(current_box), self.camera_id, str(tk), self.camera_name)
                            threading.Thread(target=_anpr_fallback_worker, args=_args, daemon=True).start()

                    # --- Vehicle identification via AI (async, won't block insert) ---
                    if frame_bgr is not None and not getattr(st, 'vehicle_info', None):
                        import threading
                        _vk = f"_vid_running_{tk}"
                        if not getattr(self, _vk, False):
                            setattr(self, _vk, True)
                            _args2 = (st, frame_bgr.copy(), list(current_box), self.camera_id, str(tk))
                            threading.Thread(target=_vehicle_id_worker, args=_args2, daemon=True).start()

                    # --- Plate-based deduplication ---
                    # If we have a plate, check if this same plate was already logged recently
                    if st.plate:
                        try:
                            from app.database import _execute, get_db_connection
                            conn = get_db_connection(timeout_s=2)
                            try:
                                c = _execute(conn,
                                    "SELECT id FROM violations WHERE camera_id=? AND plate_text=? AND timestamp > ? LIMIT 1",
                                    (str(self.camera_id), st.plate, timestamp - float(app_config.VIOLATION_COOLDOWN_SECONDS))
                                )
                                existing = c.fetchone()
                                if existing:
                                    print(f"[DEDUP] Skip: plate {st.plate} already logged at {self.camera_name}")
                                    st.logged_until = timestamp + float(app_config.VIOLATION_COOLDOWN_SECONDS)
                                    continue
                            finally:
                                conn.close()
                        except Exception:
                            pass  # If dedup check fails, proceed with insert

                    try:
                        vid = insert_violation(
                            camera_id=self.camera_id, camera_name=self.camera_name,
                            violation_type=violation_type, zone_type=ztype,
                            timestamp=timestamp, duration_s=float(dwell), zone_id=zone_id,
                            vehicle_class=vehicle_class, plate_text=st.plate,
                            plate_confidence=float(st.plate_conf or 0),
                            bbox=evidence_bbox, evidence_path=evidence_rel,
                            lat=float(self.lat) if self.lat else None,
                            lng=float(self.lng) if self.lng else None,
                            status="pending",
                        )
                        print(f"[VIOLATION DB] id={vid} | {vehicle_class} | plate={st.plate} | bbox={vw}x{vh}")
                    except Exception as e:
                        print(f"[VIOLATION ERROR] {e}")
                        vid = None

                    st.logged_until = timestamp + float(app_config.VIOLATION_COOLDOWN_SECONDS)
                    record = {
                        "id": vid, "timestamp": timestamp, "violation_type": violation_type,
                        "zone_type": ztype, "zone_id": zone_id, "zone_name": zone.get("name"),
                        "bbox": evidence_bbox, "plate_text": st.plate, "plate_confidence": st.plate_conf,
                        "duration_s": float(dwell), "camera_id": self.camera_id,
                        "camera_name": self.camera_name, "evidence_path": evidence_rel,
                        "vehicle_class": vehicle_class,
                        "vehicle_info": st.vehicle_info,
                    }
                    created.append(record)
                    with self._last_violation_lock:
                        self._last_violation_ids.append(record)
                        if len(self._last_violation_ids) > 20:
                            self._last_violation_ids = self._last_violation_ids[-20:]
                
                except Exception as e:
                    print(f"[ENFORCE] {self.camera_name} | Track error: {e}")
        
        except Exception as e:
            print(f"[ENFORCE] {self.camera_name} | Violation loop error: {e}")

        return created

    def _is_stationary(self, st: _TrackViolationState) -> bool:
        """
        Determine if a tracked vehicle is TRULY stationary.
        
        A vehicle is stationary if:
        1. We have enough position samples (at least 3)
        2. The center has NOT moved more than the configured displacement threshold
        3. Average frame-to-frame movement is <= 12px (YOLO jitter)
        
        These thresholds account for CCTV stream jitter and YOLO bbox noise,
        especially at night when detection is less stable.
        """
        if len(st.positions) < 3:
            return False  # Not enough data

        # Check max displacement from first position
        first_ts, first_cx, first_cy = st.positions[0]
        max_disp = 0.0
        for (ts, cx, cy) in st.positions:
            disp = math.hypot(cx - first_cx, cy - first_cy)
            max_disp = max(max_disp, disp)

        max_disp_threshold = min(60.0, float(getattr(app_config, "STATIC_MOVEMENT_PX", 60.0) or 60.0))

        # If vehicle moved more than the configured threshold from start,
        # it's NOT stationary.
        if max_disp > max_disp_threshold:
            return False

        # Check frame-to-frame movements
        movements = []
        for i in range(1, len(st.positions)):
            _, prev_cx, prev_cy = st.positions[i - 1]
            _, cur_cx, cur_cy = st.positions[i]
            movements.append(math.hypot(cur_cx - prev_cx, cur_cy - prev_cy))

        if not movements:
            return False

        avg_movement = sum(movements) / len(movements)

        # Avg must be <= 12px (accounts for YOLO jitter at night)
        return avg_movement <= 12.0

    def get_recent_local_violations(self, limit=10):
        with self._last_violation_lock:
            return list(self._last_violation_ids[-int(limit):])

    def _check_wrong_way(self, st, zone) -> bool:
        """Check if vehicle is TRULY moving against the allowed direction.
        
        VERY STRICT criteria to avoid false positives:
        1. Vehicle must be MOVING significantly (not parked/slow)
        2. Must have traveled far enough to determine direction reliably
        3. Direction must be CONSISTENTLY opposite (>160°) across multiple segments
        4. Must have enough data points (at least 8 positions over 6+ seconds)
        """
        # Need substantial tracking data
        if len(st.positions) < 8:
            return False
        
        # Must have been tracked for at least 6 seconds
        elapsed = st.positions[-1][0] - st.positions[0][0]
        if elapsed < 6.0:
            return False
        
        # FIRST: Check if vehicle is actually MOVING
        if self._is_stationary(st):
            return False
        
        # Calculate overall displacement
        first_ts, first_cx, first_cy = st.positions[0]
        last_ts, last_cx, last_cy = st.positions[-1]
        
        dx = last_cx - first_cx
        dy = last_cy - first_cy
        distance = math.hypot(dx, dy)
        
        # Must have moved at least 150px (very significant movement)
        if distance < 150.0:
            return False
        
        # Must be moving at decent speed (>20 px/sec)
        speed = distance / max(0.1, elapsed)
        if speed < 20.0:
            return False
        
        # Get allowed direction from zone config
        allowed_angle = self._get_zone_direction(zone)
        if allowed_angle is None:
            return False  # NO direction configured → NEVER flag
        
        # Calculate vehicle's overall direction
        vehicle_angle = math.degrees(math.atan2(dy, dx)) % 360
        
        # Angle difference
        diff = abs(vehicle_angle - allowed_angle)
        if diff > 180:
            diff = 360 - diff
        
        # VERY STRICT: only flag if >160° (nearly perfectly opposite)
        if diff <= 160.0:
            return False
        
        # CONSISTENCY CHECK: split into 3+ segments, ALL must show wrong direction
        n = len(st.positions)
        segment_size = max(3, n // 4)
        wrong_segments = 0
        total_segments = 0
        
        for seg_start in range(0, n - segment_size, segment_size):
            seg_end = min(seg_start + segment_size, n - 1)
            _, sx, sy = st.positions[seg_start]
            _, ex, ey = st.positions[seg_end]
            seg_dx = ex - sx
            seg_dy = ey - sy
            seg_dist = math.hypot(seg_dx, seg_dy)
            if seg_dist < 20.0:
                continue  # Segment too short
            
            seg_angle = math.degrees(math.atan2(seg_dy, seg_dx)) % 360
            seg_diff = abs(seg_angle - allowed_angle)
            if seg_diff > 180:
                seg_diff = 360 - seg_diff
            
            total_segments += 1
            if seg_diff > 140.0:
                wrong_segments += 1
        
        # ALL valid segments must confirm wrong-way (not just majority)
        if total_segments < 2:
            return False
        if wrong_segments < total_segments:
            return False  # Even ONE segment in correct direction = not wrong-way
        
        return True

    def _get_zone_direction(self, zone) -> float:
        """Extract allowed direction (degrees) from zone configuration.
        
        Looks for direction in:
        1. zone["direction"] field
        2. zone["notes"] containing "direction:XXX" or "arah:XXX"
        3. zone["geometry"]["direction"] if geometry is a dict
        """
        # Direct field
        d = zone.get("direction")
        if d is not None:
            try:
                return float(d) % 360
            except (ValueError, TypeError):
                pass
        
        # From notes field: "direction:90" or "arah:180"
        notes = str(zone.get("notes") or "")
        import re as _re
        m = _re.search(r'(?:direction|arah|dir)\s*[:=]\s*(\d+)', notes, _re.IGNORECASE)
        if m:
            try:
                return float(m.group(1)) % 360
            except (ValueError, TypeError):
                pass
        
        # From geometry dict
        geom = zone.get("geometry")
        if isinstance(geom, dict):
            d = geom.get("direction")
            if d is not None:
                try:
                    return float(d) % 360
                except (ValueError, TypeError):
                    pass
        
        # Default: assume right-to-left traffic (180°) if not specified
        # This is common for Indonesian roads (left-hand traffic)
        return None

    # ---------- Frame annotation (OSD) ----------

    def draw_overlay(self, frame, active_violations):
        """Draw zone outlines + violation markers + direction arrows on the frame."""
        if frame is None:
            return frame
        zones = self.get_zones_snapshot()
        frame_h, frame_w = frame.shape[:2]

        for z in zones:
            geom = z.get("geometry")
            pts = _geometry_to_polyline_for_draw(geom)
            if pts is None or len(pts) < 2:
                continue
            ztype = z.get("zone_type")
            color = _VIOLATION_COLORS.get(ztype, (200, 200, 200))
            try:
                # Scale geometry if needed
                scaled_geom = self._scale_geometry(geom, z, frame_w, frame_h)
                scaled_pts = _geometry_to_polyline_for_draw(scaled_geom)
                if scaled_pts is None:
                    scaled_pts = pts
                
                cv2.polylines(frame, [scaled_pts], isClosed=True, color=color, thickness=2)
                label = f"{ztype.upper().replace('_', ' ')}"
                lp = tuple(int(v) for v in scaled_pts[0])
                cv2.putText(frame, label, (lp[0], max(15, lp[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                
                # Draw direction arrow for wrong_way zones
                if ztype == ZONE_TYPE_WRONG_WAY:
                    direction = self._get_zone_direction(z)
                    if direction is not None:
                        # Calculate center of polygon
                        cx = int(np.mean(scaled_pts[:, 0]))
                        cy = int(np.mean(scaled_pts[:, 1]))
                        # Draw arrow showing ALLOWED direction
                        arrow_len = 40
                        rad = math.radians(direction)
                        ax = int(cx + arrow_len * math.cos(rad))
                        ay = int(cy + arrow_len * math.sin(rad))
                        cv2.arrowedLine(frame, (cx, cy), (ax, ay), color, 2, tipLength=0.35)
                        cv2.putText(frame, "ARAH BENAR", (cx - 40, cy - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            except Exception:
                pass

        for v in (active_violations or []):
            bb = v.get("bbox")
            if not bb:
                continue
            x1, y1, x2, y2 = [int(b) for b in bb]
            try:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                title = _violation_short_label(v.get("violation_type"))
                plate = v.get("plate_text") or ""
                label_text = title
                if plate:
                    label_text += f" | {plate}"
                cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x1 + len(label_text) * 9 + 12, y1), (0, 0, 160), -1)
                cv2.putText(frame, label_text, (x1 + 6, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
            except Exception:
                pass
        return frame


# -------------------------------------------------------------------
# Smart Capture: Deblurring & Enhancement
# -------------------------------------------------------------------

def _estimate_blur_score(img_gray) -> float:
    """Estimate blur level using Laplacian variance. Higher = sharper."""
    try:
        if img_gray is None or img_gray.size == 0:
            return 0.0
        blur = cv2.Laplacian(img_gray, cv2.CV_64F)
        return float(blur.var())
    except Exception:
        return 0.0


def _apply_deblur(img_bgr, bbox, speed_estimate=0.0):
    """Apply deblurring pipeline optimized for fast-moving vehicles.
    
    Args:
        img_bgr: Full frame BGR image
        bbox: (x1, y1, x2, y2) of vehicle OR None for full-frame enhancement
        speed_estimate: estimated movement speed (px/s), 0 = stationary
    
    Returns:
        Enhanced image with deblurring applied
    """
    try:
        h, w = img_bgr.shape[:2]
        
        # If bbox provided, work on the cropped area and return enhanced crop
        # If bbox is None, apply to full frame
        if bbox is None:
            # Full frame enhancement (for evidence saving)
            crop = img_bgr.copy()
            bbox_coords = [0, 0, w, h]
        else:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 20 or y2 - y1 < 20:
                return img_bgr
            crop = img_bgr[y1:y2, x1:x2].copy()
            bbox_coords = [x1, y1, x2, y2]

        # Step 1: Sharpening for all captures (enhances text/edges regardless of motion)
        sharpened, _ = preprocess_sharpen(crop, strength=1.8)

        # Step 2: Adaptive CLAHE for contrast enhancement
        enhanced, _ = preprocess_adaptive_clahe(sharpened, clip_limit=3.0, tile_size=(8, 8))

        # Step 3: Heavy deblurring for fast-moving vehicles (motion blur)
        if speed_estimate > 3.0:
            # Vehicle is moving fast — apply motion deblurring
            denoised, _ = preprocess_denoise(enhanced, h=5, hColor=5)
            blur_sigma = min(int(speed_estimate * 0.3), 15)
            if blur_sigma > 1:
                blurred = cv2.GaussianBlur(denoised, (0, 0), blur_sigma)
                deblurred = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
                enhanced = np.clip(deblurred, 0, 255).astype(np.uint8)
            # Final sharpening pass after deblurring
            final, _ = preprocess_sharpen(enhanced, strength=2.0)
            return final

        return enhanced
    except Exception as e:
        print(f"[DEBLUR] Error: {e}")
        return img_bgr


def _capture_best_frame_for_plate(frame_bgr, bbox, timestamp, track_positions, camera_id, frame_buffer=None):
    """Capture the clearest frame for plate recognition from frame buffer.
    
    Strategy: Use the best available frame for ANPR. For fast-moving vehicles,
    we estimate speed from track positions and apply appropriate deblurring.
    If frame_buffer is provided, select the sharpest frame from the buffer.
    
    Returns:
        Best crop image for OCR processing
    """
    if frame_bgr is None:
        return None, 0.0

    # If buffer available, find sharpest frame from recent history
    best_frame = frame_bgr
    best_blur_score = -1.0
    
    if frame_buffer is not None and len(frame_buffer) > 0:
        with frame_buffer._frame_buffer_lock:
            buffer_frames = list(frame_buffer._frame_buffer)
        
        for buf_ts, buf_frame in buffer_frames:
            try:
                gray = cv2.cvtColor(buf_frame, cv2.COLOR_BGR2GRAY)
                blur = _estimate_blur_score(gray)
                if blur > best_blur_score:
                    best_blur_score = blur
                    best_frame = buf_frame
            except Exception:
                continue
        
        if best_blur_score > 0 and best_frame is not frame_bgr:
            print(f"[SMART-CAPTURE] Selected sharper frame from buffer (blur={best_blur_score:.1f})")
    
    if best_frame is None:
        return None, 0.0

    try:
        # If no bbox provided, apply deblurring to full frame and return it
        if bbox is None:
            # No bbox = caller already extracted the crop
            return best_frame, best_blur_score if best_blur_score > 0 else 0.0
        
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = best_frame.shape[:2]

        # Ensure bbox is valid
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None, 0.0

        # Estimate movement speed from track positions
        speed_px_per_frame = 0.0
        if track_positions and len(track_positions) >= 2:
            try:
                last_ts, last_cx, last_cy = track_positions[-1]
                prev_ts, prev_cx, prev_cy = track_positions[-2]
                dt = abs(last_ts - prev_ts)
                if dt > 0.001:
                    dist = math.hypot(last_cx - prev_cx, last_cy - prev_cy)
                    speed_px_per_frame = dist / dt  # px per second
                    speed_px_per_frame = min(speed_px_per_frame, 200.0)  # Cap at 200px/s
            except Exception:
                speed_px_per_frame = 0.0

        # Apply deblurring pipeline
        enhanced_crop = _apply_deblur(best_frame, bbox, speed_px_per_frame)

        if enhanced_crop is None:
            return None, 0.0

        # Score the enhanced image
        try:
            gray_enhanced = cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2GRAY)
            final_blur_score = _estimate_blur_score(gray_enhanced)
        except Exception:
            final_blur_score = 0.0

        return enhanced_crop, final_blur_score

    except Exception as e:
        print(f"[SMART-CAPTURE] Error: {e}")
        return None, 0.0


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _class_to_str(cls_id):
    if cls_id is None:
        return "car"  # Default to car if class unknown (most common vehicle)
    try:
        cls_id = int(cls_id)
    except Exception:
        return "car"
    if cls_id == CLASS_CAR:
        return "car"
    if cls_id == CLASS_MOTORCYCLE:
        return "motorcycle"
    if cls_id == CLASS_BUS:
        return "bus"
    return "car"  # Default to car for any unrecognized class


def _violation_short_label(vt):
    if vt == VIOLATION_ILLEGAL_PARKING:
        return "ILLEGAL PARKING"
    if vt == VIOLATION_BUSWAY:
        return "BUSWAY VIOLATION"
    if vt == VIOLATION_BICYCLE_LANE:
        return "BIKE LANE VIOLATION"
    if vt == VIOLATION_PICKUP_DROPOFF:
        return "ILLEGAL STOP"
    if vt == VIOLATION_WRONG_WAY:
        return "WRONG WAY"
    return "VIOLATION"


def _save_evidence(frame_bgr, bbox, camera_id, timestamp, violation_type):
    """Save evidence snapshot with the violation bbox highlighted.
    
    Uses FULL RESOLUTION frame from camera capture (not the resized inference frame).
    This ensures evidence images are clear for operator review and ANPR/OCR.
    
    Includes:
    - Full frame with red bounding box around violator
    - Violation type label
    - Timestamp overlay
    - Camera name overlay
    - Zoomed inset of the vehicle (top-right corner) for clarity
    """
    if frame_bgr is None:
        return None
    
    # Validate frame is not empty/black
    try:
        if frame_bgr.size == 0 or frame_bgr.shape[0] < 100 or frame_bgr.shape[1] < 100:
            print(f"[EVIDENCE] Skipped: frame too small ({frame_bgr.shape})")
            return None
        # Check if frame is mostly black (no signal)
        mean_val = float(frame_bgr.mean())
        if mean_val < 5.0:
            print(f"[EVIDENCE] Skipped: frame is black (mean={mean_val:.1f})")
            return None
    except Exception:
        return None

    try:
        dt = datetime.datetime.fromtimestamp(float(timestamp))
        day_dir = os.path.join(EVIDENCE_DIR, dt.strftime("%Y-%m-%d"))
        os.makedirs(day_dir, exist_ok=True)
        uid = uuid.uuid4().hex[:10]
        fname = f"{dt.strftime('%H%M%S')}_{camera_id}_{violation_type}_{uid}.jpg"
        full_path = os.path.join(day_dir, fname)

        overlay = frame_bgr.copy()
        fh, fw = overlay.shape[:2]

        try:
            x1, y1, x2, y2 = [int(b) for b in bbox]
            # Clamp to frame bounds
            x1 = max(0, min(fw - 1, x1))
            y1 = max(0, min(fh - 1, y1))
            x2 = max(x1 + 1, min(fw, x2))
            y2 = max(y1 + 1, min(fh, y2))

            # Expand bbox downward to capture plate area (often cut off by YOLO)
            # Plates are typically at bottom 20-30% of vehicle
            bbox_h = y2 - y1
            bbox_w = x2 - x1
            expand_down = int(bbox_h * 0.3)  # Extend 30% below detected box
            expand_sides = int(bbox_w * 0.1)  # Slight horizontal expansion
            x1 = max(0, x1 - expand_sides)
            y2 = min(fh, y2 + expand_down)
            x2 = min(fw, x2 + expand_sides)

            # Draw main bounding box (thick red)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Violation label with background
            label = _violation_short_label(violation_type)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(overlay, (x1, max(0, y1 - 30)), (x1 + label_size[0] + 10, y1), (0, 0, 180), -1)
            cv2.putText(overlay, label, (x1 + 5, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Timestamp (bottom-left)
            ts_text = dt.strftime("%Y/%m/%d %H:%M:%S")
            cv2.putText(overlay, ts_text, (10, fh - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # Camera name (bottom-right)
            cam_label = str(camera_id).replace("_", " ").upper()
            cam_size = cv2.getTextSize(cam_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.putText(overlay, cam_label, (fw - cam_size[0] - 10, fh - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # Zoomed inset of the vehicle (top-right corner)
            # Larger inset for operator clarity (300px wide or 1/3 of frame)
            bw = x2 - x1
            bh = y2 - y1
            if bw >= 50 and bh >= 50:
                # Crop the vehicle area with generous padding for context (especially below for plate)
                pad_x = 50
                pad_top = 30
                pad_bottom = int(bh * 0.4)  # Extra padding below for plate area
                crop_x1 = max(0, x1 - pad_x)
                crop_y1 = max(0, y1 - pad_top)
                crop_x2 = min(fw, x2 + pad_x)
                crop_y2 = min(fh, y2 + pad_bottom)
                vehicle_crop = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

                if vehicle_crop.size > 0:
                    # Resize to larger inset (300px wide or 1/3 of frame width)
                    inset_w = min(300, fw // 3)
                    scale = inset_w / max(1, vehicle_crop.shape[1])
                    inset_h = int(vehicle_crop.shape[0] * scale)
                    inset_h = min(inset_h, fh // 3)
                    if inset_h > 30 and inset_w > 30:
                        inset = cv2.resize(vehicle_crop, (inset_w, inset_h), interpolation=cv2.INTER_LANCZOS4)
                        # Place in top-right corner with border
                        ix = fw - inset_w - 10
                        iy = 10
                        # Draw border
                        cv2.rectangle(overlay, (ix - 2, iy - 2),
                                      (ix + inset_w + 2, iy + inset_h + 2), (0, 0, 255), 2)
                        overlay[iy:iy + inset_h, ix:ix + inset_w] = inset
                        cv2.putText(overlay, "ZOOM", (ix + 4, iy + 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        except Exception as e:
            print(f"[EVIDENCE] Overlay error: {e}")

        quality = max(90, int(app_config.EVIDENCE_JPEG_QUALITY))
        cv2.imwrite(full_path, overlay, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        print(f"[EVIDENCE] Saved: {fname} | resolution={fw}x{fh} | quality={quality}")
        return os.path.relpath(full_path, start=os.path.dirname(EVIDENCE_DIR)).replace("\\", "/")
    except Exception as e:
        print(f"[EVIDENCE] Save error: {e}")
        return None


def auto_classify_crm(description: str) -> str | None:
    """Keyword-based classifier for CRM/public complaint free-text reports."""
    if not description:
        return None
    t = str(description).lower()
    if any(k in t for k in ("busway", "transjakarta", "trans jakarta", "jalur bus")):
        return VIOLATION_BUSWAY
    if any(k in t for k in ("sepeda", "bicycle", "jalur sepeda", "bike lane")):
        return VIOLATION_BICYCLE_LANE
    if any(k in t for k in ("parkir liar", "parkir sembarangan", "illegal park", "parkir ilegal", "parkir di trotoar", "parked illegally")):
        return VIOLATION_ILLEGAL_PARKING
    if any(k in t for k in ("ngetem", "berhenti sembarangan", "angkot stop", "pickup drop")):
        return VIOLATION_PICKUP_DROPOFF
    return None
