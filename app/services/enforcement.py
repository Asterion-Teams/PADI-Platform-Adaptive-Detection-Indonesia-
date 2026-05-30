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
)
import app.config as app_config
from app.database import (
    get_zones_for_camera,
    insert_violation,
)
from app.services.anpr import recognize_plate


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
    x1, y1, x2, y2 = box
    bw = abs(x2 - x1)
    bh = abs(y2 - y1)
    if bw <= 0 or bh <= 0:
        return False

    aspect = bw / max(1.0, bh)
    if class_id == CLASS_MOTORCYCLE:
        return 0.28 <= aspect <= 3.5
    if class_id == CLASS_BUS:
        return 0.55 <= aspect <= 5.0
    return 0.45 <= aspect <= 4.5


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
            if track_age > 2.0:
                continue  # Stale track — not detected recently
            
            bw = abs(box[2] - box[0])
            bh = abs(box[3] - box[1])
            box_area = bw * bh
            
            # SIZE FILTER: Filter out very small detections (noise/distant objects)
            if bw < 50 or bh < 35 or box_area < 3000:
                continue
            
            # Aspect ratio check
            aspect = bw / max(1, bh)
            if aspect < 0.2 or aspect > 8.0:
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
                    if ztype in {ZONE_TYPE_BUSWAY, ZONE_TYPE_BICYCLE}:
                        # Dynamic-lane violations should sit on the lane surface.
                        # Requiring at least 2 lower anchor points avoids plang /
                        # side fixtures that overlap the polygon only at the middle.
                        anchor_hits = _count_points_in_geometry(_bbox_bottom_anchors(box), scaled_geom)
                        if anchor_hits < 2:
                            continue
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
                        
                        # Early ANPR: try to read plate while vehicle is dwelling
                        # (before violation threshold is reached)
                        # Only attempt every ~8 seconds to avoid overloading OCR
                        if (st.plate is None 
                            and frame_bgr is not None
                            and (st.last_seen_ts - st.first_seen_ts) >= 3.0
                            and (timestamp - st._last_anpr_attempt) >= 8.0):
                            try:
                                _plate, _conf, _eng = recognize_plate(
                                    frame_bgr, box,
                                    region_hint=self.camera_name,
                                    seed=f"{self.camera_id}:{key}",
                                )
                                st._last_anpr_attempt = timestamp
                                if _plate and _conf >= 0.15 and _eng != "simulated":
                                    st.plate = _plate
                                    st.plate_conf = float(_conf)
                                    print(f"[ANPR-EARLY] Track {tid} | plate={_plate} | conf={_conf:.2f} | engine={_eng}")
                            except Exception:
                                st._last_anpr_attempt = timestamp
                    else:
                        track_cls = t.get("class_id")
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
            print(f"[ENFORCE-STATUS] {self.camera_name} | "
                  f"frame={frame_w}x{frame_h} | zones={n_zones} | "
                  f"yolo_tracks={n_tracks} | in_zone={vehicles_in_zone_count} | "
                  f"tracked_states={n_tracked}")

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
        for key, st in list(self._tracked.items()):
            if timestamp < st.logged_until:
                continue

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
                print(f"[ENFORCE] Track {key[0]} in zone '{zone.get('name', zone_id)}' ({ztype}) | "
                      f"dwell={dwell:.0f}s | stationary={is_stationary} | "
                      f"bbox={vw}x{vh} | samples={len(st.positions)}")

            violation_type = None
            if ztype == ZONE_TYPE_NO_PARKING:
                threshold = float(app_config.ILLEGAL_PARKING_MIN_SECONDS) + float(app_config.ILLEGAL_PARKING_GRACE_SECONDS)
                if dwell >= threshold and is_stationary:
                    violation_type = VIOLATION_ILLEGAL_PARKING
            elif ztype == ZONE_TYPE_BUSWAY:
                # Busway violation: ANY vehicle in busway zone for > threshold
                # EXCEPT buses — buses are ALLOWED to use the busway lane
                # Does NOT require stationary — moving through busway is also a violation
                
                # Check class_id from multiple sources to avoid misclassification
                if track_data_now:
                    current_cls = track_data_now.get("class_id")
                    if current_cls == CLASS_BUS:
                        track_cls_id = CLASS_BUS
                
                # Size-based heuristic: Buses are MUCH larger than cars/motorcycles
                # Any large vehicle in a BUSWAY lane is almost certainly a bus
                # Thresholds are intentionally low to catch buses from various angles
                box_w = abs(st.last_box[2] - st.last_box[0])
                box_h = abs(st.last_box[3] - st.last_box[1])
                box_area = box_w * box_h
                
                # A bus in frame is typically:
                # - Area > 20000px (even from far away / angled view)
                # - Width > 120px OR Height > 120px (bus is large in at least one dimension)
                # - Much larger than a motorcycle (area < 8000) or car (area < 15000 typically)
                is_large_vehicle = (
                    box_area > 20000 or
                    (box_w > 120 and box_h > 120) or
                    (box_w > 200) or
                    (box_h > 200)
                )
                if is_large_vehicle:
                    track_cls_id = CLASS_BUS
                
                if track_cls_id == CLASS_BUS:
                    continue  # Bus is allowed in busway — skip
                if scaled_geom is not None:
                    anchor_hits = _count_points_in_geometry(_bbox_bottom_anchors(st.last_box), scaled_geom)
                    if anchor_hits < 2:
                        continue
                motion_span = _track_motion_span(st)
                if motion_span < 12.0 and not _has_vehicle_texture(frame_bgr, st.last_box):
                    continue
                if is_stationary and dwell < max(4.0, float(app_config.DYNAMIC_LANE_MIN_SECONDS) + 2.0):
                    continue
                if dwell >= float(app_config.DYNAMIC_LANE_MIN_SECONDS):
                    violation_type = VIOLATION_BUSWAY
            elif ztype == ZONE_TYPE_BICYCLE:
                # Bicycle lane: ANY motorized vehicle in bicycle zone for > threshold
                if scaled_geom is not None:
                    anchor_hits = _count_points_in_geometry(_bbox_bottom_anchors(st.last_box), scaled_geom)
                    if anchor_hits < 2:
                        continue
                motion_span = _track_motion_span(st)
                if motion_span < 12.0 and not _has_vehicle_texture(frame_bgr, st.last_box):
                    continue
                if is_stationary and dwell < max(4.0, float(app_config.DYNAMIC_LANE_MIN_SECONDS) + 2.0):
                    continue
                if dwell >= float(app_config.DYNAMIC_LANE_MIN_SECONDS):
                    violation_type = VIOLATION_BICYCLE_LANE
            elif ztype == ZONE_TYPE_WRONG_WAY:
                # Wrong-way / lawan arah: vehicle moving AGAINST the allowed direction
                # Zone stores "allowed_direction" as angle (degrees, 0=right, 90=down, etc.)
                # If vehicle's movement direction differs by > 150° from allowed, it's wrong-way
                # Require at least 5 positions and 4s dwell to avoid flagging turning vehicles
                if dwell >= float(app_config.WRONG_WAY_MIN_SECONDS) and len(st.positions) >= 5:
                    is_wrong = self._check_wrong_way(st, zone)
                    if is_wrong:
                        violation_type = VIOLATION_WRONG_WAY

            if violation_type is None:
                continue

            # --- Violation confirmed ---
            print(f"[VIOLATION] {self.camera_name} | Track {key[0]} | "
                  f"{violation_type} | dwell={dwell:.0f}s | bbox={vw}x{vh}")

            # ANPR — aggressive plate reading with multiple attempts
            if st.plate is None and frame_bgr is not None:
                try:
                    fh, fw = frame_bgr.shape[:2]
                    bx1, by1, bx2, by2 = [int(v) for v in st.last_box]
                    box_w = bx2 - bx1
                    box_h = by2 - by1
                    
                    # For small vehicles (far from camera), use much larger crop area
                    # Small = bbox < 150px in either dimension
                    if box_w < 150 or box_h < 150:
                        # Very aggressive padding (100%) for small vehicles
                        pad_x = max(box_w, 80)
                        pad_y = max(box_h, 80)
                    else:
                        # Normal padding (50%) for larger vehicles
                        pad_x = int(box_w * 0.5)
                        pad_y = int(box_h * 0.5)
                    
                    # Attempt 1: exact vehicle bbox (the same red box shown to operator)
                    plate, conf, _engine = recognize_plate(
                        frame_bgr, st.last_box,
                        region_hint=self.camera_name,
                        seed=f"{self.camera_id}:{key}:tight",
                    )
                    if plate and conf >= 0.15 and _engine != "simulated":
                        st.plate = plate
                        st.plate_conf = float(conf)
                        print(f"[ANPR] Track {key[0]} | plate={plate} | conf={conf:.2f} | engine={_engine}")
                    else:
                        # Attempt 2: enlarged context as fallback for small/far vehicles
                        enlarged_box = [
                            max(0, bx1 - pad_x), max(0, by1 - pad_y),
                            min(fw, bx2 + pad_x), min(fh, by2 + pad_y)
                        ]
                        plate2, conf2, _engine2 = recognize_plate(
                            frame_bgr, enlarged_box,
                            region_hint=self.camera_name,
                            seed=f"{self.camera_id}:{key}:wide",
                        )
                        if plate2 and conf2 >= 0.15 and _engine2 != "simulated":
                            st.plate = plate2
                            st.plate_conf = float(conf2)
                            print(f"[ANPR-WIDE] Track {key[0]} | plate={plate2} | conf={conf2:.2f} | engine={_engine2}")
                        else:
                            print(f"[ANPR] Track {key[0]} | not readable (bbox={box_w}x{box_h})")
                except Exception as anpr_err:
                    print(f"[ANPR] Track {key[0]} | ERROR: {anpr_err}")

            evidence_bbox = [int(v) for v in st.last_box]
            evidence_rel = _save_evidence(
                frame_bgr, evidence_bbox,
                self.camera_id, timestamp, violation_type
            )

            track_data = tracks.get(st.track_id)
            cls_id = track_data.get("class_id") if track_data else st.class_id
            # Fallback to stored class_id if track no longer exists
            if cls_id is None:
                cls_id = st.class_id
            vehicle_class = _class_to_str(cls_id)

            try:
                vid = insert_violation(
                    camera_id=self.camera_id,
                    camera_name=self.camera_name,
                    violation_type=violation_type,
                    zone_type=ztype,
                    timestamp=timestamp,
                    duration_s=float(dwell),
                    zone_id=zone_id,
                    vehicle_class=vehicle_class,
                    plate_text=st.plate,
                    plate_confidence=float(st.plate_conf or 0),
                    bbox=evidence_bbox,
                    evidence_path=evidence_rel,
                    lat=float(self.lat) if self.lat else None,
                    lng=float(self.lng) if self.lng else None,
                    status="pending",
                )
                print(f"[VIOLATION DB] id={vid} | {vehicle_class} | bbox={vw}x{vh}")
            except Exception as e:
                print(f"[VIOLATION ERROR] {e}")
                vid = None

            st.logged_until = timestamp + float(app_config.VIOLATION_COOLDOWN_SECONDS)

            record = {
                "id": vid,
                "timestamp": timestamp,
                "violation_type": violation_type,
                "zone_type": ztype,
                "zone_id": zone_id,
                "zone_name": zone.get("name"),
                "bbox": evidence_bbox,
                "plate_text": st.plate,
                "plate_confidence": st.plate_conf,
                "duration_s": float(dwell),
                "camera_id": self.camera_id,
                "camera_name": self.camera_name,
                "evidence_path": evidence_rel,
                "vehicle_class": vehicle_class,
            }
            created.append(record)
            with self._last_violation_lock:
                self._last_violation_ids.append(record)
                if len(self._last_violation_ids) > 20:
                    self._last_violation_ids = self._last_violation_ids[-20:]

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

        max_disp_threshold = float(getattr(app_config, "STATIC_MOVEMENT_PX", 40.0) or 40.0)

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
        """Check if vehicle is moving against the allowed direction in a wrong_way zone.
        
        The zone stores 'allowed_direction' as degrees (0-360):
          0° = moving RIGHT (→)
          90° = moving DOWN (↓)
          180° = moving LEFT (←)
          270° = moving UP (↑)
        
        A vehicle is "wrong way" if its movement direction differs from
        allowed_direction by more than 150 degrees (i.e., clearly going opposite).
        
        CRITICAL: Stationary vehicles CANNOT be wrong-way. A parked car is not
        "going against traffic" — it's just parked. Only MOVING vehicles can
        violate wrong-way rules.
        """
        if len(st.positions) < 5:
            return False
        
        # FIRST: Check if vehicle is actually MOVING (not parked/stationary)
        # A stationary vehicle cannot be "wrong way"
        if self._is_stationary(st):
            return False
        
        # Calculate vehicle's OVERALL movement direction from positions
        first_ts, first_cx, first_cy = st.positions[0]
        last_ts, last_cx, last_cy = st.positions[-1]
        
        dx = last_cx - first_cx
        dy = last_cy - first_cy
        
        # Require significant movement distance (scaled to frame size)
        # At 3200x1800, YOLO jitter can be 10-20px per sample
        # Over 30 seconds with 15 samples, jitter alone can drift 50-80px
        # Require at least 100px to be sure it's real movement
        distance = math.hypot(dx, dy)
        if distance < 100.0:
            return False  # Not enough movement — likely stationary with jitter
        
        # Also check SPEED: must be moving at least 15 px/sec consistently
        elapsed = max(0.1, last_ts - first_ts)
        speed_px_per_sec = distance / elapsed
        if speed_px_per_sec < 15.0:
            return False  # Too slow — likely parked with bbox jitter
        
        # Vehicle's overall direction in degrees (0=right, 90=down, 180=left, 270=up)
        vehicle_angle = math.degrees(math.atan2(dy, dx)) % 360
        
        # Get allowed direction from zone config
        allowed_angle = self._get_zone_direction(zone)
        if allowed_angle is None:
            return False  # No direction configured, can't check
        
        # Calculate angle difference
        diff = abs(vehicle_angle - allowed_angle)
        if diff > 180:
            diff = 360 - diff
        
        # STRICT threshold: only flag if clearly going OPPOSITE direction (>150°)
        # This means:
        # - 0-60° difference = same direction (OK)
        # - 60-150° difference = turning/crossing (OK, belok kanan/kiri)
        # - 150-180° difference = truly opposite / lawan arah (VIOLATION)
        if diff <= 150.0:
            return False
        
        # Additional check: consistency of wrong-way movement
        # The vehicle must be consistently moving in the wrong direction,
        # not just a momentary angle from a turn.
        # Check the direction using multiple segments of the trajectory
        if len(st.positions) >= 6:
            # Split trajectory into segments and check each
            n = len(st.positions)
            wrong_segments = 0
            total_segments = 0
            
            # Check direction in 3 segments (beginning, middle, end)
            segment_size = max(2, n // 3)
            for seg_start in range(0, n - segment_size, segment_size):
                seg_end = min(seg_start + segment_size, n - 1)
                _, sx, sy = st.positions[seg_start]
                _, ex, ey = st.positions[seg_end]
                seg_dx = ex - sx
                seg_dy = ey - sy
                seg_dist = math.hypot(seg_dx, seg_dy)
                if seg_dist < 10.0:
                    continue  # Segment too short to determine direction
                
                seg_angle = math.degrees(math.atan2(seg_dy, seg_dx)) % 360
                seg_diff = abs(seg_angle - allowed_angle)
                if seg_diff > 180:
                    seg_diff = 360 - seg_diff
                
                total_segments += 1
                if seg_diff > 150.0:
                    wrong_segments += 1
            
            # At least 2 out of 3 segments must show wrong-way movement
            # This filters out vehicles that briefly go opposite while turning
            if total_segments >= 2 and wrong_segments < 2:
                return False
        
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
                # Crop the vehicle area with generous padding for context
                pad = 40
                crop_x1 = max(0, x1 - pad)
                crop_y1 = max(0, y1 - pad)
                crop_x2 = min(fw, x2 + pad)
                crop_y2 = min(fh, y2 + pad)
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
