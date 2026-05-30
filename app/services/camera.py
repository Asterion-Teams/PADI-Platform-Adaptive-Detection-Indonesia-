import threading
import time
import cv2
import numpy as np
import csv
import os
import datetime
import math
import random
import subprocess
import urllib.request
import urllib.error
from collections import deque
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from app.config import (
    YOLO_MODEL_PATH, YOLO_ONNX_PATH, YOLO_ONNX_URL, CONF_THRESHOLD, IOU_THRESHOLD, 
    VEHICLE_CLASSES, CLASS_MAPPING, CLASS_CAR, CLASS_MOTORCYCLE, CLASS_BUS,
    VEHICLE_CLASSES_CUSTOM, CLASS_MAPPING_CUSTOM,
    PROCESS_INTERVAL, STREAM_FPS, STREAM_JPEG_QUALITY, STREAM_MAX_WIDTH, INFER_IMGSZ, CAPTURE_DROP_FRAMES,
    HISTORY_MAX_LEN, DATA_LAKE_PATH,
    YOLO_CUSTOM_PATH, USE_CUSTOM_YOLO, INFER_SKIP_FRAMES,
)
import app.config as app_config
import app.globals as g
from app.utils import save_stats
from app.database import insert_history_batch
from app.services.enforcement import EnforcementEngine

def _download_file(url, dest_path):
    if os.path.exists(dest_path):
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".tmp"
    try:
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, dest_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def _ffmpeg_grab_frame(url, timeout_s=10):
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-user_agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "-allowed_extensions",
        "ALL",
        "-protocol_whitelist",
        "file,http,https,tcp,tls,crypto",
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        "5",
        "-i",
        url,
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    try:
        res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s)
        if res.returncode != 0 or not res.stdout:
            return None
        data = res.stdout
        soi = data.find(b"\xff\xd8")
        eoi = data.find(b"\xff\xd9", soi + 2)
        if soi == -1 or eoi == -1:
            return None
        jpg = data[soi : eoi + 2]
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def _opencv_dnn_cuda_available():
    try:
        if not (hasattr(cv2, "cuda") and hasattr(cv2.cuda, "getCudaEnabledDeviceCount")):
            return False
        if int(cv2.cuda.getCudaEnabledDeviceCount() or 0) <= 0:
            return False
        if not (hasattr(cv2.dnn, "DNN_BACKEND_CUDA") and hasattr(cv2.dnn, "DNN_TARGET_CUDA")):
            return False
        return True
    except Exception:
        return False

class YoloDnnEngine:
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.input_size = (640, 640)
        self.using_cuda = False
        try:
            cuda_ok = False
            if hasattr(cv2, "cuda") and hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
                cuda_ok = int(cv2.cuda.getCudaEnabledDeviceCount() or 0) > 0
            if cuda_ok and hasattr(cv2.dnn, "DNN_BACKEND_CUDA") and hasattr(cv2.dnn, "DNN_TARGET_CUDA"):
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.using_cuda = True
        except Exception:
            self.using_cuda = False

    def infer(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, self.input_size, swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        if out is None:
            return []
        if out.ndim == 3:
            out = out[0]

        boxes = []
        scores = []
        class_ids = []

        for row in out:
            obj = float(row[4])
            if obj <= 0.0:
                continue
            cls_scores = row[5:]
            cls_id = int(np.argmax(cls_scores))
            if cls_id not in VEHICLE_CLASSES:
                continue
            conf = obj * float(cls_scores[cls_id])
            if conf < float(app_config.CONF_THRESHOLD):
                continue
            cx, cy, bw, bh = row[0:4]
            x = (float(cx) - float(bw) / 2.0) * (w / self.input_size[0])
            y = (float(cy) - float(bh) / 2.0) * (h / self.input_size[1])
            bw = float(bw) * (w / self.input_size[0])
            bh = float(bh) * (h / self.input_size[1])
            boxes.append([int(x), int(y), int(bw), int(bh)])
            scores.append(float(conf))
            class_ids.append(cls_id)

        if not boxes:
            return []

        idxs = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            float(app_config.CONF_THRESHOLD),
            float(app_config.IOU_THRESHOLD),
        )
        if idxs is None or len(idxs) == 0:
            return []

        dets = []
        for i in idxs.flatten().tolist():
            x, y, bw, bh = boxes[i]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w - 1, x + bw)
            y2 = min(h - 1, y + bh)
            dets.append(
                {
                    "coco_class": int(class_ids[i]),
                    "conf": float(scores[i]),
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                }
            )
        return dets

class CameraAgent(threading.Thread):
    def __init__(self, source_config, model_ref):
        threading.Thread.__init__(self)
        self.source_id = source_config["id"]
        self.source_name = source_config["name"]
        self.source_url = source_config["url"]
        # Normalize HLS URL: fmp4 m3u8 playlists often fail with FFmpeg/OpenCV
        # Convert index.fmp4.m3u8 -> index.m3u8 for better compatibility
        if self.source_url and "index.fmp4.m3u8" in self.source_url:
            self.source_url = self.source_url.replace("index.fmp4.m3u8", "index.m3u8")
            # Also ensure tracks-v1 path is present
            if "/tracks-v1/" not in self.source_url:
                self.source_url = self.source_url.replace("/index.m3u8", "/tracks-v1/index.m3u8")
        self.mirror_id = source_config.get("mirror_id")
        self.model = model_ref
        self.running = True
        self.daemon = True
        self.last_save_time = time.time()
        self.last_log_time = 0.0
        self.last_persist_ts = 0.0
        self.prev_rects = [] # Store previous frame detections for static object filtering
        self.tracks = {}
        self.next_track_id = 1
        self.track_iou_threshold = 0.20
        # TTL: if a track isn't re-detected within 10 seconds, remove it.
        # This prevents ghost tracks from lingering after vehicles leave.
        self.track_ttl_s = 10.0
        self.cap = None
        self.cap_url = None
        self._last_ffmpeg_grab_ts = 0.0

        # Case 1 - Enforcement (illegal parking / busway / bicycle lane)
        try:
            _lat = float(source_config.get("lat")) if source_config.get("lat") not in (None, "") else None
        except Exception:
            _lat = None
        try:
            _lng = float(source_config.get("lng")) if source_config.get("lng") not in (None, "") else None
        except Exception:
            _lng = None
        self.enforcement = EnforcementEngine(
            camera_id=self.source_id,
            camera_name=self.source_name,
            lat=_lat,
            lng=_lng,
        )
        self._last_violation_records = []
        
        # Initialize stats for this camera if not exists
        if self.source_id not in g.global_stats:
            g.global_stats[self.source_id] = {
                "name": self.source_name,
                "current_count": 0,
                "current_class_counts": {str(CLASS_CAR): 0, str(CLASS_MOTORCYCLE): 0, str(CLASS_BUS): 0},
                "accumulated_count": 0,
                "accumulated_class_counts": {str(CLASS_CAR): 0, str(CLASS_MOTORCYCLE): 0, str(CLASS_BUS): 0},
                "history": deque(maxlen=HISTORY_MAX_LEN)
            }
        else:
            # Ensure name is updated if changed
            g.global_stats[self.source_id]["name"] = self.source_name
            # Ensure history exists
            if "history" not in g.global_stats[self.source_id]:
                g.global_stats[self.source_id]["history"] = deque(maxlen=HISTORY_MAX_LEN)

    def _release_cap(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.cap = None
        self.cap_url = None

    def _ensure_cap(self):
        if self.cap is not None and self.cap_url == self.source_url:
            try:
                if self.cap.isOpened():
                    return True
            except Exception:
                pass
        self._release_cap()
        try:
            if not os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"):
                # Support HLS fMP4 segments (Balitower, modern CCTV streams)
                # allowed_extensions: allow .fmp4 extension for HLS segments
                # protocol_whitelist: allow http/https/tcp/tls for HLS
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;20000|allowed_extensions;ALL|protocol_whitelist;file,http,https,tcp,tls,crypto|reconnect;1|reconnect_streamed;1|reconnect_delay_max;5"
        except Exception:
            pass
        cap = None
        try:
            # Force FFMPEG backend for HLS streams
            url = str(self.source_url or "")
            if ".m3u8" in url or "hls" in url.lower() or "balitower" in url.lower():
                cap = cv2.VideoCapture(self.source_url, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(self.source_url)
        except Exception:
            cap = None
        if cap is None:
            return False
        try:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if cap.isOpened():
                self.cap = cap
                self.cap_url = self.source_url
                return True
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        return False

    def _read_from_cap(self):
        cap = self.cap
        if cap is None:
            return None, False
        drop = int(app_config.CAPTURE_DROP_FRAMES or 0)
        if drop > 0:
            for _ in range(drop):
                try:
                    cap.grab()
                except Exception:
                    break
        try:
            ret, frame = cap.read()
        except Exception:
            ret, frame = False, None
        if not ret or frame is None:
            # If this is a local video file (MP4/AVI/etc), loop back to start (replay)
            if self._is_local_video():
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"[{self.source_name}] Video replay — looping back to start")
                        return frame, True
                except Exception:
                    pass
            return None, False
        return frame, True

    def _is_local_video(self):
        """Check if the source URL is a local video file (MP4, AVI, etc)."""
        url = str(self.source_url or "").strip().lower()
        # Local file path or file:// URL
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm')
        if any(url.endswith(ext) for ext in video_extensions):
            return True
        # Check if it's a local path (not http/rtsp)
        if not url.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
            if any(url.endswith(ext) for ext in video_extensions):
                return True
        return False

    def _capture_frame(self):
        if self._ensure_cap():
            frame, ok = self._read_from_cap()
            if ok:
                self._consecutive_failures = 0
                return frame, True
            self._release_cap()

        # Exponential backoff for repeated failures
        failures = getattr(self, '_consecutive_failures', 0)
        self._consecutive_failures = failures + 1
        # Backoff: 0.5s, 1s, 2s, 4s, max 10s
        backoff = min(10.0, 0.5 * (2 ** min(failures, 4)))

        now = time.time()
        if (now - float(self._last_ffmpeg_grab_ts or 0.0)) < backoff:
            return None, False
        self._last_ffmpeg_grab_ts = now
        frame = _ffmpeg_grab_frame(self.source_url)
        if frame is not None:
            self._consecutive_failures = 0
        return frame, frame is not None

    def _update_tracks(self, rects, rect_classes, timestamp):
        if not rects:
            expired = []
            for tid, t in self.tracks.items():
                if (timestamp - float(t.get("last_seen") or 0.0)) > self.track_ttl_s:
                    expired.append(tid)
            for tid in expired:
                del self.tracks[tid]
            return 0, {CLASS_CAR: 0, CLASS_MOTORCYCLE: 0, CLASS_BUS: 0}

        track_ids = list(self.tracks.keys())
        used_tracks = set()
        used_dets = set()
        pairs = []

        for det_i, rect in enumerate(rects):
            best_tid = None
            best_iou = 0.0
            for tid in track_ids:
                t = self.tracks.get(tid)
                if not t:
                    continue
                iou = self.get_iou(rect, t["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None and best_iou >= self.track_iou_threshold:
                pairs.append((best_iou, det_i, best_tid))

        pairs.sort(reverse=True, key=lambda x: x[0])

        for _, det_i, tid in pairs:
            if det_i in used_dets or tid in used_tracks:
                continue
            used_dets.add(det_i)
            used_tracks.add(tid)
            self.tracks[tid] = {
                "box": rects[det_i],
                "class_id": rect_classes[det_i],
                "last_seen": timestamp,
                "counted": True,
            }

        new_class_counts = {CLASS_CAR: 0, CLASS_MOTORCYCLE: 0, CLASS_BUS: 0}
        new_rects_count = 0
        for det_i, rect in enumerate(rects):
            if det_i in used_dets:
                continue
            tid = self.next_track_id
            self.next_track_id += 1
            cls_id = rect_classes[det_i]
            self.tracks[tid] = {
                "box": rect,
                "class_id": cls_id,
                "last_seen": timestamp,
                "counted": True,
            }
            new_rects_count += 1
            new_class_counts[cls_id] += 1

        expired = []
        for tid, t in self.tracks.items():
            if (timestamp - float(t.get("last_seen") or 0.0)) > self.track_ttl_s:
                expired.append(tid)
        for tid in expired:
            del self.tracks[tid]

        return new_rects_count, new_class_counts

    def log_to_datalake(self, detections, timestamp):
        """
        Simulate Big Data Ingestion:
        Write detailed detection logs to partitioned CSV files (Year/Month/Day)
        Format: timestamp, source_id, class_id, confidence, x1, y1, x2, y2
        """
        try:
            dt = datetime.datetime.fromtimestamp(timestamp)
            partition_path = os.path.join(DATA_LAKE_PATH, str(dt.year), f"{dt.month:02d}", f"{dt.day:02d}")
            os.makedirs(partition_path, exist_ok=True)
            
            filename = f"traffic_log_{self.source_id}.csv"
            filepath = os.path.join(partition_path, filename)
            
            file_exists = os.path.isfile(filepath)
            
            with open(filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "source_id", "source_name", "class_id", "confidence", "bbox"])
                
                for det in detections:
                    # det = (class_id, confidence, [x1, y1, x2, y2])
                    writer.writerow([
                        timestamp, 
                        self.source_id, 
                        self.source_name,
                        det['class_id'], 
                        f"{det['conf']:.4f}", 
                        f"{det['box']}"
                    ])
        except Exception as e:
            print(f"[ERROR] Data Lake Write Failed: {e}")

    def get_iou(self, boxA, boxB):
        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # Compute the intersection over union
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def get_traffic_multiplier(self):
        """
        Returns a multiplier to simulate realistic traffic patterns based on time of day.
        Used to augment the base video detection count for demo purposes.
        """
        now = datetime.datetime.now()
        hour = now.hour + now.minute / 60.0
        
        # Base multiplier (Video might have 5-10 cars, we want at least that)
        mult = 1.0
        
        # Morning Peak (06:30 - 09:00) - Peak at 07:30
        # Boost up to ~4x
        if 6.0 <= hour <= 9.5:
            mult += 4.0 * math.exp(-((hour - 7.5)**2) / 1.5)
            
        # Evening Peak (16:30 - 19:00) - Peak at 17:30
        # Boost up to ~5x
        if 16.0 <= hour <= 20.0:
            mult += 5.0 * math.exp(-((hour - 17.5)**2) / 2.0)
            
        # Night drop (22:00 - 05:00) - Reduce to 0.5x
        if hour >= 22.0 or hour <= 5.0:
            mult = 0.5
            
        # Random fluctuation (+/- 20%)
        mult *= random.uniform(0.8, 1.2)
        
        return max(0.5, mult)

    def run(self):
        print(f"[INFO] Started Agent for {self.source_name}")
        frame_counter = 0
        
        while self.running:
            # Mirror Mode: Copy stats from another source if configured
            if self.mirror_id and self.mirror_id in g.global_stats:
                if self.source_id not in g.global_stats:
                    break
                mirrored = g.global_stats[self.mirror_id]
                stats = g.global_stats[self.source_id]
                stats["current_count"] = mirrored.get("current_count", 0)
                stats["current_class_counts"] = mirrored.get("current_class_counts", {str(CLASS_CAR): 0, str(CLASS_MOTORCYCLE): 0, str(CLASS_BUS): 0})
                stats["accumulated_count"] = mirrored.get("accumulated_count", 0)
                stats["accumulated_class_counts"] = mirrored.get("accumulated_class_counts", {str(CLASS_CAR): 0, str(CLASS_MOTORCYCLE): 0, str(CLASS_BUS): 0})
                if "history" in mirrored:
                    stats["history"] = mirrored["history"]
                time.sleep(0.5)
                continue
            
            frame = None
            success = True
            is_active_view = self.source_url == g.VIDEO_SOURCE
            # Always capture frames for enforcement (background processing)
            # Not just when user is viewing
            should_capture = True

            if should_capture:
                frame, success = self._capture_frame()
            else:
                success = True
            
            # Update status in global stats
            if self.source_id in g.global_stats:
                if self.model is None:
                    g.global_stats[self.source_id]["status"] = "simulated"
                else:
                    g.global_stats[self.source_id]["status"] = "online" if success else "offline"
                g.global_stats[self.source_id]["last_update"] = time.time()

            capture_ok = success and frame is not None
            timestamp = time.time()
            
            # Decide if we should run inference this frame
            frame_counter += 1
            skip = int(app_config.INFER_SKIP_FRAMES or 4)
            should_infer = (frame_counter % (skip + 1) == 0)
            
            rects = []
            rect_classes = []
            datalake_batch = []

            if self.model is None:
                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                if should_infer:
                    traffic_mult = self.get_traffic_multiplier()
                    traffic_mult = max(0.6, min(2.5, float(traffic_mult)))

                    base = random.uniform(6.0, 14.0)
                    current_count = int(base * traffic_mult)
                    current_count = max(0, min(40, current_count))
                    new_rects_count = max(0, int(current_count * random.uniform(0.10, 0.30)))

                    car_ratio = min(0.9, max(0.1, 0.6 + random.uniform(-0.1, 0.1)))
                    current_cars = int(current_count * car_ratio)
                    current_motors = max(0, current_count - current_cars)
                    new_cars = int(new_rects_count * car_ratio)
                    new_motors = max(0, new_rects_count - new_cars)

                    current_class_counts = {CLASS_CAR: current_cars, CLASS_MOTORCYCLE: current_motors, CLASS_BUS: 0}
                    new_class_counts = {CLASS_CAR: new_cars, CLASS_MOTORCYCLE: new_motors, CLASS_BUS: 0}
                    self.prev_rects = []
                else:
                    # Non-inference frame: reuse last known counts
                    if self.source_id not in g.global_stats:
                        break
                    stats = g.global_stats[self.source_id]
                    current_count = stats.get("current_count", 0)
                    current_class_counts = {CLASS_CAR: 0, CLASS_MOTORCYCLE: 0, CLASS_BUS: 0}
                    new_rects_count = 0
                    new_class_counts = {CLASS_CAR: 0, CLASS_MOTORCYCLE: 0, CLASS_BUS: 0}
            elif not capture_ok:
                if frame is None and self.source_url == g.VIDEO_SOURCE:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                traffic_mult = self.get_traffic_multiplier()
                traffic_mult = max(0.6, min(2.0, float(traffic_mult)))

                base = random.uniform(5.0, 12.0)
                current_count = int(base * traffic_mult)
                current_count = max(0, min(30, current_count))
                new_rects_count = int(round(current_count * random.uniform(0.05, 0.15)))
                if current_count > 0 and new_rects_count <= 0 and random.random() < 0.35:
                    new_rects_count = 1
                new_rects_count = max(0, min(3, new_rects_count))

                car_ratio = min(0.9, max(0.1, 0.6 + random.uniform(-0.1, 0.1)))
                current_cars = int(current_count * car_ratio)
                current_motors = max(0, current_count - current_cars)
                new_cars = int(new_rects_count * car_ratio)
                new_motors = max(0, new_rects_count - new_cars)

                current_class_counts = {CLASS_CAR: current_cars, CLASS_MOTORCYCLE: current_motors, CLASS_BUS: 0}
                new_class_counts = {CLASS_CAR: new_cars, CLASS_MOTORCYCLE: new_motors, CLASS_BUS: 0}

                expired = []
                for tid, t in self.tracks.items():
                    if (timestamp - float(t.get("last_seen") or 0.0)) > self.track_ttl_s:
                        expired.append(tid)
                for tid in expired:
                    del self.tracks[tid]
            else:
                if should_infer:
                    # Use GPU batch inference — all cameras share the GPU efficiently
                    from app.services.gpu_batch import submit_inference
                    batch_dets = submit_inference(frame, self.source_id, timeout_s=5.0)
                    
                    if batch_dets:
                        for det in batch_dets:
                            x1, y1, x2, y2 = det["box"]
                            internal_class_id = det["class"]
                            conf = float(det["conf"])
                            rects.append((x1, y1, x2, y2))
                            rect_classes.append(internal_class_id)
                            datalake_batch.append({"class_id": internal_class_id, "conf": conf, "box": [x1, y1, x2, y2]})

                    if datalake_batch:
                        # Throttle: only write to data lake every 5 seconds to reduce disk I/O
                        if (timestamp - getattr(self, '_last_datalake_ts', 0.0)) >= 5.0:
                            self.log_to_datalake(datalake_batch, timestamp)
                            self._last_datalake_ts = timestamp

                    current_count = len(rects)
                    current_class_counts = {CLASS_CAR: 0, CLASS_MOTORCYCLE: 0, CLASS_BUS: 0}
                    for c_id in rect_classes:
                        current_class_counts[c_id] += 1

                    new_rects_count, new_class_counts = self._update_tracks(rects, rect_classes, timestamp)
                else:
                    # Non-inference frame: just show stream, reuse last counts
                    # Do NOT update last_seen — let tracks expire naturally
                    # This prevents stale tracks from appearing "alive" to enforcement
                    if self.source_id not in g.global_stats:
                        break
                    stats = g.global_stats[self.source_id]
                    current_count = stats.get("current_count", 0)
                    current_class_counts = {CLASS_CAR: 0, CLASS_MOTORCYCLE: 0, CLASS_BUS: 0}
                    new_rects_count = 0
                    new_class_counts = {CLASS_CAR: 0, CLASS_MOTORCYCLE: 0, CLASS_BUS: 0}

            # ---------------------------------------------------------
            # Case 1: Enforcement check (illegal parking, busway, bicycle)
            # ---------------------------------------------------------
            violation_records = []
            try:
                # Only run when we have real detections and a real frame
                # Pass is_inference_frame so enforcement only samples positions
                # when YOLO actually ran (not stale repeated boxes)
                if self.model is not None and capture_ok and frame is not None and self.tracks:
                    violation_records = self.enforcement.check_frame(
                        frame, self.tracks, timestamp,
                        is_inference_frame=should_infer
                    ) or []
            except Exception as e:
                print(f"[WARN] Enforcement check failed for {self.source_name}: {e}")
                violation_records = []
            self._last_violation_records = violation_records

            if self.source_id not in g.global_stats:
                break
            stats = g.global_stats[self.source_id]
            stats["current_count"] = current_count
            stats["current_class_counts"] = {str(k): v for k, v in current_class_counts.items()}
            persist_ok = True
            if not capture_ok and self.model is not None:
                persist_ok = (timestamp - float(self.last_persist_ts or 0.0)) >= 60.0
            # Throttle DB writes: at most once every 2 seconds
            if persist_ok and (timestamp - float(self.last_persist_ts or 0.0)) < 2.0:
                persist_ok = False
            if persist_ok:
                self.last_persist_ts = timestamp
                stats["accumulated_count"] += new_rects_count
                stats["accumulated_class_counts"].setdefault(str(CLASS_CAR), 0)
                stats["accumulated_class_counts"].setdefault(str(CLASS_MOTORCYCLE), 0)
                stats["accumulated_class_counts"].setdefault(str(CLASS_BUS), 0)
                stats["accumulated_class_counts"][str(CLASS_CAR)] += new_class_counts.get(CLASS_CAR, 0)
                stats["accumulated_class_counts"][str(CLASS_MOTORCYCLE)] += new_class_counts.get(CLASS_MOTORCYCLE, 0)
                stats["accumulated_class_counts"][str(CLASS_BUS)] += new_class_counts.get(CLASS_BUS, 0)
                stats["history"].append({
                    "ts": timestamp,
                    "count": current_count,
                    "cars": current_class_counts.get(CLASS_CAR, 0),
                    "motors": current_class_counts.get(CLASS_MOTORCYCLE, 0),
                    "new_count": new_rects_count,
                    "new_cars": new_class_counts.get(CLASS_CAR, 0),
                    "new_motors": new_class_counts.get(CLASS_MOTORCYCLE, 0)
                })

                try:
                    insert_history_batch([(
                        self.source_id,
                        timestamp,
                        current_count,
                        current_class_counts.get(CLASS_CAR, 0),
                        current_class_counts.get(CLASS_MOTORCYCLE, 0),
                        new_rects_count,
                        new_class_counts.get(CLASS_CAR, 0),
                        new_class_counts.get(CLASS_MOTORCYCLE, 0)
                    )])
                except Exception as e:
                    print(f"[{self.source_name}] DB Error: {e}")

                if timestamp - self.last_save_time > 60:
                    save_stats()
                    self.last_save_time = timestamp

                if self.source_url == g.VIDEO_SOURCE:
                    if (timestamp - self.last_log_time) > 1.0:
                        print(f"[{self.source_name}] Count: {current_count} (Total: {stats['accumulated_count']})")
                        self.last_log_time = timestamp

            # Keep a per-camera preview frame ready so /video_feed/<camera_id>
            # does not depend on whichever camera last won the global slot.
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Draw bounding boxes from last inference (persist across non-inference frames)
            if should_infer and rects:
                # Store last detections for drawing on non-inference frames
                self._last_draw_rects = list(zip(rects, rect_classes))
            
            for (rect, cls_id) in getattr(self, '_last_draw_rects', []):
                (x1, y1, x2, y2) = rect
                if cls_id == CLASS_BUS:
                    color = (0, 200, 255)  # Orange for bus
                    label = "Bus"
                elif cls_id == CLASS_MOTORCYCLE:
                    color = (255, 0, 0)    # Blue for motorcycle
                    label = "Motor"
                else:
                    color = (0, 255, 0)    # Green for car
                    label = "Car"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Enforcement overlay: zones + violation markers
            try:
                if self.enforcement.has_zones() or self._last_violation_records:
                    self.enforcement.draw_overlay(frame, self._last_violation_records)
            except Exception:
                pass

            cv2.putText(frame, f"CAM: {self.source_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Total: {stats['accumulated_count']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if not capture_ok:
                cv2.putText(frame, "NO SIGNAL", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, "Asterion", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            with g.lock:
                g.outputFrames[self.source_id] = frame
                if self.source_url == g.VIDEO_SOURCE:
                    g.outputFrame = frame

            # Sleep — target ~30 FPS for all cameras to ensure real-time
            if self.model is None:
                time.sleep(0.5)
            else:
                # Both active and background cameras run at the same FPS for real-time
                # This ensures background processing (enforcement, tracking) stays up-to-date
                time.sleep(max(0.01, 1.0 / float(app_config.STREAM_FPS or 30)))

    def stop(self):
        self.running = False

def generate_frames(camera_id):
    # Find the source URL
    target_url = None
    for src in g.CCTV_SOURCES:
        if src["id"] == camera_id:
            target_url = src["url"]
            break
            
    if target_url:
        # Set the global video source so the agent starts updating outputFrame
        g.VIDEO_SOURCE = target_url
        
        _last_frame_id = None  # Track if frame changed
        _cached_jpeg = None    # Cache encoded JPEG
        
        while True:
            with g.lock:
                frame = g.outputFrames.get(camera_id)
                if frame is None and g.VIDEO_SOURCE == target_url:
                    frame = g.outputFrame
                if frame is None:
                    time.sleep(0.01)
                    continue

            if frame is None:
                time.sleep(0.01)
                continue

            # Check if frame actually changed (avoid re-encoding same frame)
            frame_id = id(frame)
            if frame_id == _last_frame_id and _cached_jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + _cached_jpeg + b'\r\n')
                try:
                    fps = float(app_config.STREAM_FPS or 0)
                except Exception:
                    fps = 0.0
                if fps <= 0:
                    fps = 15.0
                time.sleep(max(0.001, 1.0 / fps))
                continue

            _last_frame_id = frame_id

            try:
                max_w = int(app_config.STREAM_MAX_WIDTH or 0)
            except Exception:
                max_w = 0
            if max_w > 0:
                try:
                    h, w = frame.shape[:2]
                    if w > max_w:
                        scale = float(max_w) / float(w)
                        nh = max(1, int(round(h * scale)))
                        frame = cv2.resize(frame, (int(max_w), int(nh)))
                except Exception:
                    pass

            try:
                q = int(app_config.STREAM_JPEG_QUALITY or 80)
            except Exception:
                q = 80
            q = max(30, min(95, q))
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(q)]
            (flag, encodedImage) = cv2.imencode(".jpg", frame, encode_params)
            if not flag:
                time.sleep(0.01)
                continue
            
            _cached_jpeg = bytearray(encodedImage)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + _cached_jpeg + b'\r\n')
            
            # Throttle to avoid busy loop, match process interval roughly
            try:
                fps = float(app_config.STREAM_FPS or 0)
            except Exception:
                fps = 0.0
            if fps <= 0:
                fps = 15.0
            time.sleep(max(0.001, 1.0 / fps))

def start_camera_agents():
    model_ref = None
    torch_gpu = False
    try:
        import torch
        torch_gpu = bool(torch.cuda.is_available())
    except Exception:
        torch_gpu = False

    opencv_gpu = _opencv_dnn_cuda_available()
    require_gpu = str(os.environ.get("REQUIRE_GPU") or "").strip().lower() in {"1", "true", "yes", "on"}

    g.use_gpu = bool(torch_gpu)
    g.opencv_dnn_cuda = bool(opencv_gpu)

    print(f"[INFO] Torch CUDA: {'ON' if torch_gpu else 'OFF'}")
    print(f"[INFO] OpenCV DNN CUDA: {'ON' if opencv_gpu else 'OFF'}")

    if require_gpu and (not torch_gpu) and (not opencv_gpu):
        raise RuntimeError("GPU required but not available (Torch CUDA OFF, OpenCV DNN CUDA OFF)")

    # Priority: Custom trained model (vehicle_v3_best.pt) > generic COCO (yolov8l.pt) > ONNX fallback
    preferred_model_path = None
    if USE_CUSTOM_YOLO and os.path.isfile(YOLO_CUSTOM_PATH):
        preferred_model_path = YOLO_CUSTOM_PATH
        print(f"[INFO] Custom model found: {YOLO_CUSTOM_PATH}")
    elif os.path.isfile(YOLO_MODEL_PATH):
        preferred_model_path = YOLO_MODEL_PATH

    if torch_gpu and YOLO is not None and preferred_model_path:
        print(f"[INFO] Loading YOLO model (GPU): {os.path.basename(preferred_model_path)}")
        try:
            model_ref = YOLO(preferred_model_path)
            print("[INFO] Model Loaded (Torch CUDA).")
        except Exception as e:
            print(f"[WARN] YOLO load failed, will try fallback engine: {e}")
            model_ref = None
    
    if model_ref is None and opencv_gpu:
        try:
            _download_file(YOLO_ONNX_URL, YOLO_ONNX_PATH)
            model_ref = YoloDnnEngine(YOLO_ONNX_PATH)
            print("[INFO] YOLO engine loaded (OpenCV DNN).")
            if bool(getattr(model_ref, "using_cuda", False)):
                print("[INFO] OpenCV DNN CUDA: ON")
        except Exception as e:
            print(f"[WARN] YOLO engine not available, will try fallback engine: {e}")
            model_ref = None

    if model_ref is None and (YOLO is not None) and preferred_model_path:
        print(f"[INFO] Loading YOLO model (CPU): {os.path.basename(preferred_model_path)}")
        try:
            model_ref = YOLO(preferred_model_path)
            print("[INFO] Model Loaded (CPU).")
        except Exception as e:
            print(f"[WARN] YOLO load failed, will try fallback engine: {e}")
            model_ref = None

    if model_ref is None:
        try:
            _download_file(YOLO_ONNX_URL, YOLO_ONNX_PATH)
            model_ref = YoloDnnEngine(YOLO_ONNX_PATH)
            print("[INFO] YOLO engine loaded (OpenCV DNN).")
            if bool(getattr(model_ref, "using_cuda", False)):
                print("[INFO] OpenCV DNN CUDA: ON")
        except Exception as e:
            print(f"[WARN] YOLO engine not available, running in simulation mode: {e}")
            model_ref = None

    g.yolo_model_instance = model_ref

    # Switch class mapping based on which model is loaded
    if model_ref is not None and preferred_model_path == YOLO_CUSTOM_PATH:
        app_config.VEHICLE_CLASSES = VEHICLE_CLASSES_CUSTOM
        app_config.CLASS_MAPPING = CLASS_MAPPING_CUSTOM
        print(f"[INFO] Using CUSTOM class mapping (6 classes: bus, car, microbus, motorbike, pickup-van, truck)")
    else:
        # Ensure COCO mapping is active for pretrained models (yolo11m, yolov8l, etc.)
        from app.config import VEHICLE_CLASSES_COCO, CLASS_MAPPING_COCO
        app_config.VEHICLE_CLASSES = VEHICLE_CLASSES_COCO
        app_config.CLASS_MAPPING = CLASS_MAPPING_COCO
        print(f"[INFO] Using COCO class mapping (vehicle subset: car, motorcycle, bus, truck)")

    # Start GPU batch inference worker (all cameras share one GPU efficiently)
    if model_ref is not None:
        from app.services.gpu_batch import get_batch_worker
        worker = get_batch_worker()
        print(f"[INFO] GPU Batch Inference Worker: ACTIVE (max_batch={worker.max_batch_size})")
    
    # Start agents for all sources
    for src in g.CCTV_SOURCES:
        if src["id"] not in g.camera_agents:
            agent = CameraAgent(src, g.yolo_model_instance)
            g.camera_agents[src["id"]] = agent
            agent.start()

def stop_agent(source_id):
    if source_id in g.camera_agents:
        g.camera_agents[source_id].stop()
        del g.camera_agents[source_id]
