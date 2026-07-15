"""
GPU Batch Inference Engine
---------------------------
Collects frames from all camera threads and runs YOLO inference in a single
batch on the GPU. This is much more efficient than sequential per-camera
inference because:

1. GPU utilization is maximized (batch processing is what GPUs are designed for)
2. No lock contention — cameras submit frames and get results back via queue
3. All cameras benefit from GPU acceleration simultaneously

Architecture:
  Camera Thread 1 ──┐
  Camera Thread 2 ──┼──► BatchInferenceWorker (GPU) ──► Results back to each camera
  Camera Thread 3 ──┘
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import numpy as np
import cv2

import app.globals as g
from app.config import (
    CONF_THRESHOLD, IOU_THRESHOLD, INFER_IMGSZ,
    VEHICLE_CLASSES, CLASS_MAPPING,
    VEHICLE_CLASSES_CUSTOM, CLASS_MAPPING_CUSTOM,
)
import app.config as app_config


class InferenceRequest:
    """A request from a camera thread to run inference on a frame."""
    __slots__ = ("frame", "source_id", "event", "results", "timestamp")

    def __init__(self, frame: np.ndarray, source_id: str):
        self.frame = frame
        self.source_id = source_id
        self.event = threading.Event()
        self.results = None  # Will be filled by the worker
        self.timestamp = time.time()


class BatchInferenceWorker(threading.Thread):
    """
    Dedicated GPU inference thread that processes frames in batches.
    
    Instead of each camera thread fighting for the model_lock one at a time,
    cameras submit frames to a queue. This worker collects frames (up to
    max_batch_size or max_wait_ms), runs a single batched inference call,
    and returns results to each camera thread.
    
    Benefits:
    - GPU processes N frames in ~same time as 1 frame (batch parallelism)
    - No lock contention between camera threads
    - Consistent latency for all cameras
    """

    def __init__(self, max_batch_size: int = 12, max_wait_ms: float = 60.0):
        super().__init__(daemon=True)
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.running = True
        self._queue: deque[InferenceRequest] = deque()
        self._queue_lock = threading.Lock()
        self._queue_event = threading.Event()
        self._stats = {
            "total_batches": 0,
            "total_frames": 0,
            "avg_batch_size": 0.0,
            "avg_latency_ms": 0.0,
        }

    def submit(self, frame: np.ndarray, source_id: str, timeout_s: float = 5.0) -> list | None:
        """
        Submit a frame for inference. Blocks until result is ready.
        
        Returns list of detections or None on timeout/error.
        Each detection: {"box": [x1,y1,x2,y2], "class": int, "conf": float}
        """
        req = InferenceRequest(frame, source_id)
        with self._queue_lock:
            self._queue.append(req)
        self._queue_event.set()

        # Wait for result
        if req.event.wait(timeout=timeout_s):
            return req.results
        return None

    def run(self):
        print("[GPU-BATCH] Batch inference worker started")
        while self.running:
            # Wait for at least one request
            self._queue_event.wait(timeout=0.5)
            self._queue_event.clear()

            if not self.running:
                break

            # Collect batch (wait a bit for more frames to arrive)
            batch = self._collect_batch()
            if not batch:
                continue

            # Run inference
            t0 = time.time()
            try:
                self._run_batch_inference(batch)
            except Exception as e:
                print(f"[GPU-BATCH] Inference error: {e}")
                # Return empty results on error
                for req in batch:
                    req.results = []
                    req.event.set()

            elapsed_ms = (time.time() - t0) * 1000
            self._update_stats(len(batch), elapsed_ms)

        print("[GPU-BATCH] Batch inference worker stopped")

    def _collect_batch(self) -> list[InferenceRequest]:
        """Collect up to max_batch_size requests, waiting up to max_wait_ms."""
        batch = []
        deadline = time.time() + (self.max_wait_ms / 1000.0)

        while len(batch) < self.max_batch_size:
            with self._queue_lock:
                if self._queue:
                    batch.append(self._queue.popleft())
                else:
                    break

            # If we have at least 1 frame, wait a tiny bit for more
            if len(batch) < self.max_batch_size and time.time() < deadline:
                time.sleep(0.005)  # 5ms micro-wait
                continue

        return batch

    def _run_batch_inference(self, batch: list[InferenceRequest]):
        """Run YOLO inference on a batch of frames."""
        model = g.yolo_model_instance
        if model is None:
            for req in batch:
                req.results = []
                req.event.set()
            return

        frames = [req.frame for req in batch]

        # Pre-resize frames to inference size to reduce memory transfer to GPU
        # YOLO will resize internally anyway, but pre-resizing avoids sending
        # huge frames (e.g. 3200x1800) through the pipeline
        # Track scale factors to map coordinates back to original resolution
        target_size = int(app_config.INFER_IMGSZ or 640)
        resized_frames = []
        scale_factors = []  # (scale_x, scale_y) to map back to original
        for frame in frames:
            h, w = frame.shape[:2]
            if max(h, w) > target_size * 1.5:
                # Only resize if significantly larger than inference size
                scale = target_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized_frames.append(cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR))
                # Inverse scale to map coordinates back to original frame
                scale_factors.append((w / new_w, h / new_h))
            else:
                resized_frames.append(frame)
                scale_factors.append((1.0, 1.0))  # No scaling needed

        if hasattr(model, "infer"):
            # OpenCV DNN engine — doesn't support true batching, run sequentially
            for i, req in enumerate(batch):
                try:
                    dets = model.infer(resized_frames[i])
                    req.results = self._convert_dnn_results(dets, scale_factors[i])
                except Exception as e:
                    print(f"[GPU-BATCH] DNN infer error for {req.source_id}: {e}")
                    req.results = []
                req.event.set()
        else:
            # YOLO (ultralytics) — supports batch inference natively
            try:
                call_kwargs = {
                    "conf": float(app_config.CONF_THRESHOLD),
                    "iou": float(app_config.IOU_THRESHOLD),
                    "classes": list(app_config.VEHICLE_CLASSES),
                    "verbose": False,
                    "imgsz": int(app_config.INFER_IMGSZ or 640),
                    "augment": False,
                    "agnostic_nms": False,
                    # YOLO11m optimizations
                    "half": bool(getattr(g, "use_gpu", False)),  # Half precision on GPU for ~2x speed
                    "int8": False,  # Int8 not needed, keep FP16
                }
                if bool(getattr(g, "use_gpu", False)):
                    call_kwargs["device"] = 0

                # Batch inference: pass list of frames
                try:
                    all_results = model(resized_frames, **call_kwargs)
                except TypeError:
                    # Fallback: some YOLO versions don't support all kwargs
                    call_kwargs.pop("device", None)
                    call_kwargs.pop("half", None)
                    all_results = model(frames, **call_kwargs)

                # Distribute results to each request
                for i, req in enumerate(batch):
                    try:
                        if i < len(all_results):
                            req.results = self._convert_yolo_results(all_results[i], scale_factors[i])
                        else:
                            req.results = []
                    except Exception as e:
                        print(f"[GPU-BATCH] Result parse error for {req.source_id}: {e}")
                        req.results = []
                    req.event.set()

            except Exception as e:
                print(f"[GPU-BATCH] Batch YOLO error: {e}")
                # Fallback: run one by one
                for i, req in enumerate(batch):
                    try:
                        single_result = model(resized_frames[i], **call_kwargs)
                        req.results = self._convert_yolo_results(single_result[0] if single_result else None, scale_factors[i])
                    except Exception:
                        req.results = []
                    req.event.set()

    def _convert_yolo_results(self, result, scale=(1.0, 1.0)) -> list[dict]:
        """Convert a single YOLO result to our standard format.
        Scale coordinates back to original frame resolution."""
        if result is None:
            return []
        dets = []
        sx, sy = scale
        try:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                internal_class_id = app_config.CLASS_MAPPING.get(cls_id, 0)
                # Scale coordinates back to original frame resolution
                dets.append({
                    "box": [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)],
                    "class": internal_class_id,
                    "coco_class": cls_id,
                    "conf": conf,
                })
        except Exception as e:
            print(f"[GPU-BATCH] Parse error: {e}")
        return dets

    def _convert_dnn_results(self, dets, scale=(1.0, 1.0)) -> list[dict]:
        """Convert DNN engine results to standard format.
        Scale coordinates back to original frame resolution."""
        results = []
        sx, sy = scale
        for det in (dets or []):
            x1, y1, x2, y2 = det["box"]
            coco_id = int(det["coco_class"])
            conf = float(det["conf"])
            internal_class_id = app_config.CLASS_MAPPING.get(coco_id, 0)
            results.append({
                "box": [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)],
                "class": internal_class_id,
                "coco_class": coco_id,
                "conf": conf,
            })
        return results

    def _update_stats(self, batch_size: int, latency_ms: float):
        self._stats["total_batches"] += 1
        self._stats["total_frames"] += batch_size
        # Exponential moving average
        alpha = 0.1
        self._stats["avg_batch_size"] = (
            (1 - alpha) * self._stats["avg_batch_size"] + alpha * batch_size
        )
        self._stats["avg_latency_ms"] = (
            (1 - alpha) * self._stats["avg_latency_ms"] + alpha * latency_ms
        )

    def get_stats(self) -> dict:
        return dict(self._stats)

    def stop(self):
        self.running = False
        self._queue_event.set()


# Global instance
_batch_worker: BatchInferenceWorker | None = None
_batch_worker_lock = threading.Lock()


def get_batch_worker() -> BatchInferenceWorker:
    """Get or create the global batch inference worker."""
    global _batch_worker
    if _batch_worker is None or not _batch_worker.is_alive():
        with _batch_worker_lock:
            if _batch_worker is None or not _batch_worker.is_alive():
                _batch_worker = BatchInferenceWorker(max_batch_size=12, max_wait_ms=60)
                _batch_worker.start()
    return _batch_worker


def submit_inference(frame: np.ndarray, source_id: str, timeout_s: float = 5.0) -> list | None:
    """Submit a frame for GPU batch inference. Returns detections or None."""
    worker = get_batch_worker()
    return worker.submit(frame, source_id, timeout_s)


def stop_batch_worker():
    """Stop the batch inference worker."""
    global _batch_worker
    if _batch_worker is not None:
        _batch_worker.stop()
        _batch_worker = None
