"""Test the full OCR pipeline with realistic CCTV-like images."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['ANPR_ENABLED'] = '1'
import app.config as cfg
cfg.load_persisted_settings()
print(f"ANPR_ENABLED: {cfg.ANPR_ENABLED}")

import cv2
import numpy as np

# Check PaddleOCR
print("\n=== PaddleOCR Status ===")
from app.services.anpr import _load_paddleocr
engine = _load_paddleocr()
engine_type = type(engine).__name__ if engine else "NONE"
print(f"  Engine: {engine_type}")

from app.services.anpr import recognize_plate

# Test 1: Clean plate image
print("\n=== Test 1: Clean plate ===")
img1 = np.ones((100, 300, 3), dtype=np.uint8) * 200
cv2.putText(img1, "B 2659 BUC", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
start = time.time()
plate, conf, eng = recognize_plate(img1, (0, 0, 300, 100))
print(f"  plate={plate}, conf={conf:.3f}, engine={eng}, time={time.time()-start:.1f}s")

# Test 2: Dark noisy image (night CCTV)
print("\n=== Test 2: Dark/noisy (night CCTV) ===")
img2 = np.ones((80, 200, 3), dtype=np.uint8) * 40
cv2.putText(img2, "D5678XY", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
noise = np.random.randint(0, 20, img2.shape, dtype=np.uint8)
img2 = cv2.add(img2, noise)
start = time.time()
plate, conf, eng = recognize_plate(img2, (0, 0, 200, 80))
print(f"  plate={plate}, conf={conf:.3f}, engine={eng}, time={time.time()-start:.1f}s")

# Test 3: Small plate (like from distance CCTV)
print("\n=== Test 3: Small plate (30px height) ===")
img3 = np.ones((30, 90, 3), dtype=np.uint8) * 160
cv2.putText(img3, "F912KL", (3, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (10, 10, 10), 1)
start = time.time()
plate, conf, eng = recognize_plate(img3, (0, 0, 90, 30))
print(f"  plate={plate}, conf={conf:.3f}, engine={eng}, time={time.time()-start:.1f}s")

# Test 4: Vehicle with plate at bottom
print("\n=== Test 4: Full vehicle image ===")
img4 = np.ones((400, 300, 3), dtype=np.uint8) * 100
# Car body
cv2.rectangle(img4, (30, 50), (270, 350), (80, 80, 80), -1)
# Plate area at bottom
cv2.rectangle(img4, (80, 310), (220, 340), (255, 255, 255), -1)
cv2.putText(img4, "H4321CD", (85, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
start = time.time()
plate, conf, eng = recognize_plate(img4, (30, 50, 270, 350))
print(f"  plate={plate}, conf={conf:.3f}, engine={eng}, time={time.time()-start:.1f}s")

# Test 5: AI identify vehicle
print("\n=== Test 5: ai_identify_vehicle ===")
from app.services.ai_ocr import ai_identify_vehicle
info = ai_identify_vehicle(img4)
if info:
    for k, v in info.items():
        print(f"  {k}: {v}")
else:
    print("  FAILED - returned None")

print("\nDone.")
