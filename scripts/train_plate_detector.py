"""
Train License Plate Detection Model
Uses: vehicle and license plate.v1-license-plate-only.yolov11 dataset (Roboflow)
Output: models/plate_detector_best.pt

This model detects WHERE the license plate is in the image.
Then EasyOCR reads the text from the cropped plate region.

Usage:
    python scripts/train_plate_detector.py
"""
import os
import sys

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, ".deps"))

DATASET_DIR = os.path.join(ROOT, "models", "vehicle and license plate.v1-license-plate-only.yolov11")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
OUTPUT_DIR = os.path.join(ROOT, "models", "runs", "train", "plate_detector")
FINAL_MODEL_PATH = os.path.join(ROOT, "models", "plate_detector_best.pt")

# Training config
EPOCHS = 50
IMGSZ = 640
BATCH = 8  # Adjust based on GPU VRAM (8 for 4GB, 16 for 8GB+)
BASE_MODEL = "yolo11n.pt"  # Nano for fast inference


def main():
    from ultralytics import YOLO
    import shutil

    print("=" * 60)
    print("  License Plate Detector Training")
    print("  Dataset:", DATASET_DIR)
    print("  Base model:", BASE_MODEL)
    print("  Epochs:", EPOCHS)
    print("  Image size:", IMGSZ)
    print("=" * 60)

    if not os.path.isfile(DATA_YAML):
        print(f"ERROR: data.yaml not found at {DATA_YAML}")
        sys.exit(1)

    # Fix data.yaml paths to absolute
    fixed_yaml = os.path.join(DATASET_DIR, "data_abs.yaml")
    with open(DATA_YAML, 'r') as f:
        content = f.read()
    
    content = content.replace("../train/images", os.path.join(DATASET_DIR, "train", "images").replace("\\", "/"))
    content = content.replace("../valid/images", os.path.join(DATASET_DIR, "valid", "images").replace("\\", "/"))
    content = content.replace("../test/images", os.path.join(DATASET_DIR, "test", "images").replace("\\", "/"))
    
    with open(fixed_yaml, 'w') as f:
        f.write(content)
    
    print(f"\nFixed data.yaml written to: {fixed_yaml}")
    print("Starting training...\n")

    model = YOLO(BASE_MODEL)
    results = model.train(
        data=fixed_yaml,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=os.path.join(ROOT, "models", "runs", "train"),
        name="plate_detector",
        exist_ok=True,
        device=0,  # GPU
        patience=10,
        save=True,
        plots=True,
    )

    # Copy best weights to models/
    best_pt = os.path.join(OUTPUT_DIR, "weights", "best.pt")
    if os.path.isfile(best_pt):
        shutil.copy2(best_pt, FINAL_MODEL_PATH)
        print(f"\n{'=' * 60}")
        print(f"  Training complete!")
        print(f"  Best model saved to: {FINAL_MODEL_PATH}")
        print(f"  Size: {os.path.getsize(FINAL_MODEL_PATH) / 1024 / 1024:.1f} MB")
        print(f"{'=' * 60}")
    else:
        print("WARNING: best.pt not found after training")


if __name__ == "__main__":
    main()
