"""
Train YOLOv8 on the vehicle-detection v3 dataset.

Presets:
    fast       - 1 epoch, yolov8n, img 320, batch 8 (~5-10 min on CPU)
                 Just to produce a working best.pt for integration demo.
    balanced   - 20 epochs, yolov8s, img 640, batch 16 (several hours on CPU)
    full       - 100 epochs, yolov8l, img 640, batch 16 (full GPU training)

Outputs go to models/runs/train/<name>/weights/best.pt.
A symlink/copy to models/vehicle_v3_best.pt is also created so the
running app can auto-detect it.

Usage:
    python scripts/train_vehicle_yolo.py --preset fast
    python scripts/train_vehicle_yolo.py --preset balanced --device 0
    python scripts/train_vehicle_yolo.py --epochs 50 --model yolov8m.pt --imgsz 640 --batch 16
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

# Allow importing local deps
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEPS_DIR = os.path.join(REPO_ROOT, ".deps")
if os.path.isdir(DEPS_DIR) and DEPS_DIR not in sys.path:
    sys.path.insert(0, DEPS_DIR)


PRESETS = {
    "fast": {
        "model": "yolov8n.pt",
        "epochs": 1,
        "imgsz": 320,
        "batch": 8,
        "patience": 0,
    },
    "balanced": {
        "model": "yolov8s.pt",
        "epochs": 20,
        "imgsz": 640,
        "batch": 16,
        "patience": 5,
    },
    "full": {
        "model": "yolov8l.pt",
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "patience": 20,
    },
    # YOLO11m fine-tune preset — recommended for competition
    # Best balance of accuracy vs speed for real-time CCTV processing
    "yolo11m": {
        "model": "yolo11m.pt",
        "epochs": 30,
        "imgsz": 640,
        "batch": 4,
        "patience": 10,
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=sorted(PRESETS.keys()), default=None,
                   help="Quick preset. Overrides individual flags.")
    p.add_argument("--model", default="yolov8n.pt", help="Base model or weights to start from")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu",
                   help="'cpu', '0', '0,1', etc. Default 'cpu' because no CUDA detected.")
    p.add_argument("--name", default=None,
                   help="Run name under models/runs/train. Default is <preset>_<timestamp>.")
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    if args.preset:
        cfg = PRESETS[args.preset]
        args.model = cfg["model"]
        args.epochs = cfg["epochs"]
        args.imgsz = cfg["imgsz"]
        args.batch = cfg["batch"]
        args.patience = cfg["patience"]
        if not args.name:
            args.name = f"{args.preset}_{time.strftime('%Y%m%d_%H%M%S')}"

    if not args.name:
        args.name = f"custom_{time.strftime('%Y%m%d_%H%M%S')}"

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not available. Install with: pip install ultralytics")
        return 1

    dataset_root = os.path.join(REPO_ROOT, "models", "vehicle-detection.v3i.yolov8")
    data_yaml = os.path.join(dataset_root, "data.yaml")
    if not os.path.isfile(data_yaml):
        print(f"[ERROR] data.yaml not found at {data_yaml}")
        print("        Run scripts/prepare_vehicle_dataset.py first.")
        return 1

    # Make sure val/ exists (ultralytics will fail silently otherwise)
    valid_dir = os.path.join(dataset_root, "valid", "images")
    if not os.path.isdir(valid_dir) or not os.listdir(valid_dir):
        print(f"[ERROR] valid/images empty. Run: python scripts/prepare_vehicle_dataset.py")
        return 1

    # Resolve base model: prefer local copy under models/, else let YOLO download
    base_model = args.model
    local_base = os.path.join(REPO_ROOT, "models", args.model)
    if os.path.isfile(local_base):
        base_model = local_base

    runs_dir = os.path.join(REPO_ROOT, "models", "runs")
    os.makedirs(runs_dir, exist_ok=True)

    print(f"[INFO] Base model   : {base_model}")
    print(f"[INFO] Dataset yaml : {data_yaml}")
    print(f"[INFO] Epochs/img/batch: {args.epochs} / {args.imgsz} / {args.batch}")
    print(f"[INFO] Device       : {args.device}")
    print(f"[INFO] Run name     : {args.name}")
    print(f"[INFO] Output       : {runs_dir}/train/{args.name}")

    # Train
    model = YOLO(base_model)
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=os.path.join(runs_dir, "train"),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        resume=args.resume,
        verbose=True,
    )

    # Copy best.pt to stable paths so the app can auto-detect it.
    best_src = os.path.join(runs_dir, "train", args.name, "weights", "best.pt")

    # Primary: YOLO11m fine-tuned path (main model for competition)
    stable_ft_dst = os.path.join(REPO_ROOT, "models", "vehicle_v3_yolo11m_best.pt")
    # Legacy: backward compatibility path
    stable_dst = os.path.join(REPO_ROOT, "models", "vehicle_v3_best.pt")

    if os.path.isfile(best_src):
        shutil.copy2(best_src, stable_ft_dst)
        shutil.copy2(best_src, stable_dst)
        size_mb = os.path.getsize(stable_ft_dst) / 1024 / 1024
        model_size_mb = size_mb
        print(f"\n[OK] Training done. Best weights copied to:")
        print(f"     {stable_ft_dst}  ({model_size_mb:.1f} MB)")
        print(f"     {stable_dst}  ({model_size_mb:.1f} MB)")
        print(f"     Source: {best_src}")
        print(f"\nThe Flask app will auto-pick this up on next startup.")
    else:
        print(f"[WARN] best.pt not found at {best_src}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
