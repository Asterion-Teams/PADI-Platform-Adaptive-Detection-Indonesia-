"""
OCR Accuracy Evaluation Tool - Enhanced for 90%+ Target
=======================================================
Comprehensive OCR testing with synthetic data generation and benchmarking.

Usage:
    # Generate synthetic dataset (5000 images)
    python tools/ocr_accuracy_test.py --generate --count 5000 --output data/synthetic_plates/
    
    # Run OCR test on existing dataset
    python tools/ocr_accuracy_test.py --input data/synthetic_plates/ --ground-truth data/synthetic_plates/ground_truth.csv
    
    # Full pipeline: generate + test + report
    python tools/ocr_accuracy_test.py --generate --count 5000 --output data/synthetic_plates/ --run-test --verbose

Target: >= 90% exact match accuracy on synthetic test data.
"""

import os
import sys
import csv
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def levenshtein_distance(a: str, b: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(a) < len(b):
        return levenshtein_distance(b, a)
    if len(b) == 0:
        return len(a)
    previous_row = range(len(b) + 1)
    for i, c1 in enumerate(a):
        current_row = [i + 1]
        for j, c2 in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def character_accuracy(ground_truth: str, ocr_result: str) -> float:
    """Calculate character-level accuracy between ground truth and OCR result."""
    gt = ground_truth.upper().replace(" ", "")
    ocr = ocr_result.upper().replace(" ", "")
    if not gt:
        return 1.0 if not ocr else 0.0
    max_len = max(len(gt), len(ocr))
    if max_len == 0:
        return 1.0
    matches = sum(1 for a, b in zip(gt, ocr) if a == b)
    return matches / max_len


def sequence_similarity(a: str, b: str) -> float:
    """Calculate sequence similarity (0-1)."""
    return SequenceMatcher(None, a.upper().replace(" ", ""), 
                          b.upper().replace(" ", "")).ratio()


def run_ocr_on_image(image_path: str, use_enhanced: bool = True):
    """Run the ANPR OCR pipeline on a single image."""
    import cv2
    import numpy as np
    from app.services.anpr import recognize_plate
    
    img = cv2.imread(image_path)
    if img is None:
        return None, 0.0, "read_error", None
    
    h, w = img.shape[:2]
    bbox = (0, 0, w, h)
    
    start_time = time.time()
    plate_text, confidence, engine = recognize_plate(img, bbox)
    elapsed = time.time() - start_time
    
    return plate_text, confidence, engine, elapsed


def generate_synthetic_dataset(output_dir: str, count: int = 5000, seed: int = 42):
    """Generate synthetic dataset using the synthetic plate generator."""
    print(f"\n[STEP 1] Generating {count} synthetic plate images...")
    
    try:
        from scripts.generate_synthetic_plates import generate_dataset
        stats = generate_dataset(
            output_dir=output_dir,
            count=count,
            include_vehicle_bg=False,  # Start with clean plate-only images
            seed=seed,
        )
        print(f"[OK] Synthetic dataset generated: {stats['total']} images")
        return True
    except ImportError:
        print("[WARN] Synthetic generator not available, using built-in fallback")
        return _generate_fallback_synthetic(output_dir, count, seed)


def _generate_fallback_synthetic(output_dir: str, count: int = 5000, seed: int = 42):
    """Fallback synthetic generator using PIL only."""
    import random
    from PIL import Image, ImageDraw, ImageFont
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    rng = random.Random(seed)
    
    # Indonesian plate prefixes
    prefixes = ['B', 'D', 'F', 'H', 'AB', 'BK', 'DA', 'KB', 'PA', 'EA', 'KA', 'KT', 'KU']
    suffix_chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ'  # No I, O
    
    def get_font(size=50):
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/verdana.ttf",
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    return ImageFont.truetype(fp, size)
                except Exception:
                    continue
        return ImageFont.load_default()
    
    font = get_font(60)
    
    plate_types = [
        ('black', (35, 35, 35), (255, 255, 255)),
        ('yellow', (220, 200, 50), (20, 20, 20)),
        ('white', (245, 245, 245), (20, 20, 20)),
        ('red', (200, 30, 30), (255, 255, 255)),
    ]
    
    metadata = []
    stats = {'black': 0, 'yellow': 0, 'white': 0, 'red': 0, 'total': 0}
    
    for i in range(count):
        ptype, bg_color, text_color = rng.choice(plate_types)
        
        # Generate plate text
        prefix = rng.choice(prefixes)
        num = str(rng.randint(1, 9999)).zfill(4)
        suffix_len = rng.choice([2, 3])
        suffix = ''.join(rng.choices(suffix_chars, k=suffix_len))
        plate_text = f"{prefix} {num} {suffix}"
        
        # Create plate image
        size = (580, 170)
        img = Image.new('RGB', size, bg_color)
        draw = ImageDraw.Draw(img)
        
        # Border
        draw.rectangle([0, 0, size[0]-1, size[1]-1], outline=(200, 200, 200), width=6)
        draw.rectangle([12, 12, size[0]-13, size[1]-13], outline=(180, 180, 180), width=2)
        
        # Text centered
        bbox = draw.textbbox((0, 0), plate_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (size[0] - text_w) // 2
        y = (size[1] - text_h) // 2 - 5
        draw.text((x, y), plate_text, fill=text_color, font=font)
        
        # Apply random degradation
        import numpy as np
        img_array = np.array(img)
        
        # Random blur
        if rng.random() < 0.3:
            blur_size = rng.choice([3, 5, 5])
            img_array = cv2.GaussianBlur(img_array, (blur_size, blur_size), 0)
        
        # Random noise
        if rng.random() < 0.25:
            noise_level = rng.randint(-30, 30)
            noise = np.random.randint(noise_level, -noise_level + 1, img_array.shape, dtype=np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Random darkening
        if rng.random() < 0.2:
            factor = rng.uniform(0.4, 0.7)
            img_array = (img_array * factor).astype(np.uint8)
        
        # Random rotation
        if rng.random() < 0.15:
            angle = rng.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((size[0]//2, size[1]//2), angle, 1.0)
            img_array = cv2.warpAffine(img_array, M, size, borderMode=cv2.BORDER_REPLICATE)
        
        img = Image.fromarray(img_array)
        
        # Save
        filename = f"plate_{i+1:06d}_{ptype}.jpg"
        img_path = output_path / filename
        img.save(img_path, quality=90)
        
        metadata.append({
            'filename': filename,
            'plate_text': plate_text,
            'plate_type': ptype,
        })
        stats[ptype] += 1
        stats['total'] += 1
        
        if (i + 1) % 500 == 0:
            print(f"  Generated {i+1}/{count}...")
    
    # Save ground truth
    csv_path = output_path / 'ground_truth.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("image_name,ground_truth,plate_type\n")
        for m in metadata:
            f.write(f"{m['filename']},{m['plate_text']},{m['plate_type']}\n")
    
    # Save metadata
    json_path = output_path / 'metadata.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'stats': stats, 'total': count, 'seed': seed}, f, indent=2)
    
    print(f"[OK] Generated {stats['total']} images")
    print(f"  Black: {stats['black']} | Yellow: {stats['yellow']} | White: {stats['white']} | Red: {stats['red']}")
    return True


def run_full_test(input_dir: str, ground_truth_csv: str = None,
                 output_csv: str = "hasil_ocr.csv",
                 verbose: bool = False, min_confidence: float = 0.15):
    """Run the full OCR accuracy test."""
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return None
    
    # Load ground truth
    ground_truth = {}
    plate_types = {}
    if ground_truth_csv and Path(ground_truth_csv).exists():
        with open(ground_truth_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ground_truth[row["image_name"]] = row["ground_truth"]
                plate_types[row["image_name"]] = row.get("plate_type", "unknown")
        print(f"[INFO] Loaded {len(ground_truth)} ground truth entries")
    else:
        gt_path = input_path / "ground_truth.csv"
        if gt_path.exists():
            with open(gt_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ground_truth[row["image_name"]] = row["ground_truth"]
                    plate_types[row["image_name"]] = row.get("plate_type", "unknown")
            print(f"[INFO] Loaded {len(ground_truth)} ground truth entries from {gt_path}")
    
    # Collect images
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = sorted([f for f in input_path.iterdir() if f.suffix.lower() in image_exts and f.name != "ground_truth.csv"])
    
    if not images:
        print(f"[ERROR] No images found in {input_dir}")
        return None
    
    print(f"\n[STEP 2] Running OCR on {len(images)} images...")
    print("-" * 80)
    
    # Track metrics
    results = []
    correct_count = 0
    partial_count = 0  # Within 1 char difference
    total_char_accuracy = 0.0
    total_similarity = 0.0
    failed_count = 0
    total_time = 0.0
    
    # Per-engine stats
    engine_stats = {}
    # Per-plate-type stats
    type_stats = {'black': {'correct': 0, 'total': 0},
                  'yellow': {'correct': 0, 'total': 0},
                  'white': {'correct': 0, 'total': 0},
                  'red': {'correct': 0, 'total': 0},
                  'unknown': {'correct': 0, 'total': 0}}
    
    for i, img_path in enumerate(images):
        plate_text, confidence, engine, elapsed = run_ocr_on_image(str(img_path))
        
        gt = ground_truth.get(img_path.name, "")
        ocr_result = plate_text or ""
        ptype = plate_types.get(img_path.name, "unknown")
        
        # Calculate metrics
        exact_match = False
        char_acc = 0.0
        similarity = 0.0
        lev_dist = 999
        
        if gt:
            gt_clean = gt.upper().replace(" ", "")
            ocr_clean = ocr_result.upper().replace(" ", "")
            exact_match = (gt_clean == ocr_clean)
            char_acc = character_accuracy(gt, ocr_result) if ocr_result else 0.0
            similarity = sequence_similarity(gt, ocr_result)
            lev_dist = levenshtein_distance(gt_clean, ocr_clean)
            
            if exact_match:
                correct_count += 1
                type_stats[ptype]['correct'] += 1
            elif lev_dist <= 1:
                partial_count += 1
            
            total_char_accuracy += char_acc
            total_similarity += similarity
        
        type_stats[ptype]['total'] += 1
        
        if not plate_text:
            failed_count += 1
        
        # Track engine stats
        if engine not in engine_stats:
            engine_stats[engine] = {'correct': 0, 'total': 0, 'total_conf': 0.0}
        engine_stats[engine]['total'] += 1
        if exact_match:
            engine_stats[engine]['correct'] += 1
        engine_stats[engine]['total_conf'] += confidence
        
        if elapsed:
            total_time += elapsed
        
        results.append({
            "image_name": img_path.name,
            "ground_truth": gt,
            "ocr_result": ocr_result,
            "confidence": f"{confidence:.3f}",
            "engine": engine,
            "exact_match": "YES" if exact_match else ("PARTIAL" if lev_dist <= 1 else ("NO" if gt else "N/A")),
            "char_accuracy": f"{char_acc:.3f}",
            "similarity": f"{similarity:.3f}",
            "lev_dist": lev_dist,
            "elapsed_ms": f"{(elapsed or 0)*1000:.0f}",
        })
        
        # Progress output
        status = "✓" if exact_match else ("~" if lev_dist <= 1 else ("✗" if gt else "?"))
        if verbose or (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(images)}] {status} {img_path.name}: OCR='{ocr_result}' GT='{gt}' conf={confidence:.2f} ({engine})")
    
    # Save results
    output_path = Path(output_csv)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "ground_truth", "ocr_result",
                                                "confidence", "engine", "exact_match",
                                                "char_accuracy", "similarity", "lev_dist", "elapsed_ms"])
        writer.writeheader()
        writer.writerows(results)
    
    # Print comprehensive report
    print("\n" + "=" * 80)
    print("                    OCR ACCURACY TEST REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {input_dir}")
    print(f"Total images: {len(images)}")
    print(f"Results saved: {output_path}")
    print("-" * 80)
    
    # Detection stats
    detected = len(images) - failed_count
    print(f"\n  DETECTION RATE:")
    print(f"    Plates detected:     {detected}/{len(images)} ({detected/len(images)*100:.1f}%)")
    print(f"    Failed to read:     {failed_count}/{len(images)} ({failed_count/len(images)*100:.1f}%)")
    
    if ground_truth:
        gt_tested = sum(1 for r in results if r["exact_match"] != "N/A")
        if gt_tested > 0:
            exact_accuracy = correct_count / gt_tested * 100
            avg_char_accuracy = total_char_accuracy / gt_tested * 100
            avg_similarity = total_similarity / gt_tested * 100
            error_count = gt_tested - correct_count
            avg_time_ms = (total_time / gt_tested) * 1000 if gt_tested > 0 else 0
            
            print(f"\n  ACCURACY METRICS (exact match = full character match):")
            print(f"    Exact Match:        {correct_count}/{gt_tested} = {exact_accuracy:.2f}%")
            print(f"    Within 1 char:      {partial_count}/{gt_tested} ({partial_count/gt_tested*100:.1f}%)")
            print(f"    Character Accuracy: {avg_char_accuracy:.2f}%")
            print(f"    Avg Similarity:     {avg_similarity:.2f}%")
            print(f"    Error Rate:         {error_count}/{gt_tested} = {error_count/gt_tested*100:.2f}%")
            print(f"    Avg time/image:     {avg_time_ms:.0f} ms")
            
            # Check target
            target = 90.0
            print(f"\n  TARGET CHECK: {target}% exact match")
            if exact_accuracy >= target:
                print(f"  ✓ PASSED! ({exact_accuracy:.2f}% >= {target}%)")
            else:
                deficit = target - exact_accuracy
                print(f"  ✗ Below target by {deficit:.2f}%")
            
            print(f"\n  PER PLATE TYPE:")
            print(f"    {'Type':<10} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
            print(f"    {'-'*40}")
            for ptype, stat in type_stats.items():
                if stat['total'] > 0:
                    acc = stat['correct'] / stat['total'] * 100
                    print(f"    {ptype:<10} {stat['correct']:<10} {stat['total']:<10} {acc:.1f}%")
            
            print(f"\n  PER ENGINE:")
            print(f"    {'Engine':<20} {'Correct':<10} {'Total':<10} {'Accuracy':<10} {'Avg Conf':<10}")
            print(f"    {'-'*60}")
            for eng, stat in sorted(engine_stats.items(), key=lambda x: x[1]['correct']/max(1,x[1]['total']), reverse=True):
                if stat['total'] > 0:
                    acc = stat['correct'] / stat['total'] * 100
                    avg_conf = stat['total_conf'] / stat['total']
                    print(f"    {eng:<20} {stat['correct']:<10} {stat['total']:<10} {acc:.1f}% {avg_conf:.3f}")
    
    print("\n" + "=" * 80)
    return {
        'exact_accuracy': exact_accuracy if ground_truth else 0,
        'char_accuracy': avg_char_accuracy if ground_truth else 0,
        'detection_rate': detected/len(images)*100,
        'total_images': len(images),
        'correct': correct_count,
        'failed': failed_count,
        'avg_time_ms': avg_time_ms if ground_truth else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="OCR Accuracy Test Tool - Enhanced for 90%%+ Target",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and test 5000 synthetic plates
  python tools/ocr_accuracy_test.py --generate --count 5000 -o data/synthetic_plates/ --run-test
  
  # Test existing dataset
  python tools/ocr_accuracy_test.py --input data/synthetic_plates/ --ground-truth data/synthetic_plates/ground_truth.csv
  
  # Quick test (100 images)
  python tools/ocr_accuracy_test.py --generate --count 100 -o data/test_plates/ --run-test --verbose
        """
    )
    parser.add_argument("--input", "-i", help="Directory with plate images")
    parser.add_argument("--ground-truth", "-g", help="CSV with ground truth (image_name,ground_truth)")
    parser.add_argument("--output", "-o", default="hasil_ocr.csv", help="Output CSV path")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic dataset first")
    parser.add_argument("--count", "-n", type=int, default=1000, help="Number of synthetic images to generate")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--run-test", action="store_true", help="Run OCR test after generation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--min-conf", type=float, default=0.15, help="Minimum confidence threshold")
    
    args = parser.parse_args()
    
    output_dir = args.input or f"data/synthetic_plates/"
    
    # Step 1: Generate if requested
    if args.generate:
        generate_synthetic_dataset(output_dir, args.count, args.seed)
    
    # Step 2: Run test
    gt_csv = args.ground_truth or str(Path(output_dir) / "ground_truth.csv")
    
    if args.run_test or args.input:
        result = run_full_test(
            input_dir=output_dir,
            ground_truth_csv=gt_csv if Path(gt_csv).exists() else None,
            output_csv=args.output,
            verbose=args.verbose,
            min_confidence=args.min_conf,
        )
        
        if result and result['exact_accuracy'] >= 90.0:
            print("\n✓ TARGET ACHIEVED: 90%+ exact match accuracy!")
        elif result:
            print(f"\nNote: Current accuracy is {result['exact_accuracy']:.1f}%. To reach 90%:")
            print("  - Try with AI API enabled (AI_USE_FOR_ANPR=true)")
            print("  - Use higher quality images")
            print("  - Add more synthetic training data")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
