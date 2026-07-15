"""
Synthetic Indonesian License Plate Generator
============================================
Generates realistic synthetic plate images with diverse conditions
for training and testing OCR systems.

Usage:
    python scripts/generate_synthetic_plates.py --count 5000 --output data/synthetic_plates/
    python scripts/generate_synthetic_plates.py --count 10000 --output data/synthetic_plates/ --augment

Conditions simulated:
- Normal daylight
- Night/low-light
- Rain/haze
- Motion blur
- Various rotations/skew
- Shadow/overexposure
- Partial occlusion
- Different plate types (black/yellow/white/red)
"""

import os
import sys
import random
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))


# Indonesian plate format rules
INDONESIAN_PREFIXES = [
    'A', 'B', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W',
    'AB', 'AD', 'AE', 'AG', 'BA', 'BB', 'BD', 'BE', 'BG', 'BH', 'BK', 'BL', 'BM',
    'BN', 'BP', 'DA', 'DB', 'DC', 'DD', 'DE', 'DG', 'DH', 'DK', 'DL', 'DM', 'DN',
    'DR', 'DS', 'DT', 'DW', 'EA', 'EB', 'ED', 'KA', 'KB', 'KD', 'KH', 'KT', 'KU',
    'PA', 'PB', 'QA', 'RI', 'R', 'Z'
]

PLATE_COLORS = {
    'black': {  # Private vehicle (old)
        'bg': (30, 30, 30),
        'text': (255, 255, 255),
        'border': (220, 220, 220),
    },
    'yellow': {  # Public vehicle / taxi
        'bg': (220, 200, 60),
        'text': (20, 20, 20),
        'border': (180, 160, 40),
    },
    'white': {  # New format (2022+)
        'bg': (245, 245, 245),
        'text': (20, 20, 20),
        'border': (200, 200, 200),
    },
    'red': {  # Government
        'bg': (200, 30, 30),
        'text': (255, 255, 255),
        'border': (180, 20, 20),
    },
}

# Font paths - use system fonts
FONT_PATHS = [
    # Windows paths
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/verdana.ttf",
    "C:/Windows/Fonts/verdanab.ttf",
    "C:/Windows/Fonts/ calibri.ttf",
    "C:/Windows/Fonts/ calibrib.ttf",
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/consola.ttf",
    # Fallback
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

def get_font(size=60):
    """Get a usable font for plate rendering."""
    for font_path in FONT_PATHS:
        try:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def generate_plate_number(rng: random.Random) -> str:
    """Generate a realistic Indonesian plate number."""
    prefix = rng.choice(INDONESIAN_PREFIXES)
    num_digits = rng.choice([3, 4])  # Most plates have 4 digits
    number = str(rng.randint(1, 10**num_digits - 1)).zfill(num_digits)
    suffix_len = rng.choice([1, 2, 3])
    suffix_chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ'  # Exclude I, O
    suffix = ''.join(rng.choices(suffix_chars, k=suffix_len))
    return f"{prefix} {number} {suffix}"


def render_plate_image(plate_text: str, plate_type: str, size=(600, 200),
                       rng: random.Random = None) -> Image.Image:
    """Render a single plate image with the given text and type."""
    if rng is None:
        rng = random.Random()
    
    colors = PLATE_COLORS.get(plate_type, PLATE_COLORS['black'])
    
    # Create base image
    img = Image.new('RGB', size, colors['bg'])
    draw = ImageDraw.Draw(img)
    
    # Add border/frame effect
    border_w = 8
    draw.rectangle([0, 0, size[0]-1, size[1]-1], outline=colors['border'], width=border_w)
    # Inner border
    inner_margin = 15
    draw.rectangle([inner_margin, inner_margin, size[0]-1-inner_margin, size[1]-1-inner_margin],
                   outline=colors['border'], width=3)
    
    # Split plate text for proper layout
    parts = plate_text.split()
    if len(parts) == 3:
        prefix, number, suffix = parts
    elif len(parts) == 2:
        prefix, number = parts
        suffix = ""
    else:
        prefix, number, suffix = plate_text[0], plate_text[1:4], plate_text[4:]
    
    # Font size based on plate type
    font_size = 65
    font = get_font(font_size)
    
    # Calculate positions - Indonesian plate layout
    # [PREFIX] [NUMBER] [SUFFIX] horizontally
    total_text = f"{prefix} {number}"
    if suffix:
        total_text += f" {suffix}"
    
    # Get text bounding boxes
    prefix_bbox = draw.textbbox((0, 0), prefix, font=font)
    num_bbox = draw.textbbox((0, 0), number, font=font)
    if suffix:
        suffix_bbox = draw.textbbox((0, 0), suffix, font=font)
    
    prefix_w = prefix_bbox[2] - prefix_bbox[0]
    num_w = num_bbox[2] - num_bbox[0]
    total_w = prefix_w + 40 + num_w
    if suffix:
        suffix_w = suffix_bbox[2] - suffix_bbox[0]
        total_w += 40 + suffix_w
    
    # Center the text
    start_x = (size[0] - total_w) // 2
    y = (size[1] - (num_bbox[3] - num_bbox[1])) // 2 - 5
    
    # Draw prefix (letters)
    draw.text((start_x, y), prefix, fill=colors['text'], font=font)
    x = start_x + prefix_w
    
    # Draw number
    draw.text((x + 40, y), number, fill=colors['text'], font=font)
    x += num_w + 40
    
    # Draw suffix
    if suffix:
        draw.text((x, y), suffix, fill=colors['text'], font=font)
    
    return img


def add_weather_effect(img: Image.Image, weather: str, rng: random.Random) -> Image.Image:
    """Add weather/environmental effects."""
    if weather == 'none' or weather is None:
        return img
    
    img_array = np.array(img)
    
    if weather == 'rain':
        # Rain streaks
        for _ in range(200):
            x = rng.randint(0, img_array.shape[1]-1)
            y1 = rng.randint(0, img_array.shape[0]-20)
            length = rng.randint(5, 25)
            alpha = rng.randint(100, 200)
            cv2.line(img_array, (x, y1), (x, y1 + length), (180, 180, 220, alpha), 1)
    
    elif weather == 'haze':
        # Light haze/fog overlay
        haze = rng.randint(30, 80)
        img_array = cv2.addWeighted(img_array, 0.85, 
                                     np.full_like(img_array, [haze, haze, haze]), 0.15, 0)
    
    elif weather == 'rain_haze':
        # Combined
        for _ in range(100):
            x = rng.randint(0, img_array.shape[1]-1)
            y1 = rng.randint(0, img_array.shape[0]-15)
            length = rng.randint(3, 15)
            cv2.line(img_array, (x, y1), (x, y1 + length), (170, 170, 210), 1)
        haze = rng.randint(20, 50)
        img_array = cv2.addWeighted(img_array, 0.9,
                                     np.full_like(img_array, [haze, haze, haze]), 0.1, 0)
    
    return Image.fromarray(img_array)


def add_lighting_condition(img: Image.Image, condition: str, rng: random.Random) -> Image.Image:
    """Add lighting conditions."""
    if condition == 'normal' or condition is None:
        return img
    
    img_array = np.array(img)
    
    if condition == 'dark':
        # Night / low light
        dark_factor = rng.uniform(0.3, 0.55)
        img_array = (img_array * dark_factor).astype(np.uint8)
        # Add some noise
        noise = rng.integers(-15, 15, img_array.shape, dtype=np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    elif condition == 'overexposed':
        # Bright sunlight / overexposed
        bright_factor = rng.uniform(1.3, 1.7)
        img_array = np.clip(img_array * bright_factor, 0, 255).astype(np.uint8)
    
    elif condition == 'shadow':
        # Partial shadow
        h, w = img_array.shape[:2]
        shadow_type = rng.randint(0, 2)
        if shadow_type == 0:
            # Top shadow
            overlay = np.full_like(img_array, 0)
            shadow_h = rng.randint(h//4, h//2)
            overlay[:shadow_h, :] = img_array[:shadow_h, :] // 2
            overlay[shadow_h:, :] = img_array[shadow_h:, :]
            img_array = overlay
        elif shadow_type == 1:
            # Gradient shadow
            gradient = np.linspace(0.3, 1.0, h).reshape(h, 1, 1)
            img_array = (img_array * gradient).astype(np.uint8)
        else:
            # Side shadow
            overlay = img_array.copy()
            shadow_w = rng.randint(w//4, w//2)
            overlay[:, :shadow_w] = (img_array[:, :shadow_w] * 0.4).astype(np.uint8)
            img_array = overlay
    
    return Image.fromarray(img_array)


def add_blur(img: Image.Image, blur_type: str, rng: random.Random) -> Image.Image:
    """Add blur effects."""
    if blur_type == 'none' or blur_type is None:
        return img
    
    img_array = np.array(img)
    
    if blur_type == 'gaussian_light':
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0.5)
    elif blur_type == 'gaussian_medium':
        img_array = cv2.GaussianBlur(img_array, (5, 5), 1.0)
    elif blur_type == 'gaussian_heavy':
        img_array = cv2.GaussianBlur(img_array, (7, 7), 1.5)
    elif blur_type == 'motion':
        # Simulate motion blur
        kernel_size = rng.randint(5, 15)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        img_array = cv2.filter2D(img_array, -1, kernel)
    elif blur_type == 'defocus':
        # Simulate defocus blur
        img_array = cv2.GaussianBlur(img_array, (0, 0), rng.uniform(2.0, 4.0))
    
    return Image.fromarray(img_array)


def add_noise(img: Image.Image, noise_level: str, rng: random.Random) -> Image.Image:
    """Add sensor noise."""
    if noise_level == 'none' or noise_level is None:
        return img
    
    img_array = np.array(img)
    
    if noise_level == 'light':
        noise = rng.integers(-20, 20, img_array.shape, dtype=np.int16)
    elif noise_level == 'medium':
        noise = rng.integers(-40, 40, img_array.shape, dtype=np.int16)
    elif noise_level == 'heavy':
        noise = rng.integers(-60, 60, img_array.shape, dtype=np.int16)
    elif noise_level == 'salt_pepper':
        # Salt and pepper noise
        prob = rng.uniform(0.01, 0.05)
        mask = rng.random(img_array.shape[:2])
        noise = np.zeros_like(img_array, dtype=np.int16)
        noise[mask < prob/2] = -255
        noise[mask > 1 - prob/2] = 255
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    else:
        return img
    
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def add_distortion(img: Image.Image, distortion: str, rng: random.Random) -> Image.Image:
    """Add geometric distortions."""
    if distortion == 'none' or distortion is None:
        return img
    
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    
    if distortion == 'slight_rotation':
        angle = rng.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img_array = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    elif distortion == 'moderate_rotation':
        angle = rng.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img_array = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    elif distortion == 'perspective':
        # Perspective distortion
        pts1 = np.float32([[20, 20], [w-20, 20], [20, h-20], [w-20, h-20]])
        dx = rng.uniform(-30, 30)
        dy = rng.uniform(-15, 15)
        pts2 = np.float32([[20+dx, 20+dy], [w-20-dx, 20-dy], 
                          [20-dx, h-20+dy], [w-20+dx, h-20-dy]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_array = cv2.warpPerspective(img_array, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    elif distortion == 'keystone':
        # Vertical keystone (common with angled cameras)
        skew_x = rng.uniform(-0.15, 0.15)
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([[int(w*skew_x), 0], [int(w*(1-skew_x)), 0], [0, h], [w, h]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_array = cv2.warpPerspective(img_array, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return Image.fromarray(img_array)


def generate_single_plate(rng: random.Random,
                         plate_types=['black', 'yellow', 'white', 'red'],
                         include_vehicle_bg=False) -> tuple[Image.Image, str]:
    """Generate a single synthetic plate with random conditions."""
    
    # Random plate properties
    plate_type = rng.choice(plate_types)
    plate_text = generate_plate_number(rng)
    
    # Base plate size
    base_size = (rng.randint(400, 700), rng.randint(120, 200))
    
    # Render clean plate
    plate_img = render_plate_image(plate_text, plate_type, base_size, rng)
    
    # Apply distortions FIRST
    distortion = rng.choice([
        'none', 'none', 'none',  # 30% normal
        'slight_rotation',
        'moderate_rotation',
        'perspective',
        'keystone',
    ])
    plate_img = add_distortion(plate_img, distortion, rng)
    
    # Apply blur
    blur = rng.choice([
        'none', 'none', 'none',  # 33% sharp
        'gaussian_light',
        'gaussian_medium',
        'gaussian_heavy',
        'motion',
        'defocus',
    ])
    plate_img = add_blur(plate_img, blur, rng)
    
    # Apply noise
    noise = rng.choice([
        'none', 'none',  # 25% clean
        'light', 'medium', 'heavy',
        'salt_pepper',
    ])
    plate_img = add_noise(plate_img, noise, rng)
    
    # Apply lighting
    lighting = rng.choice([
        'normal', 'normal',  # 33% normal
        'dark',
        'overexposed',
        'shadow',
    ])
    plate_img = add_lighting_condition(plate_img, lighting, rng)
    
    # Apply weather
    weather = rng.choice([
        'none', 'none',  # 33% clear
        'rain', 'haze', 'rain_haze',
    ])
    plate_img = add_weather_effect(plate_img, weather, rng)
    
    # Resize to standard output size (simulating different distances)
    output_sizes = [
        (320, 100), (400, 120), (480, 140),
        (540, 160), (600, 180), (700, 210)
    ]
    target_size = rng.choice(output_sizes)
    plate_img = plate_img.resize(target_size, Image.LANCZOS)
    
    # If including vehicle background, embed in vehicle-like image
    if include_vehicle_bg:
        bg_h, bg_w = rng.randint(400, 700), rng.randint(600, 900)
        bg_color = rng.randint(60, 180)
        bg = Image.new('RGB', (bg_w, bg_h), (bg_color, bg_color, bg_color + 20))
        
        # Paste plate at random position (bottom center typical)
        px = rng.randint(bg_w // 3, 2 * bg_w // 3)
        py = rng.randint(int(bg_h * 0.55), int(bg_h * 0.85))
        
        # For proper pasting, resize plate to realistic size
        plate_scaled = plate_img.resize((target_size[0] * 2 // 3, target_size[1] * 2 // 3), Image.LANCZOS)
        
        # Ensure plate fits
        px = min(px, bg_w - plate_scaled.width)
        py = min(py, bg_h - plate_scaled.height)
        
        bg.paste(plate_scaled, (px, py))
        plate_img = bg
        target_size = (bg_w, bg_h)
    
    return plate_img, plate_text


def generate_dataset(output_dir: str, count: int = 5000,
                   include_vehicle_bg: bool = False,
                   seed: int = 42,
                   save_metadata: bool = True):
    """Generate a full synthetic dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    rng = random.Random(seed)
    
    # Distribution: 40% black, 30% yellow, 20% white, 10% red
    plate_types = ['black'] * 40 + ['yellow'] * 30 + ['white'] * 20 + ['red'] * 10
    
    metadata = []
    stats = {'total': 0, 'black': 0, 'yellow': 0, 'white': 0, 'red': 0}
    
    print(f"Generating {count} synthetic plate images...")
    print(f"Output: {output_path}")
    print(f"Vehicle background: {include_vehicle_bg}")
    print("-" * 50)
    
    for i in range(count):
        ptype = rng.choice(plate_types)
        
        # Generate with controlled RNG
        img, plate_text = generate_single_plate(
            rng, plate_types=[ptype],
            include_vehicle_bg=include_vehicle_bg
        )
        
        # Save image
        filename = f"plate_{i+1:06d}_{ptype}.jpg"
        img_path = output_path / filename
        img.save(img_path, quality=95, optimize=True)
        
        metadata.append({
            'filename': filename,
            'plate_text': plate_text,
            'plate_type': ptype,
            'image_size': list(img.size),
        })
        
        stats['total'] += 1
        stats[ptype] += 1
        
        if (i + 1) % 500 == 0:
            print(f"  Generated {i+1}/{count}...")
    
    # Save ground truth CSV
    csv_path = output_path / 'ground_truth.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("image_name,ground_truth,plate_type\n")
        for m in metadata:
            f.write(f"{m['filename']},{m['plate_text']},{m['plate_type']}\n")
    
    # Save metadata JSON
    if save_metadata:
        json_path = output_path / 'metadata.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'stats': stats,
                'total': count,
                'seed': seed,
                'include_vehicle_bg': include_vehicle_bg,
            }, f, indent=2)
    
    print("-" * 50)
    print(f"Done! Generated {stats['total']} images")
    print(f"  Black: {stats['black']} ({stats['black']/stats['total']*100:.1f}%)")
    print(f"  Yellow: {stats['yellow']} ({stats['yellow']/stats['total']*100:.1f}%)")
    print(f"  White: {stats['white']} ({stats['white']/stats['total']*100:.1f}%)")
    print(f"  Red: {stats['red']} ({stats['red']/stats['total']*100:.1f}%)")
    print(f"Ground truth: {csv_path}")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Indonesian plate images")
    parser.add_argument("--count", "-n", type=int, default=5000,
                        help="Number of images to generate (default: 5000)")
    parser.add_argument("--output", "-o", default="data/synthetic_plates",
                        help="Output directory")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--with-bg", action="store_true",
                        help="Embed plates in vehicle-like background")
    parser.add_argument("--min-count", type=int, default=None,
                        help="Minimum plate count in dataset")
    
    args = parser.parse_args()
    
    # Check if we should expand existing dataset
    output_path = Path(args.output)
    if output_path.exists() and output_path.is_dir():
        existing_csv = output_path / 'ground_truth.csv'
        if existing_csv.exists():
            with open(existing_csv, 'r', encoding='utf-8') as f:
                existing_count = sum(1 for _ in f) - 1  # subtract header
            if args.min_count and existing_count >= args.min_count:
                print(f"Dataset already has {existing_count} images. Skipping generation.")
                print(f"Ground truth: {existing_csv}")
                sys.exit(0)
            elif args.count <= existing_count:
                print(f"Output dir has {existing_count} images, generating {args.count} more...")
                # Generate more
    
    generate_dataset(
        output_dir=args.output,
        count=args.count,
        include_vehicle_bg=args.with_bg,
        seed=args.seed,
    )
