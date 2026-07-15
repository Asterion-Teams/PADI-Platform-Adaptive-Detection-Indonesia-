"""
Enhanced ANPR Preprocessing Pipeline
=====================================
Provides 12+ specialized preprocessing variants for maximum OCR accuracy.
Each variant targets different conditions (night, rain, blur, etc.)
and the best result is selected through ensemble voting.

Usage:
    from app.services.anpr_enhanced import get_all_variants, select_best_plate
    
    variants = get_all_variants(img)
    best_plate = select_best_plate(variants, ocr_func)
"""

import cv2
import numpy as np


# ============================================================================
# CORE PREPROCESSING VARIANTS
# Each returns a processed image + descriptive tag
# ============================================================================

def preprocess_original(img):
    """Original image, no processing."""
    return img.copy(), "original"


def preprocess_adaptive_clahe(img, clip_limit=3.0, tile_size=(8,8)):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Best for: Low-light, dark plates, shadow conditions."""
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return result, f"clahe_{clip_limit}"
    except Exception:
        return img, "clahe_failed"


def preprocess_grayscale_clahe(img):
    """Grayscale + CLAHE. Best for: Overexposed plates, bright sunlight."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), "gray_clahe"
    except Exception:
        return img, "gray_clahe_failed"


def preprocess_binary_otsu(img):
    """Binary threshold (Otsu's method). Best for: High contrast, clear plates."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "binary_otsu"
    except Exception:
        return img, "binary_otsu_failed"


def preprocess_binary_inverse(img):
    """Inverted binary. Best for: White text on dark background (private plates)."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "binary_inv_otsu"
    except Exception:
        return img, "binary_inv_failed"


def preprocess_adaptive_threshold(img, block_size=11, c=2):
    """Adaptive threshold. Best for: Uneven lighting, mixed light/dark areas."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c
        )
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), f"adaptive_thresh_{block_size}"
    except Exception:
        return img, "adaptive_thresh_failed"


def preprocess_sharpen(img, strength=1.8):
    """Unsharp mask sharpening. Best for: Blurry plates, motion blur."""
    try:
        blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
        sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8), f"sharpen_{strength}"
    except Exception:
        return img, "sharpen_failed"


def preprocess_denoise(img, h=10, hColor=10):
    """Non-local means denoising. Best for: Noisy images, night CCTV."""
    try:
        denoised = cv2.fastNlMeansDenoisingColored(img, None, h, hColor, 7, 21)
        return denoised, f"denoise_{h}"
    except Exception:
        return img, "denoise_failed"


def preprocess_gamma_correction(img, gamma=1.5):
    """Gamma correction. Best for: Overexposed (gamma<1) or underexposed (gamma>1) images."""
    try:
        inv_gamma = 1.0 / max(gamma, 0.1)
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        lut = cv2.LUT(img, table)
        return lut, f"gamma_{gamma}"
    except Exception:
        return img, "gamma_failed"


def preprocess_histogram_equalization(img):
    """Histogram equalization. Best for: Low contrast images."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR), "hist_eq"
    except Exception:
        return img, "hist_eq_failed"


def preprocess_edge_enhanced(img):
    """Edge-enhanced using Sobel. Best for: Plates with faint text."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobelx, sobely)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Enhance edges in original color
        edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.addWeighted(img, 0.7, edge_bgr, 0.3, 0)
        return enhanced, "edge_enhanced"
    except Exception:
        return img, "edge_enhanced_failed"


def preprocess_morphology(img):
    """Morphological operations (close + open). Best for: Broken text, gaps in characters."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR), "morphology"
    except Exception:
        return img, "morphology_failed"


def preprocess_color_normalized(img):
    """Color normalization. Best for: Plates with unusual color casts."""
    try:
        result = img.copy()
        for i in range(3):
            channel = result[:, :, i]
            result[:, :, i] = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
        return result, "color_norm"
    except Exception:
        return img, "color_norm_failed"


def preprocess_top_hat(img):
    """Top-hat transform. Best for: Dark text on bright background, emphasizes text."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        tophat = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(tophat, cv2.COLOR_GRAY2BGR), "tophat"
    except Exception:
        return img, "tophat_failed"


def preprocess_dehaze(img):
    """Simple dehazing. Best for: Hazy, foggy, rainy conditions."""
    try:
        # CLAHE + slight gamma boost + de-noise combo
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 15)
        return denoised, "dehaze"
    except Exception:
        return img, "dehaze_failed"


def preprocess_stretched(img):
    """Full dynamic range stretch. Best for: Very dark or washed-out plates."""
    try:
        result = np.zeros_like(img)
        for c in range(3):
            channel = img[:, :, c]
            min_val = channel.min()
            max_val = channel.max()
            if max_val > min_val:
                result[:, :, c] = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                result[:, :, c] = channel
        return result, "stretched"
    except Exception:
        return img, "stretched_failed"


def preprocess_red_plate_optimized(img):
    """Optimized for red background plates (government)."""
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Boost saturation and value for better contrast
        s = cv2.add(s, 30)
        v = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(v)
        enhanced = cv2.merge([h, s, v])
        result = cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)
        return result, "red_plate_opt"
    except Exception:
        return img, "red_plate_opt_failed"


def preprocess_yellow_plate_optimized(img):
    """Optimized for yellow background plates (public/taxi)."""
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Reduce yellow dominance, enhance dark text
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        v = clahe.apply(v)
        s = cv2.subtract(s, 20)  # Desaturate yellow
        enhanced = cv2.merge([h, s, v])
        result = cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)
        return result, "yellow_plate_opt"
    except Exception:
        return img, "yellow_plate_opt_failed"


# ============================================================================
# VARIANT COMBINATIONS (most effective combinations)
# ============================================================================

def preprocess_combo_heavy_clahe_sharpen(img):
    """Heavy CLAHE + Sharpen. Best for: Very dark/low-contrast images."""
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
        sharpened = cv2.addWeighted(enhanced, 1.8, blurred, -0.8, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8), "heavy_clahe_sharpen"
    except Exception:
        return img, "combo_failed"


def preprocess_combo_denoise_clahe(img):
    """De-noise + CLAHE. Best for: Night/rain with noise."""
    try:
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 8, 8, 7, 21)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
        l = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return result, "denoise_clahe"
    except Exception:
        return img, "denoise_clahe_failed"


def preprocess_combo_binary_morphology(img):
    """Binary + Morphology + Denoise. Best for: Broken text, faint characters."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoisingColored(gray, None, 5, 5, 7, 15)
        denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Close gaps in text
        kernel = np.ones((2, 3), np.uint8)
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # Open to remove noise
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
        return cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR), "binary_morph"
    except Exception:
        return img, "binary_morph_failed"


# ============================================================================
# MASTER PIPELINE
# ============================================================================

# All preprocessing variants in priority order
ALL_VARIANTS = [
    preprocess_original,
    preprocess_adaptive_clahe,
    preprocess_grayscale_clahe,
    preprocess_binary_otsu,
    preprocess_binary_inverse,
    preprocess_adaptive_threshold,
    preprocess_sharpen,
    preprocess_denoise,
    preprocess_gamma_correction,
    preprocess_histogram_equalization,
    preprocess_edge_enhanced,
    preprocess_morphology,
    preprocess_color_normalized,
    preprocess_top_hat,
    preprocess_dehaze,
    preprocess_stretched,
    preprocess_red_plate_optimized,
    preprocess_yellow_plate_optimized,
    preprocess_combo_heavy_clahe_sharpen,
    preprocess_combo_denoise_clahe,
    preprocess_combo_binary_morphology,
]


def get_all_variants(img):
    """Generate all preprocessing variants from an input image.
    
    Returns:
        List of (processed_img, tag) tuples.
    """
    if img is None or img.size == 0:
        return []
    
    variants = []
    for variant_fn in ALL_VARIANTS:
        try:
            processed, tag = variant_fn(img)
            if processed is not None and processed.size > 0:
                variants.append((processed, tag))
        except Exception:
            continue
    
    return variants


def select_best_plate(variants, ocr_func, min_confidence=0.15):
    """Select the best plate reading from multiple preprocessing variants.
    
    Uses ensemble voting:
    1. Run OCR on all variants
    2. Filter results by confidence threshold
    3. Score by format validity + confidence
    4. Return best result
    
    Args:
        variants: List of (img, tag) from get_all_variants()
        ocr_func: Function that takes img and returns (text, confidence)
        min_confidence: Minimum confidence to consider
    
    Returns:
        (best_text, best_confidence, best_tag) or (None, 0.0, None)
    """
    import re
    
    if not variants:
        return None, 0.0, None
    
    # Valid Indonesian plate pattern
    PLATE_PATTERN = re.compile(r'^([A-Z]{1,2})\s*(\d{1,4})\s*([A-Z]{1,3})$')
    
    results = []
    
    for img, tag in variants:
        try:
            text, conf = ocr_func(img)
            if not text or conf < min_confidence:
                continue
            
            # Score the result
            score = float(conf)
            raw = re.sub(r'[^A-Z0-9]', '', str(text).upper())
            m = PLATE_PATTERN.match(raw)
            
            if m:
                # Perfect format match = high bonus
                prefix, number, suffix = m.groups()
                score += 0.3
                # Length bonuses
                if len(number) in (3, 4):
                    score += 0.1
                if 1 <= len(suffix) <= 3:
                    score += 0.05
                # Prefer single letter prefix (most common in Indonesia)
                if len(prefix) == 1:
                    score += 0.05
            else:
                # Partial match (has letters and numbers)
                has_letters = any(c.isalpha() for c in raw)
                has_digits = any(c.isdigit() for c in raw)
                if has_letters and has_digits:
                    score -= 0.2  # Penalty but not disqualify
                else:
                    continue  # Not a plate at all
            
            results.append({
                'text': text,
                'conf': float(conf),
                'score': score,
                'tag': tag,
            })
        except Exception:
            continue
    
    if not results:
        return None, 0.0, None
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    best = results[0]
    return best['text'], best['conf'], best['tag']
