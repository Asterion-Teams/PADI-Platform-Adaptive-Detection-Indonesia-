"""
AI-Enhanced OCR for License Plate Reading
-------------------------------------------
Combines PaddleOCR raw output with AI model (OpenAI-compatible API)
to improve plate reading accuracy.

Pipeline:
1. PaddleOCR reads the plate image → raw text (may have errors)
2. AI model corrects OCR errors using context (Indonesian plate format)
3. Returns cleaned, formatted plate number

The AI model understands Indonesian plate format rules:
- Prefix: 1-2 letters (B, D, F, H, AB, BK, etc.)
- Number: 1-4 digits
- Suffix: 1-3 letters
"""
import base64
import json
import os
import urllib.request
import urllib.error

import cv2
import numpy as np


def ai_enhance_plate(raw_ocr_text: str, plate_image=None) -> tuple[str | None, float]:
    """Use AI model to correct/enhance PaddleOCR plate reading.
    
    Args:
        raw_ocr_text: Raw text from PaddleOCR (may have errors)
        plate_image: Optional numpy array of plate crop (for vision models)
    
    Returns:
        (corrected_plate, confidence) or (None, 0.0) on failure
    """
    import app.config as cfg
    
    if not cfg.AI_USE_FOR_ANPR or not cfg.AI_API_KEY or not cfg.AI_BASE_URL:
        return None, 0.0
    
    try:
        # Build prompt
        prompt = f"""Kamu adalah sistem ANPR untuk plat nomor kendaraan Indonesia.

FORMAT PLAT INDONESIA (semua jenis):
1. Plat Umum/Pribadi: [HURUF] [ANGKA] [HURUF]
   Contoh: B 1234 ABC, D 5678 XY, F 912 KLM, AB 1234 CD
   
2. Plat Pejabat/Government (teks putih di atas latar merah):
   - RI = Pemerintah Pusat (Presiden, Menteri)
   - DPR = Dewan Perwakilan Rakyat
   - MPR = Majelis Permusyawaratan Rakyat
   - DPD = Dewan Perwakilan Daerah
   - POLRI = Polisi
   Contoh: RI 1234 ABC, DPR 5678 XY, POLRI 123456
   
3. Plat TNI/Militer: AB 1234 CD

4. Plat Baru (latar putih): Format sama dengan plat umum

KODE WILAYAH UMUM:
- B = Jakarta/Bekasi/Tangerang/Depok (PALING UMUM)
- D = Bandung, F = Bogor/Sukabumi
- H = Semarang, L = Surabaya
- AB = Yogyakarta, KB = Kalimantan

KESALAHAN OCR YANG SERING TERJADI:
- Angka "8" di posisi pertama biasanya huruf "B"
- Angka "0" bisa huruf "O" atau "D"
- Angka "6" bisa huruf "G"
- Huruf "I" bisa angka "1"

OCR membaca: "{raw_ocr_text}"

TUGAS: Koreksi menjadi format plat yang BENAR.
Jawab HANYA plat yang sudah dikoreksi (contoh: B 1234 ABC atau RI 5678 XY).
Jika tidak bisa ditentukan, jawab: UNKNOWN"""

        # Call AI API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.AI_API_KEY}",
        }
        
        body = {
            "model": cfg.AI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 20,
            "temperature": 0.1,
        }
        
        url = f"{cfg.AI_BASE_URL.rstrip('/')}/chat/completions"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        
        # Parse response
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        if not content or "UNKNOWN" in content.upper():
            return None, 0.0
        
        # Clean response (remove quotes, extra text)
        import re
        cleaned = re.sub(r'[^A-Z0-9\s]', '', content.upper()).strip()
        
        # IMPORTANT: Check government/police plates FIRST (before standard regex)
        # because longer prefixes like "DPR", "TNI" would be split incorrectly by standard regex
        
        # Pattern: Government/Police with suffix (DPR 5678 XY, MPR 9012 AB, TNI 1234 CD)
        # These have 3-letter prefix + 4-digit number + 2-letter suffix
        match_gov_suffix = re.match(r'^(DPR|MPR|DPD|MK|KY)\s*(\d{1,4})\s*([A-Z]{1,2})$', cleaned)
        if match_gov_suffix:
            plate = f"{match_gov_suffix.group(1)} {match_gov_suffix.group(2)} {match_gov_suffix.group(3)}"
            return plate, 0.90
        
        # Pattern: RI with suffix (RI 1234 ABC)
        match_ri = re.match(r'^(RI)\s*(\d{1,4})\s*([A-Z]{1,3})$', cleaned)
        if match_ri:
            plate = f"{match_ri.group(1)} {match_ri.group(2)} {match_ri.group(3)}".strip()
            return plate, 0.90
        
        # Pattern: Police/Military without suffix (POLRI 123456, KEJAKSAAN 1234, TNI 123456)
        match_police = re.match(r'^(POLRI|KEJAKSAAN|TNI)\s*(\d{1,6})$', cleaned)
        if match_police:
            plate = f"{match_police.group(1)} {match_police.group(2)}"
            return plate, 0.90
        
        # Pattern: Just numbers (police plate without prefix visible)
        match_num = re.match(r'^(\d{5,7})$', cleaned)
        if match_num:
            return match_num.group(1), 0.75
        
        # Pattern: Standard plates (B 1234 ABC, D 5678 XY)
        match_std = re.match(r'^([A-Z]{1,2})\s*(\d{1,4})\s*([A-Z]{0,3})$', cleaned)
        if match_std:
            plate = f"{match_std.group(1)} {match_std.group(2)} {match_std.group(3)}".strip()
            return plate, 0.90
        
        return None, 0.0
        
    except Exception as e:
        print(f"[AI-OCR] Error: {e}")
        return None, 0.0


def ai_read_plate_from_image(plate_image) -> tuple[str | None, float]:
    """Send plate image to AI model for direct plate reading.
    
    Used as final fallback when PaddleOCR completely fails.
    Encodes image as base64 and asks AI to read the plate number.
    """
    import app.config as cfg
    
    if not cfg.AI_USE_FOR_ANPR or not cfg.AI_API_KEY or not cfg.AI_BASE_URL:
        return None, 0.0
    
    if plate_image is None or plate_image.size == 0:
        return None, 0.0
    
    try:
        # Encode image to base64
        _, buf = cv2.imencode('.jpg', plate_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        
        # Try vision model (if supported)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.AI_API_KEY}",
        }
        
        prompt = """Saya memiliki gambar plat nomor kendaraan dari CCTV Indonesia. Saya perlu membaca plat nomor.

FORMAT PLAT INDONESIA (semua jenis):
1. Plat Umum/Pribadi (latar hitam): B 1234 ABC, D 5678 XY
2. Plat Tax/Umum (latar kuning): D 5678 XY, F 912 KLM
3. Plat Pejabat/Government (LATAR MERAH, teks PUTIH):
   - RI = Presiden/Menteri
   - DPR, MPR = legislative
   - POLRI = polisi
   Contoh: RI 1234 ABC, DPR 5678 XY, POLRI 123456
4. Plat TNI/Militer: AB 1234 CD
5. Plat Baru (latar putih): F 912 KLM, B 1234 ABC

Dari gambar yang dikirim, baca plat nomor yang terlihat.
JAWAB HANYA PLAT NOMOR (contoh: B 1234 ABC atau RI 5678 XY).
Jika tidak terlihat jelas, jawab: UNKNOWN"""

        # Try with image (vision model)
        body = {
            "model": cfg.AI_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }],
            "max_tokens": 30,
            "temperature": 0.1,
        }
        
        url = f"{cfg.AI_BASE_URL.rstrip('/')}/chat/completions"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            if not content or "UNKNOWN" in content.upper():
                return None, 0.0
            
            # Clean and validate - expanded patterns for government plates
            import re
            cleaned = re.sub(r'[^A-Z0-9\s]', '', content.upper()).strip()
            
            # Pattern 1: Standard [PREFIX] [NUMBER] [SUFFIX] (e.g., B 1234 ABC, RI 5678 XY)
            match = re.match(r'^([A-Z]{1,2})\s*(\d{1,4})\s*([A-Z]{0,3})$', cleaned)
            if match:
                plate = f"{match.group(1)} {match.group(2)} {match.group(3)}".strip()
                # Boost confidence for valid Indonesian plate format
                return plate, 0.90
            
            # Pattern 2: Police/Military (e.g., POLRI 123456, KEJAKSAAN 1234)
            match2 = re.match(r'^(POLRI|KEJAKSAAN|TNI)\s*(\d{1,6})$', cleaned)
            if match2:
                plate = f"{match2.group(1)} {match2.group(2)}"
                return plate, 0.90
            
            # Pattern 3: Just number (police plates without prefix visible)
            match3 = re.match(r'^(\d{5,7})$', cleaned)
            if match3:
                return match3.group(1), 0.75
            
            return None, 0.0
        except urllib.error.HTTPError as e:
            # Vision not supported — try without image
            if e.code in (400, 422):
                return None, 0.0
            raise
            
    except Exception as e:
        print(f"[AI-VISION] Error: {e}")
        return None, 0.0


def ai_identify_vehicle(vehicle_image) -> dict | None:
    """Identify vehicle details from image using AI vision model (Gemini Flash Lite).
    
    Returns dict with:
    - company: e.g. "Bluebird Group", "Grab", "GoJek" or None
    - make_model: e.g. "Toyota Avanza", "Honda Beat" 
    - vehicle_type: e.g. "Sedan", "MPV", "Motorcycle", "Pickup", "Bus", "Truck"
    - color: e.g. "Biru", "Hitam", "Putih"
    - plate: plate number if visible
    - registration_area: e.g. "Jakarta" (from plate prefix)
    - confidence scores for each field: company_conf, make_model_conf, etc.
    
    All confidence values are boosted to 85-95% for valid format matches.
    """
    import app.config as cfg
    import re
    
    if not cfg.AI_API_KEY or not cfg.AI_BASE_URL:
        return None
    
    if vehicle_image is None or vehicle_image.size == 0:
        return None
    
    try:
        _, buf = cv2.imencode('.jpg', vehicle_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.AI_API_KEY}",
        }
        
        prompt = """Analisis gambar kendaraan dari CCTV Indonesia ini. Fokus pada kendaraan di dalam kotak merah.

1. Company/Operator: Apakah ada branding perusahaan? (Bluebird, Grab, GoJek, Express, J&T, JNE, dll). Jika tidak ada branding, tulis "Private".
2. Make/Model: Merek dan model SPESIFIK kendaraan (Toyota Avanza, Honda Jazz, Suzuki Carry, Mitsubishi Xpander, Daihatsu Xenia, dll). JANGAN tulis "Unknown" - selalu estimasi dari bentuk body.
3. Vehicle Type: Sedan, MPV, SUV, Hatchback, Motorcycle, Pickup, Truck, Bus, Van, Minibus.
4. Color: Warna DOMINAN kendaraan (Putih/Hitam/Silver/Abu-abu/Merah/Biru/Kuning/Hijau/Coklat/Emas). JANGAN tulis "Unknown".
5. License Plate: Plat nomor yang terlihat (format: B 1234 ABC). Jika tidak terlihat, tulis "N/A".
6. Registration Area: Dari prefix plat (B=Jakarta, D=Bandung, F=Bogor, H=Semarang, L=Surabaya, AB=Yogyakarta, dll).

PENTING: JANGAN jawab "Unknown" atau "N/A" untuk vehicle_type, make_model, atau color. Selalu berikan estimasi terbaik.

Jawab dalam format JSON SAJA (tanpa markdown):
{"company":"Private","make_model":"Toyota Avanza","vehicle_type":"MPV","color":"Putih","plate":"B 1234 ABC","registration_area":"Jakarta"}"""

        # Use Gemini Flash Lite model for vehicle identification
        vehicle_model = os.environ.get("AI_VEHICLE_MODEL") or getattr(cfg, 'AI_VEHICLE_MODEL', None) or cfg.AI_MODEL
        
        body = {
            "model": vehicle_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }],
            "max_tokens": 200,
            "temperature": 0.1,
        }
        
        url = f"{cfg.AI_BASE_URL.rstrip('/')}/chat/completions"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        if not content:
            return None
        
        # Parse JSON response (handle markdown code fences)
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        
        vehicle_info = json.loads(content)
        
        # Calculate confidence scores for each field
        # Boost to 85-95% for valid/confirmed values
        info_with_conf = {
            'company': vehicle_info.get('company', 'Unknown'),
            'company_conf': 0.88 if vehicle_info.get('company') and vehicle_info.get('company') != 'Private' else 0.82,
            'make_model': vehicle_info.get('make_model', 'Unknown'),
            'make_model_conf': 0.85 if vehicle_info.get('make_model') else 0.0,
            'vehicle_type': vehicle_info.get('vehicle_type', 'Unknown'),
            'vehicle_type_conf': 0.88 if vehicle_info.get('vehicle_type') else 0.0,
            'color': vehicle_info.get('color', 'Unknown'),
            'color_conf': 0.85 if vehicle_info.get('color') else 0.0,
            'plate': vehicle_info.get('plate', 'N/A'),
            'plate_conf': 0.0,
            'registration_area': vehicle_info.get('registration_area', 'Unknown'),
            'registration_area_conf': 0.85 if vehicle_info.get('registration_area') else 0.0,
        }
        
        # Validate and boost plate confidence - support ALL plate types
        plate = vehicle_info.get('plate', '')
        if plate and plate != 'N/A' and 'unknown' not in plate.lower():
            raw = re.sub(r'[^A-Z0-9]', '', plate.upper())
            
            # Check government plates FIRST (longer prefix)
            # DPR 5678 XY, MPR 9012 AB, TNI 1234 CD
            m_gov = re.match(r'^(DPR|MPR|DPD|MK|KY)\s*(\d{1,4})\s*([A-Z]{1,2})$', plate.upper())
            if m_gov:
                info_with_conf['plate_conf'] = 0.92
            else:
                # RI 1234 ABC
                m_ri = re.match(r'^(RI)\s*(\d{1,4})\s*([A-Z]{1,3})$', plate.upper())
                if m_ri:
                    info_with_conf['plate_conf'] = 0.92
                else:
                    # Police/Military without suffix: POLRI 123456
                    m_pol = re.match(r'^(POLRI|KEJAKSAAN|TNI)\s*(\d{1,6})$', plate.upper())
                    if m_pol:
                        info_with_conf['plate_conf'] = 0.92
                    else:
                        # Standard plates: B 1234 ABC, D 5678 XY
                        m_std = re.match(r'^([A-Z]{1,2})\s*(\d{1,4})\s*([A-Z]{1,3})$', plate.upper())
                        if m_std:
                            info_with_conf['plate_conf'] = 0.92
                        else:
                            info_with_conf['plate_conf'] = 0.72
        
        return info_with_conf
        
    except (json.JSONDecodeError, KeyError):
        return None
    except Exception as e:
        print(f"[AI-VEHICLE] Error: {e}")
        return None


def ai_test_connection() -> dict:
    """Test AI provider connection. Returns status info."""
    import app.config as cfg
    
    if not cfg.AI_API_KEY or not cfg.AI_BASE_URL:
        return {"status": "error", "message": "API key or base URL not configured"}
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.AI_API_KEY}",
        }
        body = {
            "model": cfg.AI_MODEL,
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 5,
            "temperature": 0,
        }
        url = f"{cfg.AI_BASE_URL.rstrip('/')}/chat/completions"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {
            "status": "success",
            "message": f"Connected to {cfg.AI_PROVIDER}",
            "model": cfg.AI_MODEL,
            "response": content[:50],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
