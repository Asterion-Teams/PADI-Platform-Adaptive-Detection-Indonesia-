"""Quick test: verify AI API connectivity and model availability."""
import sys, json, urllib.request, urllib.error, base64, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.config as cfg
cfg.load_persisted_settings()

base_url = cfg.AI_BASE_URL.rstrip("/")
url = base_url + "/chat/completions"
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {cfg.AI_API_KEY}"}

print(f"Provider: {cfg.AI_PROVIDER}")
print(f"Base URL: {base_url}")
print(f"AI_MODEL: {cfg.AI_MODEL}")
print(f"AI_VEHICLE_MODEL: {cfg.AI_VEHICLE_MODEL}")
print(f"API Key: {cfg.AI_API_KEY[:15]}...")
print()

# Test 1: AI_MODEL basic
print("=== Test 1: AI_MODEL text ===")
body = {"model": cfg.AI_MODEL, "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5, "temperature": 0}
try:
    req = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        r = json.loads(resp.read())
    print(f"  OK: {r['choices'][0]['message']['content']}")
except urllib.error.HTTPError as e:
    print(f"  HTTP {e.code}: {e.read().decode()[:200]}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 2: AI_VEHICLE_MODEL basic
print("\n=== Test 2: AI_VEHICLE_MODEL text ===")
body2 = {"model": cfg.AI_VEHICLE_MODEL, "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5, "temperature": 0}
try:
    req2 = urllib.request.Request(url, json.dumps(body2).encode(), headers, method="POST")
    with urllib.request.urlopen(req2, timeout=15) as resp2:
        r2 = json.loads(resp2.read())
    print(f"  OK: {r2['choices'][0]['message']['content']}")
except urllib.error.HTTPError as e:
    print(f"  HTTP {e.code}: {e.read().decode()[:300]}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 3: AI_VEHICLE_MODEL with image (vision)
print("\n=== Test 3: AI_VEHICLE_MODEL vision ===")
# Create a dummy 100x50 test image (simulating a plate crop)
import cv2
test_img = np.ones((50, 100, 3), dtype=np.uint8) * 200
cv2.putText(test_img, "B1234AB", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
_, buf = cv2.imencode(".jpg", test_img)
img_b64 = base64.b64encode(buf.tobytes()).decode()

body3 = {
    "model": cfg.AI_VEHICLE_MODEL,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What text do you see in this image? Reply with just the text."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
    ]}],
    "max_tokens": 20,
    "temperature": 0,
}
try:
    req3 = urllib.request.Request(url, json.dumps(body3).encode(), headers, method="POST")
    with urllib.request.urlopen(req3, timeout=15) as resp3:
        r3 = json.loads(resp3.read())
    print(f"  OK: {r3['choices'][0]['message']['content']}")
except urllib.error.HTTPError as e:
    err_body = e.read().decode()[:400]
    print(f"  HTTP {e.code}: {err_body}")
    if e.code in (400, 422):
        print("  >> Vision NOT supported by this model/provider")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 4: AI_MODEL with vision (fallback check)
print("\n=== Test 4: AI_MODEL vision (fallback) ===")
body4 = dict(body3)
body4["model"] = cfg.AI_MODEL
try:
    req4 = urllib.request.Request(url, json.dumps(body4).encode(), headers, method="POST")
    with urllib.request.urlopen(req4, timeout=15) as resp4:
        r4 = json.loads(resp4.read())
    print(f"  OK: {r4['choices'][0]['message']['content']}")
except urllib.error.HTTPError as e:
    err_body = e.read().decode()[:400]
    print(f"  HTTP {e.code}: {err_body}")
    if e.code in (400, 422):
        print("  >> Vision NOT supported by AI_MODEL either")
except Exception as e:
    print(f"  ERROR: {e}")

print("\nDone.")
