"""PaddleOCR subprocess worker — reads image from stdin, outputs plate text."""
import sys, os, json, base64
import numpy as np

os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# Suppress paddle warnings
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.WARNING)

try:
    from paddleocr import PaddleOCR
    device_label = os.environ.get('PADDLE_OCR_DEVICE', 'cpu').strip() or 'cpu'
    base_kwargs = dict(
        text_detection_model_name='PP-OCRv4_server_det' if device_label.startswith('gpu') else 'PP-OCRv4_mobile_det',
        text_recognition_model_name='PP-OCRv4_server_rec' if device_label.startswith('gpu') else 'PP-OCRv4_mobile_rec',
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    try:
        engine = PaddleOCR(device=device_label, **base_kwargs)
    except TypeError:
        engine = PaddleOCR(use_gpu=device_label.startswith('gpu'), **base_kwargs)
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)

# Read base64 image from stdin
data = sys.stdin.buffer.read()
img_bytes = base64.b64decode(data)
img_array = np.frombuffer(img_bytes, np.uint8)

import cv2
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
if img is None:
    print(json.dumps({"error": "cannot decode image"}))
    sys.exit(1)

results_out = []
try:
    results = engine.predict(img)
    if results:
        for res in results:
            try:
                if hasattr(res, 'rec_texts') and hasattr(res, 'dt_polys'):
                    polys = res.dt_polys if hasattr(res, 'dt_polys') else [None] * len(res.rec_texts)
                    for i, (txt, score) in enumerate(zip(res.rec_texts, res.rec_scores)):
                        y_pos = 0
                        x_pos = 0
                        if i < len(polys) and polys[i] is not None:
                            try:
                                y_pos = float(np.mean([p[1] for p in polys[i]]))
                                x_pos = float(np.mean([p[0] for p in polys[i]]))
                            except Exception:
                                pass
                        results_out.append({"text": str(txt), "conf": float(score), "y": y_pos, "x": x_pos})
                elif hasattr(res, 'rec_texts'):
                    for txt, score in zip(res.rec_texts, res.rec_scores):
                        results_out.append({"text": str(txt), "conf": float(score), "y": 0, "x": 0})
                elif isinstance(res, dict):
                    texts = res.get('rec_texts') or res.get('rec_text') or []
                    scores = res.get('rec_scores') or res.get('rec_score') or []
                    if isinstance(texts, str):
                        texts, scores = [texts], [scores]
                    for txt, score in zip(texts, scores):
                        results_out.append({"text": str(txt), "conf": float(score), "y": 0, "x": 0})
            except Exception:
                continue
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)

print(json.dumps({"results": results_out}))
