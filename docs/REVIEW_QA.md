# Persiapan Review Dosen — Q&A Guide

## 1. Big Data: "Dimana Big Data-nya?"

**Jawaban:**
Sistem ini memenuhi 4V Big Data:

| V | Implementasi |
|---|---|
| **Volume** | Ribuan deteksi kendaraan per jam × 5 kamera × 24 jam = ~100.000+ data points/hari. Database menyimpan 248.000+ violation records. |
| **Velocity** | Real-time streaming 15 FPS dari CCTV, inference YOLO setiap ~0.2 detik, violation detection instant. |
| **Variety** | Multi-source: CCTV video stream + social media (X.com scraping) + citizen reports (CRM) + Google News. Structured (DB) + unstructured (gambar, text). |
| **Veracity** | AI-based validation: YOLO confidence threshold, enforcement dwell timer (bukan deteksi sesaat), multi-frame evidence, PaddleOCR + AI double-check untuk plat. |

**Follow-up: "Ini kan cuma 5 kamera, bukan Big Data?"**
- Arsitektur sudah dirancang untuk scale: GPU batch inference, thread-per-camera, modular. Tinggal tambah server untuk 100+ kamera.
- Data yang dihasilkan per hari sudah puluhan ribu records — ini Big Data dari sisi processing requirement.
- Real-time constraint (15 FPS × 5 camera = 75 frame/detik yang harus diproses) adalah Big Data challenge.

---

## 2. AI/ML: "Model apa yang dipakai?"

**Jawaban:**

| Komponen | Model | Fungsi |
|---|---|---|
| Vehicle Detection | YOLO11m (Ultralytics) | Deteksi & klasifikasi kendaraan (mobil, motor, bus) |
| Plate Detection | YOLO custom (plate_detector_best.pt) | Lokalisasi area plat nomor di kendaraan |
| Plate OCR | PaddleOCR PP-OCRv4 | Baca teks plat nomor |
| OCR Correction | GPT-4.1-nano (SumoPod API) | Koreksi kesalahan OCR + format Indonesia |
| Chatbot | GPT-4.1-nano (SumoPod API) | PADI Assistant untuk operator |
| Social Scraping | Playwright + Chromium | Monitoring @DishubDKI di X.com |

**Follow-up: "Kenapa YOLO bukan model lain?"**
- YOLO11m: balance terbaik antara speed dan accuracy untuk real-time video
- Single-stage detector: 1 forward pass = detection + classification (cepat)
- GPU batch inference: bisa process multiple frames sekaligus
- Pre-trained di COCO dataset, fine-tune optional untuk kendaraan Indonesia

**Follow-up: "Akurasinya berapa?"**
- Vehicle detection: ~90-95% (kendaraan dekat), ~70-80% (jauh/malam)
- Plate OCR: ~60-75% (tergantung jarak, resolusi, kondisi cahaya)
- Enforcement false positive: sangat rendah karena multi-layer validation (dwell timer, movement check, consistency check)

---

## 3. Enforcement: "Bagaimana deteksi pelanggaran bekerja?"

**Jawaban — Alur lengkap:**

```
1. YOLO detect kendaraan → bounding box + class (car/motor/bus)
2. Vehicle Tracker: match bbox antar frame (IoU matching)
3. Zone Check: apakah center point kendaraan ada di dalam polygon zona?
4. Dwell Timer: berapa lama kendaraan di zona? (parkir: 60s, busway: 5s)
5. Movement Check: apakah kendaraan diam (parkir liar) atau bergerak (busway/wrong-way)?
6. Violation Trigger → Evidence capture (full-res) + ANPR (baca plat)
7. Database insert + notification ke operator
```

**Follow-up: "Bagaimana bedakan mobil parkir vs berhenti di lampu merah?"**
- Zona digambar spesifik oleh operator (bukan seluruh jalan)
- Threshold 60 detik — lampu merah biasanya < 60 detik
- Movement check: kalau pindah >60px dari posisi awal = tidak dianggap parkir

**Follow-up: "Wrong-way detection bagaimana?"**
- Zona punya `allowed_direction` (derajat, di-set operator)
- Sistem hitung arah gerakan kendaraan dari trajectory (first → last position)
- Harus: >160° berlawanan, >150px movement, >20px/detik speed, >8 detik, ALL trajectory segments confirm
- Ini sangat ketat — hampir impossible false positive

---

## 4. ANPR: "Bisa baca plat nomor?"

**Jawaban:**
Pipeline 3 layer:
1. **AI Vision** (2-3 detik): Kirim gambar kendaraan ke GPT-4.1-nano → baca plat langsung
2. **PaddleOCR** (10-15 detik): PP-OCRv4 baca teks → AI correction format Indonesia
3. **Postprocessing**: Koreksi OCR error (8→B, 0→O), validate prefix wilayah (B=Jakarta, D=Bandung)

**Limitasi (jujur):**
- Kendaraan jauh dari kamera (plat <30px) → tidak bisa dibaca
- Malam hari / blur → akurasi turun
- Dari total violations, ~30-40% plat berhasil terbaca (sisanya terlalu kecil/blur)

---

## 5. CRM & Social Media: "Monitoring sosmed bagaimana?"

**Jawaban:**
- Playwright (headless Chromium) login ke X.com menggunakan session cookies
- Search: `@DishubDKI OR #DishubDKI OR to:DishubDKI`
- Filter hanya tweet tentang kondisi jalanan (macet, kecelakaan, parkir liar)
- Auto-classify tipe pelanggaran/masalah
- Operator bisa import ke CRM sebagai laporan
- Refresh setiap 2 menit

---

## 6. Arsitektur: "Kenapa Flask? Bukan microservice?"

**Jawaban:**
- **Monolith by design** — untuk kompetisi/PoC, simplicity > complexity
- Single process, multi-thread: camera agents + inference worker + web server
- SQLite (bukan PostgreSQL) — zero config, file-based, cukup untuk PoC
- Kalau production: bisa decompose ke microservices (camera service, inference service, API gateway)

**Follow-up: "Scalability?"**
- GPU batch inference sudah support N cameras per 1 GPU
- Thread-per-camera: tambah kamera = tambah thread (lightweight)
- Bottleneck: GPU VRAM (RTX 3050 = 4GB, cukup ~10-15 kamera)
- Untuk 100+ kamera: horizontal scaling (multiple server + load balancer)

---

## 7. Security: "Bagaimana keamanan sistem?"

**Jawaban:**
- Demo mode: halaman bisa diakses publik, action destructive butuh password
- Anti-inspect element (F12, klik kanan disabled)
- Session cookies X.com tersimpan di file lokal (bukan di kode)
- API keys tersimpan di `app_settings.json` (gitignored)
- Evidence files tersimpan lokal, bukan cloud

---

## 8. Demo Scenario (urutan yang bagus)

1. **Dashboard** — tunjukkan overview: KPI, live camera grid, violation trend
2. **Enforcement** — tunjukkan live detection + bounding box
3. **Violation detail** — klik violation, tunjukkan evidence foto + zoom kendaraan
4. **Zone Editor** — gambar zona baru (tunjukkan flexibility)
5. **OCR Test** — upload foto plat → baca dengan AI
6. **Executive Summary** — print PDF laporan
7. **CRM** — tunjukkan social media mentions @DishubDKI
8. **Settings** — tunjukkan hot-swap model, AI provider config
9. **Chatbot** — tanya "berapa total pelanggaran hari ini?" → AI jawab

---

## 9. Pertanyaan Jebakan & Jawaban

**"Ini cuma wrapper library, bukan karya sendiri?"**
- Integrasi 7+ library (YOLO, PaddleOCR, Playwright, OpenCV, Flask, Chart.js, Leaflet) menjadi 1 sistem kohesif = engineering yang signifikan
- Custom enforcement engine (zone polygon, dwell timer, wrong-way detection) = 100% original logic
- GPU batch inference worker = custom architecture
- CRM social scraping = custom Playwright integration

**"Kalau CCTV mati, sistem tetap jalan?"**
- Ya — auto-reconnect dengan exponential backoff
- Kamera lain tetap berjalan (independent threads)
- Status ditampilkan di dashboard (online/offline)

**"Data palsu?"**
- TIDAK ada data fake — enforcement hanya trigger dari deteksi real
- ANPR tidak generate plate palsu (kalau gagal baca = NULL, bukan random)
- Violation hanya tercatat kalau threshold terpenuhi (60s parkir, 5s busway)

**"Bagaimana testing/validasi?"**
- Manual validation oleh operator (Confirm/Dismiss per violation)
- False positive rate rendah karena multi-layer threshold
- Evidence foto sebagai ground truth untuk review

---

## 10. Keyword Penting untuk Disebut

- **Real-time inference** (bukan batch processing offline)
- **Edge computing** (processing di local server, bukan cloud)
- **Multi-modal AI** (vision + NLP + OCR)
- **Adaptive threshold** (configurable per zona)
- **Non-blocking architecture** (async ANPR, thread-per-camera)
- **Persist settings** (survive restart)
- **Hot-swap model** (ganti AI model tanpa downtime)
