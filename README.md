# Doc Force AI Server

Doc Force AI Server is a modular Python backend for document scanning, quality analysis, defect detection, preprocessing, and OCR. It provides a REST API that accepts images from scanners or frontends and returns processed images and structured JSON results.

Features
- Upload and validate document images (JPG, PNG, TIFF, BMP)
- Document detection and perspective correction (OpenCV baseline)
- Image quality analysis: brightness, contrast, sharpness, noise, resolution
- Defect detection rules (baseline)
- Preprocessing helpers (denoise, contrast, threshold, deskew)
- OCR using pytesseract (pluggable provider architecture)
- Modular, AI-ready abstractions for future model integration

Tech stack
- Python 3.11
- FastAPI + Uvicorn
- OpenCV, NumPy, Pillow
- pytesseract

Project structure
```
doc-force-ai-server/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── router.py
│   │   └── endpoints/
│   │       ├── health.py
│   │       ├── scan.py
│   │       ├── analyze.py
│   │       ├── ocr.py
│   │       ├── process.py
│   │       └── defects.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   ├── services/
│   │   ├── scan_service.py
│   │   ├── image_service.py
│   │   ├── document_service.py
│   │   ├── quality_service.py
│   │   ├── defect_service.py
│   │   └── ocr_service.py
│   ├── cv/
│   │   ├── blur_detector.py
│   │   ├── brightness_detector.py
│   │   ├── noise_detector.py
│   │   ├── skew_detector.py
│   │   ├── rotation_detector.py
│   │   ├── perspective.py
│   │   └── preprocessing.py
│   ├── schemas/
│   └── utils/
├── data/
│   ├── input/
│   └── output/
├── tests/
├── requirements.txt
├── .env.example
├── .gitignore
└── run.py
```

Installation (Windows PowerShell)

1. Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

Running the server

Development

```powershell
# using uvicorn directly
uvicorn app.main:app --reload

# or use helper
python run.py
```

API endpoints

Base path: /api/v1

- GET /api/v1/health — health check
- POST /api/v1/scan — upload and run full scan pipeline
- GET /api/v1/scan/{scan_id} — retrieve saved JSON result for a scan
- POST /api/v1/analyze — analyze image quality (returns document detection + quality)
- POST /api/v1/ocr — run OCR on uploaded image (form field `lang` optional)
- POST /api/v1/process — extract and save processed (perspective-corrected) image
- POST /api/v1/defects — return defects detected for uploaded image

Example curl (upload)

```powershell
curl -X POST "http://localhost:8000/api/v1/scan/" -F "file=@C:\path\to\doc.jpg"
```

Testing

Run pytest from project root:

```powershell
pytest -q
```

Configuration

Copy `.env.example` to `.env` and edit values as needed. Important variables:

- APP_NAME, APP_VERSION
- MAX_FILE_SIZE_MB
- BLUR_THRESHOLD
- OCR_LANGUAGE
- OUTPUT_DIR
- CORS_ORIGINS

Next development steps

1. Add more unit/integration tests for document detection and OCR on real samples
2. Improve defect detection rules and add ML-based detectors
3. Add persistent storage for scan metadata
4. Add async/background processing (Celery/Redis) for heavy pipelines
5. Dockerize the application

License & Notes

This repository provides baseline implementations using OpenCV and pytesseract. It is designed to be modular so that heavier AI models (PyTorch/ONNX/Yolo/PaddleOCR) can be integrated later without changing the API.

