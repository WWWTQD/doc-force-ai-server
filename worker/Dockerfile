# ─── Doc Forge AI Worker (Tesseract 5 + YOLOv8 + DocumentRestorationUNet) ──
FROM python:3.11-slim AS base

# Cài đặt thư viện hệ thống cho OpenCV, PyTorch, curl, và Tesseract OCR (Tiếng Việt + Tiếng Anh)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        curl \
        tesseract-ocr \
        tesseract-ocr-vie \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bước 1: Cài đặt PyTorch CPU-only trước (~185MB thay vì bản CUDA 2.5GB)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch \
        torchvision

# Bước 2: Cài đặt các thư viện còn lại (giữ nguyên torch CPU đã cài)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        ultralytics \
        opencv-python-headless \
        numpy \
        pillow \
        requests \
        python-docx \
        python-dotenv \
        pydantic \
        pydantic-settings \
        pyyaml \
        pytesseract

# Copy mã nguồn (bao gồm worker, app, training)
COPY . .

# Đặt thư mục làm việc tại worker để thực thi real_worker.py
WORKDIR /app/worker

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import requests, cv2, torch; print('OK')" || exit 1

CMD ["python", "-u", "real_worker.py"]
