# ─── Doc Forge AI Worker (PaddleOCR + DocumentRestorationUNet) ─────────────
FROM python:3.11-slim AS base

# Cài đặt thư viện hệ thống cần thiết cho OpenCV, PyTorch và curl
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements trước để tận dụng layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn (bao gồm worker, app, models, training)
COPY . .

# Đặt thư mục làm việc tại worker để thực thi script worker
WORKDIR /app/worker

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import requests; import cv2; print('OK')" || exit 1

CMD ["python", "real_worker.py"]
