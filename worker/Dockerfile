# ─── Doc Forge AI Worker (YOLOv8 + DocumentRestorationUNet + PaddleOCR + VietOCR) ──
FROM python:3.11-slim AS base

# Cài đặt thư viện hệ thống cho OpenCV, PyTorch, PaddleOCR và curl
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bước 1: Cài đặt PyTorch CPU-only trước (~185MB thay vì bản CUDA 2.5GB)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch \
        torchvision

# Bước 2: Cài đặt các thư viện AI & Worker (giữ nguyên torch CPU đã cài)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        ultralytics \
        "paddlepaddle<3.0.0" \
        "paddleocr>=2.7.0,<3.0.0" \
        einops \
        gdown \
        prefetch_generator \
        opencv-python-headless \
        numpy \
        pillow \
        requests \
        python-docx \
        python-dotenv \
        pydantic \
        pydantic-settings \
        pyyaml \
    && pip install --no-cache-dir --no-deps vietocr

# Copy mã nguồn (bao gồm worker, app, training)
COPY . .

# Đặt thư mục làm việc tại worker để thực thi real_worker.py
WORKDIR /app/worker

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import requests, cv2, torch; print('OK')" || exit 1

CMD ["python", "-u", "real_worker.py"]
