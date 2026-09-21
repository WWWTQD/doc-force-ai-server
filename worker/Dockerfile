# ─── Stage 1: Build dependencies ───────────────────────────────────────────
FROM python:3.10-slim AS base

# Cài libgomp (OpenCV cần) và curl (health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements trước để tận dụng layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ worker code
COPY . .

# ─── Health check ───────────────────────────────────────────────────────────
# Worker không có HTTP server riêng — script --health chỉ kiểm tra import
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import requests; import cv2; print('OK')" || exit 1

# ─── Entrypoint ─────────────────────────────────────────────────────────────
# Mặc định chạy mock_worker.py (thay bằng real_worker.py khi cài PaddleOCR xong)
CMD ["python", "mock_worker.py"]
