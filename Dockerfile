# Doc Forge AI Server — FastAPI
# Cung cấp REST API phân tích ảnh, OCR test, quality check.
FROM python:3.11-slim

# Cài Tesseract (dùng bởi pytesseract trong pipeline) và curl (healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-vie \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài dependencies trước để tận dụng Docker layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code
COPY . .

# Expose port mặc định
EXPOSE 8000

# Chạy với uvicorn production (không có --reload)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
