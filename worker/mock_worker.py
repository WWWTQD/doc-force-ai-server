"""
Mock AI Worker cho Doc Forge.

Thay thế tạm thời khi chưa có model AI thật.
Worker polling backend theo vòng lặp:
  1. POST /api/worker/jobs/claim       → nhận job
  2. sleep(5)                           → giả lập thời gian xử lý
  3. POST /api/worker/jobs/{id}/results → gửi dummy OCR JSON
  4. Quay lại bước 1

Chạy:
  pip install requests python-dotenv
  python worker/mock_worker.py

Hoặc với biến môi trường:
  BACKEND_URL=http://localhost:8080 WORKER_SECRET_KEY=worker-secret-local python worker/mock_worker.py
"""

import logging
import os
import signal
import socket
import threading
import time
import uuid

import requests
from dotenv import load_dotenv

# Đọc .env từ thư mục gốc hoặc thư mục hiện tại
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("mock_worker")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8080").rstrip("/")
WORKER_SECRET_KEY = os.environ.get("WORKER_SECRET_KEY", "worker-secret-local")
WORKER_INSTANCE_ID = os.environ.get("WORKER_INSTANCE_ID", socket.gethostname()[:100])
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL_SECONDS", "2"))
MOCK_PROCESSING_SECONDS = float(os.environ.get("MOCK_PROCESSING_SECONDS", "5"))
HEARTBEAT_INTERVAL = float(os.environ.get("HEARTBEAT_INTERVAL_SECONDS", "20"))

# ─────────────────────────────────────────────
# Stop flag — Ctrl-C / SIGTERM đặt flag này
# ─────────────────────────────────────────────
STOP = threading.Event()


def _signal_handler(*_):
    LOG.info("Shutdown signal received, stopping after current job...")
    STOP.set()


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ─────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────
def _headers():
    return {
        "X-Worker-Key": WORKER_SECRET_KEY,
        "Content-Type": "application/json",
    }


def _post(path: str, payload: dict, timeout: int = 15) -> dict | None:
    """POST tới backend, trả None nếu 204, raise nếu lỗi."""
    url = f"{BACKEND_URL}{path}"
    resp = requests.post(url, json=payload, headers=_headers(), timeout=timeout)
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────
# Dummy OCR data generator
# ─────────────────────────────────────────────
DUMMY_BLOCKS_TEMPLATE = [
    {
        "id": "b1",
        "type": "paragraph",
        "order": 1,
        "bbox": [50, 50, 750, 100],
        "text": "Đây là dòng chữ giả lập số 1 — Mock OCR Worker",
        "confidence": 0.99,
    },
    {
        "id": "b2",
        "type": "heading",
        "order": 2,
        "bbox": [50, 120, 600, 170],
        "text": "Tiêu đề văn bản mẫu (Mock AI)",
        "confidence": 0.97,
    },
    {
        "id": "b3",
        "type": "paragraph",
        "order": 3,
        "bbox": [50, 200, 750, 280],
        "text": (
            "Nội dung đoạn văn thứ hai được giả lập từ Mock OCR Worker. "
            "Chữ tiếng Việt hoạt động đúng encoding UTF-8."
        ),
        "confidence": 0.95,
    },
    {
        "id": "b4",
        "type": "paragraph",
        "order": 4,
        "bbox": [50, 310, 750, 390],
        "text": (
            "Đây là đoạn văn thứ ba. Khi bạn thay thế Mock Worker bằng model AI thật, "
            "chỉ cần cập nhật hàm ocr() trong file này."
        ),
        "confidence": 0.92,
    },
    {
        "id": "b5",
        "type": "paragraph",
        "order": 5,
        "bbox": [50, 420, 750, 480],
        "text": "Dòng văn bản số 5 — confidence thấp để test highlight cảnh báo.",
        "confidence": 0.71,
    },
]


def _make_dummy_layout(page_number: int, width: int, height: int) -> dict:
    """
    Tạo dummy OCR layout JSON đúng chuẩn LayoutValidator của Backend.
    pageNumber phải khớp với trang thực tế trong job.
    bbox phải nằm trong [0, width] x [0, height].
    """
    # Điều chỉnh bbox nếu width/height nhỏ hơn template
    w = max(width, 800)
    h = max(height, 500)
    scale_x = w / 800
    scale_y = h / 1200

    blocks = []
    for block in DUMMY_BLOCKS_TEMPLATE:
        x1, y1, x2, y2 = block["bbox"]
        # Scale và clamp bbox vào trong page
        sx1 = min(int(x1 * scale_x), w - 1)
        sy1 = min(int(y1 * scale_y), h - 1)
        sx2 = min(int(x2 * scale_x), w)
        sy2 = min(int(y2 * scale_y), h)
        # Đảm bảo x1 < x2, y1 < y2
        if sx2 <= sx1:
            sx2 = sx1 + 1
        if sy2 <= sy1:
            sy2 = sy1 + 1
        # Clamp lần nữa sau correction
        sx2 = min(sx2, w)
        sy2 = min(sy2, h)

        blocks.append({
            "id": block["id"],
            "type": block["type"],
            "order": block["order"],
            "bbox": [sx1, sy1, sx2, sy2],
            "text": block["text"],
            "confidence": block["confidence"],
        })

    return {
        "schemaVersion": 1,
        "pageNumber": page_number,
        "width": w,
        "height": h,
        "blocks": blocks,
    }


# ─────────────────────────────────────────────
# Heartbeat thread
# ─────────────────────────────────────────────
class HeartbeatThread(threading.Thread):
    """Gửi heartbeat mỗi HEARTBEAT_INTERVAL giây để giữ lease."""

    def __init__(self, job_id: str, lease_token: str):
        super().__init__(daemon=True)
        self.job_id = job_id
        self.lease_token = lease_token
        self._stop_evt = threading.Event()
        self.failed = False

    def run(self):
        while not self._stop_evt.wait(HEARTBEAT_INTERVAL):
            try:
                _post(
                    f"/api/worker/jobs/{self.job_id}/heartbeat",
                    {"leaseToken": self.lease_token, "progress": 50, "stage": "OCR"},
                )
                LOG.debug("Heartbeat OK — job %s", self.job_id)
            except Exception as exc:
                LOG.warning("Heartbeat failed — job %s: %s", self.job_id, exc)
                self.failed = True
                return

    def stop(self):
        self._stop_evt.set()


# ─────────────────────────────────────────────
# Core: process one job
# ─────────────────────────────────────────────
def _claim_job() -> dict | None:
    """POST /api/worker/jobs/claim. Trả None nếu không có job."""
    try:
        result = _post(
            "/api/worker/jobs/claim",
            {"workerInstanceId": WORKER_INSTANCE_ID},
        )
        return result  # None = 204 (no job), dict = job claimed
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 204:
            return None
        raise


def _process_job(job: dict) -> None:
    job_id = job["jobId"]
    lease_token = job["leaseToken"]
    pages = job.get("pages", [])

    LOG.info("══ Bắt đầu xử lý job %s (%d trang) ══", job_id, len(pages))

    heartbeat = HeartbeatThread(job_id, lease_token)
    heartbeat.start()

    try:
        # ── Giả lập thời gian AI xử lý ──
        LOG.info("  Đang chạy Mock OCR pipeline... (%.0fs)", MOCK_PROCESSING_SECONDS)
        elapsed = 0.0
        step = 0.5
        while elapsed < MOCK_PROCESSING_SECONDS:
            if STOP.is_set() or heartbeat.failed:
                LOG.warning("  Job %s bị hủy (stop=%s, heartbeat_failed=%s)",
                             job_id, STOP.is_set(), heartbeat.failed)
                return
            time.sleep(step)
            elapsed += step

        if heartbeat.failed:
            LOG.warning("  Heartbeat đã mất — bỏ qua kết quả job %s", job_id)
            return

        # ── Tạo dummy kết quả cho mỗi trang ──
        page_results = []
        for page in pages:
            page_id = page["pageId"]
            page_number = page["pageNumber"]
            width = page.get("width") or 800
            height = page.get("height") or 1200

            dummy_layout = _make_dummy_layout(page_number, width, height)
            page_results.append({
                "pageId": page_id,
                "restoredImageKey": None,   # Không gửi — backend validate đường dẫn
                "rawLayoutJson": dummy_layout,
            })
            LOG.info("  ✓ Trang %d (pageId=%s) — %d blocks",
                     page_number, page_id, len(dummy_layout["blocks"]))

        # ── Gửi kết quả về backend ──
        _post(
            f"/api/worker/jobs/{job_id}/results",
            {
                "leaseToken": lease_token,
                "pageResults": page_results,
                "pipelineVersion": "mock-worker-v1.0",
            },
        )
        LOG.info("══ Job %s SUCCEEDED ══", job_id)

    except requests.exceptions.RequestException as exc:
        LOG.error("  Lỗi kết nối khi xử lý job %s: %s", job_id, exc)
        # Cố gắng báo fail về backend
        _try_fail(job_id, lease_token, "NETWORK_ERROR", str(exc)[:200])

    except Exception as exc:
        LOG.error("  Lỗi không mong đợi trong job %s: %s", job_id, exc)
        _try_fail(job_id, lease_token, "WORKER_PROCESSING_FAILED", str(exc)[:200])

    finally:
        heartbeat.stop()
        heartbeat.join(timeout=5)


def _try_fail(job_id: str, lease_token: str, error_code: str, error_message: str):
    """Gửi fail về backend, bỏ qua lỗi nếu không thể gửi."""
    try:
        _post(
            f"/api/worker/jobs/{job_id}/fail",
            {
                "leaseToken": lease_token,
                "errorCode": error_code,
                "errorMessage": error_message[:500],
            },
        )
        LOG.info("  Đã báo FAILED cho job %s", job_id)
    except Exception as exc:
        LOG.warning("  Không thể báo fail cho job %s: %s", job_id, exc)


# ─────────────────────────────────────────────
# Main polling loop
# ─────────────────────────────────────────────
def main():
    LOG.info("╔══════════════════════════════════════╗")
    LOG.info("║   Doc Forge — Mock AI Worker v1.0    ║")
    LOG.info("╚══════════════════════════════════════╝")
    LOG.info("Backend URL  : %s", BACKEND_URL)
    LOG.info("Worker ID    : %s", WORKER_INSTANCE_ID)
    LOG.info("Poll interval: %.0fs", POLL_INTERVAL)
    LOG.info("Mock delay   : %.0fs", MOCK_PROCESSING_SECONDS)
    LOG.info("Nhấn Ctrl+C để dừng.\n")

    consecutive_errors = 0

    while not STOP.is_set():
        try:
            job = _claim_job()

            if job is None:
                # Không có job — chờ rồi poll tiếp
                LOG.debug("Không có job, chờ %.0fs...", POLL_INTERVAL)
                consecutive_errors = 0
                STOP.wait(POLL_INTERVAL)
                continue

            consecutive_errors = 0
            _process_job(job)

        except requests.exceptions.ConnectionError:
            consecutive_errors += 1
            wait = min(30, 5 * consecutive_errors)
            LOG.warning("Backend chưa khởi động hoặc mất kết nối. Thử lại sau %ds...", wait)
            STOP.wait(wait)

        except requests.exceptions.Timeout:
            consecutive_errors += 1
            LOG.warning("Backend timeout. Thử lại sau 5s...")
            STOP.wait(5)

        except requests.exceptions.HTTPError as exc:
            consecutive_errors += 1
            status = exc.response.status_code if exc.response is not None else "?"
            LOG.error("Backend trả HTTP %s. Thử lại sau 5s...", status)
            STOP.wait(5)

        except Exception as exc:
            consecutive_errors += 1
            LOG.error("Lỗi không xác định: %s. Thử lại sau 5s...", exc)
            STOP.wait(5)

    LOG.info("Mock worker đã dừng.")


if __name__ == "__main__":
    main()
