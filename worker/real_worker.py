"""
Real AI Worker cho Doc Forge.

Tích hợp OpenCV ImagePreprocessor và PaddleOCR (hoặc OCREngine)
để xử lý ảnh thực tế từ Backend. Đồng thời xử lý Job dạng EXPORT.

Vòng lặp:
  1. POST /api/worker/jobs/claim
  2. Nếu job type == "OCR":
       - Tải ảnh gốc: GET /api/worker/jobs/{id}/pages/{pageId}/original  (+ X-Lease-Token)
       - Tiền xử lý (OpenCV Deskew, Denoise, CLAHE)
       - Nhận dạng chữ bằng OCREngine (PaddleOCR)
       - POST /api/worker/jobs/{id}/results
  3. Nếu job type == "EXPORT":
       - Tải snapshot JSON: GET /api/worker/jobs/{id}/snapshot           (+ X-Lease-Token)
       - Render file .docx bằng python-docx (hoặc export engine)
       - POST /api/worker/jobs/{id}/artifact                             (+ X-Lease-Token)
       - POST /api/worker/jobs/{id}/complete
  4. Heartbeat tự động trong thread riêng.
"""

import io
import logging
import os
import signal
import socket
import threading
import time
import requests
from dotenv import load_dotenv

from preprocessor import ImagePreprocessor
from ocr_engine import OCREngine

# Cài đặt python-docx nếu có
DOCX_AVAILABLE = False
try:
    import docx
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_AVAILABLE = True
except ImportError:
    pass

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("real_worker")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8080").rstrip("/")
WORKER_SECRET_KEY = os.environ.get("WORKER_SECRET_KEY", "worker-secret-local")
WORKER_INSTANCE_ID = os.environ.get("WORKER_INSTANCE_ID", socket.gethostname()[:100])
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL_SECONDS", "2"))
HEARTBEAT_INTERVAL = float(os.environ.get("HEARTBEAT_INTERVAL_SECONDS", "20"))

STOP = threading.Event()


def _signal_handler(*_):
    LOG.info("Nhận tín hiệu dừng, worker sẽ ngắt sau khi hoàn thành job hiện tại...")
    STOP.set()


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def _base_headers(lease_token: str | None = None) -> dict:
    """Headers chung cho mọi request: xác thực worker + tuỳ chọn lease token."""
    h = {"X-Worker-Key": WORKER_SECRET_KEY}
    if lease_token:
        h["X-Lease-Token"] = lease_token
    return h


def _post(path: str, payload: dict, timeout: int = 15, lease_token: str | None = None) -> dict | None:
    url = f"{BACKEND_URL}{path}"
    headers = _base_headers(lease_token)
    headers["Content-Type"] = "application/json"
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    return resp.json()


def _get_bytes(path: str, lease_token: str, timeout: int = 30) -> bytes:
    """Tải bytes (ảnh gốc) — bắt buộc truyền lease_token."""
    url = f"{BACKEND_URL}{path}"
    resp = requests.get(url, headers=_base_headers(lease_token), timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _get_json(path: str, lease_token: str, timeout: int = 15) -> dict:
    """Tải JSON (snapshot) — bắt buộc truyền lease_token."""
    url = f"{BACKEND_URL}{path}"
    resp = requests.get(url, headers=_base_headers(lease_token), timeout=timeout)
    resp.raise_for_status()
    return resp.json()


class HeartbeatThread(threading.Thread):
    def __init__(self, job_id: str, lease_token: str, stage: str = "OCR"):
        super().__init__(daemon=True)
        self.job_id = job_id
        self.lease_token = lease_token
        self.stage = stage
        self._stop_evt = threading.Event()
        self.failed = False

    def run(self):
        while not self._stop_evt.wait(HEARTBEAT_INTERVAL):
            try:
                _post(
                    f"/api/worker/jobs/{self.job_id}/heartbeat",
                    {"leaseToken": self.lease_token, "progress": 50, "stage": self.stage},
                )
                LOG.debug("Heartbeat OK — job %s", self.job_id)
            except Exception as exc:
                LOG.warning("Heartbeat thất bại — job %s: %s", self.job_id, exc)
                self.failed = True
                return

    def stop(self):
        self._stop_evt.set()


def _claim_job() -> dict | None:
    try:
        return _post("/api/worker/jobs/claim", {"workerInstanceId": WORKER_INSTANCE_ID})
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 204:
            return None
        raise


def _process_ocr_job(job: dict, preprocessor: ImagePreprocessor, ocr_engine: OCREngine):
    job_id = job["jobId"]
    lease_token = job["leaseToken"]
    pages = job.get("pages", [])

    LOG.info("══ [OCR JOB %s] Bắt đầu xử lý (%d trang) ══", job_id, len(pages))
    heartbeat = HeartbeatThread(job_id, lease_token, stage="OCR")
    heartbeat.start()

    try:
        page_results = []
        for idx, page in enumerate(pages, start=1):
            if STOP.is_set() or heartbeat.failed:
                LOG.warning("  Job %s bị hủy trước khi hoàn tất", job_id)
                return

            page_id = page["pageId"]
            page_number = page["pageNumber"]
            LOG.info("  -> Tải ảnh gốc trang %d/%d (pageId=%s)...", idx, len(pages), page_id)

            # FIX: truyền lease_token theo đúng X-Lease-Token header
            image_bytes = _get_bytes(
                f"/api/worker/jobs/{job_id}/pages/{page_id}/original",
                lease_token=lease_token,
            )

            # 1. Preprocess
            LOG.info("  -> Đang tiền xử lý ảnh (OpenCV Deskew, Denoise, CLAHE)...")
            cv_img, _ = preprocessor.process(image_bytes)

            # 2. OCR Layout
            LOG.info("  -> Đang chạy nhận dạng OCR...")
            layout_json = ocr_engine.process_image(cv_img, page_number)

            page_results.append({
                "pageId": page_id,
                "restoredImageKey": None,
                "rawLayoutJson": layout_json,
            })
            LOG.info("  ✓ Trang %d hoàn tất (%d blocks nhận dạng)", page_number, len(layout_json.get("blocks", [])))

        if heartbeat.failed:
            LOG.warning("  Mất heartbeat — bỏ qua gửi kết quả job %s", job_id)
            return

        # Gửi kết quả OCR về backend qua /results
        _post(
            f"/api/worker/jobs/{job_id}/results",
            {
                "leaseToken": lease_token,
                "pageResults": page_results,
                "pipelineVersion": "real-worker-paddleocr-v1.0",
            },
        )
        LOG.info("══ [OCR JOB %s] SUCCEEDED ══", job_id)

    except Exception as exc:
        LOG.error("Lỗi khi xử lý OCR Job %s: %s", job_id, exc, exc_info=True)
        _try_fail(job_id, lease_token, "WORKER_PROCESSING_FAILED", str(exc))

    finally:
        heartbeat.stop()
        heartbeat.join(timeout=5)


def _process_export_job(job: dict):
    job_id = job["jobId"]
    lease_token = job["leaseToken"]

    LOG.info("══ [EXPORT JOB %s] Bắt đầu xuất tài liệu ══", job_id)
    heartbeat = HeartbeatThread(job_id, lease_token, stage="EXPORTING")
    heartbeat.start()

    try:
        # FIX: path /snapshot (không phải /export-snapshot) + truyền X-Lease-Token
        LOG.info("  -> Tải export snapshot JSON từ Backend...")
        snapshot = _get_json(f"/api/worker/jobs/{job_id}/snapshot", lease_token=lease_token)

        # Tạo file DOCX bằng python-docx
        docx_bytes = _generate_docx_from_snapshot(snapshot)

        # FIX: path /artifact (không phải /export-artifact) + X-Lease-Token header
        LOG.info("  -> Upload DOCX artifact (%d bytes)...", len(docx_bytes))
        url = f"{BACKEND_URL}/api/worker/jobs/{job_id}/artifact"
        files = {"file": ("document.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        headers = _base_headers(lease_token)  # bao gồm cả X-Lease-Token
        resp = requests.post(url, files=files, headers=headers, timeout=30)
        resp.raise_for_status()
        artifact_data = resp.json()
        artifact_key = artifact_data.get("storageKey")

        # Gọi /complete để hoàn tất export job
        _post(
            f"/api/worker/jobs/{job_id}/complete",
            {
                "leaseToken": lease_token,
                "exportStorageKey": artifact_key,
                "pipelineVersion": "docx-exporter-v1.0",
            },
        )
        LOG.info("══ [EXPORT JOB %s] SUCCEEDED ══", job_id)

    except Exception as exc:
        LOG.error("Lỗi khi xử lý Export Job %s: %s", job_id, exc, exc_info=True)
        _try_fail(job_id, lease_token, "EXPORT_PROCESSING_FAILED", str(exc))

    finally:
        heartbeat.stop()
        heartbeat.join(timeout=5)


def _generate_docx_from_snapshot(snapshot: dict) -> bytes:
    """Tái tạo văn bản Word .docx từ snapshot JSON các trang."""
    if not DOCX_AVAILABLE:
        # Fallback zip / minimal docx nếu không có python-docx
        raise RuntimeError("python-docx chưa được cài đặt. Hãy chạy: pip install python-docx")

    doc = docx.Document()

    # Cấu hình lề trang (1 inch)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    pages = snapshot.get("pages", [])
    for p_idx, page in enumerate(pages):
        if p_idx > 0:
            doc.add_page_break()

        layout = page.get("layout", {})
        blocks = layout.get("blocks", [])

        # Sắp xếp blocks theo order
        blocks.sort(key=lambda b: b.get("order", 0))

        for block in blocks:
            b_type = str(block.get("type", "paragraph")).lower()
            text = block.get("text", "").strip()
            if not text:
                continue

            if "heading" in b_type or b_type == "title":
                p = doc.add_heading(text, level=1)
                p.paragraph_format.space_before = Pt(12)
                p.paragraph_format.space_after = Pt(6)
            elif "table" in b_type:
                # Render table nếu có dữ liệu bảng, nếu không thì render như đoạn văn
                p = doc.add_paragraph()
                run = p.add_run(f"[BẢNG]: {text}")
                run.bold = True
                p.paragraph_format.space_after = Pt(6)
            elif "list" in b_type:
                doc.add_paragraph(text, style="List Bullet")
            else:
                p = doc.add_paragraph(text)
                p.paragraph_format.space_after = Pt(6)
                p.paragraph_format.line_spacing = 1.15

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def _try_fail(job_id: str, lease_token: str, error_code: str, error_message: str):
    try:
        _post(
            f"/api/worker/jobs/{job_id}/fail",
            {
                "leaseToken": lease_token,
                "errorCode": error_code,
                "errorMessage": error_message[:500],
            },
        )
        LOG.info("  Đã gửi báo FAILED cho job %s", job_id)
    except Exception as exc:
        LOG.warning("  Không thể báo fail cho job %s: %s", job_id, exc)


def main():
    LOG.info("╔══════════════════════════════════════╗")
    LOG.info("║   Doc Forge — Real AI Worker v1.1    ║")
    LOG.info("╚══════════════════════════════════════╝")
    LOG.info("Backend URL  : %s", BACKEND_URL)
    LOG.info("Worker ID    : %s", WORKER_INSTANCE_ID)
    LOG.info("Poll interval: %.0fs\n", POLL_INTERVAL)

    preprocessor = ImagePreprocessor()
    ocr_engine = OCREngine(lang="vi")

    consecutive_errors = 0

    while not STOP.is_set():
        try:
            job = _claim_job()

            if job is None:
                consecutive_errors = 0
                STOP.wait(POLL_INTERVAL)
                continue

            consecutive_errors = 0
            job_type = job.get("type", "OCR")

            if job_type == "EXPORT":
                _process_export_job(job)
            else:
                _process_ocr_job(job, preprocessor, ocr_engine)

        except requests.exceptions.ConnectionError:
            consecutive_errors += 1
            wait = min(30, 5 * consecutive_errors)
            LOG.warning("Chưa kết nối được Backend. Thử lại sau %ds...", wait)
            STOP.wait(wait)

        except Exception as exc:
            consecutive_errors += 1
            LOG.error("Lỗi không mong đợi trong main loop: %s. Thử lại sau 5s...", exc)
            STOP.wait(5)

    LOG.info("Real AI worker đã dừng.")


if __name__ == "__main__":
    main()
