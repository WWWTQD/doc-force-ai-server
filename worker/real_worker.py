"""
Real AI Worker cho Doc Forge.

Tích hợp OpenCV ImagePreprocessor, DocumentRestorationUNet và PaddleOCR (hoặc OCREngine)
để xử lý ảnh thực tế từ Backend. Đồng thời xử lý Job dạng EXPORT.

Vòng lặp:
  1. POST /api/worker/jobs/claim
  2. Nếu job type == "OCR":
       - Tải ảnh gốc: GET /api/worker/jobs/{id}/pages/{pageId}/original  (+ X-Lease-Token)
       - Tiền xử lý (OpenCV Deskew, Denoise, CLAHE)
       - Khôi phục chất lượng ảnh bằng DocumentRestorationUNet (nền trắng sạch, nét chữ đậm)
       - Upload ảnh đã khôi phục: POST /api/worker/jobs/{id}/pages/{pageId}/restored
       - Nhận dạng chữ bằng OCREngine (PaddleOCR) trên ảnh đã khôi phục
       - Gửi kết quả (rawLayoutJson + restoredImageKey): POST /api/worker/jobs/{id}/results
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
import sys
import threading
import time
from typing import Optional, Any
import requests
import cv2
from dotenv import load_dotenv

# Đảm bảo import được các module từ root và worker
WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORKER_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if WORKER_DIR not in sys.path:
    sys.path.insert(0, WORKER_DIR)

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


def _upload_restored(job_id: str, page_id: str, lease_token: str, image_bytes: bytes, timeout: int = 30) -> str | None:
    """Upload ảnh phục hồi lên Backend Storage — trả về storageKey hợp lệ hoặc None."""
    url = f"{BACKEND_URL}/api/worker/jobs/{job_id}/pages/{page_id}/restored"
    headers = _base_headers(lease_token)
    files = {"file": ("restored.jpg", io.BytesIO(image_bytes), "image/jpeg")}
    try:
        resp = requests.post(url, files=files, headers=headers, timeout=timeout)
        if resp.ok:
            data = resp.json()
            return data.get("storageKey")
        LOG.warning("Upload ảnh restored trang %s trả về HTTP %d: %s", page_id, resp.status_code, resp.text)
    except Exception as exc:
        LOG.warning("Lỗi kết nối khi upload ảnh restored trang %s: %s", page_id, exc)
    return None


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


def _process_ocr_job(job: dict, preprocessor: ImagePreprocessor, ocr_engine: OCREngine, restoration_service: Optional[Any] = None):
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

            image_bytes = _get_bytes(
                f"/api/worker/jobs/{job_id}/pages/{page_id}/original",
                lease_token=lease_token,
            )

            # 1. Preprocess (Deskew, Denoise, CLAHE)
            LOG.info("  -> Đang tiền xử lý ảnh (OpenCV Deskew, Denoise, CLAHE)...")
            cv_img, _ = preprocessor.process(image_bytes)

            # 2. Document Image Restoration (DocumentRestorationUNet)
            restored_image_key = None
            if restoration_service is not None and restoration_service.is_available:
                try:
                    LOG.info("  -> Đang khôi phục ảnh bằng DocumentRestorationUNet...")
                    restored_cv = restoration_service.restore_image_bgr(
                        cv_img,
                        tile_size=512,
                        overlap=64,
                        preserve_color=True,
                    )
                    success, restored_buf = cv2.imencode(".jpg", restored_cv, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if success:
                        restored_bytes = restored_buf.tobytes()
                        restored_image_key = _upload_restored(job_id, page_id, lease_token, restored_bytes)
                        if restored_image_key:
                            LOG.info("  ✓ Đã upload ảnh restored: %s", restored_image_key)
                    # Chuyển ảnh đã khôi phục sang cho OCR
                    cv_img = restored_cv
                except Exception as ex:
                    LOG.warning("  Khôi phục ảnh lỗi: %s. Dùng ảnh gốc.", ex)

            # 3. OCR Layout trên ảnh đã khôi phục
            LOG.info("  -> Đang chạy nhận dạng OCR...")
            layout_json = ocr_engine.process_image(cv_img, page_number)

            page_results.append({
                "pageId": page_id,
                "restoredImageKey": restored_image_key,
                "rawLayoutJson": layout_json,
            })
            LOG.info("  ✓ Trang %d hoàn tất (%d blocks nhận dạng, restoredImageKey=%s)", 
                     page_number, len(layout_json.get("blocks", [])), restored_image_key)

        if heartbeat.failed:
            LOG.warning("  Mất heartbeat — bỏ qua gửi kết quả job %s", job_id)
            return

        # Gửi kết quả OCR về backend qua /results
        _post(
            f"/api/worker/jobs/{job_id}/results",
            {
                "leaseToken": lease_token,
                "pageResults": page_results,
                "pipelineVersion": "real-worker-paddleocr-unet-v4",
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
        LOG.info("  -> Tải export snapshot JSON từ Backend...")
        snapshot = _get_json(f"/api/worker/jobs/{job_id}/snapshot", lease_token=lease_token)

        # Tạo file DOCX bằng python-docx
        docx_bytes = _generate_docx_from_snapshot(snapshot)

        # Upload file kết quả lên backend qua endpoint /artifact
        url = f"{BACKEND_URL}/api/worker/jobs/{job_id}/artifact"
        headers = _base_headers(lease_token)
        files = {
            "file": (
                f"export-{job_id}.docx",
                io.BytesIO(docx_bytes),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        }
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
        raise RuntimeError("python-docx chưa được cài đặt. Hãy chạy: pip install python-docx")

    doc = docx.Document()

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
    LOG.info("╔═════════════════════════════════════════════════╗")
    LOG.info("║ Doc Forge — Real AI Worker (U-Net v4 + OCR)     ║")
    LOG.info("╚═════════════════════════════════════════════════╝")
    LOG.info("Backend URL  : %s", BACKEND_URL)
    LOG.info("Worker ID    : %s", WORKER_INSTANCE_ID)
    LOG.info("Poll interval: %.0fs\n", POLL_INTERVAL)

    preprocessor = ImagePreprocessor()
    ocr_engine = OCREngine(lang="vi")

    # Khởi tạo mô hình DocumentRestorationUNet
    restoration_service = None
    try:
        from app.services.restoration_service import RestorationService
        restoration_service = RestorationService.get_instance()
        restoration_service.initialize()
        if restoration_service.is_available:
            LOG.info("DocumentRestorationUNet đã sẵn sàng phục vụ AI Worker.")
        else:
            LOG.warning("RestorationService khởi tạo nhưng mô hình chưa sẵn sàng (sẽ fallback ảnh gốc).")
    except Exception as ex:
        LOG.warning("Không thể khởi tạo RestorationService: %s", ex)

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
                _process_ocr_job(job, preprocessor, ocr_engine, restoration_service=restoration_service)

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
