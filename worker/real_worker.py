"""
Real AI Worker cho Doc Forge.

Tích hợp DocumentScanner, OpenCV ImagePreprocessor, DocumentRestorationUNet
và OCREngine (YOLO Layout + PaddleOCR + VietOCR) để xử lý ảnh thực tế từ Backend.
Đồng thời xử lý Job dạng EXPORT.

Vòng lặp:
  1. POST /api/worker/jobs/claim
  2. Nếu job type == "OCR":
       - Tải ảnh gốc: GET /api/worker/jobs/{id}/pages/{pageId}/original  (+ X-Lease-Token)
       - Phát hiện tờ giấy (YOLO document_detector.pt) -> nắn phẳng -> xoay đúng chiều
       - Tiền xử lý (OpenCV Deskew, Denoise, Magic Color khử bóng)
       - Khôi phục chất lượng ảnh bằng DocumentRestorationUNet (làm trắng nền & rõ nét chữ)
       - Upload ảnh đã khôi phục lên Backend (/restored) hoặc lưu vào SHARED_STORAGE_ROOT
       - Phân tích bố cục (YOLO layout_best_backup.pt) + nhận dạng chữ (PaddleOCR + VietOCR) trên chính ảnh đã khôi phục
       - POST /api/worker/jobs/{id}/results
  3. Nếu job type == "EXPORT":
       - Tải snapshot JSON: GET /api/worker/jobs/{id}/snapshot           (+ X-Lease-Token)
       - Render file .docx bằng python-docx (hoặc export engine)
       - POST /api/worker/jobs/{id}/artifact                             (+ X-Lease-Token)
       - POST /api/worker/jobs/{id}/complete
  4. Heartbeat tự động trong thread riêng.
"""

import os
import io
import sys
import logging
import signal
import socket
import subprocess
import tempfile
import threading
import time

# Đảm bảo import được các module từ root và worker
WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORKER_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if WORKER_DIR not in sys.path:
    sys.path.insert(0, WORKER_DIR)

# Cấu hình UTF-8 cho stdout/stderr trên Windows để tránh lỗi UnicodeEncodeError khi ghi log
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Cấu hình logging chuẩn ra stdout
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for h in root_logger.handlers[:]:
    root_logger.removeHandler(h)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
root_logger.addHandler(console_handler)
LOG = logging.getLogger("real_worker")
LOG.setLevel(logging.INFO)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
try:
    import torch
except Exception:
    pass

import cv2
import requests
from dotenv import load_dotenv

from preprocessor import ImagePreprocessor
from ocr_engine import OCREngine
from document_scanner import DocumentScanner, map_layout_to_original
from job_progress import VALID_HEARTBEAT_STAGES, compute_progress, parse_scan_options
from scan_pipeline import ScanPipeline

# Cài đặt python-docx nếu có
DOCX_AVAILABLE = False
try:
    import docx
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_AVAILABLE = True
except ImportError:
    pass

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

def _configure_worker_logging(clear_file: bool = False):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers[:]:
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    try:
        log_file = os.path.join(PROJECT_ROOT, "worker_live.log")
        fh = logging.FileHandler(log_file, mode="w" if clear_file else "a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        pass


_configure_worker_logging(clear_file=True)
LOG = logging.getLogger("real_worker")
LOG.setLevel(logging.INFO)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8080").rstrip("/")
WORKER_SECRET_KEY = os.environ.get("WORKER_SECRET_KEY", "worker-secret-local")
WORKER_INSTANCE_ID = os.environ.get("WORKER_INSTANCE_ID", socket.gethostname()[:100])
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL_SECONDS", "2"))
HEARTBEAT_INTERVAL = float(os.environ.get("HEARTBEAT_INTERVAL_SECONDS", "20"))

# Thư mục storage dùng chung với Backend (vd: ...\PBL4-doc_forge-Backend\storage hoặc /data/storage).
# Khi đặt, worker ghi ảnh đã nắn phẳng/làm sạch làm "ảnh phục hồi" và giữ bbox theo ảnh đó.
_default_storage = "/data/storage" if os.path.isdir("/data/storage") else ""
SHARED_STORAGE_ROOT = os.environ.get("SHARED_STORAGE_ROOT", _default_storage).strip()
ENABLE_MAGIC_FILTER = os.environ.get("ENABLE_MAGIC_FILTER", "true").lower() != "false"
ENABLE_DOCUMENT_SCAN = os.environ.get("ENABLE_DOCUMENT_SCAN", "true").lower() != "false"
_env_restoration = os.environ.get("ENABLE_IMAGE_RESTORATION", "true")
ENABLE_IMAGE_RESTORATION = _env_restoration.strip().lower() not in ("0", "false", "no", "off")
ENABLE_LLM_POSTPROCESS = os.environ.get("ENABLE_LLM_POSTPROCESS", os.environ.get("LLM_POSTPROCESS_ENABLED", "true")).lower() != "false"
PIPELINE_VERSION = "real-worker-unet-v5.1-gemini-postocr-v2.3"

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
    def __init__(self, job_id: str, lease_token: str, stage: str = "OCR", progress: int = 5):
        super().__init__(daemon=True)
        self.job_id = job_id
        self.lease_token = lease_token
        self._lock = threading.Lock()
        self.stage = stage if stage in VALID_HEARTBEAT_STAGES else "OCR"
        self.progress = max(0, min(99, int(progress)))
        self._stop_evt = threading.Event()
        self.consecutive_fails = 0
        self.failed = False

    def update(self, progress: int | None = None, stage: str | None = None) -> None:
        with self._lock:
            if stage is not None and stage in VALID_HEARTBEAT_STAGES:
                self.stage = stage
            if progress is not None:
                clamped = max(5, min(95, int(progress)))
                if clamped > self.progress:
                    self.progress = clamped
        self._send_beat()

    def _send_beat(self):
        if self._stop_evt.is_set():
            return
        with self._lock:
            stage = self.stage
            progress = self.progress
        try:
            _post(
                f"/api/worker/jobs/{self.job_id}/heartbeat",
                {"leaseToken": self.lease_token, "progress": progress, "stage": stage},
                lease_token=self.lease_token,
            )
            LOG.debug("Heartbeat OK — job %s (stage=%s, progress=%d)", self.job_id, stage, progress)
            self.consecutive_fails = 0
        except requests.exceptions.HTTPError as exc:
            if self._stop_evt.is_set() or (exc.response is not None and exc.response.status_code == 409):
                self._stop_evt.set()
                return
            self.consecutive_fails += 1
            LOG.warning("Heartbeat cảnh báo (lần %d) — job %s: %s", self.consecutive_fails, self.job_id, exc)
        except Exception as exc:
            if self._stop_evt.is_set():
                return
            self.consecutive_fails += 1
            LOG.warning("Heartbeat cảnh báo (lần %d) — job %s: %s", self.consecutive_fails, self.job_id, exc)

    def run(self):
        # Gửi ngay heartbeat đầu tiên để reset thời hạn lease ngay khi claim
        self._send_beat()
        while not self._stop_evt.wait(10):
            if self._stop_evt.is_set():
                break
            self._send_beat()

    def stop(self):
        self._stop_evt.set()


def _claim_job() -> dict | None:
    try:
        return _post("/api/worker/jobs/claim", {"workerInstanceId": WORKER_INSTANCE_ID})
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 204:
            return None
        raise


def _upload_restored(job_id: str, page_id: str, lease_token: str, image_bytes: bytes, timeout: int = 30) -> str | None:
    """Upload ảnh đã được U-Net làm sạch lên Backend (/restored) để lấy restoredImageKey."""
    url = f"{BACKEND_URL}/api/worker/jobs/{job_id}/pages/{page_id}/restored"
    headers = _base_headers(lease_token)
    files = {"file": ("restored.jpg", io.BytesIO(image_bytes), "image/jpeg")}
    try:
        resp = requests.post(url, files=files, headers=headers, timeout=timeout)
        if resp.ok:
            data = resp.json()
            key = data.get("storageKey")
            if key:
                LOG.info("  ✓ Đã upload ảnh restored qua API (/restored): %s", key)
                return key
        LOG.warning("  Upload ảnh restored trang %s trả về HTTP %d: %s", page_id, resp.status_code, resp.text)
    except Exception as exc:
        LOG.warning("  Lỗi kết nối khi upload ảnh restored qua API (trang %s): %s", page_id, exc)
    return None


def _sync_to_docker_backend(local_path: str, storage_key: str):
    """
    Nếu Spring Boot Backend đang chạy trong Docker container (ceabf1005847 / pbl4-doc_forge-backend-app-1),
    sao chép trực tiếp file ảnh đã phục hồi vào /data/storage của container để frontend tải được ảnh sạch.
    """
    container_names = ["pbl4-doc_forge-backend-app-1", "docforge-app"]
    dest_dir = "/data/storage/" + "/".join(storage_key.split("/")[:-1])

    for cname in container_names:
        try:
            # Tạo thư mục đích trong container
            subprocess.run(
                ["docker", "exec", cname, "mkdir", "-p", dest_dir],
                capture_output=True, timeout=5
            )
            # Copy file vào container
            dest_file = f"{cname}:/data/storage/{storage_key}"
            res = subprocess.run(["docker", "cp", local_path, dest_file], capture_output=True, timeout=5)
            if res.returncode == 0:
                LOG.info("  ✓ Đã đồng bộ restored image vào Docker backend (%s): %s", cname, storage_key)
                return True
        except Exception:
            pass
    return False


def _save_restored_image(job: dict, page_id: str, img) -> str | None:
    """
    1. Ưu tiên upload ảnh đã được U-Net làm sạch lên Backend qua POST /api/worker/jobs/{jobId}/pages/{pageId}/restored.
    2. Fallback: Ghi ảnh phục hồi vào storage dùng chung theo đúng key Backend chấp nhận:
       documents/{documentId}/jobs/{jobId}/{leaseToken}/pages/{pageId}/restored.jpg
    Trả về storage key, hoặc None nếu không ghi được.
    """
    try:
        ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
        if not ok:
            raise ValueError("Không encode được ảnh phục hồi")
        image_bytes = encoded.tobytes()
    except Exception as exc:
        LOG.warning("  Không encode được ảnh phục hồi: %s", exc)
        return None

    # Bước 1: Nếu có SHARED_STORAGE_ROOT (/data/storage trong Docker), ghi trực tiếp cực nhanh
    if job.get("documentId") and SHARED_STORAGE_ROOT and os.path.isdir(SHARED_STORAGE_ROOT):
        key = (f"documents/{job['documentId']}/jobs/{job['jobId']}/{job['leaseToken']}"
               f"/pages/{page_id}/restored.jpg")
        target = os.path.join(SHARED_STORAGE_ROOT, *key.split("/"))
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            temp = target + ".tmp"
            with open(temp, "wb") as fh:
                fh.write(image_bytes)
            os.replace(temp, target)
            LOG.info("  ✓ Đã lưu ảnh phục hồi vào shared storage: %s", key)
            return key
        except Exception as exc:
            LOG.warning("  Không ghi được vào shared storage (%s): %s", target, exc)

    # Bước 2: Upload trực tiếp qua HTTP API (/restored) của Backend
    job_id = job.get("jobId")
    lease_token = job.get("leaseToken")
    if job_id and lease_token:
        uploaded_key = _upload_restored(job_id, page_id, lease_token, image_bytes)
        if uploaded_key:
            return uploaded_key

    # Bước 3: Fallback ghi vào temp + docker cp
    if not job.get("documentId"):
        return None
    key = (f"documents/{job['documentId']}/jobs/{job['jobId']}/{job['leaseToken']}"
           f"/pages/{page_id}/restored.jpg")

    root = os.path.join(tempfile.gettempdir(), "docforge_storage")
    os.makedirs(root, exist_ok=True)

    target = os.path.join(root, *key.split("/"))
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        temp = target + ".tmp"
        with open(temp, "wb") as fh:
            fh.write(image_bytes)
        os.replace(temp, target)

        # Đồng bộ sang Docker backend container nếu backend đang chạy qua docker-compose
        _sync_to_docker_backend(target, key)
        return key
    except Exception as exc:
        LOG.warning("  Không ghi được ảnh phục hồi (%s): %s — dùng toạ độ ảnh gốc.", target, exc)
        return None


def _process_ocr_job(job: dict, pipeline: ScanPipeline):
    job_id = job["jobId"]
    lease_token = job["leaseToken"]
    pages = job.get("pages", [])
    n_pages = max(1, len(pages))
    job_options = parse_scan_options(
        job,
        default_llm=ENABLE_LLM_POSTPROCESS,
        default_unet=ENABLE_IMAGE_RESTORATION,
    )

    LOG.info(
        "══ [OCR JOB %s] Bắt đầu xử lý (%d trang, useUnet=%s, useLlm=%s) ══",
        job_id,
        len(pages),
        job_options.get("useUnet"),
        job_options.get("llmPostprocess"),
    )
    heartbeat = HeartbeatThread(
        job_id,
        lease_token,
        stage="PREPROCESSING",
        progress=compute_progress(0, n_pages, "PREPROCESSING"),
    )
    heartbeat.start()

    try:
        page_results = []
        for idx, page in enumerate(pages, start=1):
            if STOP.is_set():
                LOG.warning("  Job %s bị hủy do nhận tín hiệu dừng", job_id)
                return

            page_id = page["pageId"]
            page_number = page["pageNumber"]
            LOG.info("  -> Tải ảnh gốc trang %d/%d (pageId=%s)...", idx, len(pages), page_id)

            # FIX: truyền lease_token theo đúng X-Lease-Token header
            image_bytes = _get_bytes(
                f"/api/worker/jobs/{job_id}/pages/{page_id}/original",
                lease_token=lease_token,
            )

            # 1. Bắt tờ giấy + nắn phẳng + xoay -> 2. Deskew/Denoise/Magic Color hoặc U-Net v5.1 -> 3. Layout + OCR -> 4. Gemini Post-OCR
            LOG.info(
                "  -> Scan tài liệu + %s + OCR%s...",
                "U-Net v5.1 Hybrid" if job_options.get("useUnet") else "OpenCV Magic Enhance",
                " + Gemini Post-OCR" if job_options.get("llmPostprocess") else "",
            )
            result = pipeline.process_bytes(
                image_bytes,
                page_number,
                options=job_options,
                progress_cb=lambda stg, _i=idx - 1: heartbeat.update(
                    progress=compute_progress(_i, n_pages, stg),
                    stage=stg,
                ),
            )

            # Upload ảnh phục hồi lên Backend (/restored) hoặc ghi trực tiếp vào SHARED_STORAGE_ROOT (/data/storage)
            restored_key = _save_restored_image(job, page_id, result.restored_image)
            if result.geometry.rotation in (90, 180, 270) and job.get("documentId"):
                try:
                    ok_orig, enc_orig = cv2.imencode(".jpg", result.geometry.image, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
                    if ok_orig:
                        orig_key = f"documents/{job['documentId']}/pages/{page_id}/original.jpg"
                        orig_bytes = enc_orig.tobytes()
                        if SHARED_STORAGE_ROOT and os.path.isdir(SHARED_STORAGE_ROOT):
                            orig_target = os.path.join(SHARED_STORAGE_ROOT, *orig_key.split("/"))
                            os.makedirs(os.path.dirname(orig_target), exist_ok=True)
                            with open(orig_target, "wb") as f_orig:
                                f_orig.write(orig_bytes)
                            LOG.info("  ✓ Đã cập nhật ảnh gốc đã xoay (%d°) vào storage: %s", result.geometry.rotation, orig_key)
                        else:
                            tmp_orig = os.path.join(tempfile.gettempdir(), f"rotated_orig_{page_id}.jpg")
                            with open(tmp_orig, "wb") as f_orig:
                                f_orig.write(orig_bytes)
                            _sync_to_docker_backend(tmp_orig, orig_key)
                        rh, rw = result.restored_image.shape[:2]
                        subprocess.run(
                            [
                                "docker", "exec", "pbl4-doc_forge-backend-db-1",
                                "psql", "-U", "docforge", "-d", "docforge", "-c",
                                f"UPDATE document_pages SET width = {int(rw)}, height = {int(rh)} WHERE id = '{page_id}';",
                            ],
                            capture_output=True, timeout=5,
                        )
                except Exception as exc_orig:
                    LOG.debug("Không thể đồng bộ ảnh gốc đã xoay: %s", exc_orig)
            layout_json = result.layout if restored_key else result.layout_in_original_coords()

            page_results.append({
                "pageId": page_id,
                "restoredImageKey": restored_key,
                "rawLayoutJson": layout_json,
            })
            LOG.info("  ✓ Trang %d hoàn tất (%d blocks, scan=%s, xoay=%d°)", page_number,
                     len(layout_json.get("blocks", [])), result.geometry.method, result.geometry.rotation)

        if STOP.is_set():
            LOG.warning("  Dừng worker — bỏ qua gửi kết quả job %s", job_id)
            return

        # Gửi kết quả OCR về backend qua /results
        _post(
            f"/api/worker/jobs/{job_id}/results",
            {
                "leaseToken": lease_token,
                "pageResults": page_results,
                "pipelineVersion": PIPELINE_VERSION,
            },
        )
        # Dọn dẹp kết quả cũ (nếu trang được chạy lại OCR) để tránh trùng version
        try:
            subprocess.run(
                [
                    "docker", "exec", "pbl4-doc_forge-backend-db-1",
                    "psql", "-U", "docforge", "-d", "docforge", "-c",
                    f"DELETE FROM page_results WHERE page_id IN (SELECT id FROM document_pages WHERE document_id = '{job.get('documentId')}') AND processing_job_id <> '{job_id}';",
                ],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
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

        export_format = str(
            snapshot.get("format")
            or (job.get("options") or {}).get("format")
            or "DOCX"
        ).upper()

        if "PDF" in export_format:
            LOG.info("  -> Kết xuất file PDF chuẩn A4 nền trắng tinh khiết (format=%s)...", export_format)
            artifact_bytes = _generate_pdf_from_snapshot(snapshot)
            filename = "document.pdf"
            mime_type = "application/pdf"
            pipe_ver = "pdf-exporter-v2.0"
        else:
            LOG.info("  -> Kết xuất file Word DOCX chuẩn định dạng (format=%s)...", export_format)
            artifact_bytes = _generate_docx_from_snapshot(snapshot)
            filename = "document.docx"
            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            pipe_ver = "docx-exporter-v2.0"

        LOG.info("  -> Upload %s artifact (%d bytes)...", filename, len(artifact_bytes))
        url = f"{BACKEND_URL}/api/worker/jobs/{job_id}/artifact"
        files = {"file": (filename, artifact_bytes, mime_type)}
        headers = _base_headers(lease_token)
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
                "pipelineVersion": pipe_ver,
            },
        )
        LOG.info("══ [EXPORT JOB %s (%s)] SUCCEEDED ══", job_id, export_format)

    except Exception as exc:
        LOG.error("Lỗi khi xử lý Export Job %s: %s", job_id, exc, exc_info=True)
        _try_fail(job_id, lease_token, "EXPORT_PROCESSING_FAILED", str(exc))

    finally:
        heartbeat.stop()
        heartbeat.join(timeout=5)


try:
    from export_layout import (
        append_formatted_runs as _append_formatted_runs,
        generate_docx_from_snapshot as _export_generate_docx_from_snapshot,
        generate_pdf_from_snapshot as _export_generate_pdf_from_snapshot,
        resolve_ttf_font as _resolve_ttf_font,
        strip_markdown_tokens as _strip_markdown_tokens,
    )
except ImportError:
    from worker.export_layout import (
        append_formatted_runs as _append_formatted_runs,
        generate_docx_from_snapshot as _export_generate_docx_from_snapshot,
        generate_pdf_from_snapshot as _export_generate_pdf_from_snapshot,
        resolve_ttf_font as _resolve_ttf_font,
        strip_markdown_tokens as _strip_markdown_tokens,
    )


def _generate_pdf_from_snapshot(snapshot: dict) -> bytes:
    """
    Tái tạo file PDF (.pdf) nhiều trang chuẩn khổ A4 (200 DPI: 1654 x 2339 px),
    giữ đúng hàng nhiều cột (Quốc hiệu 2 bên / biểu mẫu), khối chữ ký nửa phải và bảng biểu.
    """
    return _export_generate_pdf_from_snapshot(snapshot)


def _generate_docx_from_snapshot(snapshot: dict) -> bytes:
    """Tái tạo văn bản Word .docx từ snapshot JSON, giữ đúng hàng nhiều cột & chữ ký nửa phải."""
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx chưa được cài đặt. Hãy chạy: pip install python-docx")
    return _export_generate_docx_from_snapshot(snapshot)


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
    print("╔══════════════════════════════════════╗", flush=True)
    print("║   Doc Forge — Real AI Worker v1.3    ║", flush=True)
    print("╚══════════════════════════════════════╝", flush=True)
    print(f"Backend URL  : {BACKEND_URL}", flush=True)
    print(f"Worker ID    : {WORKER_INSTANCE_ID}", flush=True)
    print(f"Poll interval: {POLL_INTERVAL:.0f}s\n", flush=True)

    # Đảm bảo logging ra console và worker_live.log không bị Ultralytics hay PaddleOCR đè level
    _configure_worker_logging(clear_file=True)
    LOG.info("Backend URL: %s | Worker ID: %s | Poll interval: %.0fs", BACKEND_URL, WORKER_INSTANCE_ID, POLL_INTERVAL)

    # Khởi tạo sẵn DocumentRestorationUNet v5.1 trong RAM để sẵn sàng khi người dùng bật useUnet=true trên UI
    restoration_service = None
    try:
        from app.services.restoration_service import RestorationService
        restoration_service = RestorationService.get_instance()
        if restoration_service.is_available:
            LOG.info(
                "✓ DocumentRestorationUNet v5.1 (Stroke-Preserving Hybrid) đã sẵn sàng! (Mặc định tự động: %s)",
                "BẬT" if ENABLE_IMAGE_RESTORATION else "TẮT - bật linh hoạt qua useUnet",
            )
        else:
            LOG.warning("DocumentRestorationUNet chưa có trọng số khả dụng.")
    except Exception as exc:
        LOG.warning("Không thể khởi tạo RestorationService: %s", exc)

    LOG.info(
        "✓ Gemini Post-OCR (llm_postprocessor): mặc định=%s | model=%s | fallback=%s",
        "BẬT" if ENABLE_LLM_POSTPROCESS else "TẮT",
        os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite"),
        os.environ.get(
            "LLM_FALLBACK_MODEL",
            "gemini-3.1-flash-lite,gemini-3.5-flash,gemini-2.5-flash,gemini-2.5-flash-lite,gemini-2.0-flash",
        ),
    )

    preprocessor = ImagePreprocessor(enable_magic=ENABLE_MAGIC_FILTER)
    ocr_engine = OCREngine(lang="vi")
    pipeline = ScanPipeline(
        ocr_engine=ocr_engine,
        preprocessor=preprocessor,
        restoration_service=restoration_service,
        enable_document_scan=ENABLE_DOCUMENT_SCAN,
        enable_restoration=ENABLE_IMAGE_RESTORATION,
    )

    # Đảm bảo logging ra console và worker_live.log không bị Ultralytics hay PaddleOCR đè level
    _configure_worker_logging(clear_file=False)
    LOG.info("✓ AI Worker đã sẵn sàng! Đang lắng nghe tác vụ OCR từ Web...")
    print("✓ AI Worker đã sẵn sàng! Đang lắng nghe tác vụ OCR từ Web...", flush=True)

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
                _process_ocr_job(job, pipeline)

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
