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

# Thư mục storage dùng chung với Backend (vd: ...\PBL4-doc_forge-Backend\storage hoặc /data/storage).
# Khi đặt, worker ghi ảnh đã nắn phẳng/làm sạch làm "ảnh phục hồi" và giữ bbox theo ảnh đó.
SHARED_STORAGE_ROOT = os.environ.get("SHARED_STORAGE_ROOT", "").strip()
ENABLE_MAGIC_FILTER = os.environ.get("ENABLE_MAGIC_FILTER", "false").lower() == "true"
ENABLE_DOCUMENT_SCAN = os.environ.get("ENABLE_DOCUMENT_SCAN", "true").lower() != "false"
ENABLE_IMAGE_RESTORATION = os.environ.get("ENABLE_IMAGE_RESTORATION", "true").lower() != "false"
PIPELINE_VERSION = "real-worker-unet-docscan-paddleocr-v2.1"

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
        self.consecutive_fails = 0
        self.failed = False

    def _send_beat(self):
        try:
            _post(
                f"/api/worker/jobs/{self.job_id}/heartbeat",
                {"leaseToken": self.lease_token, "progress": 50, "stage": self.stage},
                lease_token=self.lease_token,
            )
            LOG.debug("Heartbeat OK — job %s", self.job_id)
            self.consecutive_fails = 0
        except Exception as exc:
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
        ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("Không encode được ảnh phục hồi")
        image_bytes = encoded.tobytes()
    except Exception as exc:
        LOG.warning("  Không encode được ảnh phục hồi: %s", exc)
        return None

    # Bước 1: Upload trực tiếp qua HTTP API (/restored) của Backend
    job_id = job.get("jobId")
    lease_token = job.get("leaseToken")
    if job_id and lease_token:
        uploaded_key = _upload_restored(job_id, page_id, lease_token, image_bytes)
        if uploaded_key:
            return uploaded_key

    # Bước 2: Fallback ghi vào SHARED_STORAGE_ROOT / docker cp
    if not job.get("documentId"):
        return None
    key = (f"documents/{job['documentId']}/jobs/{job['jobId']}/{job['leaseToken']}"
           f"/pages/{page_id}/restored.jpg")

    root = SHARED_STORAGE_ROOT
    if not root or not os.path.isdir(root):
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

    LOG.info("══ [OCR JOB %s] Bắt đầu xử lý (%d trang) ══", job_id, len(pages))
    heartbeat = HeartbeatThread(job_id, lease_token, stage="OCR")
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

            # 1. Bắt tờ giấy + nắn phẳng + xoay -> 2. Deskew/Denoise/Magic Color -> 3. U-Net Restoration -> 4. Layout + OCR
            LOG.info("  -> Scan tài liệu (detect, warp, orient, enhance) + U-Net Restoration + OCR...")
            result = pipeline.process_bytes(image_bytes, page_number)

            # Upload ảnh phục hồi lên Backend (/restored) -> giữ toạ độ theo ảnh đó; nếu không có -> ánh xạ bbox về ảnh gốc
            restored_key = _save_restored_image(job, page_id, result.restored_image)
            if result.geometry.rotation in (90, 180, 270) and job.get("documentId"):
                try:
                    ok_orig, enc_orig = cv2.imencode(".jpg", result.geometry.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    if ok_orig:
                        orig_key = f"documents/{job['documentId']}/pages/{page_id}/original.jpg"
                        tmp_orig = os.path.join(tempfile.gettempdir(), f"rotated_orig_{page_id}.jpg")
                        with open(tmp_orig, "wb") as f_orig:
                            f_orig.write(enc_orig.tobytes())
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


def _append_formatted_runs(p, text: str, style_meta: dict):
    """Phân tách chuỗi markdown (**bold**, *italic*, <u>underline</u>) thành các runs trong docx."""
    import re
    block_bold = style_meta.get("bold", False)
    block_italic = style_meta.get("italic", False)
    block_underline = style_meta.get("underline", False)

    tokens = re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*|<u>.*?</u>|__[^_]+__)', text)
    for token in tokens:
        if not token:
            continue
        is_b = block_bold
        is_i = block_italic
        is_u = block_underline
        clean_text = token

        if token.startswith("**") and token.endswith("**") and len(token) >= 4:
            is_b = True
            clean_text = token[2:-2]
        elif token.startswith("*") and token.endswith("*") and len(token) >= 2:
            is_i = True
            clean_text = token[1:-1]
        elif token.startswith("<u>") and token.endswith("</u>") and len(token) >= 7:
            is_u = True
            clean_text = token[3:-4]
        elif token.startswith("__") and token.endswith("__") and len(token) >= 4:
            is_u = True
            clean_text = token[2:-2]

        run = p.add_run(clean_text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
        if is_b:
            run.bold = True
        if is_i:
            run.italic = True
        if is_u:
            run.underline = True


def _generate_docx_from_snapshot(snapshot: dict) -> bytes:
    """Tái tạo văn bản Word .docx từ snapshot JSON các trang với chuẩn định dạng sách/văn bản."""
    if not DOCX_AVAILABLE:
        # Fallback zip / minimal docx nếu không có python-docx
        raise RuntimeError("python-docx chưa được cài đặt. Hãy chạy: pip install python-docx")

    doc = docx.Document()

    # Cấu hình lề trang theo chuẩn nghị định và xuất bản (Top 2cm, Bottom 2cm, Left 2.5cm, Right 2cm)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.79)     # 2.0 cm
        section.bottom_margin = Inches(0.79)  # 2.0 cm
        section.left_margin = Inches(0.98)    # 2.5 cm
        section.right_margin = Inches(0.79)   # 2.0 cm

    # Cấu hình Style Normal mặc định: font Times New Roman, 12pt
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(12)
    font.color.rgb = RGBColor(16, 17, 20)

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

            style_meta = block.get("style", {}) or {}

            if "heading" in b_type or b_type == "title":
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(14)
                p.paragraph_format.space_after = Pt(8)
                p.paragraph_format.line_spacing = 1.2
                run = p.add_run(text.upper())
                run.font.name = 'Times New Roman'
                run.font.size = Pt(14)
                run.bold = True
            elif "table" in b_type:
                # Render table nếu có dữ liệu bảng, nếu không thì render như đoạn văn
                p = doc.add_paragraph()
                run = p.add_run(f"[BẢNG]: {text}")
                run.font.name = 'Times New Roman'
                run.font.size = Pt(11)
                run.bold = True
                p.paragraph_format.space_after = Pt(6)
            elif "list" in b_type:
                doc.add_paragraph(text, style="List Bullet")
            else:
                p = doc.add_paragraph()
                p.paragraph_format.space_after = Pt(4)
                p.paragraph_format.line_spacing = 1.25

                # Kiểm tra căn giữa hoặc căn phải từ metadata hoặc text
                is_centered = style_meta.get("textAlign") == "center"
                is_right = style_meta.get("textAlign") == "right" or "NHÀ XUẤT BẢN" in text.upper()
                is_indented = style_meta.get("isIndented", False)

                if is_centered:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                elif is_right:
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                else:
                    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                    # Thụt dòng đầu đoạn (1 cm = Pt(28)) nếu block được đánh dấu indented hoặc là dòng mở đoạn
                    if is_indented or (block.get("order", 0) > 1 and len(text) > 40 and not is_right):
                        p.paragraph_format.first_line_indent = Pt(28)

                # Parse formatted text (bold, italic, underline)
                _append_formatted_runs(p, text, style_meta)

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
    print("╔══════════════════════════════════════╗", flush=True)
    print("║   Doc Forge — Real AI Worker v1.3    ║", flush=True)
    print("╚══════════════════════════════════════╝", flush=True)
    print(f"Backend URL  : {BACKEND_URL}", flush=True)
    print(f"Worker ID    : {WORKER_INSTANCE_ID}", flush=True)
    print(f"Poll interval: {POLL_INTERVAL:.0f}s\n", flush=True)

    # Khởi tạo DocumentRestorationUNet
    restoration_service = None
    if ENABLE_IMAGE_RESTORATION:
        try:
            from app.services.restoration_service import RestorationService
            restoration_service = RestorationService.get_instance()
            if restoration_service.is_available:
                LOG.info("✓ DocumentRestorationUNet đã sẵn sàng!")
            else:
                LOG.warning("DocumentRestorationUNet chưa có trọng số khả dụng.")
        except Exception as exc:
            LOG.warning("Không thể khởi tạo RestorationService: %s", exc)

    preprocessor = ImagePreprocessor(enable_magic=ENABLE_MAGIC_FILTER)
    ocr_engine = OCREngine(lang="vi")
    pipeline = ScanPipeline(
        ocr_engine=ocr_engine,
        preprocessor=preprocessor,
        restoration_service=restoration_service,
        enable_document_scan=ENABLE_DOCUMENT_SCAN,
        enable_restoration=ENABLE_IMAGE_RESTORATION,
    )

    # Đảm bảo logging ra console không bị Ultralytics hay PaddleOCR đè level
    logging.getLogger().setLevel(logging.INFO)
    LOG.setLevel(logging.INFO)

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
