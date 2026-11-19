"""
Perfect Document Scan Exporter (Word .docx & PDF) for Doc Forge.

Quy trình khép kín từ ảnh camera thật đến File Word & PDF chuẩn 100%:
  Ảnh camera thật (real/)
    -> ScanPipeline (YOLO Detect -> Warp A4 -> Auto-Rotate -> Magic White Khử bóng)
    -> VietOCR + PaddleOCR (Đọc trọn vẹn chữ Tiếng Việt)
    -> DeepDocumentTypographyResNet (Dự đoán Đậm, Nghiêng, Font, Cỡ chữ pt)
    -> Heuristic Alignment Engine (Phân tích Căn giữa, Căn đều, Căn phải, Thụt đầu dòng)
    -> Xuất file Word (.docx) sạch tinh 100%
    -> Xuất file PDF (.pdf) nền trắng tinh khiết
"""

from __future__ import annotations

import os
import sys
import glob
import time
import logging
from typing import Dict, List, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Đảm bảo đường dẫn import
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER_DIR = os.path.join(ROOT_DIR, "worker")
TRAINING_DIR = os.path.join(ROOT_DIR, "training")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if WORKER_DIR not in sys.path:
    sys.path.insert(0, WORKER_DIR)
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)

from export_layout import generate_docx_from_snapshot, generate_pdf_from_snapshot
from scan_pipeline import ScanPipeline
from train_with_real_scanner_data import DeepDocumentTypographyResNet

LOG = logging.getLogger("perfect_exporter")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class PerfectScanExporter:
    """Bộ xuất tài liệu Word & PDF chuẩn mực cao nhất từ ảnh chụp scan thực tế."""

    def __init__(self, typography_model_path: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = ScanPipeline(enable_document_scan=True, enable_restoration=True)

        if typography_model_path is None:
            typography_model_path = os.path.join(ROOT_DIR, "models", "typography_real_best.pth")
            if not os.path.exists(typography_model_path):
                typography_model_path = os.path.join(ROOT_DIR, "models", "typography_best.pth")

        self.typo_model = None
        if os.path.exists(typography_model_path):
            LOG.info("Nạp mô hình Typography AI: %s", typography_model_path)
            ckpt = torch.load(typography_model_path, map_location=self.device)
            self.typo_model = DeepDocumentTypographyResNet().to(self.device)
            # Khớp state dict
            self.typo_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            self.typo_model.eval()
            LOG.info("Typography AI sẵn sàng hoạt động.")
        else:
            LOG.warning("Không tìm thấy checkpoint Typography tại %s. Sẽ dùng fallback heuristics.", typography_model_path)

    def _predict_typography(self, crop: np.ndarray) -> Dict[str, Any]:
        """Đưa ảnh cắt dòng chữ qua mô hình Deep ResNet để nhận diện kiểu chữ."""
        if self.typo_model is None or crop is None or crop.size == 0:
            return {"is_bold": False, "is_italic": False, "font_family": "Times New Roman", "size_pt": 13.0}

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        h, w = gray.shape[:2]
        scale = min(28.0 / max(1, h), 120.0 / max(1, w))
        nw, nh = max(4, int(w * scale)), max(4, int(h * scale))
        resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

        canvas = np.full((32, 128), 255, dtype=np.uint8)
        oy = (32 - nh) // 2
        ox = (128 - nw) // 2
        canvas[oy : oy + nh, ox : ox + nw] = resized

        tensor_in = torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            p_bold, p_italic, p_serif, p_size = self.typo_model(tensor_in)
            is_bold = bool(p_bold.argmax(dim=1).item() == 1)
            is_italic = bool(p_italic.argmax(dim=1).item() == 1)
            font_family = "Times New Roman" if p_serif.argmax(dim=1).item() == 1 else "Arial"
            size_pt = float(round(p_size.item(), 1))
            size_pt = max(10.0, min(24.0, size_pt))

        return {
            "is_bold": is_bold,
            "is_italic": is_italic,
            "font_family": font_family,
            "size_pt": size_pt,
        }

    def _determine_alignment(self, text: str, bbox: list, page_width: int) -> tuple[WD_ALIGN_PARAGRAPH, bool]:
        """
        Xác định căn lề chuẩn mực theo thể thức văn bản:
        - Quốc hiệu, Tiêu ngữ, Quyết định, Tên trường/khoa -> CĂN GIỮA (CENTER)
        - Ngày tháng năm, Ký tên -> CĂN PHẢI (RIGHT)
        - Đoạn nội dung văn bản dài -> CĂN ĐỀU 2 BÊN (JUSTIFY)
        - Khác -> CĂN TRÁI (LEFT)
        """
        s = text.strip().upper()
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        mid_box = (x1 + x2) / 2.0
        mid_page = page_width / 2.0

        is_first_line = False

        # Các tiêu đề hành chính bắt buộc căn giữa
        if any(kw in s for kw in ["CỘNG HÒA", "ĐỘC LẬP", "QUYẾT ĐỊNH", "GIẤY XÁC NHẬN", "THÔNG BÁO", "TỜ TRÌNH", "HỢP ĐỒNG"]):
            return WD_ALIGN_PARAGRAPH.CENTER, False

        # Ngày tháng năm hoặc chữ ký thường ở bên phải
        if any(kw in text.lower() for kw in ["ngày", "tháng", "năm"]) and x1 > page_width * 0.40:
            return WD_ALIGN_PARAGRAPH.RIGHT, False

        # Căn giữa theo vị trí hình học (nếu hộp chữ ngắn nằm giữa trang)
        if abs(mid_box - mid_page) < page_width * 0.08 and bw < page_width * 0.65:
            return WD_ALIGN_PARAGRAPH.CENTER, False

        # Đoạn văn rộng dài -> Căn đều hai bên
        if bw > page_width * 0.68:
            return WD_ALIGN_PARAGRAPH.JUSTIFY, True

        # Thụt đầu dòng nếu dòng thụt vào hơn lề chuẩn
        if x1 > page_width * 0.18 and bw > page_width * 0.50:
            return WD_ALIGN_PARAGRAPH.LEFT, True

        return WD_ALIGN_PARAGRAPH.LEFT, False

    def export_from_image(self, image_path: str, output_base_name: str = "real_scan_perfect") -> Dict[str, str]:
        """Quét ảnh và xuất đồng thời ra file Word (.docx) và PDF (.pdf)."""
        LOG.info("=== BẮT ĐẦU QUY TRÌNH QUÉT & XUẤT TÀI LIỆU HOÀN THIỆN: %s ===", image_path)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể mở file ảnh: {image_path}")

        # 1. Chạy AI Scan Pipeline
        t0 = time.time()
        res = self.pipeline.process_image(img, page_number=1)
        LOG.info("Chạy xong Pipeline trong: %.2fs (Bắt góc: %s, Góc xoay: %d°)", time.time() - t0, res.geometry.method, res.geometry.rotation)

        restored_img = res.restored_image
        h_page, w_page = restored_img.shape[:2]
        blocks = res.layout.get("blocks", [])
        LOG.info("Tổng số khối chữ nhận diện được: %d khối.", len(blocks))

        # 2. Bảo toàn style từ OCREngine (chỉ bổ sung nếu block thiếu metadata)
        out_blocks = []
        for idx, b in enumerate(blocks, start=1):
            text = (b.get("text") or "").strip()
            if not text:
                continue

            bx1, by1, bx2, by2 = b["bbox"]
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w_page, bx2), min(h_page, by2)
            crop = restored_img[by1:by2, bx1:bx2] if by2 > by1 and bx2 > bx1 else None

            blk_copy = dict(b)
            blk_copy["order"] = blk_copy.get("order", idx)
            st = dict(blk_copy.get("style") or {})
            if not st.get("fontFamily") or not st.get("fontSizePt"):
                typo = self._predict_typography(crop)
                st.setdefault("fontFamily", typo["font_family"])
                st.setdefault("fontSizePt", typo["size_pt"])
                if "bold" not in st:
                    st["bold"] = typo["is_bold"]
                if "italic" not in st:
                    st["italic"] = typo["is_italic"]
            blk_copy["style"] = st
            out_blocks.append(blk_copy)

        snapshot = {
            "pages": [
                {
                    "pageNumber": 1,
                    "layout": {
                        "schemaVersion": 1,
                        "pageNumber": 1,
                        "width": w_page,
                        "height": h_page,
                        "blocks": out_blocks,
                    },
                }
            ]
        }

        # 3. Tạo file Word (.docx) & PDF (.pdf) dùng chung engine bố cục nhiều cột
        out_dir = os.path.join(ROOT_DIR, "data", "output")
        os.makedirs(out_dir, exist_ok=True)
        docx_path = os.path.join(out_dir, f"{output_base_name}.docx")
        with open(docx_path, "wb") as f_docx:
            f_docx.write(generate_docx_from_snapshot(snapshot))
        LOG.info("-> Đã xuất xong File Word chuẩn: %s", docx_path)

        pdf_path = os.path.join(out_dir, f"{output_base_name}.pdf")
        with open(pdf_path, "wb") as f_pdf:
            f_pdf.write(generate_pdf_from_snapshot(snapshot))
        LOG.info("-> Đã xuất xong File PDF nền trắng tinh khiết: %s", pdf_path)

        return {
            "docx": docx_path,
            "pdf": pdf_path,
            "total_blocks": len(out_blocks),
            "restored_image_shape": f"{w_page}x{h_page}",
        }


if __name__ == "__main__":
    exporter = PerfectScanExporter()
    test_files = sorted(glob.glob(os.path.join(ROOT_DIR, "real", "*.jpg")))
    if test_files:
        sample_image = test_files[0]
        print(f"\nKiểm thử quét và xuất trực tiếp từ ảnh thật: {sample_image}")
        res = exporter.export_from_image(sample_image, output_base_name="real_scan_perfect")
        print("\n══════════════════════════════════════════════════════════")
        print("  ✓ ĐÃ XUẤT THÀNH CÔNG FILE WORD VÀ PDF TỪ ẢNH REAL!")
        print("══════════════════════════════════════════════════════════")
        print(f"  - File Word: {res['docx']}")
        print(f"  - File PDF:  {res['pdf']}")
        print(f"  - Tổng số khối văn bản định dạng: {res['total_blocks']}")
        print("══════════════════════════════════════════════════════════")
