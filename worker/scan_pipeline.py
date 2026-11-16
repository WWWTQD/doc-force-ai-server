"""
Scan Pipeline cho Doc Forge AI Worker.

Ghép toàn bộ chuỗi xử lý một trang ảnh chụp từ camera:
  ảnh gốc
    -> DocumentScanner (YOLO document_detector.pt: bắt tờ giấy, nắn phẳng, xoay đúng chiều)
    -> ImagePreprocessor (deskew, denoise, Magic Color khử bóng / làm trắng nền)
    -> DocumentRestorationUNet (RestorationService: khôi phục chất lượng ảnh, làm trắng nền & rõ nét chữ)
    -> OCREngine (YOLO layout_best_backup.pt + PaddleOCR + VietOCR tiếng Việt)
    -> rawLayoutJson

Dùng chung cho real_worker.py và scripts/evaluate_scanner.py.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

# Tắt MKLDNN & PIR API cho PaddlePaddle trên CPU Windows để không bị lỗi xung đột OneDnnContext
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

import numpy as np

# Đảm bảo có thể import app.services.restoration_service từ thư mục gốc dự án
WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORKER_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if WORKER_DIR not in sys.path:
    sys.path.insert(0, WORKER_DIR)

from document_scanner import DocumentScanner, ScanGeometry, map_layout_to_original, rotate_image, rotation_matrix
from ocr_engine import OCREngine
from preprocessor import ImagePreprocessor

LOG = logging.getLogger("scan_pipeline")


@dataclass
class PageScanResult:
    layout: dict  # toạ độ theo ảnh đã xử lý (restored_image)
    restored_image: np.ndarray  # ảnh đã nắn phẳng + khôi phục bằng U-Net, chính là ảnh đưa vào OCR
    geometry: ScanGeometry
    transform: np.ndarray  # 3x3: ảnh gốc -> restored_image

    def layout_in_original_coords(self) -> dict:
        """Bản sao layout với bbox ánh xạ về ảnh gốc (khi frontend chỉ có ảnh gốc)."""
        import copy
        return map_layout_to_original(copy.deepcopy(self.layout), self.transform, self.geometry.original_size)


class ScanPipeline:
    def __init__(
        self,
        ocr_engine: OCREngine | None = None,
        preprocessor: ImagePreprocessor | None = None,
        scanner: DocumentScanner | None = None,
        restoration_service: Any | None = None,
        enable_document_scan: bool = True,
        enable_restoration: bool = True,
    ):
        self.ocr_engine = ocr_engine or OCREngine(lang="vi")
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.enable_document_scan = enable_document_scan
        self.enable_restoration = enable_restoration
        self.restoration_service = restoration_service

        if self.enable_restoration and self.restoration_service is None:
            try:
                from app.services.restoration_service import RestorationService
                self.restoration_service = RestorationService.get_instance()
            except Exception as exc:
                LOG.warning("Không nạp được RestorationService trong ScanPipeline: %s", exc)

        scorer = self.ocr_engine.score_orientation if self.ocr_engine.can_score_orientation else None
        self.scanner = scanner or DocumentScanner(orientation_scorer=scorer)

    def process_image(self, img: np.ndarray, page_number: int) -> PageScanResult:
        h, w = img.shape[:2]
        if self.enable_document_scan:
            # Giữ nguyên 100% khung ảnh gốc (không cắt lẹm lề/đầu/cuối trang và bỏ qua YOLO detect_corners chậm trên CPU)
            # Chỉ thực hiện xoay tự động 0°/90°/180°/270° về đúng chiều đọc tiếng Việt (< 0.4s)
            warped_img = img
            warp_transform = np.eye(3, dtype=np.float64)
            method = "original"
            conf = 1.0
            corners_list = None

            rotation = self.scanner.orient(warped_img)
            if rotation:
                rh, rw = warped_img.shape[:2]
                warp_transform = rotation_matrix(rotation, rw, rh) @ warp_transform
                warped_img = rotate_image(warped_img, rotation)

            try:
                warped_img = self.preprocessor.clean_handwriting_and_book_gutter(warped_img)
            except Exception as exc:
                LOG.warning("Lỗi khi lọc chữ viết tay / gáy sách: %s", exc)

            LOG.info(
                "  [ScanPipeline] method=%s conf=%.2f rotation=%d° (%dx%d -> %dx%d)",
                method, conf, rotation, w, h, warped_img.shape[1], warped_img.shape[0],
            )
            geometry = ScanGeometry(
                image=warped_img,
                transform=warp_transform,
                original_size=(w, h),
                method=method,
                detection_confidence=conf,
                corners=corners_list,
                rotation=rotation,
            )
        else:
            geometry = ScanGeometry(image=img, transform=np.eye(3), original_size=(w, h), method="disabled")

        use_unet = (
            self.enable_restoration
            and self.restoration_service is not None
            and getattr(self.restoration_service, "is_available", False)
        )

        if use_unet:
            # Khi có DocumentRestorationUNet v5: chỉ xoay thẳng hình học (deskew),
            # giữ nguyên phân bố pixel tự nhiên đưa vào U-Net v5 (tránh bị bilateralFilter/CLAHE làm nhòe nét hoặc tăng nhiễu nền)
            if getattr(self.preprocessor, "enable_deskew", True):
                try:
                    restored, deskew_matrix = self.preprocessor.deskew_with_matrix(geometry.image)
                except Exception as exc:
                    LOG.warning("Lỗi khi deskew ảnh: %s", exc)
                    restored, deskew_matrix = geometry.image, np.eye(3, dtype=np.float64)
            else:
                restored, deskew_matrix = geometry.image, np.eye(3, dtype=np.float64)
            transform = deskew_matrix @ geometry.transform
            pre_unet_table_img = restored

            try:
                LOG.info("  -> Khôi phục ảnh bằng DocumentRestorationUNet v5 (tile=512, overlap=64)...")
                restored_h, restored_w = restored.shape[:2]
                unet_restored = self.restoration_service.restore_image_bgr(
                    restored,
                    tile_size=512,
                    overlap=64,
                    preserve_color=True,
                )
                if unet_restored.shape[:2] != (restored_h, restored_w):
                    import cv2
                    unet_restored = cv2.resize(
                        unet_restored,
                        (restored_w, restored_h),
                        interpolation=cv2.INTER_CUBIC,
                    )
                restored = unet_restored
                ocr_input_img = restored
            except Exception as exc:
                LOG.warning("  Cảnh báo: Khôi phục ảnh U-Net v5 gặp lỗi (%s), giữ nguyên ảnh camera tự nhiên cho OCR.", exc)
                ocr_input_img = pre_unet_table_img
                restored, deskew_matrix = self.preprocessor.enhance(geometry.image)
                transform = deskew_matrix @ geometry.transform
        else:
            # Tắt bilateralFilter + cắt ngưỡng cứng cho OCR: giữ trọn nét dấu thanh tiếng Việt sắc nét từ ảnh camera gốc
            import cv2
            restored, deskew_matrix = self.preprocessor.enhance(geometry.image)
            transform = deskew_matrix @ geometry.transform
            pre_unet_table_img = cv2.warpAffine(
                geometry.image,
                deskew_matrix[:2],
                (restored.shape[1], restored.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            ocr_input_img = pre_unet_table_img

        layout = self.ocr_engine.process_image(ocr_input_img, page_number, table_ref_img=pre_unet_table_img)
        layout["scan"] = geometry.to_metadata()
        return PageScanResult(layout=layout, restored_image=restored, geometry=geometry, transform=transform)

    def process_bytes(self, image_bytes: bytes, page_number: int) -> PageScanResult:
        return self.process_image(self.preprocessor.decode_image(image_bytes), page_number)
