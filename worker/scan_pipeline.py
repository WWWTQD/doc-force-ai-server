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
            # Bước 1: Kiểm tra xem có YOLO document_detector rõ ràng không (conf >= 0.35).
            # Nếu không có model hoặc fallback opencv cắt lẹm trang (< 80%), giữ nguyên 100% khung trang gốc
            # để bảo toàn đầu/đuôi trang; sau đó BẮT BUỘC tự động xoay hướng 0°/90°/180°/270° về chiều đứng!
            try:
                corners, method, conf = self.scanner.detect_corners(img)
            except Exception as exc:
                LOG.warning("Document detection lỗi: %s — giữ nguyên khung ảnh.", exc)
                corners, method, conf = None, "none", 0.0

            use_warp = False
            warped_img = img
            warp_transform = np.eye(3, dtype=np.float64)
            corners_list = None

            if corners is not None and method not in ("none", "opencv") and conf >= 0.35:
                cand_img, cand_transform = self.scanner.warp(img, corners)
                ch, cw = cand_img.shape[:2]
                if (ch * cw) >= (h * w * 0.75) and ch >= min(h, w) * 0.80 and cw >= min(h, w) * 0.80:
                    use_warp = True
                    warped_img = cand_img
                    warp_transform = cand_transform
                    corners_list = corners.tolist()

            if not use_warp:
                method = "original"
                conf = 0.0

            rotation = self.scanner.orient(warped_img)
            if rotation:
                rh, rw = warped_img.shape[:2]
                warp_transform = rotation_matrix(rotation, rw, rh) @ warp_transform
                warped_img = rotate_image(warped_img, rotation)

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

        restored, deskew_matrix = self.preprocessor.enhance(geometry.image)
        transform = deskew_matrix @ geometry.transform

        # Đưa qua mô hình DocumentRestorationUNet của mình trước khi vào Model Nhận diện chữ của manoi
        if (
            self.enable_restoration
            and self.restoration_service is not None
            and getattr(self.restoration_service, "is_available", False)
        ):
            try:
                LOG.info("  -> Khôi phục ảnh bằng DocumentRestorationUNet (tile=512, overlap=64)...")
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
            except Exception as exc:
                LOG.warning("  Cảnh báo: Khôi phục ảnh U-Net gặp lỗi (%s), tiếp tục với ảnh tiền xử lý.", exc)

        layout = self.ocr_engine.process_image(restored, page_number)
        layout["scan"] = geometry.to_metadata()
        return PageScanResult(layout=layout, restored_image=restored, geometry=geometry, transform=transform)

    def process_bytes(self, image_bytes: bytes, page_number: int) -> PageScanResult:
        return self.process_image(self.preprocessor.decode_image(image_bytes), page_number)
