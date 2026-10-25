"""
Scan Pipeline cho Doc Forge AI Worker.

Ghép toàn bộ chuỗi xử lý một trang ảnh chụp từ camera:
  ảnh gốc
    -> DocumentScanner (YOLO document_detector.pt: bắt tờ giấy, nắn phẳng, xoay đúng chiều)
    -> ImagePreprocessor (deskew, denoise, Magic Color khử bóng / làm trắng nền)
    -> OCREngine (YOLO layout_best_backup.pt + PaddleOCR tiếng Việt)
    -> rawLayoutJson

Dùng chung cho real_worker.py và scripts/evaluate_scanner.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from document_scanner import DocumentScanner, ScanGeometry, map_layout_to_original
from ocr_engine import OCREngine
from preprocessor import ImagePreprocessor

LOG = logging.getLogger("scan_pipeline")


@dataclass
class PageScanResult:
    layout: dict  # toạ độ theo ảnh đã xử lý (restored_image)
    restored_image: np.ndarray  # ảnh đã nắn phẳng + làm sạch, chính là ảnh đưa vào OCR
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
        enable_document_scan: bool = True,
    ):
        self.ocr_engine = ocr_engine or OCREngine(lang="vi")
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.enable_document_scan = enable_document_scan
        scorer = self.ocr_engine.score_orientation if self.ocr_engine.can_score_orientation else None
        self.scanner = scanner or DocumentScanner(orientation_scorer=scorer)

    def process_image(self, img: np.ndarray, page_number: int) -> PageScanResult:
        h, w = img.shape[:2]
        if self.enable_document_scan:
            geometry = self.scanner.scan(img)
        else:
            geometry = ScanGeometry(image=img, transform=np.eye(3), original_size=(w, h))

        restored, deskew_matrix = self.preprocessor.enhance(geometry.image)
        transform = deskew_matrix @ geometry.transform

        layout = self.ocr_engine.process_image(restored, page_number)
        layout["scan"] = geometry.to_metadata()
        return PageScanResult(layout=layout, restored_image=restored, geometry=geometry, transform=transform)

    def process_bytes(self, image_bytes: bytes, page_number: int) -> PageScanResult:
        return self.process_image(self.preprocessor.decode_image(image_bytes), page_number)
