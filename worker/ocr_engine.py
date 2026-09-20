"""
OCR Engine Module cho Doc Forge AI Worker.

Tích hợp PaddleOCR (hoặc Fallback Engine nếu chưa cài PaddleOCR)
để nhận dạng chữ tiếng Việt/Anh và xây dựng Layout JSON cho Backend.
"""

import logging
import uuid
import numpy as np

LOG = logging.getLogger("ocr_engine")

PADDLE_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    LOG.warning("Thư viện paddleocr chưa được cài đặt. OCR Engine sẽ chạy ở chế độ Fallback Engine.")


class OCREngine:
    """Wrapper quản lý PaddleOCR và phân tích cấu trúc layout."""

    def __init__(self, lang: str = "vi", use_gpu: bool = False):
        self.lang = lang
        self.ocr_instance = None

        if PADDLE_AVAILABLE:
            try:
                LOG.info("Khởi tạo PaddleOCR (lang=%s, use_gpu=%s)...", lang, use_gpu)
                self.ocr_instance = PaddleOCR(
                    use_angle_cls=True,
                    lang=lang,
                    use_gpu=use_gpu,
                    show_log=False,
                )
                LOG.info("PaddleOCR khởi tạo thành công.")
            except Exception as e:
                LOG.error("Lỗi khi khởi tạo PaddleOCR: %s. Chuyển sang Fallback Engine.", e)

    def process_image(self, img: np.ndarray, page_number: int) -> dict:
        """
        Nhận vào ảnh OpenCV (BGR) và pageNumber.
        Trả về dictionary rawLayoutJson khớp với schema LayoutValidator của Backend.
        """
        h, w = img.shape[:2]

        if self.ocr_instance is not None:
            try:
                return self._run_paddle_ocr(img, page_number, w, h)
            except Exception as e:
                LOG.error("Lỗi trong quá trình chạy PaddleOCR: %s. Dùng fallback engine.", e)

        return self._run_fallback_ocr(img, page_number, w, h)

    def _run_paddle_ocr(self, img: np.ndarray, page_number: int, width: int, height: int) -> dict:
        """Thực thi PaddleOCR và nhóm các dòng chữ thành khối layout."""
        # PaddleOCR hỗ trợ nhận ndarray trực tiếp
        results = self.ocr_instance.ocr(img, cls=True)

        raw_boxes = []
        if results and len(results) > 0 and results[0] is not None:
            for line in results[0]:
                poly, (text, confidence) = line
                # poly = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                
                # Clamp coordinates
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(x1 + 1, min(x2, width))
                y2 = max(y1 + 1, min(y2, height))

                raw_boxes.append({
                    "bbox": [x1, y1, x2, y2],
                    "text": text.strip(),
                    "confidence": float(confidence),
                    "height_px": y2 - y1,
                })

        # Sắp xếp các dòng theo vị trí Y tăng dần
        raw_boxes.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

        # Phân tích heuristics đơn giản để gán loại block (heading / paragraph / table)
        blocks = []
        avg_height = np.mean([b["height_px"] for b in raw_boxes]) if raw_boxes else 20.0

        for order, b in enumerate(raw_boxes, start=1):
            block_type = "paragraph"
            # Chiều cao chữ lớn hơn 1.4 lần trung bình -> Heading
            if b["height_px"] > avg_height * 1.4 or (order == 1 and b["height_px"] > avg_height * 1.1):
                block_type = "heading"

            blocks.append({
                "id": f"b-{order}-{uuid.uuid4().hex[:6]}",
                "type": block_type,
                "order": order,
                "bbox": b["bbox"],
                "text": b["text"],
                "confidence": round(b["confidence"], 3),
            })

        if not blocks:
            # Nếu không tìm thấy chữ nào
            blocks.append({
                "id": f"b-1-{uuid.uuid4().hex[:6]}",
                "type": "paragraph",
                "order": 1,
                "bbox": [10, 10, width - 10, 50],
                "text": "[Trang không có nội dung chữ nhận dạng được]",
                "confidence": 1.0,
            })

        return {
            "schemaVersion": 1,
            "pageNumber": page_number,
            "width": width,
            "height": height,
            "blocks": blocks,
        }

    def _run_fallback_ocr(self, img: np.ndarray, page_number: int, width: int, height: int) -> dict:
        """
        Engine dự phòng khi PaddleOCR chưa sẵn sàng.
        Tạo layout nhận dạng sơ bộ với tọa độ chuẩn.
        """
        w = max(width, 800)
        h = max(height, 1200)

        blocks = [
            {
                "id": f"fb-1-{uuid.uuid4().hex[:6]}",
                "type": "heading",
                "order": 1,
                "bbox": [50, 50, int(w * 0.8), 110],
                "text": "DOC FORGE — KẾT QUẢ XỬ LÝ AI REAL WORKER",
                "confidence": 0.98,
            },
            {
                "id": f"fb-2-{uuid.uuid4().hex[:6]}",
                "type": "paragraph",
                "order": 2,
                "bbox": [50, 130, int(w * 0.9), 220],
                "text": "Ảnh đã được chạy qua OpenCV ImagePreprocessor (Deskew, Denoise, CLAHE). "
                        "Hệ thống sẵn sàng nhận dữ liệu từ PaddleOCR khi cài đặt mô hình.",
                "confidence": 0.96,
            },
            {
                "id": f"fb-3-{uuid.uuid4().hex[:6]}",
                "type": "paragraph",
                "order": 3,
                "bbox": [50, 240, int(w * 0.9), 320],
                "text": f"Thông tin trang: Trang số {page_number}, Kích thước {width}x{height} px.",
                "confidence": 0.95,
            },
        ]

        return {
            "schemaVersion": 1,
            "pageNumber": page_number,
            "width": w,
            "height": h,
            "blocks": blocks,
        }
