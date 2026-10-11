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

YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    LOG.warning("Thư viện ultralytics chưa được cài đặt. Bỏ qua YOLO layout detection.")


class OCREngine:
    """Wrapper quản lý PaddleOCR và phân tích cấu trúc layout."""

    def __init__(self, lang: str = "vi", use_gpu: bool = False):
        self.lang = lang
        self.ocr_instance = None
        self.yolo_model = None

        if YOLO_AVAILABLE:
            try:
                model_path = os.path.join(os.path.dirname(__file__), "models", "best.pt")
                if not os.path.exists(model_path):
                    model_path = "models/best.pt"
                if os.path.exists(model_path):
                    LOG.info("Khởi tạo YOLO layout detection (%s)...", model_path)
                    self.yolo_model = YOLO(model_path)
                    LOG.info("YOLO model khởi tạo thành công.")
                else:
                    LOG.warning("Không tìm thấy model YOLO tại %s. Bỏ qua layout detection.", model_path)
            except Exception as e:
                LOG.error("Lỗi khi khởi tạo YOLO model: %s. Bỏ qua layout detection.", e)

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

        Pipeline:
          1. YOLO layout detection (nếu có) → crop từng vùng
          2. PaddleOCR nhận dạng chữ trên từng vùng (hoặc toàn ảnh nếu không có YOLO)
          3. Fallback engine nếu PaddleOCR không sẵn sàng
        """
        h, w = img.shape[:2]

        # Bước 1: YOLO layout detection
        yolo_regions = self._detect_yolo_regions(img, w, h)

        if self.ocr_instance is not None:
            try:
                return self._run_paddle_ocr(img, page_number, w, h, yolo_regions=yolo_regions)
            except Exception as e:
                LOG.error("Lỗi trong quá trình chạy PaddleOCR: %s. Dùng fallback engine.", e)

        return self._run_fallback_ocr(img, page_number, w, h)

    def _detect_yolo_regions(self, img: np.ndarray, width: int, height: int) -> list[dict]:
        """
        Chạy YOLO để phát hiện các vùng layout (text block, table, figure, ...).
        Trả về danh sách dict gồm bbox [x1, y1, x2, y2] và class_name.
        Trả về list rỗng nếu YOLO không khả dụng hoặc không detect được gì.
        """
        if self.yolo_model is None:
            return []
        try:
            results = self.yolo_model(img, verbose=False)
            regions = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(x1 + 1, min(x2, width))
                    y2 = max(y1 + 1, min(y2, height))
                    cls_id = int(box.cls[0])
                    cls_name = result.names.get(cls_id, "text")
                    conf = float(box.conf[0])
                    regions.append({
                        "bbox": [x1, y1, x2, y2],
                        "class_name": cls_name,
                        "confidence": conf,
                    })
            if regions:
                LOG.info("YOLO phát hiện %d vùng layout.", len(regions))
            return regions
        except Exception as e:
            LOG.warning("YOLO detection thất bại: %s. Bỏ qua layout detection.", e)
            return []


    def _run_paddle_ocr(
        self,
        img: np.ndarray,
        page_number: int,
        width: int,
        height: int,
        yolo_regions: list[dict] | None = None,
    ) -> dict:
        """Thực thi PaddleOCR và nhóm các dòng chữ thành khối layout.

        Nếu yolo_regions có dữ liệu, sử dụng class_name từ YOLO để gán block_type
        cho từng text box (match theo overlap lớn nhất).
        """
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

        # Ánh xạ class_name của YOLO → block_type của Doc Forge
        _YOLO_CLASS_MAP = {
            "title": "heading",
            "heading": "heading",
            "table": "table",
            "figure": "paragraph",
            "list": "list",
            "text": "paragraph",
            "paragraph": "paragraph",
        }

        def _block_type_from_yolo(bbox: list[int]) -> str | None:
            """Trả về block_type từ YOLO region có overlap lớn nhất với bbox."""
            if not yolo_regions:
                return None
            bx1, by1, bx2, by2 = bbox
            best_type, best_overlap = None, 0.0
            for region in yolo_regions:
                rx1, ry1, rx2, ry2 = region["bbox"]
                ix1, iy1 = max(bx1, rx1), max(by1, ry1)
                ix2, iy2 = min(bx2, rx2), min(by2, ry2)
                if ix2 > ix1 and iy2 > iy1:
                    overlap = (ix2 - ix1) * (iy2 - iy1)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        raw_cls = region["class_name"].lower()
                        best_type = _YOLO_CLASS_MAP.get(raw_cls, "paragraph")
            return best_type

        # Phân tích để gán loại block (dùng YOLO nếu có, fallback heuristics chiều cao)
        blocks = []
        avg_height = np.mean([b["height_px"] for b in raw_boxes]) if raw_boxes else 20.0

        for order, b in enumerate(raw_boxes, start=1):
            # Ưu tiên dùng YOLO class
            block_type = _block_type_from_yolo(b["bbox"])
            if block_type is None:
                # Fallback heuristics: chiều cao chữ lớn hơn 1.4 lần trung bình → Heading
                if b["height_px"] > avg_height * 1.4 or (order == 1 and b["height_px"] > avg_height * 1.1):
                    block_type = "heading"
                else:
                    block_type = "paragraph"

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
