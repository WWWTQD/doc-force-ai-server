"""
OCR Engine Module cho Doc Forge AI Worker.

Tích hợp PaddleOCR (hoặc Fallback Engine nếu chưa cài PaddleOCR)
để nhận dạng chữ tiếng Việt/Anh và xây dựng Layout JSON cho Backend.
"""

import os
import logging
import uuid

try:
    from vietnamese_normalizer import normalize_vietnamese_text
except ImportError:
    def normalize_vietnamese_text(t: str) -> str:
        return t

# Đặt KMP_DUPLICATE_LIB_OK và import torch trước paddle để tránh xung đột OpenMP DLL trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import torch
    torch.set_num_threads(max(4, os.cpu_count() or 4))
except Exception:
    torch = None

import cv2
import numpy as np

LOG = logging.getLogger("ocr_engine")

YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    LOG.warning("Không thể nạp YOLO: %s. Bỏ qua YOLO layout detection.", e)

PADDLE_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception as e:
    LOG.warning("Không thể nạp PaddleOCR: %s. OCR Engine sẽ chạy ở chế độ Fallback Engine.", e)

# VietOCR: mô hình nhận dạng chữ tiếng Việt (đủ dấu, đ/ư/ơ) — dùng thay bộ nhận dạng Latin của PaddleOCR.
# PaddleOCR khi đó chỉ làm nhiệm vụ phát hiện vị trí dòng chữ.
VIETOCR_AVAILABLE = False
try:
    from PIL import Image
    from vietocr.tool.predictor import Predictor as VietOCRPredictor
    from vietocr.tool.config import Cfg as VietOCRCfg
    VIETOCR_AVAILABLE = True
except Exception as e:
    LOG.warning("Không thể nạp VietOCR: %s. Dùng bộ nhận dạng Latin của PaddleOCR (kém chính xác với tiếng Việt).", e)

# vgg_seq2seq: nhanh (~0.6s/dòng CPU); vgg_transformer: chậm hơn ~5 lần
VIETOCR_MODEL = os.environ.get("VIETOCR_MODEL", "vgg_seq2seq")
USE_VIETOCR = os.environ.get("OCR_RECOGNIZER", "vietocr").lower() == "vietocr"
# Dòng có tỉ lệ rộng/cao lớn hơn ngưỡng này sẽ được tách đôi tại khoảng trắng giữa hai từ,
# vì mô hình seq2seq hay bỏ sót chữ trên chuỗi quá dài.
MAX_LINE_ASPECT = float(os.environ.get("VIETOCR_MAX_ASPECT", "16"))


def _order_poly(pts) -> np.ndarray:
    """Sắp xếp 4 đỉnh tứ giác theo thứ tự chuẩn: top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    idx = np.argsort(pts[:, 0])
    left = pts[idx[:2]]
    right = pts[idx[2:]]
    tl, bl = left[np.argsort(left[:, 1])]
    tr, br = right[np.argsort(right[:, 1])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _rotate_crop(img: np.ndarray, pts, pad_x: int = 6, pad_y: int = 4) -> np.ndarray:
    """Cắt dòng chữ theo đúng tứ giác (kể cả dòng nghiêng) và kéo thẳng thành hình chữ nhật."""
    pts = _order_poly(pts)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    w, h = max(w, 1), max(h, 1)
    dst = np.array([
        [pad_x, pad_y],
        [w + pad_x, pad_y],
        [w + pad_x, h + pad_y],
        [pad_x, h + pad_y],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(
        img, M, (w + 2 * pad_x, h + 2 * pad_y),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _resolve_layout_model_path() -> str:
    """
    Model phân tích bố cục (heading/paragraph/table/...).
    Ưu tiên biến môi trường LAYOUT_MODEL_PATH, sau đó layout_best_backup.pt,
    cuối cùng best.pt (tên cũ) để tương thích ngược.
    """
    env_path = os.environ.get("LAYOUT_MODEL_PATH")
    if env_path:
        return env_path
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    for name in ("layout_best_backup.pt", "best.pt"):
        candidate = os.path.join(models_dir, name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(models_dir, "layout_best_backup.pt")


class OCREngine:
    """Wrapper quản lý PaddleOCR và phân tích cấu trúc layout."""

    def __init__(self, lang: str = "vi", use_gpu: bool = False):
        self.lang = lang
        self.ocr_instance = None
        self.yolo_model = None

        if YOLO_AVAILABLE:
            try:
                model_path = _resolve_layout_model_path()
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
                    use_angle_cls=False,
                    lang=lang,
                    use_gpu=use_gpu,
                    show_log=False,
                    det_limit_side_len=2200,
                    det_db_thresh=0.20,
                    det_db_box_thresh=0.18,
                    det_db_unclip_ratio=1.35,
                )
                LOG.info("PaddleOCR khởi tạo thành công.")
            except Exception as e:
                LOG.error("Lỗi khi khởi tạo PaddleOCR: %s. Chuyển sang Fallback Engine.", e)

        self.viet_recognizer = None
        if USE_VIETOCR and VIETOCR_AVAILABLE and self.ocr_instance is not None:
            try:
                LOG.info("Khởi tạo VietOCR (%s)...", VIETOCR_MODEL)
                cfg = VietOCRCfg.load_config_from_name(VIETOCR_MODEL)
                cfg["device"] = "cuda" if use_gpu else "cpu"
                cfg["predictor"]["beamsearch"] = False
                self.viet_recognizer = VietOCRPredictor(cfg)
                LOG.info("VietOCR khởi tạo thành công — dùng làm bộ nhận dạng chữ chính.")
            except Exception as e:
                LOG.error("Lỗi khởi tạo VietOCR: %s. Dùng bộ nhận dạng của PaddleOCR.", e)

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

    @property
    def can_score_orientation(self) -> bool:
        return self.ocr_instance is not None

    def score_orientation(self, img: np.ndarray) -> float:
        """
        Chấm điểm "đọc được" của ảnh ở hướng hiện tại (0..1), dùng để chọn 0° hay 180°.
        Chạy PaddleOCR với cls=False trên bản thu nhỏ: chữ lộn ngược cho confidence thấp.
        """
        if self.ocr_instance is None:
            return 0.0
        h, w = img.shape[:2]
        # Thu nhỏ 640px để chấm điểm hướng đọc cực nhanh (1s thay vì 20s)
        scale = min(1.0, 640.0 / max(h, w))
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
        results = self.ocr_instance.ocr(small, cls=False)
        confidences = []
        if results and results[0]:
            for _, (text, confidence) in results[0]:
                if len(text.strip()) >= 3:
                    confidences.append(float(confidence))
        if not confidences:
            return 0.0
        coverage = min(1.0, len(confidences) / 5.0)
        return float(np.mean(confidences)) * coverage

    def _detect_yolo_regions(self, img: np.ndarray, width: int, height: int) -> list[dict]:
        """
        Chạy YOLO để phát hiện các vùng layout (text block, table, figure, ...).
        Trả về danh sách dict gồm bbox [x1, y1, x2, y2] và class_name.
        Trả về list rỗng nếu YOLO không khả dụng hoặc không detect được gì.
        """
        if self.yolo_model is None:
            return []
        try:
            # imgsz=640 tăng tốc độ phát hiện bố cục trên CPU
            results = self.yolo_model(img, imgsz=640, verbose=False)
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


    def _detect_and_recognize_vietocr(self, img: np.ndarray, width: int) -> list:
        """
        PaddleOCR phát hiện tứ giác từng dòng -> lọc nhiễu lề / gáy sách -> cắt thẳng theo tứ giác
        -> tách các box bị gộp dọc 2 dòng -> khử trùng lặp Y -> VietOCR đọc.
        Trả về list [poly, (text, confidence)] cùng định dạng PaddleOCR.
        """
        dt_boxes, _ = self.ocr_instance.text_detector(img)
        if dt_boxes is None or len(dt_boxes) == 0:
            return []

        # Ước lượng đường lề phải của cột chữ in theo phương trình tuyến tính right(y) = slope * y + intercept
        clean_wide = []
        for b in dt_boxes:
            xs = [p[0] for p in b]
            ys = [p[1] for p in b]
            bw = max(xs) - min(xs)
            if bw > width * 0.45 and max(xs) < width * 0.96:
                clean_wide.append((float(np.mean(ys)), float(max(xs))))

        if len(clean_wide) >= 4:
            ys_w = np.array([p[0] for p in clean_wide])
            xs_w = np.array([p[1] for p in clean_wide])
            mid = len(clean_wide) // 2
            order_idx = np.argsort(ys_w)
            top_half = order_idx[:mid]
            bot_half = order_idx[mid:]
            y1_m, x1_m = float(np.median(ys_w[top_half])), float(np.percentile(xs_w[top_half], 70))
            y2_m, x2_m = float(np.median(ys_w[bot_half])), float(np.percentile(xs_w[bot_half], 70))
            slope = (x2_m - x1_m) / max(1.0, (y2_m - y1_m))
            intercept = x1_m - slope * y1_m
        else:
            slope, intercept = 0.0, width * 0.92

        def _right_margin_at(y_val: float) -> float:
            return slope * y_val + intercept

        raw_items = []
        heights = []
        for b in dt_boxes:
            pts = _order_poly(b)
            xs = pts[:, 0]
            ys = pts[:, 1]
            x1, x2 = float(min(xs)), float(max(xs))
            ym = float(np.mean(ys))
            bw = x2 - x1
            rm = _right_margin_at(ym)

            # Bỏ mảnh chữ rác ở gáy sách mép trái
            if x2 < 120 and bw < 100:
                continue
            # Bỏ các mảnh chữ viết tay rời ở lề phải
            if x1 > rm - 65 and bw < 220:
                continue
            # Cắt bớt đuôi hộp nếu dòng in bị nối dính vào chữ viết tay ở lề phải
            if x1 < 450 and x2 > rm + 18:
                target_r = rm + 6
                for right_i, left_i in [(1, 0), (2, 3)]:
                    dx = pts[right_i, 0] - pts[left_i, 0]
                    if dx > 10:
                        ratio = (target_r - pts[left_i, 0]) / dx
                        pts[right_i, 0] = target_r
                        pts[right_i, 1] = pts[left_i, 1] + ratio * (pts[right_i, 1] - pts[left_i, 1])

            h_box = float(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
            raw_items.append((ym, pts, h_box))
            heights.append(h_box)

        if not raw_items:
            return []

        median_h = float(np.median(heights)) if heights else 55.0
        raw_items.sort(key=lambda x: x[0])

        final_polys = []
        final_pieces = []
        final_ys = []

        for ym, pts, _ in raw_items:
            crop = _rotate_crop(img, pts)
            ch, cw = crop.shape[:2]
            if ch > median_h * 1.65:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
                _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                row_proj = ink[:, int(cw * 0.05):int(cw * 0.95)].sum(axis=1)
                lo, hi = int(ch * 0.30), int(ch * 0.70)
                split_y = lo + int(np.argmin(row_proj[lo:hi]))
                r = float(split_y) / max(1.0, float(ch))
                tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]
                mid_l = tl + r * (bl - tl)
                mid_r = tr + r * (br - tr)
                poly_top = np.array([tl, tr, mid_r, mid_l], dtype=np.float32)
                poly_bot = np.array([mid_l, mid_r, br, bl], dtype=np.float32)
                parts = []
                top_crop = crop[:split_y + 2, :]
                bot_crop = crop[max(0, split_y - 2):, :]
                top_active = int((row_proj[:split_y] > cw * 0.015).sum())
                bot_active = int((row_proj[split_y:] > cw * 0.015).sum())
                min_active = max(8, int(median_h * 0.25))
                if top_active >= min_active:
                    parts.append((ym - ch * 0.22, poly_top, top_crop))
                if bot_active >= min_active:
                    parts.append((ym + ch * 0.22, poly_bot, bot_crop))
                if not parts:
                    parts = [(ym, pts, crop)]
            else:
                parts = [(ym, pts, crop)]

            for sub_y, sub_poly, sub_crop in parts:
                if any(abs(sub_y - ey) < median_h * 0.42 for ey in final_ys):
                    continue
                final_ys.append(sub_y)
                final_polys.append(sub_poly)
                final_pieces.append(Image.fromarray(cv2.cvtColor(sub_crop, cv2.COLOR_BGR2RGB)))

        if not final_pieces:
            return []

        if torch is not None:
            with torch.inference_mode():
                texts = self.viet_recognizer.predict_batch(final_pieces)
        else:
            texts = self.viet_recognizer.predict_batch(final_pieces)

        out = []
        for poly, text in zip(final_polys, texts):
            cleaned = (text or "").strip()
            if cleaned:
                out.append([np.asarray(poly).tolist(), (cleaned, 0.95)])
        return out

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
        if self.viet_recognizer is not None:
            results = [self._detect_and_recognize_vietocr(img, width)]
        else:
            results = self.ocr_instance.ocr(img, cls=False)

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

        # Tính toán lề trang chuẩn hóa
        substantial_boxes = [b for b in raw_boxes if (b["bbox"][2] - b["bbox"][0]) > width * 0.25]
        if substantial_boxes:
            left_coords = sorted([b["bbox"][0] for b in substantial_boxes])
            dominant_left = left_coords[int(len(left_coords) * 0.20)]
            right_coords = sorted([b["bbox"][2] for b in substantial_boxes])
            dominant_right = right_coords[int(len(right_coords) * 0.85)]
        else:
            dominant_left = width * 0.15
            dominant_right = width * 0.85

        # Lọc sạch các mảnh chữ rác ở gáy sách mép trái (gutter noise)
        valid_boxes = []
        for b in raw_boxes:
            bx1, by1, bx2, by2 = b["bbox"]
            bw = bx2 - bx1
            # Các mảnh chữ thừa của trang bên cạnh luôn sát lề trái (bx2 < 120 và bw < 100)
            if bx2 < 120 and bw < 100:
                continue
            valid_boxes.append(b)
        raw_boxes = valid_boxes

        col_width = max(100, dominant_right - dominant_left)
        col_center = (dominant_left + dominant_right) / 2

        left_m = min([b["bbox"][0] for b in raw_boxes]) if raw_boxes else 0
        top_m = min([b["bbox"][1] for b in raw_boxes]) if raw_boxes else 0
        right_m = max([b["bbox"][2] for b in raw_boxes]) if raw_boxes else width
        bottom_m = max([b["bbox"][3] for b in raw_boxes]) if raw_boxes else height

        page_margins = {
            "top": round((top_m / height) * 100, 2),
            "bottom": round(((height - bottom_m) / height) * 100, 2),
            "left": round((left_m / width) * 100, 2),
            "right": round(((width - right_m) / width) * 100, 2),
        }

        for order, b in enumerate(raw_boxes, start=1):
            bx1, by1, bx2, by2 = b["bbox"]
            b_width = bx2 - bx1
            b_center = (bx1 + bx2) / 2
            txt = b["text"].strip()

            normalized_text = normalize_vietnamese_text(b["text"]).strip()
            if not normalized_text:
                continue

            is_centered = (abs(b_center - col_center) < col_width * 0.08) and (b_width < col_width * 0.80)
            is_uppercase = normalized_text.isupper() and len(normalized_text) > 2
            diff_from_left = bx1 - dominant_left
            is_indented = (diff_from_left >= 70) and (diff_from_left <= 180)

            # Ưu tiên nhận diện tiêu đề Chương/Mục hoặc dòng căn giữa in hoa ở đầu trang
            if (is_centered and (is_uppercase or order <= 2)) or normalized_text.startswith("Chương "):
                block_type = "heading"
            else:
                block_type = _block_type_from_yolo(b["bbox"])
                if block_type is None:
                    if is_centered and (is_uppercase or b["height_px"] > avg_height * 1.15):
                        block_type = "heading"
                    elif b["height_px"] > avg_height * 1.35 or (order == 1 and b["height_px"] > avg_height * 1.1):
                        block_type = "heading"
                    else:
                        block_type = "paragraph"

            style_meta = {}
            if block_type == "heading" or is_centered:
                style_meta["textAlign"] = "center"
                style_meta["bold"] = True
            elif "nhà xuất bản" in normalized_text.lower():
                style_meta["textAlign"] = "right" if bx2 > dominant_right - 40 else "center"
                style_meta["bold"] = True
            else:
                if is_indented:
                    style_meta["isIndented"] = True
                if is_uppercase or (len(normalized_text) > 3 and normalized_text[0].isdigit() and ". " in normalized_text[:8]):
                    style_meta["bold"] = True

            blocks.append({
                "id": f"b-{order}-{uuid.uuid4().hex[:6]}",
                "type": block_type,
                "order": len(blocks) + 1,
                "bbox": b["bbox"],
                "text": normalized_text,
                "confidence": round(b["confidence"], 3),
                "style": style_meta,
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
            "pageMargins": page_margins,
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
